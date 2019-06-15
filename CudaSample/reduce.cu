#include <stdio.h>
#include <iostream>

using namespace std;


__global__ void global_reduce_kernel(float * d_out, float * d_in){
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    // do reduction in global mem
    for(unsigned int s = blockDim.x / 2; s > 0; s >>=1){
        if(tid < s){
            d_in[myId] += d_in[myId + s]; //直接操作全局内存 d_in （内存一直在那里）
        }

        // 单个线程执行的过程中 对应的第一个小片段执行完后，等待block中所有的线性在第一个小片段都运行完后，
        // 再进入第二个小片段执行达到对折相加的效果
        // 备注: block线程还是原来的那些线程 只是在线程运行中每一个片段等待了一下
        __syncthreads();
    }

    // 如果是第0号线程 将对应的结果 输出给 线程对应block块， 做为第一个给block块的最终结果
    if(tid == 0){
        d_out[blockIdx.x] = d_in[myId];
    }
}


__global__ void shmem_reduce_kernel(float * d_out, const float * d_in){
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();  // make sure entire block is loaded!

    // do reduction is shread mem
    for(unsigned int s = blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }

        // 这一个批次所有循环都进行结束 一起进入下一个循环
        __syncthreads(); // make sure all adds at one stage are done!

    }

    // 第0个值，就是这一个线程块所有的求和
    // only thread 0 writes result for this block back to global mem
    if(tid ==0 ){
        d_out[blockIdx.x] = sdata[0];
    }
}


void reduce(float * d_out, float * d_intermediate, float * d_in, int size, bool usesSharedMemory){

    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = size / maxThreadsPerBlock;

    if(usesSharedMemory){
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
    }else{
        global_reduce_kernel<<<blocks, threads>>>(d_intermediate, d_in);
    }

    // now we're down to one block left, so reduce it
    threads = blocks;
    blocks = 1;
    if(usesSharedMemory){
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_intermediate);
    }else{
        global_reduce_kernel<<<blocks, threads>>>(d_out, d_intermediate);
    }


}

/**
 *
tonye@tonye-Lenovo-V4400:~/cuda-workspace/cuda-app/CudaSample$ nvcc -o reduce reduce.cu
nvcc warning : The 'compute_20', 'sm_20', and 'sm_21' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
tonye@tonye-Lenovo-V4400:~/cuda-workspace/cuda-app/CudaSample$ ./reduce 0
 #### deviceCount:1
 #### deviceCount:1
Using device 0:
GeForce GT 740M; global mem: 2101542912B; compute v3.5; clock: 1032500 KHz
 ####  ARRAY_SIZE:1048576
Running global reduce
average time elapsed: 1.404125
tonye@tonye-Lenovo-V4400:~/cuda-workspace/cuda-app/CudaSample$ ./reduce 1
 #### deviceCount:1
 #### deviceCount:1
Using device 0:
GeForce GT 740M; global mem: 2101542912B; compute v3.5; clock: 1032500 KHz
 ####  ARRAY_SIZE:1048576
Running reduce with shread mem
average time elapsed: 0.974607
 */
int main(int argc, char **argv) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    std::cout << "#### deviceCount:" << deviceCount << std::endl;

    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;

    if (cudaGetDeviceProperties(&devProps, dev) == 0) {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d KHz\n",
               devProps.name,
               (int) devProps.totalGlobalMem,
               (int) devProps.major, (int) devProps.minor,
               (int) devProps.clockRate);

    }

    const int ARRAY_SIZE = 1 << 20;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
    printf("#### ARRAY_SIZE:%d \n", ARRAY_SIZE);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    float sum = 0.0f;
    for(int i=0;i<ARRAY_SIZE;i++){
        // generate random float in [-1.0f, 1.0f]
        h_in[i] = -1.0f + (float)random()/((float)RAND_MAX/2.0f);
        sum += h_in[i];
    }

    printf("#### CPU sum求和：%f \n", sum);

    // declare GPU memory pointers
    float * d_in, * d_intermediate, * d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_intermediate, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, sizeof(float));

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    int whichKernel = 0;
    if(argc == 2){
        whichKernel = atoi(argv[1]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //launch the kernel
    switch(whichKernel){
        case 0:
            printf("Running global reduce \n");
            cudaEventRecord(start, 0);
            //修改了全局内存  如果进行幂等操作 GPU执行reduce的值 和 cpu计算sum的值就 不一致
            for(int i=0;i<100;i++){
                reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, false);
            }
            cudaEventRecord(stop, 0);
            break;
        case 1:
            printf("Running reduce with shread mem\n");
            cudaEventRecord(start, 0);
            //因为使用了 共享内存 可以进行幂等操作  多次GPU执行reduce的值 和 cpu计算sum的值 一致
            for(int i=0;i<100;i++){
                reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
            }
            cudaEventRecord(stop, 0);
            break;
        default:
            fprintf(stderr, "error: ran no kernel\n");
            exit(EXIT_FAILURE);
    }

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    elapsedTime /= 100.0f;      // 100 trials

    // copy back the sum from GPU
    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("average time elapsed: %f\n", elapsedTime);

    printf("#### GPU reduce求和: %f\n", h_out);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);

    return 0;


}