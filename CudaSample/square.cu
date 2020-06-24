#include <stdio.h>

__global__ void square(float* d_out, float* d_in){
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f * f;
}


/**
 * cd ./CudaSample
 * nvcc square.cu -o square -Wno-deprecated-gpu-targets
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv){

    const int ARRAY_SIZE = 128;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // 1.CPU分配内存空间
    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    for(int i=0;i<ARRAY_SIZE;i++){
        h_in[i] = float(i);
    }

    float h_out[ARRAY_SIZE];

    // 2.GPU分配内存空间
    // declare GPU memory pointers
    float* d_in;
    float* d_out;

    // allocate GPU memory
    cudaMalloc((void**) &d_in, ARRAY_BYTES);
    cudaMalloc((void**) &d_out, ARRAY_BYTES);

    // 3.将CPU数据复制到GPU中
    // transfer the array to GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // 4.加载kernel给GPU做计算
    // launch the kernel   1 表示创建多少个线程块  ARRAY_SIZE 表示每个线程块运行多少个线程
    square<<<1, ARRAY_SIZE>>>(d_out, d_in);

    // 5.将GPU运算结果复制到CPU中
    // copy back the result array to the GPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print out the resulting array
    for(int i=0;i<ARRAY_SIZE;i++){
        printf("%f", h_out[i]);
        printf(((i%4) != 3)? "\t" : "\n");
    }

    // 6.释放GPU分配的内存空间
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;


}