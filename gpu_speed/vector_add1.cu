#include <stdio.h>
#include "../include/gputimer.h"

//单block单thread向量加
__global__ void vector_add_gpu_1(float *d_a, float * d_b, float *d_c, int n){
    for(int i=0;i<n;i++){
        d_c[i] = d_a[i] + d_b[i];
    }
}


// 单block多thread向量加
__global__ void vector_add_gpu_2(float *d_a, float *d_b, float *d_c, int n){
    int tid = threadIdx.x;
    const int t_n = blockDim.x;
    // 每个线程完成1次向量加法后索引tid根据线程总数(blockDim.x)进行跳步
    while(tid<n){
        d_c[tid] = d_a[tid] + d_b[tid];
        tid += t_n;
    }
}

//多block多thread向量加
__global__ void vector_add_gpu_3(float *d_a, float *d_b, float *d_c, int n){
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int t_n = gridDim.x * blockDim.x;
    int tid = bidx * blockDim.x + tidx;
    while(tid<n){
        d_c[tid] = d_a[tid] + d_b[tid];
        tid += t_n;
    }
}

/*
cd /home/tonye/cuda-workspace/cuda-app/gpu_speed
nvcc vector_add1.cu -o vector_add1
./vector_add1
 */
int main(int argc, char** argv){

    const int ARRAY_SIZE = 65536;
    const int ARRAY_BYTES = sizeof(ARRAY_SIZE) * ARRAY_SIZE;

    float a[ARRAY_SIZE], b[ARRAY_SIZE], c[ARRAY_SIZE];

    for(int i=0;i<ARRAY_SIZE;i++){
        a[i] = float(i);
        b[i] = float(i);
    }

    float *d_a, *d_b, *d_c;

    GpuTimer timer;

    cudaMalloc((void **)&d_a, ARRAY_BYTES);
    cudaMalloc((void **)&d_b, ARRAY_BYTES);
    cudaMalloc((void **)&d_c, ARRAY_BYTES);

    cudaMemcpy(d_a, a, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, ARRAY_BYTES, cudaMemcpyHostToDevice);

    timer.Start();
    //vector_add_gpu_1<<<1, 1>>>(d_a, d_b, d_c, ARRAY_SIZE);
    //vector_add_gpu_2<<<1, 5>>>(d_a, d_b, d_c, ARRAY_SIZE);
    vector_add_gpu_3<<<100,100000>>>(d_a, d_b, d_c, ARRAY_SIZE);
    timer.Stop();

    printf("time elapsed: %f\n", timer.Elapsed());

    cudaMemcpy(c, d_c, ARRAY_BYTES, cudaMemcpyDeviceToHost);

//    for(int i=0;i<ARRAY_SIZE;i++){
//        printf("%f", c[i]);
//        printf(((i%4)!=3)?"\t":"\n");
//    }

    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);



    return 0;
}