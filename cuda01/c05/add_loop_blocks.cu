#include "../common/book.h"

#define N 10

__global__ void add(int *a, int *b, int *c){
    int tid = threadIdx.x;
    if(tid < N){
        c[tid] = a[tid] + b[tid];
    }
}

/**
 *  cd /home/tonye/cuda-workspace/cuda-app/cuda01/c05
 *  nvcc -o add_loop_blocks add_loop_blocks.cu
 */
int main(void){
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    //在GPu上分配内存
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N  *sizeof(int)));

    //在cpu赋值
    for(int i=0;i<N;i++){
        a[i] = i;
        b[i] = i * i;
    }

    //从cpu复制GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice));

    add<<<1, N>>>(dev_a, dev_b, dev_c);

    //从gpu复制到cpu
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    //显示结果
    for(int i=0;i<N;i++){
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    //释放GPU内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}