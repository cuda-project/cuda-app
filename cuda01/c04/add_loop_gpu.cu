#include "../common/book.h"

#define N 10;

__global__ add(int *a, int *b, int *c){
    int tid = blockIdx.x;
    while(tid < N){
        c[tid] = a[tid] + b[tid];
    }
}

/**
 * cd /home/tonye/cuda-workspace/cuda-app/cuda01/c04
 * nvcc -o add_loop_gpu add_loop_gpu.cu
 *
 */
int main(void){
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    for(int i=0;i<N; i++){
        a[i] = -i;
        b[i] = i * i;
    }

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(int)));

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(N), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(N), cudaMemcpyHostToDevice));


    add<<<N, 1>>>(dev_a, dev_b, dev_c);


    HANDLE_ERROR(cudaMemcpy(c, dev_c, N*sizeof(N), cudaMemcpyDeviceToHost));


    for(int i=0;i<N;i++){
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    return 0;


}