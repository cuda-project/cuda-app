#include "../common/book.h"

#define N  (33 * 1024)

__global__ void add(int *a, int *b, int *c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < N){
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}


/**
 *  cd /home/tonye/cuda-workspace/cuda-app/cuda01/c05
 *  nvcc -o add_loop_lang_blocks add_loop_lang_blocks.cu
 */
int main(void){
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;

    a = (int *)malloc(N * sizeof(int));
    b = (int *)malloc(N * sizeof(int));
    c = (int *)malloc(N * sizeof(int));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    for(int i=0;i<N;i++){
        a[i] = i;
        b[i] = 2 * i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(N), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(N), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_c, c, N * sizeof(N), cudaMemcpyHostToDevice));

    add<<<128, 128>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(N), cudaMemcpyDeviceToHost));

    bool success = true;
    for(int i=0;i<N;i++){
        if((a[i] + b[i]) != c[i]){
            printf("Error: %d + %d != %d\n", a[i], b[i], c[i]);
            success = false;
        }
    }

    if(success){
        printf("We dic it!\n");
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    free(a);
    free(b);
    free(c);

    return 0;

}