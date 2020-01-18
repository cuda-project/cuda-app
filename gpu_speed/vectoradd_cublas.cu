#include <stdio.h>
#include <cublas_v2.h>
#define DATATYPE float

/*
cd /home/tonye/cuda-workspace/cuda-app/gpu_speed
nvcc vectoradd_cublas.cu -o vectoradd_cublas -lcublas
./vectoradd_cublas
 */
int main(int argc, char ** argv){
    const int ARRAY_SIZE = 10;
    const int ARRAY_BYTIES = sizeof(DATATYPE) * ARRAY_SIZE;

    DATATYPE a[ARRAY_SIZE], b[ARRAY_SIZE], c[ARRAY_SIZE];

    for(int i =0;i<ARRAY_SIZE;i++){
        a[i] = DATATYPE(i);
        b[i] = DATATYPE(i);
    }\

    DATATYPE * d_a, * d_b;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMalloc((void**)&d_a, ARRAY_BYTIES);
    cudaMalloc((void**)&d_b, ARRAY_BYTIES);
    DATATYPE alpha = 1.0;
    cublasSetVector(ARRAY_SIZE, sizeof(DATATYPE), a, 1, d_a, 1);
    cublasSetVector(ARRAY_SIZE, sizeof(DATATYPE), b, 1, d_b, 1);
    cublasSaxpy_v2(handle, ARRAY_SIZE, &alpha, d_a, 1, d_b, 1);
    cublasGetVector(ARRAY_SIZE, sizeof(DATATYPE), d_b, 1, c, 1);


    for(int i=0;i<ARRAY_SIZE;i++){
        printf("%f", c[i]);
        printf(((i%4)!=3)?"\t":"\n");
    }

    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cublasDestroy(handle);
}