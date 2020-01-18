#include "../common/book.h"

__device__ addem(int a, int b){
    return a + b;
}

__global__ add(int a, int b, int *c){
    *c = addem(a, b);
}

int main(void){
    int c;
    int *dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

    add<<<1, 1>>>(2, 7, dev_c);

    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

    printf("2 + 7 = %d\n", c);

    HANDLE_ERROR(cudaFree(dev_c));


}