
#include "../common/book.h"

__global__ add(int a, int b, int *c){


}

int main(void){

    int c;
    int *dev_c;
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

    add<<<1, 1>>>(2, 7, dev_c);

    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

    printf("2 + 7 = %d\n", c);

    cudaFree( dev_c);

    printf("hello world!\n");
    return 0;
}