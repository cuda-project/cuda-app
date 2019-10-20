#include "../common/book.h"

__global__ void kernel(void){

}

/**
 * nvcc -o simple_kernel simple_kernel.cu
 * ./simple_kernel.cu
 * @return
 */
int main(void){

    kernel<<<1, 1>>>();

    printf("hello world!\n");
    return 0;
}