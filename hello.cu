#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU(void){
    printf("Hello World from GPU!\n");
}

int main(void) {
    //nvcc hello.cu -o hello
    //printf("Hello World from CPU!\n");

    // nvcc hello.cu -o hello -Wno-deprecated-gpu-targets
    cudaSetDevice(0);
    //一个kernel是由一组线程执行，所有线程执行相同的代码。上面一行三对尖括号中的1和10 表明了该function将有10个线程
    helloFromGPU <<<1, 1000>>>();
    //用来显式的摧毁清理CUDA程序占用的资源
    cudaDeviceReset();

    return 0;

}