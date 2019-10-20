#include "../common/book.h"

int main(void){
    cudaDeviceProp prop;
    int dev;
    HANDLE_ERROR(cudaGetDevice(&dev));
    printf("ID of current CUDA device: %d\n", dev);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minjor = 3;
    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
    printf("ID of CUDA device closet to revision 1.3: %d\n");
    HANDLE_ERROR(cudaSetDevice(dev));
}