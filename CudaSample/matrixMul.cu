#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_WIDTH 16

__global__ void MatrixMul(float *Md, float *Nd, float *Pd, const int WIDTH){

    unsigned int col = TILE_WIDTH * blockIdx.x + threadIdx.x;
    unsigned int row = TILE_WIDTH * blockIdx.y + threadIdx.y;

    for(int k =0;k<WIDTH;k++){
        Pd[row*WIDTH+col] += Md[row*WIDTH+k]*Nd[k*WIDTH+col];
    }

}

__global__ void MatrixMulSh(float *Md, float *Nd, float *Pd, const int WIDTH){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int Pervalue = 0;

    unsigned  int col = TILE_WIDTH * blockIdx.x + threadIdx.x;
    unsigned int row = TILE_WIDTH * blockIdx.y + threadIdx.y;

    for(int m = 0; m<WIDTH/TILE_WIDTH;m++){
        Mds[threadIdx.y][threadIdx.x] = Md[row*WIDTH+(m * TILE_WIDTH+threadIdx.x)];
        Nds[threadIdx.y][threadIdx.x] = Nd[(m*TILE_WIDTH+threadIdx.y)*WIDTH + col];
        __syncthreads();
        for(int k=0;k<TILE_WIDTH;k++){
            Pervalue += Mds[threadIdx.x][k]*Nds[threadIdx.y][k];

        }
        __syncthreads();
    }
    Pd[row * WIDTH + col] = Pervalue;
}

int main(){

    printf("hello\n");

    const int WIDTH = 4;
    //定义CPU上的矩阵
    float array1_h[WIDTH][WIDTH], array2_h[WIDTH][WIDTH];
    float result_array_h[WIDTH][WIDTH];

    //定义GPU上的矩阵
    float *array1_d, *array2_d, *result_array_d;
    int i,j;

    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //初始化矩阵
    for(i=0;i<WIDTH;i++){
        for(j=0;j<WIDTH;j++){
            array1_h[i][j] = 1;
            array2_h[i][j] = 2;
        }
    }

    //在GPU上给Array分配空间
    int size = WIDTH * WIDTH * sizeof(int);
    cudaMalloc((void**) &array1_d, size);
    cudaMalloc((void**) &array2_d, size);

    //将CPU上的数据传输到GPU显存
    cudaMemcpy(array1_d, array1_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(array2_d, array2_h, size, cudaMemcpyHostToDevice);

    //在GPU上存储结果的矩阵分配空间
    cudaMalloc((void**) &result_array_d, size);

    //定义kernel函数的执行设置
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH,1);
    dim3 dimGrid(WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH, 1);

    //执行kernel函数
    cudaEventRecord(start, 0);
    MatrixMul<<<dimGrid, dimBlock>>>(array1_d, array2_d, result_array_d, WIDTH);
    //MatrixMulSh<<<dimGrid, dimBlock>>>(array1_d, array2_d, result_array_d, WIDTH);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("elapsedTime:%f\n",elapsedTime);

    cudaMemcpy(result_array_h, result_array_d, size, cudaMemcpyDeviceToHost);
    printf("result_array_h: %d\n", result_array_h);

    cudaFree(array1_d);
    cudaFree(array2_d);
    cudaFree(result_array_d);
    return 0;
}