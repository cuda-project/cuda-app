#include "../common/book.h"

#define N (33 * 1024)
#define imin(a, b) (a>b?a:b)
#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);

__global__ void dot(float *a, float *b, float *c){
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while(tid < N){
        temp += a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // 设置cache中相应位置的值
    cache[cacheIndex] = temp;

    __syncthreads();

    //归约
    int i = blockDim.x/2;
    while(i != 0){
        if(cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];

        }
        __syncthreads();
        i /= 2;
    }

    if(cacheIndex == 0){
        c[blockIdx.x] = cache[0];
    }

}


/**
 *  cd /home/tonye/cuda-workspace/cuda-app/cuda01/c05
 *  nvcc -o dot dot.cu
 */
int main(void){
    float *a, *b, c_cpu, c_gpu, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

    c_cpu = 0;
    for(int i =0;i<N; i++){
        a[i] = i;
        b[i] = i * 2;
        c_cpu += i*i*2;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    // 在cpu上完成最终的求和运算
    c_gpu = 0;
    for(int i =0;i<blocksPerGrid;i++){
        c_gpu += partial_c[i];
    }


    printf("Does GPU value %.6g = %.6g?\n", c_gpu, 2*sum_squares((float)(N-1)));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    free(a);
    free(b);
    free(partial_c);


}