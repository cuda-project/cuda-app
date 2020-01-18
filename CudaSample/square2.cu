#include <stdio.h>

__global__ void square(float * d_in, float * d_out){
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f * f;
}

int main(int arg, char** argv){

    const int ARRAY_SIZE = 8;
    const int ARRAY_BITES = ARRAY_SIZE * sizeof(float);

    float h_in[ARRAY_SIZE];
    float h_out[ARRAY_SIZE];

    for(int i=0;i<ARRAY_SIZE;i++){
        h_in[i] = float(i);
    }

    float * d_in;
    float * d_out;

    cudaMalloc((void**)&d_in, ARRAY_BITES);
    cudaMalloc((void**)&d_out, ARRAY_BITES);

    cudaMemcpy(d_in, h_in, ARRAY_BITES, cudaMemcpyHostToDevice);
    square<<<1, ARRAY_SIZE>>>(d_in, d_out);

    cudaMemcpy(h_out, d_out, ARRAY_BITES, cudaMemcpyDeviceToHost);

    for(int i=0;i<ARRAY_SIZE; i++){
        printf("%f", h_out[i]);
        printf(((i%4)!=3?"\t":"\n"));
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;

}