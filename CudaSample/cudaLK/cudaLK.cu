#include "cudaLK.h"
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

__global__ void convertToGrey(unsigned char *d_in, float *d_out, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        d_out[idx] = d_in[idx*3]*0.229f + d_in[idx*3+1]*0.587f + d_in[idx*3+2]*0.114f;
    }
}

void cudaLK::checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err){
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void cudaLK::run(unsigned char *prev, unsigned char *cur, int _w, int _h) {
    w = _w;
    h = _h;
    initMem();

    int blocks1D = (w*h)/256 + (w*h%256?1:0);

    int start = getTimeNow();
    int s;

    //Copy image to GPU
    s = getTimeNow();
    cudaMemcpy(gpu_img_prev_RGB, prev, w*h*3, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_img_cur_RGB, cur, w*h*3, cudaMemcpyHostToDevice);
    checkCUDAError("start");

    printf("Copying 2 images from CPU to GPU: %d ms\n", getTimeNow()-s);

    // RGB -> grey
    s = getTimeNow();
    convertToGrey<<<blocks1D, 256>>>(gpu_img_prev_RGB, gpu_img_pyramid_prev[0], w*h);
    convertToGrey<<<blocks1D, 256>>>(gpu_img_cur_RGB, gpu_img_pyramid_cur[0], w*h);
    cudaThreadSynchronize();
    checkCUDAError("convertToGrey");
    printf("Converting from RGB to greyscala: %d ms\n", getTimeNow()-s);


    char *res_prev;
    char *res_cur;
    res_prev = (char *)malloc(w * h * sizeof(char));
    res_cur = (char *)malloc(w * h * sizeof(char));
    cudaMemcpy(res_prev, gpu_img_pyramid_prev[0], w*h, cudaMemcpyDeviceToHost);
    cudaMemcpy(res_cur, gpu_img_pyramid_cur[0], w*h, cudaMemcpyDeviceToHost);

    IplImage *gray_img1 = cvCreateImage(cvSize(w, h), 8, 1);
    IplImage *gray_img2 = cvCreateImage(cvSize(w, h), 8, 1);

    gray_img1->imageData = res_prev;
    gray_img2->imageData = res_cur;


    cvSaveImage("image/gray_img1.png", gray_img1);
    cvSaveImage("image/gray_img2.png", gray_img2);


}

void cudaLK::initMem(){
    cudaMalloc((void**)&gpu_img_prev_RGB, sizeof(char)*w*h*3);
    cudaMalloc((void**)&gpu_img_cur_RGB, sizeof(char)*w*h*3);
    cudaMalloc((void**)&gpu_img_pyramid_prev[0], sizeof(char)*w*h);
    cudaMalloc((void**)&gpu_img_pyramid_cur[0], sizeof(char)*w*h);

}

cudaLK::cudaLK() {

}

cudaLK::~cudaLK() {
    for(int i =0;i<LEVELS; i++){
        cudaFree(gpu_img_pyramid_prev[i]);
        cudaFree(gpu_img_pyramid_cur[i]);
    }

}