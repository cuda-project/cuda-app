#include <stdio.h>
#include "cudaLK.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>

/**
 * cd  /home/tonye/cuda-workspace/cuda-app/CudaSample/cudaLK
 */
int main(int argc, char **argv){
    if(argc != 3){
        printf("./cudaLK img1.png img2.png\n");
        return 0;
    }

    IplImage *img1 = cvLoadImage(argv[1]);
    IplImage *img2 = cvLoadImage(argv[2]);

    if(!img1 || !img2){
        printf("Error loading images\n");
        return 1;
    }

    cudaLK LK;

    LK.run((unsigned char*)img1->imageData, (unsigned char*)img2->imageData, img1->width, img1->height);


}