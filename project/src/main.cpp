#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

#include "../include/cputools.h"

using namespace cv;
using namespace std;

void loadGrayImage(String filename, Mat &image)
{
    image = imread(filename, IMREAD_GRAYSCALE); // Read the file
    
    if(image.empty())
    { // Check for invalid input
        cout << "Could not open or find the image" << endl;
    }
    else{
        cout << "The image has " << image.rows << " rows and "  << image.cols << " columns." << endl;
    }
}

void saveImage(String filename, Mat image)
{
    if(!image.empty())
    {
        // cout << "The image has " << image.rows << " rows and "  << image.cols << " columns." << endl;
        imwrite(filename, image);
    }

}

int getElementVal(Mat &image, int row, int col)
{
    if(image.type() == 0)
    {
        return static_cast<int>(image.at<uchar>(row, col));
    }
    else
    {
        cout << "Can't return type! This is not as 8-bit image." << endl;
        return -1;
    }

}



void getGaussianKernel(double kernel[])
{
     int gaussVals[5] =  {1, 4, 6, 4, 1};
     double norm = 1.0/16.0;

     for (int i = 0; i < 5; i++)
     {
        kernel[i] = gaussVals[i] * norm;
     }
}

int main( int argc, char** argv ){


    // load the image
    String imageName("mc-escher-01.jpg");
    if( argc > 1)
    {
        imageName = argv[1];
    }
    
    Mat image;
    loadGrayImage(imageName, image);
    
    
    // first apply gaussian blurring
    double gaussianKernel[5];
    Mat xfilteredIm(image.rows, image.cols, 0), smoothedIm(image.rows, image.cols, 0);
    
    getGaussianKernel(gaussianKernel);
    
    convCPU(image, xfilteredIm, gaussianKernel, false);
    convCPU(xfilteredIm, smoothedIm, gaussianKernel, true);

    // now apply a sobel filter for edge detection
    int sobelKernel[3] = {1, 0, -1};
    Mat xEdges(image.rows, image.cols, 0), yEdges(image.rows, image.cols, 0), mag(image.rows, image.cols, 0), angle(image.rows, image.cols, 0), final;
    
    convCPU(smoothedIm, xEdges, sobelKernel, false);
    convCPU(smoothedIm, yEdges, sobelKernel, true);

    final = xEdges + yEdges;
    cartToPolar(xEdges, yEdges, mag, angle);
    saveImage(imageName + "_gray.jpg", image);
    saveImage(imageName + "_smooth.jpg", smoothedIm);
    saveImage(imageName + "_yfiltered.jpg", yEdges);
    saveImage(imageName + "_xfiltered.jpg", xEdges);
    saveImage(imageName + "_final.jpg", final);

    return 0;
}

