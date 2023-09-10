#pragma once

#include <opencv2/core.hpp>

using namespace cv;

#ifndef CPUTOOLS_H
#define CPUTOOLS_H

template <typename T, size_t N> 
void convCPU(Mat &srcIm, Mat &dstIM, T (&filter)[N], int transpose);
#endif