#include <iostream>
#include <opencv2/core.hpp>

#include "../include/cputools.h"

using namespace cv;
using namespace std;

template <typename T, size_t N>
void cputools<T, N>::convCPU(Mat &srcIm, Mat &dstIM, T (&filter)[N], int transpose)
{
    int temp, row_offset, col_offset;
    int halo = N / 2;
    if (N % 2 == 0)
        cout << "The size of the filter must be and odd number." << endl;

    // go over each row and col in the image
    for (int row = 0; row < srcIm.rows; row++)
        for (int col = 0; col < srcIm.cols; col++)
        {
            temp = 0;

            // go over each row and col in the filter.  anchor_offsets account for how 
            // far we need to move the image index to account for the halo size
            for (int i = 0; i < N; i++)
            {
                row_offset = row - halo + i;
                col_offset = col - halo + i; 
                // cout << "row: " << row << " col: " << col << " row_offset: " << row_offset << " col_offset: " << col_offset << " i: " << i << endl; 
                // assume image is padded with zeros.  We ignore the computation 
                // in the padded area when anchor offsets are negative.
                if (row_offset >= 0 && row_offset <= srcIm.rows && col_offset >= 0 && col_offset <= srcIm.cols)
                    if (!transpose)
                    {
                        temp += srcIm.at<uchar>(row_offset, col) * filter[i];
                    }
                    else
                    {
                        temp += srcIm.at<uchar>(row, col_offset) * filter[i];
                    } 
            }
        dstIM.at<uchar>(row, col) = temp;
        }
}
