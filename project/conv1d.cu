#include <iostream>
#include <stdlib.h>
#include <assert.h>

// Function to check for CUDA errors
#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__
void convolution(int *array, int *mask, int *result, int n, int m)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = m / 2;
    int start = threadId - offset;
    
    int temp = 0;
     
    // each thread assigned to a element of the output array computes 
    // the solution of the convolution of the input array with the mask.
    // each thread 
    for(int i=0; i < m; i++){
        if((start + i >= 0 ) && (start + i < n)){
            temp += array[start + i] * mask[i];
        }
    }

    // printf("threadId: %d", threadId);
    result[threadId] = temp;

}

void verify_result(int *array, int *mask, int *result, int n, int m){
    int offset = m / 2;
    int temp;
    int start;

    for(int i=0; i < n; i++){
        temp = 0;
        start = i - offset;
        for(int j=0; j < m; j++){
            if((start + j >= 0 ) && (start + j < n)){
                temp += array[start + j] * mask[j];
            }
        }
        assert(temp == result[i]);
    }
}


int main(int argc, char *argv[])
{
    // 1d array
    int n = 1 << 20;
    int bytes_n = n * sizeof(int);
    int *h_array = (int *)malloc(bytes_n);
    int *h_result = (int *)malloc(bytes_n);

    for (int i = 0; i < n; i++)
        h_array[i] = rand() % 100;
        

    int m = 7;
    int bytes_m = m * sizeof(int); 
    int *h_mask = (int *)malloc(bytes_m);

    for (int i = 0; i < m; i++)
        h_mask[i] = rand() % 10;



    // device
    int *d_array, *d_mask , *d_result;
    checkCudaError(cudaMalloc(&d_array, bytes_n));
    checkCudaError(cudaMalloc(&d_mask, bytes_m));
    checkCudaError(cudaMalloc(&d_result, bytes_n));

    // std::cout << "1" << std::endl;
    checkCudaError(cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_mask, h_mask, bytes_m, cudaMemcpyHostToDevice));

    const unsigned int blockSize = 256;
    const unsigned int gridSize = (n * blockSize  - 1) / blockSize;

    
    convolution<<<gridSize, blockSize>>>(d_array, d_mask, d_result, n, m);  
    checkCudaError(cudaPeekAtLastError());
    // checkCudaError(cudaDeviceSynchronize());

    
    checkCudaError(cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost));

    verify_result(h_array, h_mask, h_result, n, m);

    std::cout << "COMPLETED SUCCESSFULLY" << std::endl;

return 0;

}
