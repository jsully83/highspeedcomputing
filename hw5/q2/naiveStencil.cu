
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

#define n 4

/**
 * @brief Copied from cuda_samples/Common/helper_cuda.h
 * 
 */
static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief initialize arrays on CPU
 * 
 * @param b 
 */
void initArrayCPU(float b[n][n][n]){
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++)
                {b[i][j][k] = k;}
}

/**
 * @brief initialize array b for the device by copying from array a on the host
 * 
 * @param a original array on the host
 * @param b destination array for the device
 */

void initArrayDevice(float a[n][n][n], float *b){
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++)
                b[i*n*n + j*n + k] = a[i][j][k];
}
/**
 * @brief check that the values in the array on the device match the values in the array on the host
 * 
 * @param a 
 * @param b 
 */
void checkArray(float a[n][n][n], float *b){
    int errorNum = 0;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++)
                if(a[i][j][k] != b[i*n*n + j*n + k]){
                    errorNum++;
                    // printf("Error at i = %i, j = %i, k = %i\n", i, j, k);
                    // printf("a[%i][%i][%i] = %f\n", i, j, k, a[i][j][k]);
                    // printf("b[%i][%i][%i] = %f\n", i, j, k, b[i*n*n + j*n + k]);
                }
                printf("There were: %i errors.\n", errorNum);
}

void cpuStencil(float a[n][n][n], float b[n][n][n]){
    for (int i=1; i<n-1; i++)
        for (int j=1; j<n-1; j++)
            for (int k=1; k<n-1; k++) {
                a[i][j][k]= 0.8 * (b[i-1][j][k] + b[i+1][j][k] + 
                                   b[i][j-1][k] + b[i][j+1][k] + 
                                   b[i][j][k-1] + b[i][j][k+1]);
    }
}

__device__ int getAdjacent(int i, int j, int k){
    return (i*n*n + j*n + k);
}

__global__ void naiveStencil(float *d_a, float *d_b){


    

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // if(threadIdx.x == 0 || threadIdx.x == n - 1 || threadIdx.y == 0 || threadIdx.y == n - 1 || threadIdx.z == 0 || threadIdx.z == n - 1)
        printf("i=%i, j=%i, k=%i\n", i, j, k);
        // return;
    
    int tid = i*n*n + j*n + k;
    printf("tid = %i\n", tid);
    d_a[tid] = 0.8 * (d_b[getAdjacent(i-1,j,k)] + d_b[getAdjacent(i+1,j,k)] + 
                      d_b[getAdjacent(i,j-1,k)] + d_b[getAdjacent(i,j+1,k)] + 
                      d_b[getAdjacent(i,j,k-1)] + d_b[getAdjacent(i,j,k+1)]);
                    
    return;

}

int main(int argc, char* argv[]){
    
    float *h_a, *h_b;
    float *d_a, *d_b;
    float a[n][n][n], b[n][n][n];
    // size_t tensorSize = n * n * n * sizeof(float);
    size_t tensorSize = n * n * sizeof(float);

    // allocate memory on host
    checkCudaErrors(cudaMallocHost(&h_a, tensorSize));
    checkCudaErrors(cudaMallocHost(&h_b, tensorSize));

    // allocate memory on device
    checkCudaErrors(cudaMalloc(&d_a, tensorSize));
    checkCudaErrors(cudaMalloc(&d_b, tensorSize));

    // init arrays
    initArrayCPU(b);
    initArrayDevice(b, h_b);
    checkArray(b, h_b);
    
    cpuStencil(a, b);
    
    // ============  CUDA SETUP ============
    // device event objects
    float gpu_elapsed_time_ms;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // copy arrays to device
    checkCudaErrors(cudaMemcpy(d_a, h_a, tensorSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, tensorSize, cudaMemcpyHostToDevice));

    const unsigned int blockSize = 2;
    const unsigned int gridSize = (n + blockSize - 1) / blockSize;
    
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(gridSize, gridSize, gridSize);

    printf("A block has %i threads.\n", blockSize);
    printf("A grid has %d blocks.\n", gridSize);
    printf("There are a total of %i threads.\n",blockSize*gridSize);  
    

    // ============  KERNEL LAUNCH ============
    checkCudaErrors(cudaEventRecord(start, 0));

    naiveStencil<<<dimGrid, dimBlock>>>(d_a, d_b); 
    getLastCudaError("Kernel Error");
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_a, d_a, tensorSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));

    checkArray(a, h_a);
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop));

    printf("Elapsed time: %f ms.\n", gpu_elapsed_time_ms/10);
    
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    getLastCudaError("Last Error");
    return 0;
}

