
#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define N 32
#define K 5
#define RADIUS (K / 2) 
#define T_BLOCK_SIZE 16
#define TILE_SIZE_OUT (T_BLOCK_SIZE - (K - 1))
#define T_GRID_SIZE ((N - 1) / TILE_SIZE_OUT + 1)

#define N_TRIALS 1

__constant__ float kernel[K];

namespace cg = cooperative_groups;

// Function to check for CUDA errors
#define checkCudaError(err) __checkCudaErrors(err, __FILE__, __LINE__)

void __checkCudaErrors(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
            (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void print_array(int *array, int size){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            // std::cout << i * size + j << ": "<<array[i * size + j] << " ";
            std::cout <<array[i * size + j] << ", ";
        }
        std::cout << std::endl;
    }
}

void sep_kernel(float *h_kernel) {
    float sum = 0.0;
    float stdDev = 1.0;

    for (int x = -RADIUS; x <= RADIUS; x++) {
            float value = exp(-((x * x)) / (2 * stdDev * stdDev)) / sqrt(2* M_PI * stdDev * stdDev);
            h_kernel[(x + RADIUS)] = value;
            sum += value;
        
    }

    // Normalize the Kernel
    for (int i = 0; i < K; ++i) {
        h_kernel[i] /= sum;

    }
}


bool compare_results(float *cpu, float *device){
    float tol = 1e-3;
    for(int i = 0; i < N * N; i++){
        if(abs(cpu[i] - device[i]) > tol){
            return false;
        }
    }
    return true;
}

__global__ void rowConvolution(int *d_input, float *d_output) {
    // tile index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output index
    int row_out = blockIdx.y * TILE_SIZE_OUT + ty;
    int col_out = blockIdx.x * TILE_SIZE_OUT + tx;

    // input index
    int row_in = row_out;
    int col_in = col_out - RADIUS;
    
    __shared__ float input_sh[T_BLOCK_SIZE][T_BLOCK_SIZE];
    
    if((row_in >= 0) && (row_in < N) && (col_in >= 0) && (col_in < N)){

        for(int i = 0; i < T_BLOCK_SIZE; i++){
            innput[ty][tx] = d_input[];




        input_sh[ty][tx] = d_input[row_in * N + col_in];
        printf("1: blockidy: %i, blockidx: %i, ty: %i, tx: %i, row_in: %i, col_in: %i, row_out: %i, col_out: %i, %d = input[%i]\n", blockIdx.y, blockIdx.x, ty, tx, row_in, col_in, row_out, col_out, d_input[row_in * N + col_in], row_in * N + col_in); 
    }

    else{

        input_sh[ty][tx] = 0.0f;
    }

    // printf("1: blockidy: %i, blockidx: %i, ty: %i, tx: %i, row_in: %i, col_in: %i, row_out: %i, col_out: %i, %d = input[%i]\n", blockIdx.y, blockIdx.x, ty, tx, row_in, col_in, row_out, col_out, d_input[row_in * N + col_in], row_in * N + col_in);


    __syncthreads();

    printf("2: blockidy: %i, blockidx: %i, ty: %i, tx: %i, row_in: %i, col_in: %i, %f = input[%i]\n", blockIdx.y, blockIdx.x, ty, tx, row_in, col_in, input_sh[ty][tx], row_in * N + col_in);
    
    float temp = 0.0f;
    if((ty < TILE_SIZE_OUT) && (tx < TILE_SIZE_OUT)){
        #pragma unroll
        for(int i = 0; i < K; i++){
                temp += input_sh[ty][i+tx] * kernel[i * K];
            }
        
        if((row_out < N) && (col_out < N)){
            d_output[row_out * N + col_out] = temp;
        }
    }
}


int main(int argc, char *argv[]) {

    // Size in bytes for matrix and kernel
    size_t matrixSize = N * N * sizeof(int);
    size_t sepKernelSize = K * sizeof(float);


    //use unified memory
    int *array = (int *)malloc(matrixSize);
    float *unifiedResult;

    checkCudaError(cudaMallocManaged(&array, matrixSize));
    checkCudaError(cudaMallocManaged(&unifiedResult, matrixSize));
    
    // initilize the matrix and kernel
    for (int i = 0; i < N * N; i++) {
        array[i] = rand() % 255;
    }

    // print_array(array, N);

    float *sep_kernel = (float *)malloc(sepKernelSize);
    checkCudaError(cudaMemcpyToSymbol(kernel, sep_kernel, sepKernelSize));

    checkCudaError(cudaMemPrefetchAsync(array, matrixSize, 0));
    checkCudaError(cudaMemPrefetchAsync(unifiedResult, matrixSize, 0));

    dim3 tiledBlockDim(T_BLOCK_SIZE, T_BLOCK_SIZE);
    dim3 tiledGridDim(T_GRID_SIZE, T_GRID_SIZE, 1);

    // // execute cuda command
    for (int i = 0; i < N_TRIALS; i++){
        rowConvolution<<<tiledGridDim, tiledBlockDim>>>(array, unifiedResult);
        checkCudaError(cudaPeekAtLastError());
        checkCudaError(cudaDeviceSynchronize());
    }

    // Free allocated memory

    // free device memory
    cudaFree(array);
    cudaFree(kernel);
    cudaFree(unifiedResult);

    return 0;
}