
#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define N 16384
#define K 5
#define RADIUS (K / 2) 
#define T_BLOCK_SIZE 32
#define TILE_SIZE_OUT (T_BLOCK_SIZE - (K - 1))
#define T_GRID_SIZE ((N - 1) / TILE_SIZE_OUT + 1)

#define N_TRIALS 100

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

void print_array(float *array, int size){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            // std::cout << i * size + j << ": "<<array[i * size + j] << " ";
            std::cout <<array[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

void initialize_gaussian_kernel(float *h_kernel) {
    float sum = 0.0;

    for (int x = -RADIUS; x <= RADIUS; x++) {
        for (int y = -RADIUS; y <= RADIUS; y++) {
            float value = exp(-(x * x + y * y)/2) / (2 * M_PI);
            h_kernel[(x + RADIUS) * K + (y + RADIUS)] = value;
            sum += value;
        }
    }

    // Normalize the Kernel
    for (int i = 0; i < K * K; ++i) {
        h_kernel[i] /= sum;

    }
}

// Function to perform the convolution on the CPU
void convolutionCPU(int *input, float *kernel, float *output) {
    float temp;
    int r_off;
    int c_off;

    // go over rows and columns of the input
    #pragma unroll
    for (int row = 0; row < N; row++){
        #pragma unroll
        for (int col = 0; col < N; col++){
            temp = 0.0f;

            // go over rows and columns of the kernel.
            // offset moves the kernel it's RADIUS off the input
            // so anchor element is at the corner element
            #pragma unroll
            for(int k_row = 0; k_row < K; k_row++ ){
                r_off = row - RADIUS + k_row; 
                
                #pragma unroll
                for(int k_col = 0; k_col < K; k_col++ ){
                    c_off = col - RADIUS + k_col;

                    // zero padding. don't accumulate if the kernel hangs off the input
                    if((r_off >= 0) && (r_off < N) && (c_off >= 0) && (c_off < N)){
                        temp += input[r_off * N + c_off] * kernel[k_row * K + k_col];
                    }
                }
            }
            output[row * N + col] = temp;
        } 
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

__global__ void naiveConvolution(int *d_input, float *d_kernel, float *d_output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // drop threads if grid is larger than array size
    if (row >= N || col >= N){return;}

    int startRow = row - RADIUS;
    int startCol = col - RADIUS;
    float temp = 0.0f;

    // #pragma unroll
    for(int i = 0; i < K; i++) {
        // #pragma unroll
        for(int j = 0; j < K; j++) {  
            if((startRow + i >= 0) && (startRow + i < N)) {
                if((startCol + j >= 0) && (startCol + j < N)) {
                    temp += d_input[(startRow + i) * N + (startCol + j)] * d_kernel[i * K + j];
                }
            }
            
        }
    }
    d_output[row * N + col] = temp;
}

__constant__ float const_kernel[K * K];

__global__ void constantConvolution(int *d_input, float *d_output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // drop threads if grid is larger than array size
    if (row >= N || col >= N){return;}

    int startRow = row - RADIUS;
    int startCol = col - RADIUS;
    float temp = 0.0f;

    // #pragma unroll
    for(int i = 0; i < K; i++) {
        // #pragma unroll
        for(int j = 0; j < K; j++) {  
            if((startRow + i >= 0) && (startRow + i < N)) {
                if((startCol + j >= 0) && (startCol + j < N)) {
                    temp += d_input[(startRow + i) * N + (startCol + j)] * const_kernel[i * K + j];
                }
            }
        }
    }
    d_output[row * N + col] = temp;
    // printf("row=%i col=%i temp[%i]=%d\n", row, col, (row * N + col), temp);
}

__global__ void tiledConvolution(int *d_input, float *d_output) {
    // tile index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output index
    int row_out = blockIdx.y * TILE_SIZE_OUT + ty;
    int col_out = blockIdx.x * TILE_SIZE_OUT + tx;

    // input index
    int row_in = row_out - RADIUS;
    int col_in = col_out - RADIUS;
    
    __shared__ float input_sh[T_BLOCK_SIZE][T_BLOCK_SIZE];
    
    if((row_in >= 0) && (row_in < N) && (col_in >= 0) && (col_in < N)){
        input_sh[ty][tx] = d_input[row_in * N + col_in];
    }
    else{
        input_sh[ty][tx] = 0.0f;
    }
    __syncthreads();
    
    float temp = 0.0f;
    if((ty < TILE_SIZE_OUT) && (tx < TILE_SIZE_OUT)){
        // #pragma unroll
        for(int i = 0; i < K; i++){
            // #pragma unroll
            for(int j = 0; j < K; j++){
                temp += input_sh[i+ty][j+tx] * const_kernel[i * K +j];
            }
        }

        if((row_out < N) && (col_out < N)){
            d_output[row_out * N + col_out] = temp;
        }
    }
}

__global__ void unifiedConvolution(int *d_input, float *d_output) {
    // tile index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output index
    int row_out = blockIdx.y * TILE_SIZE_OUT + ty;
    int col_out = blockIdx.x * TILE_SIZE_OUT + tx;

    // input index
    int row_in = row_out - RADIUS;
    int col_in = col_out - RADIUS;
    
    __shared__ float input_sh[T_BLOCK_SIZE][T_BLOCK_SIZE];
    
    if((row_in >= 0) && (row_in < N) && (col_in >= 0) && (col_in < N)){
        input_sh[ty][tx] = d_input[row_in * N + col_in];
    }
    else{
        input_sh[ty][tx] = 0.0f;
    }
    __syncthreads();
    
    float temp = 0.0f;
    if((ty < TILE_SIZE_OUT) && (tx < TILE_SIZE_OUT)){
        // #pragma unroll
        for(int i = 0; i < K; i++){
            // #pragma unroll
            for(int j = 0; j < K; j++){
                temp += input_sh[i+ty][j+tx] * const_kernel[i * K +j];
            }
        }

        if((row_out < N) && (col_out < N)){
            d_output[row_out * N + col_out] = temp;
        }
    }
}


__global__ void coopConvolution(int *d_input, float *d_output) {
    // tile index

    cg::thread_block cta = cg::this_thread_block();

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output index
    int row_out = blockIdx.y * TILE_SIZE_OUT + ty;
    int col_out = blockIdx.x * TILE_SIZE_OUT + tx;

    // input index
    int row_in = row_out - RADIUS;
    int col_in = col_out - RADIUS;
    
    __shared__ float input_sh[T_BLOCK_SIZE][T_BLOCK_SIZE];
    
    if((row_in >= 0) && (row_in < N) && (col_in >= 0) && (col_in < N)){
        input_sh[ty][tx] = d_input[row_in * N + col_in];
    }
    else{
        input_sh[ty][tx] = 0.0f;
    }
    cg::sync(cta);
    
    float temp = 0.0f;
    if((ty < TILE_SIZE_OUT) && (tx < TILE_SIZE_OUT)){
        // #pragma unroll
        for(int i = 0; i < K; i++){
            // #pragma unroll
            for(int j = 0; j < K; j++){
                temp += input_sh[i+ty][j+tx] * const_kernel[i * K +j];
            }
        }

        if((row_out < N) && (col_out < N)){
            d_output[row_out * N + col_out] = temp;
        }
    }
}

int main(int argc, char *argv[]) {

    // Size in bytes for matrix and kernel
    size_t matrixSize = N * N * sizeof(int);
    size_t kernelSize = K * K * sizeof(float);

    // Allocate host memory
    int *h_input = (int *)malloc(matrixSize);
    float *h_kernel = (float *)malloc(kernelSize);
    float *cpuResult = (float *)malloc(matrixSize);

    // initilize the matrix and kernel
    for (int i = 0; i < N * N; i++) {
        h_input[i] = rand() % 255;
    }
    // print_array(h_input, N);
    initialize_gaussian_kernel(h_kernel);

    // =============== CPU Convolution ===============
    // auto startCPU = std::chrono::high_resolution_clock::now();
    // convolutionCPU(h_input, h_kernel, cpuResult);
    // auto endCPU = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float, std::milli> cpuDuration = endCPU - startCPU;
    // std::cout << "CPU Convolution time: " << cpuDuration.count() << " ms\n"<< std::endl;
    // print_array(cpuResult, N);
    // =============== Naive GPU Convolution ===============
    // device variables
    int *d_input; 
    float *d_kernel, *d_output;
    checkCudaError(cudaMalloc(&d_input, matrixSize));
    checkCudaError(cudaMalloc(&d_kernel, kernelSize));
    checkCudaError(cudaMalloc(&d_output, matrixSize));

    // copy to device
    checkCudaError(cudaMemcpy(d_input, h_input, matrixSize, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice));
    
    // Set up execution configuration
    const unsigned int blockSize = 32;
    const unsigned int gridSize = (N + blockSize - 1) / blockSize;  
    
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim(gridSize, gridSize);
    
    // execute cuda command
    for (int i = 0; i < N_TRIALS; i++){
        naiveConvolution<<<gridDim, blockDim>>>(d_input, d_kernel, d_output);
        checkCudaError(cudaPeekAtLastError());
        checkCudaError(cudaDeviceSynchronize());
    }

    // copy result from device
    float *naiveResult = (float *)malloc(matrixSize);
    checkCudaError(cudaMemcpy(naiveResult, d_output, matrixSize, cudaMemcpyDeviceToHost));
    
    // if(compare_results(cpuResult, naiveResult)){
    //     std::cout << "Naive result complete." << std::endl;
    // }
    // else{
    //     std::cout << "Naive result failed." << std::endl;
    // }

    // =============== Constant GPU Convolution ===============
    // copy kernel to constant memory
    checkCudaError(cudaMemcpyToSymbol(const_kernel, h_kernel, kernelSize));

    // execute cuda command
    for (int i = 0; i < N_TRIALS; i++){
        constantConvolution<<<gridDim, blockDim>>>(d_input, d_output);
        checkCudaError(cudaPeekAtLastError());
        checkCudaError(cudaDeviceSynchronize());
    }

    // copy result from device
    float *constantResult = (float *)malloc(matrixSize);
    checkCudaError(cudaMemcpy(constantResult, d_output, matrixSize, cudaMemcpyDeviceToHost));

    // if(compare_results(cpuResult, constantResult)){
    //     std::cout << "Constant result complete." << std::endl;
    // }
    // else{
    //     std::cout << "Constant result failed." << std::endl;
    // }

    // =============== Tiled GPU Convolution ===============
    dim3 tiledBlockDim(T_BLOCK_SIZE, T_BLOCK_SIZE);
    dim3 tiledGridDim(T_GRID_SIZE, T_GRID_SIZE, 1);

    // execute cuda command
    for (int i = 0; i < N_TRIALS; i++){
        tiledConvolution<<<tiledGridDim, tiledBlockDim>>>(d_input, d_output);
        checkCudaError(cudaPeekAtLastError());
        checkCudaError(cudaDeviceSynchronize());
    }

    // copy result from device
    float *tiledResult = (float *)malloc(matrixSize);
    checkCudaError(cudaMemcpy(tiledResult, d_output, matrixSize, cudaMemcpyDeviceToHost));

    // if(compare_results(cpuResult, tiledResult)){
    //     std::cout << "Tiled result complete." << std::endl;
    // } 
    // else{
    //     std::cout << "Tiled result failed." << std::endl;
    // }

    // =============== Unified Memory GPU Convolution ===============

    //use unified memory
    int *array;
    float *mask, *unifiedResult;

    checkCudaError(cudaMallocManaged(&array, matrixSize));
    checkCudaError(cudaMallocManaged(&mask, kernelSize));
    checkCudaError(cudaMallocManaged(&unifiedResult, matrixSize));
    
    for(int i = 0; i < N * N; i++)
        array[i] = h_input[i];

    for(int i = 0; i < K * K; i++)
        mask[i] = h_kernel[i];

    checkCudaError(cudaMemPrefetchAsync(array, matrixSize, 0));
    checkCudaError(cudaMemPrefetchAsync(mask, kernelSize, 0));
    checkCudaError(cudaMemPrefetchAsync(unifiedResult, matrixSize, 0));

    // // execute cuda command
    for (int i = 0; i < N_TRIALS; i++){
        unifiedConvolution<<<tiledGridDim, tiledBlockDim>>>(array, unifiedResult);
        checkCudaError(cudaPeekAtLastError());
        checkCudaError(cudaDeviceSynchronize());
    }
    // if(compare_results(cpuResult, unifiedResult)){
    //     std::cout << "Unified result complete." << std::endl;
    // } 
    // else{
    //     std::cout << "Unified result failed." << std::endl;
    // }


     // =============== Coop Memory GPU Convolution ===============

    //use unified memory
    float *coopResult;

    checkCudaError(cudaMallocManaged(&coopResult, matrixSize));
    checkCudaError(cudaMemPrefetchAsync(coopResult, matrixSize, 0));

    // // execute cuda command
    for (int i = 0; i < N_TRIALS; i++){
        coopConvolution<<<tiledGridDim, tiledBlockDim>>>(array, coopResult);
        checkCudaError(cudaPeekAtLastError());
        checkCudaError(cudaDeviceSynchronize());
    }
    // if(compare_results(cpuResult, coopResult)){
    //     std::cout << "coop result complete." << std::endl;
    // } 
    // else{
    //     std::cout << "coop result failed." << std::endl;
    // }

    // Free allocated memory
    free(h_input);
    free(h_kernel);
    free(cpuResult);
    free(naiveResult);
    free(constantResult);
    free(tiledResult);

    // free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    cudaFree(array);
    cudaFree(mask);
    cudaFree(unifiedResult);
    cudaFree(coopResult);

    return 0;
}