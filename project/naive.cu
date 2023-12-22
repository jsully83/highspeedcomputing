
#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// #define N 16384
#define K 7
#define RADIUS (K / 2) 
#define T_BLOCK_SIZE 32

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
    // mean=0 std=1
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
void convolutionCPU(int *input, float *kernel, float *output, int N) {
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

bool compare_results(float *cpu, float *device, int N){
    float tol = 1e-3;
    for(int i = 0; i < N * N; i++){
        if(abs(cpu[i] - device[i]) > tol){
            return false;
        }
    }
    return true;
}

__global__ void naiveConvolution(int *d_input, float *d_kernel, float *d_output, int N) {
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

__global__ void constantConvolution(int *d_input, float *d_output, int N) {
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


int main(int argc, char *argv[]) {
    int N;
    if (argc < 2) {
        std::cout << "Please enter the array size N." << std::endl;
        return 1;
    }
    else {
        N = std::atoi(argv[1]);
    }

    cudaError_t err = cudaSetDevice(0);
    if(cudaSuccess != err){
        std::cerr << "No CUDA device found!" << cudaGetErrorString(err) << std::endl;
    }
    else {
        checkCudaError(cudaSetDevice(0));
    }


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
    auto startCPU = std::chrono::high_resolution_clock::now();
    convolutionCPU(h_input, h_kernel, cpuResult, N);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpuDuration = endCPU - startCPU;
    std::cout << "CPU Convolution time: " << cpuDuration.count() << " ms\n"<< std::endl;
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
        naiveConvolution<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, N);
        checkCudaError(cudaPeekAtLastError());
        checkCudaError(cudaDeviceSynchronize());
    }

    // copy result from device
    float *naiveResult = (float *)malloc(matrixSize);
    checkCudaError(cudaMemcpy(naiveResult, d_output, matrixSize, cudaMemcpyDeviceToHost));
    
    if(compare_results(cpuResult, naiveResult, N)){
        std::cout << "Naive result complete." << std::endl;
    }
    else{
        std::cout << "Naive result failed." << std::endl;
    }

    // =============== Constant GPU Convolution ===============
    // copy kernel to constant memory
    checkCudaError(cudaMemcpyToSymbol(const_kernel, h_kernel, kernelSize));

    // execute cuda command
    for (int i = 0; i < N_TRIALS; i++){
        constantConvolution<<<gridDim, blockDim>>>(d_input, d_output, N);
        checkCudaError(cudaPeekAtLastError());
        checkCudaError(cudaDeviceSynchronize());
    }

    // copy result from device
    float *constantResult = (float *)malloc(matrixSize);
    checkCudaError(cudaMemcpy(constantResult, d_output, matrixSize, cudaMemcpyDeviceToHost));

    if(compare_results(cpuResult, constantResult, N)){
        std::cout << "Constant result complete." << std::endl;
    }
    else{
        std::cout << "Constant result failed." << std::endl;
    }

    // Free allocated memory
    free(h_input);
    free(h_kernel);
    free(cpuResult);
    free(naiveResult);
    free(constantResult);

    // free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}