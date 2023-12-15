
#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#define N (1 << 10)
#define K 5
#define RADIUS (K / 2) 


// Function to check for CUDA errors
#define checkCudaError(err) __checkCudaErrors(err, __FILE__, __LINE__)

void __checkCudaErrors(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
            (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void print_array(double *array, int size){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            std::cout << array[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

void initialize_gaussian_kernel(double *h_kernel) {
    double sum = 0.0f;
    double stdDev = 1.0f;

    for (int x = -RADIUS; x <= RADIUS; x++) {
        for (int y = -RADIUS; y <= RADIUS; y++) {
            double value = exp(-(x * x + y * y) / (2 * stdDev * stdDev)) / sqrt(2 * M_PI * stdDev * stdDev);
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
void convolutionCPU(double *input, double *kernel, double *output) {
    int temp;
    int r_off;
    int c_off;

    // go over rows and columns of the input
    for (int row = 0; row < N; row++){
        for (int col = 0; col < N; col++){
            temp = 0;

            // go over rows and columns of the kernel.
            // offset moves the kernel it's RADIUS off the input
            // so anchor element is at the corner element
            for(int k_row = 0; k_row < K; k_row++ ){
                r_off = row - RADIUS + k_row; 

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

bool compare_results(double *cpu, double *device){
    double tol = 1e-5;
    for (int i = 0; i < N * N; i++){
        if(abs(cpu[i] - device[i]) > tol){
            return false;
        }
    }
    return true;
}

__global__ void naiveConvolution(double *d_input, double *d_kernel, double *d_output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int startRow = row - RADIUS;
    int startCol = col - RADIUS;
    int temp = 0;

    for(int i = 0; i < K; i++) {
        for(int j = 0; j < K; j++) {  
            if((startRow + i >= 0) && (startRow + i < N)) {
                if((startCol + j >= 0) && (startCol + j < N)) {
                    temp += d_input[(startRow + i) * N + (startCol + j)] * d_kernel[i * K + j];
                    // printf("temp=%f\n", temp);
                }
            }
            
        }
    }
    d_output[row * N + col] = temp;
    // printf("row=%i col=%i temp[%i]=%d\n", row, col, (row * N + col), temp);
}

int main(int argc, char *argv[]) {

    // Size in bytes for matrix and kernel
    size_t matrixSize = N * N * sizeof(double);
    size_t kernelSize = K * K * sizeof(double);

    // Allocate host memory
    double *h_input = (double *)malloc(matrixSize);
    double *h_kernel = (double *)malloc(kernelSize);
    double *cpuResult = (double *)malloc(matrixSize);

    // initilize the matrix and kernel
    for (int i = 0; i < N * N; i++) {
        h_input[i] = rand() % 255;
    }
    initialize_gaussian_kernel(h_kernel);

    // =============== CPU Convolution ===============
    auto startCPU = std::chrono::high_resolution_clock::now();
    convolutionCPU(h_input, h_kernel, cpuResult);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endCPU - startCPU;
    std::cout << "CPU Convolution time: " << cpuDuration.count() << " ms\n"<< std::endl;

    // =============== Naive GPU Convolution ===============
    // device variables
    double *d_input, *d_kernel, *d_output;
    checkCudaError(cudaMalloc(&d_input, matrixSize));
    checkCudaError(cudaMalloc(&d_kernel, kernelSize));
    checkCudaError(cudaMalloc(&d_output, matrixSize));

    // copy to device
    checkCudaError(cudaMemcpy(d_input, h_input, matrixSize, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice));
    
    // Set up execution configuration
    const unsigned int blockSize = 16;
    const unsigned int gridSize = (N + blockSize - 1) / blockSize;  
    
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim(gridSize, gridSize);
    
    // execute cuda command
    naiveConvolution<<<gridDim, blockDim>>>(d_input, d_kernel, d_output);
    checkCudaError(cudaPeekAtLastError());
    checkCudaError(cudaDeviceSynchronize());

    // copy result from device
    double *naiveResult = (double *)malloc(matrixSize);
    checkCudaError(cudaMemcpy(naiveResult, d_output, matrixSize, cudaMemcpyDeviceToHost));
    
    if(compare_results(cpuResult, naiveResult)){
        std::cout << "Naive result complete." << std::endl;
    }
    else{
        std::cout << "Naive result failed." << std::endl;
    }

    // Free allocated memory
    free(h_input);
    free(h_kernel);
    free(cpuResult);

    // free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}