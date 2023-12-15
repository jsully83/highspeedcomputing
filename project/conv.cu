
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}
bool compareMatrices(const float* a, const float* b, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; ++i) {
        if (abs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void convolutionCPU(int* array, int* kernel, float* output) {

}
__global__ void naiveConvolution2D(float *input, float *kernel, float *output, int width, int height, int kernelWidth, int kernelHeight) {
    // Calculate the global thread positions
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    // Initialize the output for this thread
    float sum = 0;

    // Only compute if within the dimensions of the image
    if (x >= halfKernelWidth && x < (width - halfKernelWidth) && y >= halfKernelHeight && y < (height - halfKernelHeight)) {
        // Iterate over the kernel
        for (int ky = -halfKernelHeight; ky <= halfKernelHeight; ky++) {
            for (int kx = -halfKernelWidth; kx <= halfKernelWidth; kx++) {
                // Calculate the input image index
                int imageIndex = (y + ky) * width + (x + kx);

                // Calculate the kernel index
                int kernelIndex = (ky + halfKernelHeight) * kernelWidth + (kx + halfKernelWidth);

                // Perform the convolution operation
                sum += input[imageIndex] * kernel[kernelIndex];
            }
        }
    }

    // Write the output
    output[y * width + x] = sum;
    printf("from cuda: %f\n", output[y * width + x]);
}

__global__ void separableConvolution(float *input, float *output, int width, int height, const float* gaussianKernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within bounds
    if (x >= width || y >= height) return;

    // Perform the horizontal convolution
    float horizontalResult = 0.0f;
    for (int kx = -2; kx <= 2; kx++) {
        int col = min(max(x + kx, 0), width - 1);
        horizontalResult += input[y * width + col] * gaussianKernel[2 + kx];
    }

    // Perform the vertical convolution
    float result = 0.0f;
    for (int ky = -2; ky <= 2; ky++) {
        int row = min(max(y + ky, 0), height - 1);
        result += horizontalResult * gaussianKernel[2 + ky];
    }

    // Write the final output
    output[y * width + x] = result;
}

void printResults(float* result){
    for(int i=0; i<sizeof(result); i++){
        std::cout << result[i] << std::endl;
    }
}

int main() {
    // Matrix and kernel dimensions
    const int width = 8;
    const int height = 8;
    const int kernelWidth = 5;
    const int kernelHeight = 5;

    // Size in bytes for matrix and kernel
    size_t matrixSize = width * height * sizeof(float);
    size_t kernelSize = kernelWidth * kernelHeight * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(matrixSize);
    float *h_kernel = (float *)malloc(kernelSize);
    float *cpuResult = (float *)malloc(matrixSize);

    // Initialize host memory with random data
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < kernelWidth * kernelHeight; i++) {
        h_kernel[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    //================= CPU Convolution ================= 
    auto startCPU = std::chrono::high_resolution_clock::now();
    convolutionCPU(h_input, h_kernel, cpuResult, width, height, kernelWidth, kernelHeight);
    auto endCPU = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpuDuration = endCPU - startCPU;
    std::cout << "CPU Convolution time: " << cpuDuration.count() << " ms\n";

    printResults(cpuResult);


    //================= Naive GPU Convolution ================= 
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    checkCudaError(cudaMalloc(&d_input, matrixSize));
    checkCudaError(cudaMalloc(&d_kernel, kernelSize));
    checkCudaError(cudaMalloc(&d_output, matrixSize));

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_input, h_input, matrixSize, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 blockSize(8, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Create events for timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    // Launch the kernel and record time
    checkCudaError(cudaEventRecord(start));
    naiveConvolution2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, width, height, kernelWidth, kernelHeight);
    cudaDeviceSynchronize();
    checkCudaError(cudaEventRecord(stop));              
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaGetLastError());
    
    // Comparing GPU results with CPU results
    float *naiveResult = (float *)malloc(matrixSize);
    checkCudaError(cudaMemcpy(naiveResult, d_output, matrixSize, cudaMemcpyDeviceToHost));
    
    if (compareMatrices(cpuResult, naiveResult, width * height)) {
        std::cout << "GPU and CPU results match.\n";
    } else {
        std::cout << "GPU and CPU results do not match.\n";
    }

    printResults(naiveResult);

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Naive Convolution execution time: " << milliseconds << " milliseconds" << std::endl;

    //================= Separable Filter GPU Convolution ================= 
    const float gaussianKernel[5] = {0.03107,0.23642,0.46503,0.23642,0.03107};

    // Allocate memory for Gaussian kernel on device
    float *d_gaussianKernel;
    checkCudaError(cudaMalloc(&d_gaussianKernel, 5 * sizeof(float)));
    checkCudaError(cudaMemcpy(d_gaussianKernel, gaussianKernel, 5 * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaError(cudaEventRecord(start));
    separableConvolution<<<gridSize, blockSize>>>(d_input, d_output, width, height, d_gaussianKernel);
    cudaDeviceSynchronize();
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaGetLastError());

    milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Separable Filter Convolution execution time: " << milliseconds << " milliseconds" << std::endl;

    // Clean up device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    // Clean up host memory
    free(h_input);
    free(h_kernel);
    free(cpuResult);
    free(naiveResult);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
