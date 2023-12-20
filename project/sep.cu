
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

__constant__ float kernel[K];

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
            std::cout <<array[i * size + j] << ", ";
        }
        std::cout << std::endl;
    }
}

void print_kernel(float *array, int size){
    for (int i = 0; i < size; i++){
            std::cout <<array[i ] << ", "<< std::endl;

    }
}

void initKernel(float *h_kernel) {
    float sum = 0.0;

    for (int x = -RADIUS; x <= RADIUS; x++) {
            float value = exp(-(x * x) / 2) / sqrt(2* M_PI);
            h_kernel[(x + RADIUS)] = value;
            printf("%f\n", value);
            sum += value;
        
    }

    printf("%f\n", sum);
    // // // Normalize the Kernel
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

__global__ void rowConvolution(float *d_input, float *d_output) {
    // tile index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output index
    int row_out = blockIdx.y * TILE_SIZE_OUT + ty;
    int col_out = blockIdx.x * TILE_SIZE_OUT + tx;

    // input index
    int row_in = row_out;
    int col_in = col_out - RADIUS;
    
    __shared__ float tile[T_BLOCK_SIZE][T_BLOCK_SIZE];
    
    if((row_in >= 0) && (row_in < N) && (col_in >= 0) && (col_in < N)){
        tile[ty][tx] = d_input[row_in * N + col_in];
    }
    else{
        tile[ty][tx] = 0.0f;
    }
    __syncthreads();

    float temp = 0.0f;
    if((ty < TILE_SIZE_OUT) && (tx < TILE_SIZE_OUT)){
        #pragma unroll
        for(int i = 0; i < K; i++){
                temp += tile[ty][i+tx] * kernel[i];
                // printf("blockidy: %i, blockidx: %i, temp[%i][%i], (kernel[%i]): %f tile: %f  temp: %f\n", blockIdx.y, blockIdx.x, ty, tx, i, kernel[i], tile[ty][i+tx], temp);        

            }
            if((row_in < N) && (col_in < N)){
            d_output[row_out * N + col_out] = temp;
        }
    }
}

void row_convolutionGPU(float* in, float* out){
    dim3 rowBlockDim(T_BLOCK_SIZE, T_BLOCK_SIZE, 1);
    dim3 rowGridDim(T_GRID_SIZE, T_GRID_SIZE, 1);
    rowConvolution<<<rowGridDim, rowBlockDim>>>(in, out);
    checkCudaError(cudaPeekAtLastError());
}


__global__ void colConvolution(float *d_input, float *d_output) {
    // printf("1:\n");
    // tile index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output index
    int row_out = blockIdx.y * TILE_SIZE_OUT + ty;
    int col_out = blockIdx.x * TILE_SIZE_OUT + tx;

    // input index
    int row_in = row_out - RADIUS;
    int col_in = col_out;
    
    __shared__ float tile[T_BLOCK_SIZE][T_BLOCK_SIZE];
    
    if((row_in >= 0) && (row_in < N) && (col_in >= 0) && (col_in < N)){
        tile[ty][tx] = d_input[row_in * N + col_in];
        // tile[ty][tx] = 0.0f;
    }
    else{
        tile[ty][tx] = 0.0f;
    }
    // printf("0: blockidy: %i, blockidx: %i, ty:%i, tx:%i, \n", blockIdx.y, blockIdx.x, ty, tx);            
    __syncthreads();
    // printf("1: blockidy: %i, blockidx: %i, ty:%i, tx:%i, \n", blockIdx.y, blockIdx.x, ty, tx);            
    float temp = 0.0f;
    if((ty < TILE_SIZE_OUT) && (tx < TILE_SIZE_OUT)){
        // printf("3\n");
        #pragma unroll
        for(int i = 0; i < K; i++){
            // printf("2: blockidy: %i, blockidx: %i, temp[%i][%i], (kernel[%i]): %f tile: %f  temp: %f\n", blockIdx.y, blockIdx.x, ty, tx, i, kernel[i], tile[ty][i+tx], temp);            
            temp += tile[i+ty][tx] * kernel[i];
            
            }
            if((row_in < N) && (col_in < N)){
            d_output[row_out * N + col_out] = temp;
        }
    }
}

void col_convolutionGPU(float* in, float* out){
    dim3 colBlockDim(T_BLOCK_SIZE, T_BLOCK_SIZE);
    dim3 colGridDim(T_GRID_SIZE, T_GRID_SIZE);
    colConvolution<<<colGridDim, colBlockDim>>>(in, out);
    checkCudaError(cudaPeekAtLastError());
    
}

int main(int argc, char *argv[]) {

    cudaError_t err = cudaSetDevice(0);
    if(cudaSuccess != err){
        std::cerr << "No CUDA device found!" << cudaGetErrorString(err) << std::endl;
    }
    else {
        checkCudaError(cudaSetDevice(0));
    }

    // Size in bytes for matrix and kernel
    size_t matrixSize = N * N * sizeof(float);

    //assign memory
    float *h_in = (float *)malloc(matrixSize);
    
    float *d_in = (float *)malloc(matrixSize);
    float *d_buffer = (float *)malloc(matrixSize);
    float *d_out = (float *)malloc(matrixSize);

    checkCudaError(cudaMalloc(&d_in, matrixSize));
    checkCudaError(cudaMalloc(&d_buffer, matrixSize));
    checkCudaError(cudaMalloc(&d_out, matrixSize));
    
    // initilize the matrix and kernel
    float sep_kernel[5] = {0.05448868, 0.24420134, 0.40261995, 0.24420134, 0.05448868};
    
    for (int i = 0; i < N * N; i++) {
        h_in[i] = rand() % 255;
    }

    checkCudaError(cudaMemcpyToSymbol(kernel, sep_kernel, K * sizeof(float)));
    checkCudaError(cudaMemcpy(d_in, h_in, matrixSize, cudaMemcpyHostToDevice));

    // // execute cuda command

    for (int i = 0; i < N_TRIALS; i++){
        row_convolutionGPU(d_in, d_buffer);
        col_convolutionGPU(d_buffer, d_out);
    }
    checkCudaError(cudaDeviceSynchronize());

    // free  memory
    cudaFree(h_in);
    cudaFree(kernel);
    cudaFree(d_in);
    cudaFree(d_buffer);
    cudaFree(d_out);
    return 0;
}