
#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// #define N 16384
#define K 5
#define RADIUS (K / 2) 
#define ROW_BLOCK_Y 16
#define ROW_BLOCK_X (1024 / ROW_BLOCK_Y)
#define ROW_TILE_X (ROW_BLOCK_X - (K - 1))

#define COL_BLOCK_X 64 
#define COL_BLOCK_Y (1024 / COL_BLOCK_X)
#define COL_TILE_Y (COL_BLOCK_Y - (K - 1))

#define N_TRIALS 100

__constant__ float kernel[K];

// Function to check for CUDA errors
#define checkCudaError(err) __checkCudaError(err, __FILE__, __LINE__)

void __checkCudaError(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
            (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__global__ void rowConvolution(float *d_input, float *d_output, int N) {
// __global__ void rowConvolution(float *d_input, float *d_output) {
    // tile index
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    

    // output index
    int row_out = blockIdx.y * ROW_BLOCK_Y + ty;
    int col_out = blockIdx.x * ROW_TILE_X + tx;

    // we only need a halo region in the rows direction.
    int row_in = row_out;
    int col_in = col_out - RADIUS;
    
    // tile size is as large as a block
    __shared__ float tile[ROW_BLOCK_Y][ROW_BLOCK_X];
    
    if((row_in >= 0) && (row_in < N) && (col_in >= 0) && (col_in < N)){
        tile[ty][tx] = d_input[row_in * N + col_in];
    }
    else{
        tile[ty][tx] = 0.0f;
    }
    __syncthreads();

    float temp = 0.0f;
    if((ty < ROW_BLOCK_Y) && (tx < ROW_TILE_X)){
        #pragma unroll
        for(int i = 0; i < K; i++){
                temp += tile[ty][i+tx] * kernel[i];
            }
            if((row_in < N) && (col_in < N)){
                d_output[row_out * N + col_out] = temp;
        }
    }
}

// void row_convolutionGPU(float* in, float* out){
void row_convolutionGPU(float* in, float* out, int N){

    const unsigned int rowNumBlocks_x = (N + ROW_BLOCK_X - 1) / ROW_BLOCK_X;
    const unsigned int rowNumBlocks_y = (N + ROW_BLOCK_Y - 1) / ROW_BLOCK_Y;

    dim3 rowBlockDim(ROW_BLOCK_X, ROW_BLOCK_Y);
    dim3 rowGridDim(rowNumBlocks_x,rowNumBlocks_y);
    rowConvolution<<<rowGridDim, rowBlockDim>>>(in, out, N);
    checkCudaError(cudaPeekAtLastError());
}

// __global__ void colConvolution(float *d_input, float *d_output) {
__global__ void colConvolution(float *d_input, float *d_output, int N) {
    // tile index
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    // output index
    int row_out = blockIdx.y * COL_TILE_Y + ty;
    int col_out = blockIdx.x * COL_BLOCK_X + tx;

    // input index
    int row_in = row_out - RADIUS;
    int col_in = col_out;
    
    __shared__ float tile[COL_BLOCK_Y][COL_BLOCK_X];
    
    if((row_in >= 0) && (row_in < N) && (col_in >= 0) && (col_in < N)){
        tile[ty][tx] = d_input[row_in * N + col_in];
    }
    else{
        tile[ty][tx] = 0.0f;
    }
    
    __syncthreads();

    float temp = 0.0f;
    if((ty < COL_TILE_Y) && (tx < COL_BLOCK_X)){
        #pragma unroll
        for(int i = 0; i < K; i++){
            temp += tile[i+ty][tx] * kernel[i];
            
        }
        if((row_in < N) && (col_in < N)){
            d_output[row_out * N + col_out] = temp;
        }
    }
}

// void col_convolutionGPU(float* in, float* out){
void col_convolutionGPU(float* in, float* out, int N){

    const unsigned int colNumBlocks_x = (N + COL_BLOCK_X - 1) / COL_BLOCK_X + 1;
    const unsigned int colNumBlocks_y = (N + COL_BLOCK_Y - 1) / COL_BLOCK_Y + 1;
    
    dim3 colBlockDim(COL_BLOCK_X, COL_BLOCK_Y);
    dim3 colGridDim(colNumBlocks_x, colNumBlocks_y);
    colConvolution<<<colGridDim, colBlockDim>>>(in, out, N);
    checkCudaError(cudaPeekAtLastError());
    
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

    float gpu_elapsed_time_ms;
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    checkCudaError(cudaDeviceSynchronize());
    checkCudaError(cudaEventRecord(start, 0));
    // execute cuda command
    for (int i = 0; i < N_TRIALS; i++){
        row_convolutionGPU(d_in, d_buffer, N);
        col_convolutionGPU(d_buffer, d_out, N);
    }
    checkCudaError(cudaEventRecord(stop, 0));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop));
    std::cout << "Average time elasped for "<< N_TRIALS << " trials: "<< gpu_elapsed_time_ms/N_TRIALS << " ms." << std::endl;

    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));

    // free  memory
    cudaFree(h_in);
    cudaFree(kernel);
    cudaFree(d_in);
    cudaFree(d_buffer);
    cudaFree(d_out);
    return 0;
}