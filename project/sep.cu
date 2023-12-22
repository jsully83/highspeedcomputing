#include <iostream>
#include <cuda_runtime.h>

#define N_TRIALS 100

// convolution mask
#define M 5
#define RADIUS (M / 2)

// 1D row convolution
#define ROW_BLOCK_Y 16 
#define ROW_BLOCK_X (1024 / ROW_BLOCK_Y) //use max threads allowed
#define ROW_TILE_X (ROW_BLOCK_X - (M - 1))

// 1D column convolution
#define COL_BLOCK_X 32 
#define COL_BLOCK_Y (1024 / COL_BLOCK_X) //use max threads allowed
#define COL_TILE_Y (COL_BLOCK_Y - (M - 1))

// 2D convolution
#define T_BLOCK_SIZE 32
#define T_TILE_SIZE_OUT (T_BLOCK_SIZE - (M - 1))
#define T_GRID_SIZE ((N - 1) / T_TILE_SIZE_OUT + 1)

// Function to check for CUDA errors
#define checkCudaError(err) __checkCudaError(err, __FILE__, __LINE__)
void __checkCudaError(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
            (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__constant__ float d_mask1d[M];
__constant__ float d_mask2d[M * M];

__global__ void row_convolution(float *d_input, float *d_output, int N) {

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
        for(int i = 0; i < M; i++){
                temp += tile[ty][i+tx] * d_mask1d[i];
            }
            if((row_in < N) && (col_in < N)){
                d_output[row_out * N + col_out] = temp;
        }
    }
}

void launch_row_convolution(float* in, float* out, int N){

    const unsigned int rowNumBlocks_x = (N + ROW_BLOCK_X - 1) / ROW_BLOCK_X;
    const unsigned int rowNumBlocks_y = (N + ROW_BLOCK_Y - 1) / ROW_BLOCK_Y;

    dim3 rowBlockDim(ROW_BLOCK_X, ROW_BLOCK_Y);
    dim3 rowGridDim(rowNumBlocks_x,rowNumBlocks_y);
    row_convolution<<<rowGridDim, rowBlockDim>>>(in, out, N);
    checkCudaError(cudaPeekAtLastError());
}

__global__ void col_convolution(float *d_input, float *d_output, int N) {

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
        for(int i = 0; i < M; i++){
            temp += tile[i+ty][tx] * d_mask1d[i];
        }
        if((row_in < N) && (col_in < N)){
            d_output[row_out * N + col_out] = temp;
        }
    }
}

void launch_col_convolution(float* in, float* out, int N){

    const unsigned int colNumBlocks_x = (N + COL_BLOCK_X - 1) / COL_BLOCK_X + 1;
    const unsigned int colNumBlocks_y = (N + COL_BLOCK_Y - 1) / COL_BLOCK_Y + 1;
    
    dim3 colBlockDim(COL_BLOCK_X, COL_BLOCK_Y);
    dim3 colGridDim(colNumBlocks_x, colNumBlocks_y);
    col_convolution<<<colGridDim, colBlockDim>>>(in, out, N);
    checkCudaError(cudaPeekAtLastError());
    
}

__global__ void tiled_convolution(float *d_input, float *d_output, int N) {
    // tile index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output index
    int row_out = blockIdx.y * T_TILE_SIZE_OUT + ty;
    int col_out = blockIdx.x * T_TILE_SIZE_OUT + tx;

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
    if((ty < T_TILE_SIZE_OUT) && (tx < T_TILE_SIZE_OUT)){
        #pragma unroll
        for(int i = 0; i < M; i++){
            #pragma unroll
            for(int j = 0; j < M; j++){
                temp += input_sh[i+ty][j+tx] * d_mask2d[i * M +j];
            }
        }

        if((row_out < N) && (col_out < N)){
            d_output[row_out * N + col_out] = temp;
        }
    }
}

void launch_tiled_convolution(float* in, float* out, int N){

    dim3 tiledBlockDim(T_BLOCK_SIZE, T_BLOCK_SIZE);
    dim3 tiledGridDim(T_GRID_SIZE, T_GRID_SIZE, 1);
    tiled_convolution<<<tiledGridDim, tiledBlockDim>>>(in, out, N);
    checkCudaError(cudaPeekAtLastError());
    
}

void initialize_gaussian_kernel(float *mask) {
    // mean=0 std=1
    float sum = 0.0;
    for (int x = -RADIUS; x <= RADIUS; x++) {
        for (int y = -RADIUS; y <= RADIUS; y++) {
            float value = exp(-(x * x + y * y)/2) / (2 * M_PI);
            mask[(x + RADIUS) * M + (y + RADIUS)] = value;
            sum += value;
        }
    }
    // Normalize the Kernel
    for (int i = 0; i < M * M; ++i) {
        mask[i] /= sum;

    }
}


int main(int argc, char *argv[]) {

    // command line argument
    int N;
    if (argc < 2) {
        std::cout << "Please enter the array size N." << std::endl;
        return 1;
    }
    else {
        N = std::atoi(argv[1]);
    }

    std::cout <<"Array size: " << N << "x" << N << std::endl;
    std::cout << "Mask size: " << M << std::endl;

    // warm-up the first gpu
    cudaError_t err = cudaSetDevice(0);
    if(cudaSuccess != err){
        std::cerr << "No CUDA device found!" << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    else {
        checkCudaError(cudaSetDevice(0));
    }
    
    // create vars
    float *h_in;
    float *d_in, *d_buffer, *d_out;

    // assign memory
    size_t matrixSize = N * N * sizeof(float);
    h_in = (float *)malloc(matrixSize);

    checkCudaError(cudaMalloc(&d_in, matrixSize));
    checkCudaError(cudaMalloc(&d_buffer, matrixSize));
    checkCudaError(cudaMalloc(&d_out, matrixSize));
    
    // initilize the matrix and mask (precalculated 1d norm distribution)
    float h_mask1d[5] = {0.05448868, 0.24420134, 0.40261995, 0.24420134, 0.05448868};
    float *h_mask2d = (float *)malloc(M * M * sizeof(float));
    
    initialize_gaussian_kernel(h_mask2d);

    for (int i = 0; i < N * N; i++) {
        h_in[i] = rand() % 255;
    }

    checkCudaError(cudaMemcpyToSymbol(d_mask1d, h_mask1d, M * sizeof(float)));
    checkCudaError(cudaMemcpyToSymbol(d_mask2d, h_mask2d, M * M * sizeof(float)));
    checkCudaError(cudaMemcpy(d_in, h_in, matrixSize, cudaMemcpyHostToDevice));

    float gpu_elapsed_time_ms;
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    
    //============= SEPARABLE CONVOLUTION ==================//
    checkCudaError(cudaDeviceSynchronize());
    checkCudaError(cudaEventRecord(start, 0));
    // execute cuda command
    for (int i = 0; i < N_TRIALS; i++){
        launch_row_convolution(d_in, d_buffer, N);
        launch_col_convolution(d_buffer, d_out, N);

    }
    checkCudaError(cudaEventRecord(stop, 0));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop));
    std::cout << "Separable convolution mean time elasped for "<< N_TRIALS << " trials: "<< gpu_elapsed_time_ms/N_TRIALS << " ms." << std::endl;

    //============= 2D CONVOLUTION ==================//
    checkCudaError(cudaDeviceSynchronize());
    checkCudaError(cudaEventRecord(start, 0));
    // execute cuda command
    for (int i = 0; i < N_TRIALS; i++){
        launch_tiled_convolution(d_in, d_out, N);
    
    }
    checkCudaError(cudaEventRecord(stop, 0));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop));
    std::cout << "2D Convolution mean time elasped for "<< N_TRIALS << " trials: "<< gpu_elapsed_time_ms/N_TRIALS << " ms." << std::endl;

    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));

    // free  memory
    free(h_in);
    free(h_mask2d);

    cudaFree(d_mask1d);
    cudaFree(d_mask2d);
    cudaFree(d_in);
    cudaFree(d_buffer);
    cudaFree(d_out);
    return 0;
}