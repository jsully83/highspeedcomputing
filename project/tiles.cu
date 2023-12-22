#include <iostream>
#include <cuda_runtime.h>

#define N_TRIALS 100

// 1D row convolution
#define ROW_BLOCK_Y 8 
#define ROW_BLOCK_X (256 / ROW_BLOCK_Y) 

// 1D column convolution
#define COL_BLOCK_X 32 
#define COL_BLOCK_Y (512 / COL_BLOCK_X) //use max threads allowed

// 2D convolution
#define T_BLOCK_SIZE 32

// Function to check for CUDA errors
#define checkCudaError(err) __checkCudaError(err, __FILE__, __LINE__)
void __checkCudaError(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
            (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// max mask size 15x15
__constant__ float d_mask[256];

void copy_to_constant(float *host, int size){
    
    float *clear_mem[256] = {};
    checkCudaError(cudaMemcpyToSymbol(d_mask, clear_mem, 256 * sizeof(float)));
    checkCudaError(cudaMemcpyToSymbol(d_mask, host, size));

}

__global__ void row_convolution(float *d_input, float *d_output, int N, int M, int radius) {
    // tile index
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int tile_size = ROW_BLOCK_X - (M - 1);
    
    // output index
    int row_out = blockIdx.y * ROW_BLOCK_Y + ty;
    int col_out = blockIdx.x * tile_size + tx;

    // we only need a halo region in the rows direction.
    int row_in = row_out;
    int col_in = col_out - radius;

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
    if((ty < ROW_BLOCK_Y) && (tx < tile_size)){
        #pragma unroll
        for(int i = 0; i < M; i++){
                temp += tile[ty][i+tx] * d_mask[i];
            }
            if((row_in < N) && (col_in < N)){
                d_output[row_out * N + col_out] = temp;
        }
    }
}

void launch_row_convolution(float* in, float* out, int N, int M, int radius){
    
    const unsigned int rowNumBlocks_x = (N + ROW_BLOCK_X - 1) / ROW_BLOCK_X;
    const unsigned int rowNumBlocks_y = (N + ROW_BLOCK_Y - 1) / ROW_BLOCK_Y;
    dim3 rowBlockDim(ROW_BLOCK_X, ROW_BLOCK_Y);
    dim3 rowGridDim(rowNumBlocks_x,rowNumBlocks_y);

    row_convolution<<<rowGridDim, rowBlockDim>>>(in, out, N, M, radius);
    checkCudaError(cudaPeekAtLastError());
}

__global__ void col_convolution(float *d_input, float *d_output, int N, int M, int radius) {

    // tile index
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int tile_size =  COL_BLOCK_Y - (M - 1);
    
    // output index
    int row_out = blockIdx.y * tile_size + ty;
    int col_out = blockIdx.x * COL_BLOCK_X + tx;

    // input index
    int row_in = row_out - radius;
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
    if((ty < tile_size) && (tx < COL_BLOCK_X)){
        #pragma unroll
        for(int i = 0; i < M; i++){
            temp += tile[i+ty][tx] * d_mask[i];
        }
        if((row_in < N) && (col_in < N)){
            d_output[row_out * N + col_out] = temp;
        }
    }
}

void launch_col_convolution(float* in, float* out, int N, int M, int radius){
    
    const unsigned int colNumBlocks_x = (N + COL_BLOCK_X - 1) / COL_BLOCK_X + 1;
    const unsigned int colNumBlocks_y = (N + COL_BLOCK_Y - 1) / COL_BLOCK_Y + 1;
    dim3 colBlockDim(COL_BLOCK_X, COL_BLOCK_Y);
    dim3 colGridDim(colNumBlocks_x, colNumBlocks_y);

    col_convolution<<<colGridDim, colBlockDim>>>(in, out, N, M, radius);
    checkCudaError(cudaPeekAtLastError());
    
}

__global__ void tiled_convolution(float *d_input, float *d_output, int N, int M, int tile_size, int radius) {
    // tile index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output index
    int row_out = blockIdx.y * tile_size + ty;
    int col_out = blockIdx.x * tile_size + tx;

    // input index
    int row_in = row_out - radius;
    int col_in = col_out - radius;
    
    __shared__ float input_sh[T_BLOCK_SIZE][T_BLOCK_SIZE];
    
    if((row_in >= 0) && (row_in < N) && (col_in >= 0) && (col_in < N)){
        input_sh[ty][tx] = d_input[row_in * N + col_in];
    }
    else{
        input_sh[ty][tx] = 0.0f;
    }
    __syncthreads();
    
    float temp = 0.0f;
    if((ty < tile_size) && (tx < tile_size)){
        #pragma unroll
        for(int i = 0; i < M; i++){
            #pragma unroll
            for(int j = 0; j < M; j++){
                temp += input_sh[i+ty][j+tx] * d_mask[i * M +j];
            }
        }

        if((row_out < N) && (col_out < N)){
            d_output[row_out * N + col_out] = temp;
        }
    }
}

void launch_tiled_convolution(float* in, float* out, int N, int M, int radius){
    
    const unsigned int grid_size ((N - 1) / (T_BLOCK_SIZE - (M - 1)) + 1);
    int tile_size = T_BLOCK_SIZE - (M - 1);
    dim3 tiledBlockDim(T_BLOCK_SIZE, T_BLOCK_SIZE);
    dim3 tiledGridDim(grid_size, grid_size);

    tiled_convolution<<<tiledGridDim, tiledBlockDim>>>(in, out, N, M, tile_size, radius);
    checkCudaError(cudaPeekAtLastError());
    
}

void initialize_gaussian_kernel(float *mask, int M, int radius) {
    // mean=0 std=1
    float sum = 0.0;
    for (int x =  radius; x <= radius; x++) {
        for (int y =  radius; y <= radius; y++) {
            float value = exp(-(x * x + y * y)/2) / (2 * M_PI);
            mask[(x + radius) * M + (y + radius)] = value;
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
    int N, M, radius;
    if (argc < 3) {
        std::cout << "Please enter the array size, N (arg1) and mask size M, (arg2)" << std::endl;
        return 1;
    }
    
    if (std::atoi(argv[2]) % 2 !=1){
        std::cout << "Mask size must be odd." << std::endl;
        return 1;
    }
    else {
        N = std::atoi(argv[1]);
        M = std::atoi(argv[2]);
        radius = M / 2;
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
    
    // initilize masks (precalculated 1d norm distribution)
    float h_mask1d[5] = {0.05448868, 0.24420134, 0.40261995, 0.24420134, 0.05448868};
    float *h_mask2d = (float *)malloc(M * M * sizeof(float));
    
    initialize_gaussian_kernel(h_mask2d, M, radius);

    // initilize matrix
    for (int i = 0; i < N * N; i++) {
        h_in[i] = rand() % 255;
    }

    checkCudaError(cudaMemcpy(d_in, h_in, matrixSize, cudaMemcpyHostToDevice));

    // create timer
    float gpu_elapsed_time_ms;
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    
    //============= SEPARABLE CONVOLUTION ==================//
    // copy mask to constant memory
    copy_to_constant(h_mask1d, M * sizeof(float));

    checkCudaError(cudaDeviceSynchronize());
    checkCudaError(cudaEventRecord(start, 0));
    // execute cuda command
    for (int i = 0; i < N_TRIALS; i++){
        launch_row_convolution(d_in, d_buffer, N, M, radius);
        launch_col_convolution(d_buffer, d_out, N, M, radius);
    }
    checkCudaError(cudaEventRecord(stop, 0));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop));
    std::cout << "Separable convolution mean time elasped for "<< N_TRIALS << " trials: "<< gpu_elapsed_time_ms/N_TRIALS << " ms." << std::endl;

    //============= 2D CONVOLUTION ==================//
    // copy mask to constant memory
    copy_to_constant(h_mask2d, M * M * sizeof(float));

    checkCudaError(cudaDeviceSynchronize());
    checkCudaError(cudaEventRecord(start, 0));
    // execute cuda command
    for (int i = 0; i < N_TRIALS; i++){
        launch_tiled_convolution(d_in, d_out, N, M, radius);
    
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

    cudaFree(d_in);
    cudaFree(d_buffer);
    cudaFree(d_out);
    return 0;
}