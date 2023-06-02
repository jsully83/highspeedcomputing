
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

#define n 4
#define BLOCKSIZE 2
#define TILESIZE 2
#define RADIUS 1

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


void initArrayCPU(float a[n][n][n], float b[n][n][n]){
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++){
                a[i][j][k] = 0;
                b[i][j][k] = i*n*n + j*n + k;
    }
}


void initArrayDevice(float *b, float a[n][n][n]){
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++){
                b[i*n*n + j*n + k] = a[i][j][k];
                // printf("b[%i] = %f\n", i*n*n + j*n + k, b[i*n*n + j*n + k]);
            }
}

void checkArray(float *b, float a[n][n][n]){
    int errorNum = 0;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++)
                if(b[i*n*n + j*n + k] != a[i][j][k]){
                    errorNum++;
                    printf("Error at i = %i, j = %i, k = %i\n", i, j, k);
                    printf("a[%i][%i][%i] = %f\n", i, j, k, a[i][j][k]);
                    printf("b[%i][%i][%i] = %f\n", i, j, k, b[i*n*n + j*n + k]);
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
    
    int tid = i*n*n + j*n + k;
    // printf("d_b[%i] = %f\n", tid, d_b[tid]);

    

    if(i == 0 || i == n - 1 || j == 0 || j == n - 1 || k == 0 || k == n - 1)
        return;

    // printf("i=%i, j=%i, k=%i\n", i, j, k);
    // printf("blockIdx.x=%i, blockIdx.y=%i, blockIdx.z=%i\n", blockIdx.x, blockIdx.y, blockIdx.z);
    // printf("blockDim.x=%i, blockDim.y=%i, blockDim.z=%i\n", blockDim.x, blockDim.y, blockDim.z);
    // printf("threadIdx.x=%i, threadIdx.y=%i, threadIdx.z=%i\n", threadIdx.x, threadIdx.y, threadIdx.z);
    

    d_a[tid] = 0.8 * (d_b[getAdjacent(i-1,j,k)] + d_b[getAdjacent(i+1,j,k)] + 
                      d_b[getAdjacent(i,j-1,k)] + d_b[getAdjacent(i,j+1,k)] + 
                      d_b[getAdjacent(i,j,k-1)] + d_b[getAdjacent(i,j,k+1)]);
    
                    
    return;

}

void launchNaiveStencil(float *d_a, float *d_b, float *h_a, size_t tensorBytes){


    // ============  CUDA SETUP ============
    // device event objects
    float gpu_elapsed_time_ms;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    const unsigned int gridSize = (n + BLOCKSIZE - 1) / BLOCKSIZE;
    
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 dimGrid(gridSize, gridSize, gridSize);

    // ============  KERNEL LAUNCH ============
    checkCudaErrors(cudaEventRecord(start, 0));

    naiveStencil<<<dimGrid, dimBlock>>>(d_a, d_b); 
    getLastCudaError("Kernel Error");
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_a, d_a, tensorBytes, cudaMemcpyDeviceToHost));
    
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop));
    printf("Naive stencil elapsed time: %f ms.\n", gpu_elapsed_time_ms/10);
    
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    checkCudaErrors(cudaMemset(d_a, 0, tensorBytes));

    // checkArray(d_a, a);

}





__global__ void tiledStencil(float *d_a, float *d_b){
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int ten = blockIdx.z * blockDim.z + threadIdx.z;


    extern __shared__ float tile[];
    
    // we need a global ID for each thread and a tile ID for each thread
    // each tile needs to be shifted by RADIUS in each direction to account for the halo
    // each tile is the same size as a block so we index it with the thread ID


    int maxThreadId = ((n * n * n) / 3) - 1;
    int gridIdx = (row * n * n) + (col * n) + ten;
    int tileIdx = (threadIdx.x + RADIUS) * (BLOCKSIZE + 2 * RADIUS) +
                  (threadIdx.y + RADIUS) * (BLOCKSIZE + 2 * RADIUS) + 
                  threadIdx.z + RADIUS;
    
    // we shifted everything by RADIUS so we don't need the last thread in each dimension
    if (row >= n - 1) return;
    if (col >= n - 1) return;
    if (ten >= n - 1) return;
    
    // load the data from global memory into shared memory
    tile[tileIdx] = d_b[gridIdx];
    __syncthreads();    



    printf("row=%i, col=%i, ten=%i\n", row, col, ten);
    printf("gridIDx=%i, tileIdx = %i\n", gridIdx, tileIdx);
    printf("blockIdx.x=%i, blockIdx.y=%i, blockIdx.z=%i\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("threadIdx.x=%i, threadIdx.y=%i, threadIdx.z=%i\n", threadIdx.x, threadIdx.y, threadIdx.z);


    // d_a[tid] = 0.8 * (d_b[getAdjacent(i-1,j,k)] + d_b[getAdjacent(i+1,j,k)] + 
    //                   d_b[getAdjacent(i,j-1,k)] + d_b[getAdjacent(i,j+1,k)] + 
    //                   d_b[getAdjacent(i,j,k-1)] + d_b[getAdjacent(i,j,k+1)]);
    
                    
    return;

}

// void launchTiledStencil(float *d_a, float *d_b, float *h_a, size_t tensorBytes){  
    



int main(int argc, char* argv[]){
    
    float *h_a, *h_b;
    float *d_a, *d_b;
    float a[n][n][n], b[n][n][n];
    size_t tensorBytes = n * n * n * sizeof(float);

    // allocate memory on host
    checkCudaErrors(cudaMallocHost(&h_a, tensorBytes));
    checkCudaErrors(cudaMallocHost(&h_b, tensorBytes));

    // allocate memory on device
    checkCudaErrors(cudaMalloc(&d_a, tensorBytes));
    checkCudaErrors(cudaMalloc(&d_b, tensorBytes));

    // init arrays
    initArrayCPU(a,b);
    initArrayDevice(h_a,a);
    initArrayDevice(h_b,b);
    checkArray(h_b, b);

    // run cpu stencil
    cpuStencil(a, b);
 


    // copy arrays to device
    checkCudaErrors(cudaMemcpy(d_a, h_a, tensorBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, tensorBytes, cudaMemcpyHostToDevice));

    launchNaiveStencil(d_a, d_b, h_a, tensorBytes);
    
    


       // ============  CUDA SETUP ============
    // device event objects
    float gpu_elapsed_time_ms;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // lets just make the tile size equal to the block size
    // for easier indexing.
    const unsigned int gridSize = (n + TILESIZE - 1) / TILESIZE;
    
    dim3 dimBlock(TILESIZE, TILESIZE, TILESIZE);
    dim3 dimGrid(gridSize, gridSize, gridSize);

    // allocate shared memory dynamically to the kernel.  
    // we need to take into account the tile size, the halo region, and the size of float
    size_t tileBytes = (TILESIZE + 2 * RADIUS) * 
                       (TILESIZE + 2 * RADIUS) * 
                       (TILESIZE + 2 * RADIUS) * sizeof(float);

    printf("A block has %i threads.\n", BLOCKSIZE*BLOCKSIZE*BLOCKSIZE);
    printf("A grid has %d blocks.\n", gridSize*gridSize*gridSize);
    printf("There are a total of %i threads.\n",BLOCKSIZE*BLOCKSIZE*BLOCKSIZE*gridSize*gridSize*gridSize);  



    // ============  KERNEL LAUNCH ============
    checkCudaErrors(cudaEventRecord(start, 0));

    tiledStencil<<<dimGrid, dimBlock, tileBytes>>>(d_a, d_b);
    getLastCudaError("Kernel Error");
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_a, d_a, tensorBytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));

    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop));

    printf("Naive stencil elapsed time: %f ms.\n", gpu_elapsed_time_ms/10);
    
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));




    // checkArray(d_a, a);



    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    getLastCudaError("Last Error");
    return 0;
}

