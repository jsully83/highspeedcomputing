
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define n 8
#define BLOCKSIZE 4
#define TILESIZE 4
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
                    printf("h_a[%i][%i][%i] = %f\n", i, j, k, b[i*n*n + j*n + k]);
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

    if(i == 0 || i == n - 1 || j == 0 || j == n - 1 || k == 0 || k == n - 1) return;

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
    
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
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

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));

}





// __global__ void tiledStencil(float *d_a, float *d_b){
    
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     int col = blockIdx.y * blockDim.y + threadIdx.y;
//     int tensor = blockIdx.z * blockDim.z + threadIdx.z;

//     // extern __shared__ float tile[];
//     __shared__ float tile[TILESIZE][TILESIZE][TILESIZE];

    

//     int gridIdx = (row * n * n) + (col * n) + tensor;

//     if(gridIdx >= n*n*n) return;

//     // printf("row: %i, col: %i, tensor: %i tileIdx_row: %i, tileIdx_col: %i, tileIdx_tensor: %i\n", row, col, tensor, tileIdx_row, tileIdx_col, tileIdx_tensor);
    

//     // int tileIdx = blockIdx.x * blockDim.x + (threadIdx.x + RADIUS) + 
//     //               blockIdx.y * blockDim.y + (threadIdx.y + RADIUS) +
//     //               blockIdx.z * blockDim.z + (threadIdx.z + RADIUS);
                  

    
//     // if(row >= n - 1) return;
//     // if(col >= n - 1) return;
//     // if(tensor >= n - 1) return;

//     tile[threadIdx.x][threadIdx.y][threadIdx.z]= d_b[gridIdx];
//     __syncthreads(); 

//     // printf("d_b[%i] = %f, tile[%i][%i][%i] = %f\n", gridIdx, d_b[gridIdx], threadIdx.x, threadIdx.y, threadIdx.z, tile[threadIdx.x][threadIdx.y][threadIdx.z]);

//     // int tileIdx = (blockIdx.x * pow(blockDim.x, 2)) + 
//     //              (blockIdx.y * pow(blockDim.y, 1)) + 
//     //              (blockIdx.z * pow(blockDim.z, 0));

//     // we shifted everything by RADIUS so we don't need the last thread in each dimension
//     if (threadIdx.x < 1 || threadIdx.x >= n - 1) return;
//     if (threadIdx.y < 1 || threadIdx.y >= n - 1) return;
//     if (threadIdx.z < 1 || threadIdx.z >= n - 1) return;

    
//     // load the data from global memory into shared memory
//     // tile[tileIdx] = d_b[gridIdx];
    
//        // use the 0 thread to load the halo data
//     // if(threadIdx.x < RADIUS)
//     //     tile[tileIdx - RADIUS] = d_b[gridIdx - RADIUS];
//     //     tile[tileIdx + blockDim.x] = d_b[gridIdx + blockDim.x];
    

//     // if(row == 1 && col == 1 && tensor == 1)
//     //     for(int i = 0; i <= TILESIZE + 1; i++)
//     //         printf("tile[%i] = %f\n", i, tile[i]);
    

//     // printf("row=%i, col=%i, ten=%i\n", row, col, tensor);
//     // printf("d_b[%i] = %f, tile[%i] = %f, blockIdx.x=%i, blockIdx.y=%i, blockIdx.z=%i threadIdx.x=%i, threadIdx.y=%i, threadIdx.z=%i,  \n", gridIdx, d_b[gridIdx], tileIdx, tile[tileIdx], blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
//     // printf("blockIdx.x=%i, blockIdx.y=%i, blockIdx.z=%i\n", blockIdx.x, blockIdx.y, blockIdx.z);
//     // printf("threadIdx.x=%i, threadIdx.y=%i, threadIdx.z=%i\n", threadIdx.x, threadIdx.y, threadIdx.z);


//     d_a[gridIdx] = 0.8 * (tile[threadIdx.x-1][threadIdx.y][threadIdx.z] + tile[threadIdx.x+1][threadIdx.y][threadIdx.z] + 
//                       tile[threadIdx.x][threadIdx.y-1][threadIdx.z] + tile[threadIdx.x][threadIdx.y+1][threadIdx.z] + 
//                       tile[threadIdx.x][threadIdx.y][threadIdx.z-1] + tile[threadIdx.x][threadIdx.y][threadIdx.z+1]);
    
//     // __syncthreads();
//     return;

// }


__device__ int getTileAdjacent(int i, int j, int k){
    return ((j) * blockDim.x * blockDim.x + (k) * blockDim.x + (i));
}


__global__ void tiledStencil(float *d_a, float *d_b){

    // create the tiles
    extern __shared__ float tile[];


    // global row column and slice indicies
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tensor = blockIdx.z * blockDim.z + threadIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    float temp;

    tile[getTileAdjacent(tx, ty, tz)] = d_b[getAdjacent(row,col,tensor)];

    __syncthreads();

    // printf("blockIdx.x=%i, blockIdx.y=%i, blockIdx.z=%i threadIdx.x=%i, threadIdx.y=%i, threadIdx.z=%i, row = %i, col = %i, ten = %i, d_b[%i] = %f, tile[%i] = %f\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, row, col, tensor, getAdjacent(row,col,tensor), d_b[getAdjacent(row,col,tensor)], getTileAdjacent(tx, ty, tz), tile[getTileAdjacent(tx, ty, tz)]);
 

    if(row > RADIUS - 1 && row < n - RADIUS && col > RADIUS - 1 && col < n - RADIUS  && tensor > RADIUS - 1 && tensor < n - RADIUS){
        if(tx < blockDim.x - RADIUS && ty < blockDim.y - RADIUS && tz < blockDim.z - RADIUS){
            if(blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0){
                if(tx > RADIUS - 1 && ty > RADIUS - 1 && tz > RADIUS - 1){
                    temp = 0.8 * (tile[getTileAdjacent(tx-1, ty, tz)] + tile[getTileAdjacent(tx+1, ty, tz)] + 
                                  tile[getTileAdjacent(tx, ty-1, tz)] + tile[getTileAdjacent(tx, ty+1, tz)] + 
                                  tile[getTileAdjacent(tx, ty, tz-1)] + tile[getTileAdjacent(tx, ty, tz+1)]);
                    // printf("blockIdx.x=%i, blockIdx.y=%i, blockIdx.z=%i threadIdx.x=%i, threadIdx.y=%i, threadIdx.z=%i, row = %i, col = %i, ten = %i\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, row, col, tensor);
                    // printf("temp = %f, %f, %f, %f, %f, %f, %f\n", temp, tile[getTileAdjacent(tx-1, ty, tz)], tile[getTileAdjacent(tx+1, ty, tz)], tile[getTileAdjacent(tx, ty-1, tz)], tile[getTileAdjacent(tx, ty+1, tz)], tile[getTileAdjacent(tx, ty, tz-1)], tile[getTileAdjacent(tx, ty, tz+1)]); 
                }
            }
            if(blockIdx.y == 0 && blockIdx.x != 0){
                if(tz > RADIUS - 1){
                    temp = 0.8 * (tile[getTileAdjacent(tx-1, ty, tz)] + tile[getTileAdjacent(tx+1, ty, tz)] + 
                                  tile[getTileAdjacent(tx, ty-1, tz)] + tile[getTileAdjacent(tx, ty+1, tz)] + 
                                  tile[getTileAdjacent(tx, ty, tz-1)] + tile[getTileAdjacent(tx, ty, tz+1)]);
                    // printf("blockIdx.x=%i, blockIdx.y=%i, blockIdx.z=%i threadIdx.x=%i, threadIdx.y=%i, threadIdx.z=%i, row = %i, col = %i, ten = %i\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, row, col, tensor);
                    // printf("seed = %f, %f, %f, %f, %f, %f, %f\n", tile[getTileAdjacent(tx, ty, tz)], tile[getTileAdjacent(tx-1, ty, tz)], tile[getTileAdjacent(tx+1, ty, tz)], tile[getTileAdjacent(tx, ty-1, tz)], tile[getTileAdjacent(tx, ty+1, tz)], tile[getTileAdjacent(tx, ty, tz-1)], tile[getTileAdjacent(tx, ty, tz+1)]); 
                }   
            }
            if(blockIdx.x == 0 && blockIdx.y != 0){
                if(ty > RADIUS - 1){
                    temp = 0.8 * (tile[getTileAdjacent(tx-1, ty, tz)] + tile[getTileAdjacent(tx+1, ty, tz)] + 
                                  tile[getTileAdjacent(tx, ty-1, tz)] + tile[getTileAdjacent(tx, ty+1, tz)] + 
                                  tile[getTileAdjacent(tx, ty, tz-1)] + tile[getTileAdjacent(tx, ty, tz+1)]);
                    // printf("blockIdx.x=%i, blockIdx.y=%i, blockIdx.z=%i threadIdx.x=%i, threadIdx.y=%i, threadIdx.z=%i, row = %i, col = %i, ten = %i\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, row, col, tensor);
                    // printf("seed = %f, %f, %f, %f, %f, %f, %f\n", tile[getTileAdjacent(tx, ty, tz)], tile[getTileAdjacent(tx-1, ty, tz)], tile[getTileAdjacent(tx+1, ty, tz)], tile[getTileAdjacent(tx, ty-1, tz)], tile[getTileAdjacent(tx, ty+1, tz)], tile[getTileAdjacent(tx, ty, tz-1)], tile[getTileAdjacent(tx, ty, tz+1)]); 
                }   
            }
        }
    __syncthreads();
    d_a[getAdjacent(row,col,tensor)] = temp;
    }
}



int main(int argc, char* argv[]){
    
    float *h_a, *h_b;
    float *d_a, *d_b;
    float a[n][n][n], b[n][n][n];
    size_t tensorBytes = n * n * n * sizeof(float);

    // allocate memory on host
    checkCudaErrors(cudaMallocHost(&h_a, tensorBytes));
    checkCudaErrors(cudaMallocHost(&h_b, tensorBytes));



    // init arrays
    initArrayCPU(a,b);
    initArrayDevice(h_a,a);
    initArrayDevice(h_b,b);
    checkArray(h_b, b);

    // run cpu stencil
    cpuStencil(a, b);
 
    // allocate memory on device
    checkCudaErrors(cudaMalloc(&d_a, tensorBytes));
    checkCudaErrors(cudaMalloc(&d_b, tensorBytes));

    // copy arrays to device
    checkCudaErrors(cudaMemcpy(d_a, h_a, tensorBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, tensorBytes, cudaMemcpyHostToDevice));

    // launchNaiveStencil(d_a, d_b, h_a, tensorBytes);

    // checkArray(h_a, a);


       // ============  CUDA SETUP ============
    // device event objects
    float gpu_elapsed_time_ms;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaMemcpy(d_a, h_a, tensorBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, tensorBytes, cudaMemcpyHostToDevice));

    // lets just make the tile size equal to the block size
    // for easier indexing.
    const unsigned int gridSize = (n + TILESIZE - 1) / TILESIZE;
    
    dim3 dimBlock(TILESIZE, TILESIZE, TILESIZE);
    dim3 dimGrid(gridSize, gridSize, gridSize);


    printf("Blocks/Grid = %i\tThreads/Block = %i\tThreads: %i\n", dimGrid.x*dimGrid.y*dimGrid.z, dimBlock.x*dimBlock.y*dimBlock.z, dimGrid.x*dimGrid.y*dimGrid.z*dimBlock.x*dimBlock.y*dimBlock.z);


    // allocate shared memory dynamically to the kernel.  
    // we need to take into account the tile size, the halo region, and the size of float
    size_t tileBytes = TILESIZE * TILESIZE * TILESIZE * sizeof(float);    

    // ============  KERNEL LAUNCH ============
    checkCudaErrors(cudaEventRecord(start, 0));

    tiledStencil<<<dimGrid, dimBlock, tileBytes>>>(d_a, d_b);
    getLastCudaError("Kernel Error");
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_a, d_a, tensorBytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));

    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop));

    printf("Tiled stencil elapsed time: %f ms.\n", gpu_elapsed_time_ms/10);
    
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));


    // checkArray(h_a, a);



    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    getLastCudaError("Last Error");
    return 0;
}

