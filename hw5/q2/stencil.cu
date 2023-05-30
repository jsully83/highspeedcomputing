
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


#define n 8
#define BLOCKSIZE 2




// void initArrays(float *a, float *b){
//     for(int i = 0; i < n; i++){
//         for(int j = 0; j < n; j++){
//             for(int k =0; k < n; k++){
//                 a[i * j * n + k] = 0.0;
//                 b[i * j * n + k] = 1.0;
//             }
//         }
//     }
// }



void initArrays(float *a, float *b){
    for(int i = 0; i < n; i++){
        a[i] = 0.0;
        b[i] = 1.0;
    }
}


// __global__ void naiveStencilKernel(float *d_a, float *d_b){
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     int k = blockIdx.z * blockDim.z + threadIdx.z;

//     if(i > 0 && i < n-1 && j > 0 && j < n-1 && k > 0 && k < n-1){
//         d_a[i][j][k] = 1 * (d_b[i-1][j][k] +
//                             d_b[i+1][j][k] +
//                             d_b[i][j-1][k] +
//                             d_b[i][j+1][k] +
//                             d_b[i][j][k-1] +
//                             d_b[i][j][k+1]);
//     }
// }


__global__ void naiveStencilKernel(float *d_a, float *d_b){

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int tidz = blockIdx.z * blockDim.z + threadIdx.z;

    // printf("tidx = %i\t tidy = %i\t tidz = %i\n", tidx, tidy, tidz);
    
    // the boundry condition is elements 1 to n-1
    if (tidx > 0 && tidx < n-1 && tidy > 0 && tidy < n-1 && tidz > 0 && tidz < n-1){
        d_a[tidx + tidy + tidz - 2] = (d_b[tidx-1] + d_b[tidy-1] + d_b[tidz-1] +
                                       d_b[tidx]   + d_b[tidy]   + d_b[tidz]   +
                                        d_b[tidx+1] + d_b[tidy+1] + d_b[tidz+1]);
    }
}


// __global__ void tiledStencilKernel(float *d_a, float *d_b){
    
//     __shared__ float tile[BLOCKSIZE];
//     int tid = blockIdx.x * BLOCKSIZE + threadIdx.x;
//     float tmp;
//     // printf("threadIdx.x = %i\n", threadIdx.x); // 0-3

//     // printf("blockIdx.x = %i\n", blockIdx.x); // 0-3
//     // printf("blockDim.x = %i\n", blockDim.x); // 4
//     // printf("gridDim.x = %i\n", gridDim.x); // 4
//     // printf("tid = %i\n", tid); // 0-15

//     // load the thread's data element into shared memory
//     for(int i = 0; i < n; i += blockDim.x){
        
        
//         tile[threadIdx.x] = d_b[i + threadIdx.x];
//         // printf("myBlock[%i] = %f\n", tid, myBlock[tid]);
//         // printf("d_b[%i] = %f\n", tid, d_b[tid]);

//         // ensure that all the threads have loaded their values into
//         // shared memory; otherwise, one thread might be computing on
//         // unitialized data.
//         __syncthreads();

//         //  compute the average of this thread's left and right neightbors
//         // tmp = (tile[threadIdx.x>0?tid-1:BLOCKSIZE] + tile[tid<BLOCKSIZE?tid+1:0]);
//         // tmp = (myBlock[tid>0?tid-1:BLOCKSIZE] + myBlock[tid<BLOCKSIZE?tid+1:0]) * 0.8f;
//         if(threadIdx.x>0&&threadIdx.x<blockDim.x){
//             tmp = (tile[threadIdx.x-1] + tile[threadIdx.x] + tile[threadIdx.x+1]);
//         }
        

//         // square the previous result and add my value, squared
//         // tile[tid] = tmp * tmp + tile[tid] * tile[tid];

//         // write the result back to global memory
       
//         d_a[tid] = tmp;

//     }


//         printf("myBlock[%i] = %f\t d_a[%i] = %f\n", tid, tile[tid], tid, d_a[tid]);


// }
int main(int argc, char* argv[]){

    
    float *h_a, *h_b;
    float *d_a, *d_b;
    size_t tensorSize = n * n * n * sizeof(float);
    // size_t tensorSize = n * n * n * sizeof(float);

    // allocate memory on host
    cudaMallocHost((void **) &h_a, tensorSize);
    cudaMallocHost((void **) &h_b, tensorSize);

    // allocate memory on device
    cudaMalloc(&d_a, tensorSize);
    cudaMalloc(&d_b, tensorSize);

    // device event objects
    float gpu_elapsed_time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // init arrays
    initArrays(h_a, h_b);

    // for(int i = 0; i < n; i++){
    //     printf("h_a[%i] = %f\t h_b[%i] = %f\n", i, h_a[i], i, h_b[i]);
    // }

    // copy arrays to device
    cudaMemcpy(d_a, h_a, tensorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, tensorSize, cudaMemcpyHostToDevice);


    // compute grid and block dimensions
    // unsigned int gridRows = (n + BLOCKSIZE - 1) / BLOCKSIZE;
    // unsigned int gridCols = (n + BLOCKSIZE - 1) / BLOCKSIZE;
    // unsigned int gridTensors = (n + BLOCKSIZE - 1) / BLOCKSIZE;

    // dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, BLOCKSIZE);
    dim3 dimGrid(gridRows, gridCols, gridTensors);

    unsigned int gridSize = (n + BLOCKSIZE - 1) / BLOCKSIZE;
    printf("A block has %i threads in a grid with %i blocks for a total of %i threads.\n", BLOCKSIZE, gridSize, BLOCKSIZE*gridSize);
    
    // ============  KERNEL LAUNCH ============
    cudaEventRecord(start, 0);
    // tiledStencilKernel<<<gridSize, BLOCKSIZE>>>(d_a, d_b); 
    // naiveStencilKernel<<<dimGrid, dimBlock>>>(d_a, d_b); 

    cudaMemcpy(h_a, d_a, tensorSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, tensorSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    // ========================================

    for(int i = 0; i < n  ; i++){
        printf("h_a[%i] = %f\n", i, h_a[i]);   

    }
    printf("Elapsed time: %f ms.\n", gpu_elapsed_time_ms/10);
    // printf("A block has %d threads in a grid with %d blocks for a total of %d threads.\n", BLOCKSIZE, dimGrid, dimBlock);

    // wait for the device to finish so that we see the message
    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(h_a);
    cudaFree(h_b);

    return 0;
}

