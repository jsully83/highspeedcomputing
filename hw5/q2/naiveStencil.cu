
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


#define n 8

void initArrays(float *a, float *b){
    for(int i = 0; i < n * n; i++){
        a[i] = 0.0;
        b[i] = 1.0;
    }
}
__global__ void naiveStencilKernel(float *d_a, float *d_b){

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    // int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    // int tidz = blockIdx.z * blockDim.z + threadIdx.z;
    
    printf("blockiIdx.x = %i\n", blockIdx.x);
    printf("blockDim.x = %i\n", blockDim.x);
    printf("threadIdx.x = %i\n", threadIdx.x);

    if(tidx > 0 && tidx < n-1){
        d_a[tidx] = d_b[tidx];
    }



}


int main(int argc, char* argv[]){
    
    float *h_a, *h_b;
    float *d_a, *d_b;
    // size_t tensorSize = n * n * n * sizeof(float);
    size_t tensorSize = n * n * sizeof(float);

    

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

    // copy arrays to device
    cudaMemcpy(d_a, h_a, tensorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, tensorSize, cudaMemcpyHostToDevice);

    const unsigned int blockSize = n * n ;
    const unsigned int gridSize = (n + blockSize - 1) / blockSize;
    dim3 dimGrid(gridSize, gridSize);
    dim3 dimBlock(blockSize, blockSize);

    // ============  KERNEL LAUNCH ============
    cudaEventRecord(start, 0);

    naiveStencilKernel<<<dimGrid, dimBlock>>>(d_a, d_b); 

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
    printf("A block has %i threads in a grid with %i blocks for a total of %i threads.\n", blockSize, gridSize, blockSize*gridSize);    

    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(h_a);
    cudaFree(h_b);

    return 0;
}

