
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <random>

__global__ void histogram(int N, int *nums, int *bins, int divisor, int n_bins){
    // global thread id
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // check for out of bounds
    if(id < N){
        atomicAdd(&bins[(nums[id]/divisor) % n_bins], 1);
    }
}

int main(int argc, char* argv[]){
    
    srand(time(NULL));

    // number of integers to generate
    int N = atoi(argv[1]);
    int N_RANGE  = 10000000;
    size_t n_bytes = N*sizeof(int);

    // number of bins
    int N_BINS = 32;
    size_t bin_bytes = N_BINS*sizeof(int);

    // allocate memory on the host
    int *h_nums, *h_bins;
    
    h_nums = (int*)malloc(n_bytes);
    h_bins = (int*)malloc(bin_bytes);

    // allocate shared memory on the device
    int *d_nums;
    int *d_bins;

    cudaMalloc(&d_nums, n_bytes);
    cudaMalloc(&d_bins, bin_bytes);

    // what we should divide by to get the bin
    int divisor = N_RANGE/(N_BINS);

    // number of threads in a block
    int blockSize = atoi(argv[2]);
    
    // number of blocks in a grid
    int gridSize = (int)ceil((float)N/blockSize);

    // device event objects
    float gpu_elapsed_time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // generate N random numbers onthe host
    for (int i = 0; i < N; i++) {
        h_nums[i] = rand() % N_RANGE + 1;
    }

    // set bins to zero
    for (int i = 0; i < N_BINS; i++) {
        h_bins[i] = 0;
    }


    // copy N integers to device
    cudaMemcpy(d_nums, h_nums, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, h_bins, bin_bytes, cudaMemcpyHostToDevice);
    
    // start timing all the device operations
    cudaEventRecord(start, 0);
    for(int i=0; i < 10; i++){
        histogram<<<gridSize, blockSize>>>(N, d_nums, d_bins, divisor, N_BINS);
    }
    
    
    cudaMemcpy(h_nums, d_nums, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bins, d_bins, bin_bytes, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop, 0);

    // print the bin counts
    for (int i = 0; i < N_BINS; i++) {
        printf("bin[%i] = %i\n", i, h_bins[i]/10);
    }

    // check how many elements were binned
    int elements=0;
    for(int i=0; i < N_BINS; i++){
        elements += h_bins[i];
    }
    elements = elements / 10;

    // check if all elements were binned
    if(elements == N){
        printf("Success! The histogram counted %i numbers!\n", elements);
    }
    else{
        printf("Not all numbers counted in the histogram! Only got %i\n", elements);
    }

    // compute time elapse on GPU computing
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Averaged elapsed time for 10 loops on the GPU: %f ms.\n", gpu_elapsed_time_ms/10);
    printf("A block has %i threads in a grid with %i blocks for a total of %i threads.\n", blockSize, gridSize, blockSize*gridSize);


    // wait for the device to finish so that we see the message
    cudaDeviceSynchronize();
    cudaFree(d_nums);
    cudaFree(d_bins);
    free(h_nums);
    free(h_bins);

    return 0;
}