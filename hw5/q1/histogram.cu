
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <random>

#define N_RANGE 10000000
#define N_BINS 32

void gen_random_nums(int N, int* nums){
    // Generate N random integers.
    for (int i = 0; i < N; i++) {
        nums[i] = rand() % N_RANGE + 1;
    }
}

void bin_nums_CPU(int N, int *nums, int *bins){

    int bin;

    for (int i = 0; i < N; i++){ 
        bin = round(nums[i] / (N_RANGE/N_BINS));
        bins[bin]++;
        // printf("bin[%i] = %i\n", bin, bins[bin]);
    }
}

// __global__ void bin_nums_GPU()

int main(int argc, char* argv[]){
    
    srand(time(NULL));
    // Number of integers to generate
    int N = atoi(argv[1]); 

    int *nums;
    int *bins;


    // Allocation unified memory
    (int*)cudaMallocManaged(&nums, N*sizeof(int));
    (int*)cudaMallocManaged(&bins, N_BINS*sizeof(int));

    // device event objects
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // printf("generating numbers...");
    gen_random_nums(N, nums);
    bin_nums_CPU(N, nums, bins);

    // start timing all the device operations
    cudaEventRecord(start, 0);

    

    

    for (int i = 0; i < N_BINS; i++) {
        printf("bin[%i] = %i\t", i, bins[i]);
        printf("\n");
    }

    // wait for the device to finish so that we see the message
    cudaDeviceSynchronize();
    cudaFree(nums);
    cudaFree(bins);
    

    return 0;
}