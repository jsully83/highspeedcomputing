#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

int main(int argc, char *argv[]) {
    srand(time(NULL));

    // number of integers to generate
    int N = atoi(argv[1]);
    int N_RANGE  = 10000000;
    size_t n_bytes = N*sizeof(int);

    // number of bins
    int N_BINS = 32;
    size_t bin_bytes = N_BINS*sizeof(int);

    // allocate memory on the host
    int *nums, *bins;

    nums = (int*)malloc(n_bytes);
    bins = (int*)malloc(bin_bytes);

    int divisor = N_RANGE/(N_BINS);

    // generate N random numbers on the host
    for (int i = 0; i < N; i++) {
        nums[i] = rand() % N_RANGE + 1;
    }

    // set bins to zero
    for (int i = 0; i < N_BINS; i++) {
        bins[i] = 0;
    }

    // start the timer
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // parallelize the binning and run it 10 times
    int loops = 10;
    for(int i=0; i < loops; i++){
        #pragma omp parallel
        {
            int nthreads = omp_get_num_threads();
            int ithread = omp_get_thread_num();

            // each thread has its own private bins
            int *private_bins = (int*)malloc(bin_bytes);
            for (int i = 0; i < N_BINS; i++) {
                private_bins[i] = 0;
            }

            // each thread gets its own chunk of the array
            int chunk_size = N/nthreads;
            int start = ithread*chunk_size;
            int end = start + chunk_size;

            // each thread bins its own chunk
            for (int i = start; i < end; i++) {
                private_bins[(nums[i] / divisor) % N_BINS]++;
            }

            // if there are leftover elements, the master thread bins them
            if (ithread == 0) {
                for (int i = N - (N % nthreads); i < N; i++) {
                    private_bins[(nums[i] / divisor) % N_BINS]++;
                }
            }

            // reduction
            #pragma omp critical
            {
                for (int i = 0; i < N_BINS; i++) {
                    bins[i] += private_bins[i];
                }
            }
        }
    }

    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;

    // print the bin counts
    for (int i = 0; i < N_BINS; i++) {
        printf("bin[%i] = %i\n", i, bins[i]/10);
    }

    // check how many elements were binned
    int elements=0;
    for(int i=0; i < N_BINS; i++){
        elements += bins[i];
    }
    elements = elements/loops;

    if(elements == N){
        printf("Success! The histogram counted %i numbers!\n", elements);
    }
    else{
        printf("Not all numbers counted in the histogram! Only got %i\n", elements);
    }

    printf("Average elapsed time for 10 loops on the CPU: Time: %.10f ms.\n", elapsed*1000/loops);
    

    return 0;
}
