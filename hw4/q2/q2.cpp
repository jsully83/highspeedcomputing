#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

using namespace std;

int main(int argc, char* argv[]) {
    mt19937 rng(time(0));

    int N = 2000000; // Number of integers to generate
    int B = atoi(argv[1]); // Number of bins
    int root = 0; // Root process
    int sum = 0; // Sum of counts. Used to calculate displacements
    int bins[B]; // Array to store the counts of each bin
    int rootbin[B];
    int rank, size; // for storing this process' rank, and the number of processes
    
    // Initialize MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check that the number of processes is equal to the number of bins.
    // If not, abort.
    if(size!=B){
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate memory for arrays.
    int *nums = (int *)malloc(N*sizeof(int));
    int *displs = (int *)malloc(size*sizeof(int));
    int *sendcounts = (int *)malloc(size*sizeof(int));
    

    // Calculate send counts and displacements, initialize bins to 0
    for (int i=0; i<size; i++) {
        bins[i] = 0;
        rootbin[i] = 0;
        sendcounts[i] = (N/B);
        displs[i] = sum;
        sum += sendcounts[i];
    }

    // Allocate memory for the receive buffer
    int *recvbuf = (int *)malloc(sendcounts[rank]*sizeof(int));

    // Generate N random integers.
    for (int i = 0; i < N; i++) {
        // Get a random integer between one and one million.
        int n = rng() % N + 1;
        nums[i] = n;
    }

    // Print numbers, displacements and calculated for the root
    // if(rank == root && size <= 10 && N <= 100){
    //     for(int i=0; i<N; i++){
    //         printf("nums[%i] = %i\n", i, nums[i]);
    //     }
    //     for(int i=0; i<size; i++){
    //         printf("sendcounts[%i] = %i\tdispls[%i] = %i\n",i ,sendcounts[i], i, displs[i]);
    //     }
    // }
    

    // start the timer
    double start = MPI_Wtime();


    // Scatter the numbers to each process
    MPI_Scatterv(nums, sendcounts, displs, MPI_INT, recvbuf, sendcounts[rank], MPI_INT, root, MPI_COMM_WORLD);
    
    // printf("Rank %d: ", rank);
    // for (int i = 0; i < sendcounts[rank]; i++) {
    //     printf("%i\t", recvbuf[i]);
    // }
    // printf("\n");

    int bin;
    for (int i = 0; i < sendcounts[rank]; i++){ 
        bin = round(recvbuf[i] / (N/B));
        bins[bin]++;
    }

    // printf("Rank %d:", rank);
    // for (int i = 0; i < sendcounts[rank]; i++) {
    //     printf("bin[%i] = %i\t", i, bins[i]);
    // }
    // printf("\n");


    // Reduce the bins to the root process
    MPI_Reduce(&bins, &rootbin, B, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

    // stop the timer
    double end = MPI_Wtime();
    
    if(rank == root){
        printf("Time taken: %f\n", (end - start));
        for(int i=0; i<B; i++){
            printf("bin[%i] = %i\n", i, rootbin[i]);
        }
    }

    free(nums);
    free(recvbuf);
    free(sendcounts);
    free(displs);

    MPI_Finalize();

    return 0;
}