// HW4 Q1 Uses MPI to pass a counter to each process which prints then increments until
// it's passed to all processes.  Then the counter is decremented and passed back to the
// original process.  This repeats until the counter reaches 0.
#include "mpi.h"
#include <stdio.h>

int main (int argc, char** argv) {
    const int PINGPONG_LIMIT = 32;
    int rank, size, namelen;
    int pingpongs = 0;

    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &namelen);

    // process 0 starts by printing the value, increments it then sends to the next process
    if(rank == 0){
        printf("Process %d at Node %s starts with value %d\n", rank, processor_name, pingpongs);
    }

    // Each process that != 0 should be set to recieve the value from the previous process.
    // Process 0 skips this increments the value and sends to process 1.  
    if(rank!=0){
        MPI_Recv(&pingpongs, 1, MPI_INT, (rank - 1) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d at Node %s received the value %d. ", rank, processor_name, pingpongs);
    }

    // We don't want the last process to send back to process 0. Every other process should 
    // now be set to receive the decremented value from the next higher process. Process size - 1
    // skips this and decrements the value to send to process size - 2.
    if(rank!=size-1){
        pingpongs++;
        printf("Process %d sends value %d to process %d.\n", rank, pingpongs, (rank + 1) % size);
        MPI_Send(&pingpongs, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
    }
    else{    
        
        MPI_Recv(&pingpongs, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d at Node %s received the value %d. ", rank, processor_name, pingpongs);
    }

    // We don't want process 0 to send again.
    if(rank!=0){
        pingpongs--;
        printf("Process %d sends value %d to process %d.\n", rank, pingpongs, (rank - 1) % size);
        MPI_Send(&pingpongs, 1, MPI_INT, (rank - 1) % size, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
