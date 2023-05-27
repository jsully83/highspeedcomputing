// To measure effects of memory coalescing. Coalescing.cu 
// B. Wilkinson Jan 30, 2011

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define BlockSize 16 		// Size of blocks, 32 x 32 threads, fixed, used globally

__global__ void gpu_Comput (int *h, int N, int T) {

// Array loaded with global thread ID that acesses that location

	int col = threadIdx.x + blockDim.x * blockIdx.x; 
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	int threadID = col + row * N;
	int index = row + col * N;	// sequentially down each row

	for (int t = 0; t < T; t++)	// loop to repeat to reduce other time effects
	   h[index] = threadID;  	// load array with flattened global thread ID
}

void printArray(int *h, int N) {

	printf("Results of computation, every N/8 numbers, eight numbers\n");

	for (int row = 0; row < N; row += N/8) {
	  for (int col = 0; col < N; col += N/8) 
	 	printf("%6d  ", h[col + row * N]);
	  printf("\n"); 
	}
}

int main(int argc, char *argv[])  {

	int T = 100;		 	// number of iterations, entered at keyboard
	int B = 1;		 	// number of blocks, entered at keyboard
	char key;

	int *h, *dev_h;			// ptr to array holding numbers on host and device

	cudaEvent_t start, stop;     	// cuda events to measure time
	float elapsed_time_ms1;
	cudaEventCreate( &start ); 
	cudaEventCreate( &stop );

/* ------------------------- Keyboard input -----------------------------------*/

do {  // loop to repeat complete program	

	printf("Grid Structure 2-D grid, 2-D blocks\n");
	printf("Blocks fixed at 16 x 16 threads, 512 threads, max for compute cap. 1.x\n");
	printf("Enter number of blocks in grid, each dimension, currently %d\n",B);
	scanf("%d",&B);
	printf("Enter number of iterations, currently %d\n",T);
	scanf("%d",&T);

	int N = B * BlockSize;		// size of data array, given input data

	printf("Array size (and total grid-block size) %d x %d\n", N, N);

	dim3 Block(BlockSize, BlockSize);	//Block structure, 32 x 32 max
	dim3 Grid(B, B);			//Grid structure, B x B	

/* ------------------------- Allocate Memory-----------------------------------*/

	int size = N * N * sizeof(int);		// number of bytes in total in array
	h = (int*) malloc(size);		// Array on host

	cudaMalloc((void**)&dev_h, size);	// allocate device memory

/* ------------------------- GPU Computation -----------------------------------*/
			
	cudaEventRecord( start, 0 );
	
	gpu_Comput<<< Grid, Block >>>(dev_h, N, T);
	
	cudaEventRecord( stop, 0 );     	// instrument code to measue end time
	cudaEventSynchronize( stop );		// wait for all work done by threads
	cudaEventElapsedTime( &elapsed_time_ms1, start, stop );

	cudaMemcpy(h,dev_h, size ,cudaMemcpyDeviceToHost);  //Get results to check

	printArray(h,N);
	printf("\nTime to calculate results on GPU: %f ms.\n", elapsed_time_ms1);

/* -------------------------REPEAT PROGRAM INPUT-----------------------------------*/

	printf("\nEnter c to repeat, return to terminate\n");

	scanf("%c",&key);
	scanf("%c",&key);

} while (key == 'c'); // loop of complete program
 
/* --------------  clean up  ---------------------------------------*/
	
	free(h);
	cudaFree(dev_h);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
  
	return 0;
}
