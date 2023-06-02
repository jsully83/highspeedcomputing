__global__ void tiledStencilKernel(float *data){
    
    __shared__ float myblock[1024];
    int tid = threadIdx.x;
    float tmp;

    // load the thread's data element into shared memory
    myBlock[tid] = data[tid];

    // ensure that all the threads have loaded their values into
    // shared memory; otherwise, one thread might be computing on
    // unitialized data.
    __syncthreads();

    //  compute the average of this thread's left and right neightbors
    tmp = (myBlock[tid>0?tid-1:1023] + myBlock[tid<1023?tid+1:0]) * 0.5f;

    // square the previous result and add my value, squared
    myBlock[tid] = tmp * tmp + myBlock[tid] * myBlock[tid];

    // write the result back to global memory
    data[tid] = myBlock[tid];
}