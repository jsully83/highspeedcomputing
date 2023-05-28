# highspeedcomputing

To test the programs in HW5 first run

    sh hw5/runInteractiveGPU.sh

to start an interactive GPU sessions.  Then type:

    . startup.sh

to load the correct modules for compilation.

Question 1

in hw5/q1 You can run:
    
    ./deviceQuery

to get information about the GPU.  

Next, run:

    make

to compile both programs.  There are two files called runCudaHistogram.sh and runOMPHistogram.sh  You can edit the number of integrers used and the number of threads in the cuda program used.  To run type:

    sh runCudaHistogram.sh
    sh runOMPHistogram.sh



    
