# Convolution with CUDA

### First start an interactive GPU node by sourcing runInteractiveGPU.sh

```
cd /highspeedcomputing/project
. runInteractiveGPU.sh
```

### This code uses gcc 11.1 and cuda 12.1.  To load these dependencies source the start up script

```
. startup.sh
```

### Compile the code by running make

```
make
```

### Run the naive code by using a command line argument to specify the matrix size.


```
nvprof ./naive 1024
```

### The output will be
```
CPU Convolution time: 199.928 ms
Naive result complete.
Constant result complete.

...nvprof text...

```

### The code is defaulted to a filter size of 5.  Change the global variable K to define a different <br>
### filter size and recompile the code.

```
make clean
make
```

### Run the tile and separable filter code by defining command line arguments for matrix size then filter size

```
nvprof ./tiles 1024 7
```

### the output will be 

```
Array size: 1024x1024
Mask size: 7
Separable convolution mean time elasped for 100 trials: 0.0311574 ms.
2D Convolution mean time elasped for 100 trials: 0.0599139 ms.

...nvprof text...
```