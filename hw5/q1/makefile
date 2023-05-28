all:
	nvcc histogram.cu -o histogram
	gcc -fopenmp histogramOMP.c -o histogramOMP

clean:
	rm histogram
	rm histogramOMP