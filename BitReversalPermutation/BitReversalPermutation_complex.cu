# Copyright 2019, Dimitra S. Kaitalidou, All rights reserved

#include "stdio.h"
#include "stdlib.h"
#include "cuda.h"
#include "math.h"

#define N 256
#define THR_PER_BL 8
#define BL_PER_GR 32

__global__ void kernel1(int* D, int* Q, int k) {

	// Find index
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int block = (int)(i / (2 * k));
	int j;

	if (i % 2 == 0) { j = 2 * block * k + (int)(i / 2) - k * ((int)(i / (2 * k))); }
	else { j = (2 * block + 1) * k + (int)(i / 2) - k * ((int)(i / (2 * k))); }

	// Assign the values to the output array
	Q[j] = D[i];
}

__global__ void kernel2(int* D, int* Q) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Assign the values of the output array back to the input array
	D[i] = Q[i];
}

int main(int argc, char** argv)
{

	// Initialize variables
	int *D;
	int *Q;
	int *dev_d;
	int *dev_q;
	int bits = (int)(log(N) / log(2));

	// Allocate the memory on the CPU
	D = (int*)malloc(N * sizeof(int));
	if (D == NULL) { printf("Allocation failure"); }

	// Allocate the memory on the CPU
	Q = (int*)malloc(N * sizeof(int));
	if (Q == NULL) { printf("Allocation failure"); }

	// printf("INPUT\n");
	for (int i = 0; i < N; i++)
	{
		D[i] = rand() % 100;
		Q[i] = 0;

		// Print the initial values - optional
		// printf("The number %d is: ", i);
		// printf("%d \n", D[i]);
	}

	// Allocate memory in the GPU
	cudaMalloc((void**)&dev_d, N * sizeof(int));
	cudaMalloc((void**)&dev_q, N * sizeof(int));

	// Parallel Programming in CUDA C
	// Copy the array 'd' to the GPU
	// Copy the array 'q' to the GPU
	cudaMemcpy(dev_d, D, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_q, Q, N * sizeof(int), cudaMemcpyHostToDevice);
	dim3 dimBlock(THR_PER_BL, 1);
	dim3 dimGrid(BL_PER_GR, 1);

	for (int i = 1; i < bits; i++)
	{
		kernel1 << <dimGrid, dimBlock >> > (dev_d, dev_q, N / (int)(pow(2.0, (double)i)));
		kernel2 << <dimGrid, dimBlock >> > (dev_d, dev_q);
	}

	// Copy the array 'd' back from the GPU to the CPU
	// Copy the array 'q' back from the GPU to the CPU
	cudaMemcpy(Q, dev_q, N * sizeof(int), cudaMemcpyDeviceToHost);

	/* Print results - optional
	printf("RESULT\n");
	for (int i = 0; i < N; i++)
	{
		// Print the final values - optional
		printf("The number %d is: ", i);
		printf("%d \n", Q[i]);
	}
	*/

	// Free the memory allocated on the GPU
	cudaFree(dev_d);
	cudaFree(dev_q);

	// Return
	return 0;
}
