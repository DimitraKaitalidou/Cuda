// Copyright 2019, Dimitra S. Kaitalidou, All rights reserved

#include "stdio.h"
#include "stdlib.h"
#include "cuda.h"

#define N 256
#define THR_PER_BL 8
#define BL_PER_GR 32

__global__ void kernel(int* D, int* q, int k) {

	// Find index of i row and j column of the distance array
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(D[i * N + j] > D[i * N + k] + D[k * N + j])
	{
		D[i * N + j] = D[i * N + k] + D[k * N + j];
		q[i * N + j] = k;
	}
}

int main(int argc, char** argv)
{

	// Initialize variables
	int **D;
	int *d;
	int *dev_d;
	int *q;
	int *dev_q;

	// Allocate the memory on the CPU
	D = (int**)malloc(N * sizeof(int*));
	if (D == NULL) { printf("Allocation failure"); }
	for(int i = 0; i < N; i++)
	{
		D[i] = (int*)malloc(N * sizeof(int));
		if(D[i] == NULL) { printf("Allocation failure"); }
	}
	d = (int*)malloc(N * N * sizeof(int));
	if(d == NULL) { printf("Allocation failure"); }
	q = (int*)malloc(N * N * sizeof(int));
	if(q == NULL) { printf("Allocation failure"); }

	// Fill the arrays 'd' on the CPU
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			if(i == j)
			{
				D[i][j] = 0;
			}
			else
			{
				D[i][j] = rand() % 100;
			}
		}
	}

	// printf("The initial distances are:\n");
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			d[i * N + j] = D[i][j];
			q[i * N + j] = N + 1;

			// Print the distances - optional
			// printf("%d\n", d[i * N + j]);
		}
	}

	// Allocate memory in the GPU
	cudaMalloc((void**)&dev_d, N * N * sizeof(int));
	cudaMalloc((void**)&dev_q, N * N * sizeof(int));

	// Parallel Programming in CUDA C
	// Copy the array 'd' to the GPU
	// Copy the array 'q' to the GPU
	cudaMemcpy(dev_d, d, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_q, q, N * N * sizeof(int), cudaMemcpyHostToDevice);
	dim3 dimBlock(THR_PER_BL, THR_PER_BL);
	dim3 dimGrid(BL_PER_GR, BL_PER_GR);

	for(int k = 0; k<N; k++)
	{
		kernel << <dimGrid, dimBlock >> >(dev_d, dev_q, k);
	}

	// Copy the array 'd' back from the GPU to the CPU
	// Copy the array 'q' back from the GPU to the CPU
	cudaMemcpy(d, dev_d, N * N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(q, dev_q, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	/* Print results - optional
	printf("The smallest distances are:\n");
	for(int i = 0; i < N * N; i++)
	{
		printf("%d\n", d[i]);
	}

	printf("The intermediate nodes are:\n");
	for(int i = 0; i < N * N; i++)
	{
		printf("%d\n", q[i]);
	}
	*/

	// Free the memory allocated on the GPU
	cudaFree(dev_d);
	cudaFree(dev_q);

	// Return
	return 0;
}
