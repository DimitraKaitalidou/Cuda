// Copyright 2019, Dimitra S. Kaitalidou, All rights reserved

#include "cuda.h"
#include "stdio.h"
#include "stdlib.h"

#define N 256
#define THR_PER_BL 8
#define BL_PER_GR 32

__global__ void kernel1(int* D, int* q, int b) {
	
   int i = threadIdx.x + b * THR_PER_BL;
   int j = threadIdx.y + b * THR_PER_BL;

   float d, f, e;
   for(int k = b * THR_PER_BL; k < (b + 1) * THR_PER_BL; k++)
      {
         d = D[i * N + j];
         f = D[i * N + k];
         e = D[k * N + j];

         __syncthreads();
            
         if(d > f + e)
            {
               D[i * N + j] = f + e;
               q[i * N + j] = k;
            }
      }
}

__global__ void kernel2(int* D, int* q, int b) {

   int i, j;
   if (blockIdx.y == 0)
      {
         j = b * blockDim.y + threadIdx.y;
         if (blockIdx.x >= b)
            {
               i = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
            }
         else
            {
               i = blockIdx.x * blockDim.x + threadIdx.x;
            }
      }
   else
      {
         i = b * blockDim.y + threadIdx.y;
         if (blockIdx.x >= b)
            {
               j = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
            }
         else
            {
               j = blockIdx.x * blockDim.x + threadIdx.x;
            }
      }

   float d, f, e;
   for(int k = b * THR_PER_BL; k < (b + 1) * THR_PER_BL; k++)
      {
         d = D[i * N + j];
         f = D[i * N + k];
         e = D[k * N + j];
         
         __syncthreads();
         
         if (d > f + e)
            {
               D[i * N + j] = f + e;
               q[i * N + j] = k;
            }
      }
}

__global__ void kernel3(int* D, int* q, int b) {

   int i, j;

   if(blockIdx.x >= b)
      {
         i = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
      }
   else
      {
         i = blockIdx.x * blockDim.x + threadIdx.x;
      }
   if(blockIdx.y >= b)
      {
         j = (blockIdx.y + 1) * blockDim.y + threadIdx.y;
      }
   else
      {
         j = blockIdx.y * blockDim.y + threadIdx.y;
      }

   float d, f, e;
   for(int k = b * THR_PER_BL; k < (b + 1) * THR_PER_BL; k++)
      {
         d = D[i * N + j];
         f = D[i * N + k];
         e = D[k * N + j];

         __syncthreads();

         if(d > f + e)
            {
               D[i * N + j] = f + e;
               q[i * N + j] = k;
            }
      }
}

int main(int argc, char** argv){

   // Initialize variables
   int **D;
   int *d;
   int *dev_d;
   int *q;
   int *dev_q;

   // Allocate the memory on the CPU
   D = (int**)malloc(N * sizeof(int*));
   if(D == NULL) { printf("Allocation failure"); }
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

   // Allocate the memory on the GPU
   cudaMalloc((void**)&dev_d, N * N * sizeof(int));
   cudaMalloc((void**)&dev_q, N * N * sizeof(int));

   // Parallel Programming in CUDA C
   // Copy the array 'd' to the GPU
   // Copy the array 'q' to the GPU
   cudaMemcpy(dev_d, d, N * N * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(dev_q, q, N * N * sizeof(int), cudaMemcpyHostToDevice);
   dim3 dimBlock(THR_PER_BL, THR_PER_BL);
   dim3 dimGrid1(N / THR_PER_BL - 1, 2);
   dim3 dimGrid2(N / THR_PER_BL - 1, N / THR_PER_BL - 1);
   for(int b = 0; b < (N / THR_PER_BL); b++)
      {
         kernel1 << <1, dimBlock >> > (dev_d, dev_q, b);
         kernel2 << <dimGrid1, dimBlock >> > (dev_d, dev_q, b);
         kernel3 << <dimGrid2, dimBlock >> > (dev_d, dev_q, b);
      }

   // Copy the array 'd' back from the GPU to the CPU
   // Copy the array 'q' back from the GPU to the CPU
   cudaMemcpy(d, dev_d, N * N * sizeof(int), cudaMemcpyDeviceToHost);
   cudaMemcpy(q, dev_q, N * N * sizeof(int), cudaMemcpyDeviceToHost);

   /* // Print results - optional
   printf("The smallest distances are:\n");
   for (int i = 0; i < N * N; i++)
      {
         printf("%d\n", d[i]);
      }

   printf("The intermediate nodes are:\n");
   for (int i = 0; i < N * N; i++)
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
