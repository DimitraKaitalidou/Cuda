// Copyright 2019, Dimitra S. Kaitalidou, All rights reserved

#include "stdio.h"
#include "stdlib.h"
#include "cuda.h"
#include "math.h"

#define N 256
#define THR_PER_BL 8
#define BL_PER_GR 32

__global__ void kernel(int* D, int* Q, int bits) {

   // Find index
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   // Initialize variables that will be shifted left and right
   int shifted_right = i;
   int shifted_left = shifted_right;

   // Perform bit reversal permutation
   for (int a = 1; a < bits; a++)
      {
         shifted_right >>= 1;
         shifted_left <<= 1;
         shifted_left |= shifted_right & 1;
      }
   shifted_left &= N - 1;

   // Assign the values to the bit reversed positions
   Q[shifted_left] = D[i];
}

int main(int argc, char** argv){

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

   kernel << <dimGrid, dimBlock >> >(dev_d, dev_q, bits);

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
