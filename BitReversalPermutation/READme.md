# Parallel implementation of Bit Reversal Permutation algorithm

Bit Reversal Permutation is the permutation of the elements of an array of size N, where N is a power of 2. The new position of an element is derived by reversing the binary representation of its 
position. Demo A shows how the new position is calculated:

### Demo A

For N = 8, all positions have a 3-bit binary representation. We want to permute the element at position 1:
- Initialize: i = 1(001)<sub>2</sub>, irev = i
- k = 1
   - i >>= 1 // i = (000)<sub>2</sub>
   - irev <<= 1 // irev = (010)<sub>2</sub>
   - irev |= (i & 1) // i & 1 = (000)<sub>2</sub>, irev = (010)<sub>2</sub>
- k = 2
   - i >>= 1 // i = (000)<sub>2</sub>
   - irev <<= 1 // irev = (100)<sub>2</sub>
   - irev |= (i & 1) // i & 1 = (000)<sub>2</sub>, irev = (100)<sub>2</sub>
- irev &= N - 1 // (100)<sub>2</sub> & (110)<sub>2</sub> = (100)<sub>2</sub>

Indeed, 1 = (001)<sub>2</sub> when reversed gives 4 = (100)<sub>2</sub>

## Simple implementation

Array description:
- D: input array
- Q: output array, which contains the values of D at the permuted positions
- dev_d: copied from d (GPU)
- dev_q: copied from q (GPU)

The global variable definition of dim3 type is done to divide the dataset into blocks and threads and to calculate the permuted array faster. The global variables are such that THR_PER_BL x BL_PER_GR = N.
The kernel<<...>>(...) is executed N times, according to Demo A.

### Example with a few numeric data for validation

| - | INPUT | 
| --- | --- |
| The number 0 is:  | 41 | 
| The number 1 is:  | 67 |
| The number 2 is:  | 34 |
| The number 3 is:  | 0  |
| The number 4 is:  | 69 |
| The number 5 is:  | 24 |
| The number 6 is:  | 78 |
| The number 7 is:  | 58 |
| The number 8 is:  | 62 |
| The number 9 is:  | 64 |
| The number 10 is: | 5  |
| The number 11 is: | 45 |
| The number 12 is: | 81 |
| The number 13 is: | 27 |
| The number 14 is: | 61 |
| The number 15 is: | 91 |

| - | RESULT | 
| --- | --- |
| The number 0 is:  | 41 | 
| The number 1 is:  | 62 |
| The number 2 is:  | 69 |
| The number 3 is:  | 81 |
| The number 4 is:  | 34 |
| The number 5 is:  | 5  |
| The number 6 is:  | 78 |
| The number 7 is:  | 61 |
| The number 8 is:  | 67 |
| The number 9 is:  | 64 |
| The number 10 is: | 24 |
| The number 11 is: | 27 |
| The number 12 is: | 0  |
| The number 13 is: | 45 |
| The number 14 is: | 58 |
| The number 15 is: | 91 |

## Complex implementation

The implementation in stages is more complex. It is used in the FFT algorithm. Specically, FFT divides the time signal of N elements into N signals of one element. The division is done according to Demo B. The procedure is as follows:
- The elements whose position is even number are placed in the group on the left and the elements whose position is odd number are placed in the group on the right 
- Then, the array is split in half and the same procedure it repeated, until N signals of one element are created
- As shown in Demo B, the elements are placed in the desired position one stage before the end

The CPU implementation is the same as in the simple implementation. But the GPU implementation is different, because the complex implementation utilizes 2 kernels. 
Specifically,
- kernel1: the block number (it should not be confused with the GPU block) and the odd or even number of the position of an element define its position in the output array. The block number is defined as follows: each time a stage is executed, 
the output array is virtually separated into blocks. In the 1<sup>st</sup> stage there are 2 blocks, in the 2<sup>nd</sup> stage 4 blocks, in the 3<sup>rd</sup> stage 8 blocks, e.t.c. The number of blocks in each stage is calculated from 
formula (1):
```
k = N / (2 ^ i) (1), (1)
```
where i is the number of the stage and the number of the group is calculated from formula (2):
```
block = (int)(i / 2 * k), (2) 
```
where i is the id of the current thread. The precise position of an element in the output array depends on whether its current position is an odd or an even number:
```
if(i % 2 == 0) {j = 2 * block * k + (int)(i / 2) – k * ((int)(i / (2 * k)))}
else {j = (2 * block + 1) * k + (int)(i / 2) – k * ((int)(i / (2 * k)))}
```
- kernel2: Each time kernel1 is finished, the output array needs to be copied to the input array
so that the latter can be used as input in the next stage. 

For details regarding the complex implementation, please refer to this [source](http://www.dspguide.com/ch12/2.htm)

### Demo B

**Stage 0**

| Block 0 |
| --- |
| 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 |

**Stage 1**

| Block 0 | Block 1 |
| --- | --- |
| 0 2 4 6 8 10 12 14 | 1 3 5 7 9 11 13 15 |

**Stage 2**

| Block 0 | Block 1 | Block 2 | Block 3 |
| --- | --- | --- | --- | 
| 0 4 8 12 | 2 6 10 14 | 1 5 9 13 | 3 7 11 15 |

**Stage 3**

| Block 0 | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 | Block 6 | Blok 7 |
| --- | --- | --- | --- | --- |  --- | --- | --- |
| 0 8 | 4 12 | 2 10 | 6 14 | 1 9 | 5 13 | 3 11 | 7 15 |

**Stage 4**

| Block 0 | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 | Block 6 | Block 7 | Block 8 | Block 9 | Block 10 | Block 11 | Block 12 | Block 13 | Block 14 | Block 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 8 | 4 | 12 | 2 | 10 | 6 | 14 | 1 | 9 | 5 | 13 | 3 | 11 | 7 | 15 |

### Example with a few numeric data for validation

| - | INPUT | 
| --- | --- |
| The number 0 is:  | 41 | 
| The number 1 is:  | 67 |
| The number 2 is:  | 34 |
| The number 3 is:  | 0  |
| The number 4 is:  | 69 |
| The number 5 is:  | 24 |
| The number 6 is:  | 78 |
| The number 7 is:  | 58 |
| The number 8 is:  | 62 |
| The number 9 is:  | 64 |
| The number 10 is: | 5  |
| The number 11 is: | 45 |
| The number 12 is: | 81 |
| The number 13 is: | 27 |
| The number 14 is: | 61 |
| The number 15 is: | 91 |

| - | RESULT | 
| --- | --- |
| The number 0 is:  | 41 | 
| The number 1 is:  | 62 |
| The number 2 is:  | 69 |
| The number 3 is:  | 81 |
| The number 4 is:  | 34 |
| The number 5 is:  | 5  |
| The number 6 is:  | 78 |
| The number 7 is:  | 61 |
| The number 8 is:  | 67 |
| The number 9 is:  | 64 |
| The number 10 is: | 24 |
| The number 11 is: | 27 |
| The number 12 is: | 0  |
| The number 13 is: | 45 |
| The number 14 is: | 58 |
| The number 15 is: | 91 |

## References
1. https://en.wikipedia.org/wiki/Bit-reversal_permutation
2. http://www.katjaas.nl/bitreversal/bitreversal.html
3. http://www.dspguide.com/ch12/2.htm
