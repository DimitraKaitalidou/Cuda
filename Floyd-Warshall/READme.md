# Parallel implementation of Floyd-Warshall algorithm

Given N nodes and a matrix D of size N x N, where element (i, j) contains the distance between nodes i and j, the Floyd-Warshall algorithm calculates the smallest distances for all source-destination combinations. The description of the algorithm can be found [here](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)

The final minimum distances are calculated according to the formula:
```
Dij(k+1) = min{Dij(k), Di(k+1)(k) + D(k+1)j(k)}, for every i, j where k is the intermediate node (1)
```

### Simple implementation

Array description:
- d: distances (CPU), initialized randomly
- q: intermediate node for which the distance becomes minimum (CPU), initialized with N + 1
- dev_d: copied from d (GPU)
- dev_q: copied from q (GPU)

The global variable definition of dim3 type is done to divide the dataset into blocks and threads and to calculate the final distance matrix faster. The global variables are such that THR_PER_BL x BL_PER_GR = N.

The kernel<<...>>(...) is executed N x N times for every k, where k is the intermediate node.
The execution is performed in GPU, according to formula (1).

## Example with a few numeric data for validation

The initial distances are:

| - | 0 | 1 | 2 | 3 | 4 | 5 | 
| --- | --- | --- | --- | --- | --- | --- |
| **0** | 0  | 41 | 67 | 34 | 0  | 69 |
| **1** | 24 | 0  | 78 | 58 | 62 | 64 |
| **2** | 5  | 45 | 0  | 81 | 27 | 61 |
| **3** | 91 | 95 | 42 | 0  | 27 | 36 |
| **4** | 91 | 4  | 2  | 53 | 0  | 92 |
| **5** | 82 | 21 | 16 | 18 | 95 | 0  |

The smallest distances are:

| - | 0 | 1 | 2 | 3 | 4 | 5 | 
| --- | --- | --- | --- | --- | --- | --- |
| **0** | 0  | 4  | 2  | 34 | 0  | 63 |
| **1** | 24 | 0  | 26 | 58 | 24 | 64 |
| **2** | 5  | 9  | 0  | 39 | 5  | 61 |
| **3** | 34 | 31 | 29 | 0  | 27 | 36 |
| **4** | 7  | 4  | 2  | 41 | 0  | 63 | 
| **5** | 21 | 21 | 16 | 18 | 21 | 0  |

The intermediate nodes are:

| - | 0 | 1 | 2 | 3 | 4 | 5 | 
| --- | --- | --- | --- | --- | --- | --- |
| **0** | 7  | 4  | 4  | 7  | 7  | 4 | 
| **1** | 7  | 7  | 4  | 7  | 0  | 7 | 
| **2** | 7  | 4  | 7  | 0  | 0  | 7 |
| **3** | 4  | 4  | 4  | 7  | 7  | 7 |
| **4** | 2  | 7  | 7  | 2  | 7  | 2 |
| **5** | 2  | 7  | 7  | 7  | 2  | 7 |

### Complex implementation

The CPU implementation is the same as in the simple implementation. But the GPU implementation is different, because the complex implementation utilizes the shared memory and uses 3 kernels. 
Specifically,
- kernel1: the primary block is the current block and only its data are loaded to memory
- kernel2: in the 2nd phase the blocks that lie on the same row and column as the primary block are loaded to memory. The grid of the 2nd kernel is of size 2 x (N - 1). Before the application 
of the algorithm to each element, the blocks are parsed ignoring the primary. This process is done by comparing blockIdx.y with the number of the primary block. If it is greater or equal, it is increased by 1.
- kernel3: in the 3rd phase the rest of the blocks are loaded to memory. The grid of the 3rd kernel is of size (N - 1) x (N - 1). Before the application of the algorithm to each element, the blocks
are parsed ignoring the primary. The difference with the 2nd phase is that the blockIdx is compared in both directions with the number of the primary block. If it is greater or equal, it is increased by 1 (blockIdx.x + 1 or blockIdx.y + 1 respectively).

For details regarding the complex implementation, please refer to this [paper](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1213&context=hms)

## Example with a few numeric data for validation

The initial distances are:

| - | 0 | 1 | 2 | 3 | 4 | 5 | 
| --- | --- | --- | --- | --- | --- | --- |
| **0** | 0  | 41 | 67 | 34 | 0  | 69 |
| **1** | 24 | 0  | 78 | 58 | 62 | 64 |
| **2** | 5  | 45 | 0  | 81 | 27 | 61 | 
| **3** | 91 | 95 | 42 | 0  | 27 | 36 | 
| **4** | 91 | 4  | 2  | 53 | 0  | 92 | 
| **5** | 82 | 21 | 16 | 18 | 95 | 0  |

The smallest distances are:

| - | 0 | 1 | 2 | 3 | 4 | 5 | 
| --- | --- | --- | --- | --- | --- | --- |
| **0** | 0  | 4  | 2  | 34 | 0  | 63 |
| **1** | 24 | 0  | 26 | 58 | 24 | 64 | 
| **2** | 5  | 9  | 0  | 39 | 5  | 61 | 
| **3** | 34 | 31 | 29 | 0  | 27 | 36 |
| **4** | 7  | 4  | 2  | 41 | 0  | 63 | 
| **5** | 21 | 21 | 16 | 18 | 21 | 0  |

The intermediate nodes are:

| - | 0 | 1 | 2 | 3 | 4 | 5 | 
| --- | --- | --- | --- | --- | --- | --- |
| **0** | 7  | 4  | 4  | 7  | 7  | 4 |
| **1** | 7  | 7  | 4  | 7  | 0  | 7 |
| **2** | 7  | 4  | 7  | 0  | 0  | 7 |
| **3** | 4  | 4  | 4  | 7  | 7  | 7 |
| **4** | 2  | 7  | 7  | 0  | 7  | 2 |
| **5** | 2  | 7  | 7  | 7  | 0  | 7 |

## References
1. https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
2. Katz, G. J., and Kider, Jr, J. T. All-pairs shortest-paths for large graphs on the gpu. In Proceedings of the 23rd ACM SIGGRAPH/EUROGRAPHICS symposium on Graphics hardware (Aire-la-Ville, Switzerland, Switzerland, 2008), GH '08, Eurographics Association, pp. 47--55.
