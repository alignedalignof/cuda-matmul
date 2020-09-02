Matrix multiplication with CUDA
===============================

Program to explore CUDA and GPU related performance characteristics when multiplying matrices.
Following results are for multiplying 1000x1000 matrices on a i7-8750H CPU and a GTX1060 GPU.

| Method            | Running time*, ms  | Kernel time, ms
| ---               |:---                |:---
| CPU cell wise     | 2800               | -
| CPU layer wise    | 300                | -
| GPU cell wise     | 14                 | 11
| GPU block wise    | 8.0                | 5.0
| GPU layer wise    | 4.0                | 1.2
| cuBLAS gemm       | 3.5                | 0.80

*For GPUs running time includes data download from the GPU
