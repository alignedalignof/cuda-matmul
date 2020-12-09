Matrix multiplication with CUDA
===============================

Program to explore CUDA and GPU related performance characteristics when multiplying matrices.
Following results are for multiplying 3073x3073 matrices on a i7-8750H CPU and a GTX1060 GPU.

| Method            | Running time*, ms  | Kernel time, ms
| ---               |:---                |:---
| CPU cell wise     | 121000             | -
| CPU layer wise    | 10900              | -
| GPU cell wise     | 326                | 311
| GPU block wise    | 134                | 119
| GPU layer wise    | 37.1               | 23.7
| cuBLAS gemm       | 36.8               | 22.0

*For GPUs running time includes data download from the GPU
