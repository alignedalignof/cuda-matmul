#include <cuda_runtime.h>

struct KernelMat {
	int rows;
	int cols;
	int stride;
	__align__(128) float data[];
};
