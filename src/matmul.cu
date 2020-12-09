#include "kernelmat.h"

extern "C" __global__ void MatMulCell(const KernelMat* A, const KernelMat* B, KernelMat* C)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if (row >= C->rows || col >= C->cols)
		return;
	const float* a_data = A->data + row*A->stride;
	const float* b_data = B->data;
	int b_stride = B->stride;
	int n = A->cols;
	float acc = 0.0;
	for (int i = 0; i < n; i++)
		acc += a_data[i]*(b_data + i*b_stride)[col];
	(C->data + row*C->stride)[col] = acc;
}

extern "C" __global__ void MatMulBlock(const KernelMat* A, const KernelMat* B, KernelMat* C)
{
	__shared__ float a[16][16];
	__shared__ float b[16][16];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = blockIdx.y*blockDim.y + ty;
	int col = blockIdx.x*blockDim.x + tx;
	int a_stride = A->stride;
	int b_stride = B->stride;
	int a_rows = A->rows;
	int a_cols = A->cols;
	int b_rows = B->rows;
	int b_cols = B->cols;
	const float* a_data = A->data;
	const float* b_data = B->data;
	float acc = 0.0;
	for (int i = 0; i < a_stride; i += 16) {
		if ((row < a_rows) && (i + tx < a_cols))
			a[ty][tx] = (a_data + row*a_stride)[i + tx];
		else
			a[ty][tx] = 0.0;
		if ((col < b_cols) && (i + ty < b_rows))
			b[ty][tx] = (b_data + (i + ty)*b_stride)[col];
		else
			b[ty][tx] = 0.0;
		__syncthreads();
		#pragma unroll
		for (int k = 0; k < 16; ++k)
			acc += a[ty][k]*b[k][tx];
		__syncthreads();
	}
	if (row < C->rows && col < C->cols)
		(C->data + row*C->stride)[col] = acc;
}

extern "C" __global__ void MatMulLayer(const KernelMat* const A, const KernelMat* const B, KernelMat* const C)
{
	int ax = threadIdx.x & 1;
	int ay = threadIdx.x >> 1;
	const float* a_data = A->data + (128*blockIdx.y + 16*threadIdx.y + ay)*A->stride + 4*ax;
	const float* b_data = B->data + threadIdx.y*B->stride + blockIdx.x*128 + 4*threadIdx.x;
	const float* b_end = B->data + B->rows*B->stride;
	const float* a_end = A->data + A->rows*A->stride;
	
	__shared__ float4 b[8][32];
	__shared__ __align__(alignof(float4)) float a[8][4][32];
	
	float4 acc[16];
	for (int i = 16; i --> 0;)
		acc[i] = float4{ 0, 0, 0, 0 };

	int N = A->cols;
	int b_stride = B->stride;
	while (N > 0)
	{	
		N -= 8;
		
		b[threadIdx.y][threadIdx.x] = (b_data < b_end) ? ((float4*)b_data)[0] : float4{0, 0, 0, 0};
		b_data += 8*b_stride;

		float4 a4 = (a_data < a_end) ? ((float4*)a_data)[0] : float4{ 0, 0, 0, 0 };
		int x = threadIdx.x;
		x = ((x & 1) << 4) | (x >> 1);
		a[threadIdx.y][0][x] = a4.x;
		a[threadIdx.y][1][x] = a4.y;
		a[threadIdx.y][2][x] = a4.z;
		a[threadIdx.y][3][x] = a4.w;
		a_data += 8;
		
		__syncthreads();
		
		#pragma unroll
		for (x = 0; x < 8; x++)
		{
			float4 bx = ((float4*)&b[x])[threadIdx.x];
			#pragma unroll
			for (int y = 0; y < 16; y += 4)
			{
				a4 = ((float4*)&a[threadIdx.y][x & 3][y + ((x & 4) << 2)])[0];

				acc[y].x += a4.x*bx.x;
				acc[y].y += a4.x*bx.y;
				acc[y].z += a4.x*bx.z;
				acc[y].w += a4.x*bx.w;
				
				acc[y + 1].x += a4.y*bx.x;
				acc[y + 1].y += a4.y*bx.y;
				acc[y + 1].z += a4.y*bx.z;
				acc[y + 1].w += a4.y*bx.w;
				
				acc[y + 2].x += a4.z*bx.x;
				acc[y + 2].y += a4.z*bx.y;
				acc[y + 2].z += a4.z*bx.z;
				acc[y + 2].w += a4.z*bx.w;
				
				acc[y + 3].x += a4.w*bx.x;
				acc[y + 3].y += a4.w*bx.y;
				acc[y + 3].z += a4.w*bx.z;
				acc[y + 3].w += a4.w*bx.w;
			}
		}
		
		__syncthreads();
	}
	
	int c_stride = C->stride;
	if (128*blockIdx.x + 4*threadIdx.x >= c_stride)
		return;
	float* c_data = C->data + (128*blockIdx.y + 16*threadIdx.y)*c_stride + 128*blockIdx.x + 4*threadIdx.x;
	float* c_end = C->data + C->rows*c_stride;
	for (int y = 0; y < 16; y++)
	{
		if (c_data >= c_end)
			return;
		((float4*)c_data)[0] = acc[y];
		c_data += c_stride;
	}
}