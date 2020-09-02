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

extern "C" __global__ void MatMulLayer(const KernelMat* A, const KernelMat* B, KernelMat* C)
{
	__shared__ __align__(4) float b[8][128];
	__shared__ __align__(4) float a[8][128];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int t = blockDim.x*ty + tx;
	int row = blockIdx.y*8*blockDim.y;
	int col = blockIdx.x*8*blockDim.x;

	int a_stride = A->stride;
	int b_stride = B->stride;
	int a_rows = A->rows;
	int a_cols = A->cols;
	int b_rows = B->rows;
	int b_cols = B->cols;
	const float* a_data = A->data;
	const float* b_data = B->data;
	
	__align__(4) float acc[8][8];
	for (int i = 0; i < 16; ++i)
		((float4*)acc)[i] = float4{0, 0, 0, 0};
	for (int ofs = 0; ofs < a_stride; ofs += 8) {
		int ax = t&1;
		int ay = row + (t/2);
		float4 a4 = (ay < a_rows) ? ((float4*)(a_data + ay*a_stride + ofs))[ax] : float4{0, 0, 0, 0};
		#pragma unroll
		for (int i = 0; i < 4; i++)
			a[4*ax + i][t/2] = (ofs + 4*ax + i < a_cols) ? ((float*)&a4)[i] : 0.0;
		
		int bx = t&31;
		int by = ofs + (t/32);
		float4 b4 = (by < b_rows) ? ((float4*)(b_data + by*b_stride + col))[bx] : float4{0, 0, 0, 0};
		#pragma unroll
		for (int i = 0; i < 4; i++)
			if (col + 4*bx + i >= b_cols)
				((float*)&b4)[i] = 0.0;
		((float4*)&b[t/32])[bx] = b4;
		__syncthreads();
		#pragma unroll
		for (int layer = 0; layer < 8; layer++) {
			__align__(4) float b1[8];
			__align__(4) float a1[8];
			((float4*)a1)[0] = ((float4*)&a[layer])[ty];
			((float4*)a1)[1] = ((float4*)&a[layer])[ty + 16];
			((float4*)b1)[0] = ((float4*)&b[layer])[tx];
			((float4*)b1)[1] = ((float4*)&b[layer])[tx + 16];
			#pragma unroll
			for (int i = 0; i < 8; ++i) {
				acc[i][0] += b1[0]*a1[i];
				acc[i][1] += b1[1]*a1[i];
				acc[i][2] += b1[2]*a1[i];
				acc[i][3] += b1[3]*a1[i];
				acc[i][4] += b1[4]*a1[i];
				acc[i][5] += b1[5]*a1[i];
				acc[i][6] += b1[6]*a1[i];
				acc[i][7] += b1[7]*a1[i];
			}
		}
		__syncthreads();
	}
	float* c_data = C->data;
	int c_stride = C->stride;
	int c_rows = C->rows;
	int c_cols = C->cols;
	for (int y = 0; y < 8; y++) {
		for (int x = 0; x < 8; x++) {
				int cy = row + 4*ty + 64*(y/4) + (y&3);
				int cx = col + 4*tx + 64*(x/4) + (x&3);
				if (cy < c_rows && cx < c_cols)
					c_data[cy*c_stride + cx] = acc[y][x];
		}
	}

}