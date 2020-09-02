#include <unordered_map>
#include <stdexcept>

#include <cuda.h>
#include <cublas_v2.h>

#include "kernelmat.h"
#include "cudamat.h"

#define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)) * (S))

using namespace std;

struct Kernel {
	const char* name;
	CUfunction func;
};

static unordered_map<MatMulMethod, Kernel> METHOD_TO_KERNEL = {
	{GPU_CELL_WISE, {"MatMulCell"}},
	{GPU_BLOCK_WISE, {"MatMulBlock"}},
	{GPU_LAYER_WISE, {"MatMulLayer"}},
};

static struct Cuda {
	CUdevice dev;
	CUcontext ctx;
	cublasHandle_t cublas;
	Cuda();
	~Cuda();
} cuda;

Cuda::Cuda() {
	dev = 0;
	ctx = 0;
	cuInit(0);
	if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
		cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
		cuCtxPushCurrent(ctx) != CUDA_SUCCESS) {
		if (ctx)
			cuDevicePrimaryCtxRelease(dev);
		throw runtime_error("No CUDA runtime");
	}

	char gpu[1000] = "Unknown";
	volatile CUresult cuda = cuDeviceGetName(gpu, sizeof(gpu), dev);
	printf("Using %s\n", gpu);

	CUmodule module;
	if (cuModuleLoad(&module, "matmul.ptx") != CUDA_SUCCESS)
		return;
	for (auto& kv : METHOD_TO_KERNEL)
		if (cuModuleGetFunction(&kv.second.func, module, kv.second.name) != CUDA_SUCCESS)
			kv.second.func = 0;
	if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS)
		cublas = 0;
}

Cuda::~Cuda() {
	if (ctx)
		cuDevicePrimaryCtxRelease(dev);
	if (cublas)
		cublasDestroy(cublas);
}

static CUdeviceptr device_mat_ptr(void* ptr) {
	return ((CUdeviceptr*)ptr)[0];
}

static CUdeviceptr device_data_ptr(void* ptr) {
	return device_mat_ptr(ptr) + sizeof(KernelMat);
}

static CudaMat cudamat_multiply_layer(const CudaMat& a, const CudaMat& b) {
	CudaMat c(a.rows(), b.columns());
	for (int com = 0; com < a.columns(); ++com)
		for (int row = 0; row < c.rows(); ++row)
			for (int col = 0; col < c.columns(); ++col)
				c.row(row)[col] += a.row(row)[com]*b.row(com)[col];
	return c;
}

static CudaMat cudamat_multiply_cell(const CudaMat& a, const CudaMat& b) {
	CudaMat c(a.rows(), b.columns());
	for (int row = 0; row < c.rows(); ++row)
		for (int col = 0; col < c.columns(); ++col)
			for (int com = 0; com < a.columns(); ++com)
				c.row(row)[col] += a.row(row)[com]*b.row(com)[col];
	return c;
}

static void cudamat_gpu_multiply(void* a, void* b, void* c, int rows, int cols, MatMulMethod method) {
	auto it = METHOD_TO_KERNEL.find(method);
	if (it == METHOD_TO_KERNEL.end())
		throw invalid_argument("Unknown kernel");
	if (!it->second.func)
		throw invalid_argument("Kernel not loaded");
	void* args[] = { a, b, c };
	int tpbx = 16;
	int tpby = 16;
	int bpgx = ROUND_UP(cols, tpbx)/tpbx;
	int bpgy = ROUND_UP(rows, tpby)/tpby;
	if (it->second.name == "MatMulLayer") {
		bpgx = ROUND_UP(cols, 128)/128;
		bpgy = ROUND_UP(rows, 128)/128;
	}
	if (cuLaunchKernel(it->second.func, bpgx, bpgy, 1, tpbx, tpby, 1, 0, NULL, args, NULL) != CUDA_SUCCESS)
		throw runtime_error("Kernel launch failed");
}

static void cudamat_gpu_cublas(void* a, void* b, void* c, int m, int n, int k, int lda, int ldb, int ldc) {
	if (!cuda.cublas)
		throw runtime_error("cuBLAS not found");
	float alpha = 1.0f;
	float beta = 0.0f;
	float* A = (float*)device_data_ptr(a);
	float* B = (float*)device_data_ptr(b);
	float* C = (float*)device_data_ptr(c);
	if (cublasSgemm(cuda.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
					m, n, k,
					&alpha, A, lda, B, ldb, &beta, C, ldc) != CUBLAS_STATUS_SUCCESS)
		throw runtime_error("cuBLIN");
}

CudaMat CudaMat::multiply(const CudaMat& a, const CudaMat& b, MatMulMethod method) {
	if (a.columns() != b.rows())
		throw invalid_argument("Invalid matrix dimensions");
	if (method == CPU_LAYER_WISE)
		return cudamat_multiply_layer(a, b);
	if (method == CPU_CELL_WISE)
		return cudamat_multiply_cell(a, b);
	CudaMat c(a.rows(), b.columns());
	if (method == GPU_CUBLAS)
		cudamat_gpu_cublas(b.dev_, a.dev_, c.dev_, b.columns(), a.rows(), b.rows(), b.stride(), a.stride(), c.stride());
	else
		cudamat_gpu_multiply(a.dev_, b.dev_, c.dev_, c.rows(), c.columns(), method);
	return c;
}

CudaMat::CudaMat(int rows, int cols) {
	if (rows < 1 || cols < 1)
		throw invalid_argument("Invalid matrix size");
	rows_ = rows;
	cols_ = cols;

	const int PAD = alignof(KernelMat::data)/sizeof(float);
	stride_ = ROUND_UP(cols, PAD);
	host_ = vector<float>(rows_*stride_);
	bytes_ = host_.size()*sizeof(float);
	dev_ = malloc(sizeof(CUdeviceptr));
	if (!dev_)
		throw runtime_error("java.lang.OutOfMemoryError");
	CUdeviceptr dev = 0;
	KernelMat mat;
	mat.rows = rows_;
	mat.cols = cols_;
	mat.stride = stride_;
	if (cuMemAlloc(&dev, sizeof(KernelMat) + bytes_) != CUDA_SUCCESS ||
		cuMemcpyHtoD(dev, &mat, sizeof(KernelMat)) != CUDA_SUCCESS) {
		free(dev_);
		cuMemFree(dev);
		throw runtime_error("No GPU memory");
	}
	((CUdeviceptr*)dev_)[0] = dev;
}

CudaMat::CudaMat(CudaMat&& temp) {
	rows_ = temp.rows_;
	cols_ = temp.cols_;
	stride_ = temp.stride_;
	host_ = move(temp.host_);
	bytes_ = temp.bytes_;
	dev_ = temp.dev_;
	temp.dev_ = 0;
}

CudaMat::~CudaMat() {
	if (!dev_)
		return;
	cuMemFree(device_mat_ptr(dev_));
	free(dev_);
}

void CudaMat::upload() const {
	if (cuMemcpyHtoD(device_data_ptr(dev_), host_.data(), bytes_) != CUDA_SUCCESS)
		throw runtime_error("Upload failed");
}

void CudaMat::download() {
	CUresult r = cuMemcpyDtoH(host_.data(), device_data_ptr(dev_), bytes_);
	if (r != CUDA_SUCCESS)
		throw runtime_error("Download failed");
}
