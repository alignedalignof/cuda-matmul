#include <vector>

enum MatMulMethod {
	GPU_CELL_WISE,
	GPU_CELL_WISE_PREFETCH,
	GPU_BLOCK_WISE,
	GPU_LAYER_WISE,
	GPU_CUBLAS,

	CPU,
	CPU_CELL_WISE = CPU,
	CPU_LAYER_WISE,
};

class CudaMat {
public:
	CudaMat(int rows, int cols);
	CudaMat(const CudaMat& other) = delete;
	CudaMat& operator=(const CudaMat& other) = delete;
	CudaMat& operator=(CudaMat&& other) = delete;
	CudaMat(CudaMat&& temp);

	~CudaMat();

	void upload() const;
	void download();

	int columns() const {
		return cols_;
	}
	int rows() const {
		return rows_;
	}
	float* row(int i) {
		return host_.data() + i*stride_;
	}
	const float* row(int i) const {
		return host_.data() + i*stride_;
	}
	const float* data() const {
		return host_.data();
	}
	int stride() const {
		return stride_;
	}
	float* data() {
		return host_.data();
	}

	static CudaMat multiply(const CudaMat& a, const CudaMat& b, MatMulMethod method);
private:
	int rows_;
	int cols_;
	int stride_;
	size_t bytes_;

	std::vector<float> host_;
	mutable void* dev_;
};

