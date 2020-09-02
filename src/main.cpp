#include <stdexcept>
#include <ctime>
#include <cstdio>
#include <random>
#include <chrono>

#include "cudamat.h"

using namespace std;
using namespace chrono;

static float calc_mse(const CudaMat& a, const CudaMat& b) {
	if (a.columns() != b.columns() || a.rows() != b.rows())
		throw invalid_argument("Invalid matrix dimensions");
	float mse = 0;
	for (int row = 0; row < a.rows(); ++row) {
		const float* a_row = a.row(row);
		const float* b_row = b.row(row);
		for (int col = 0; col < a.columns(); ++col) {
			float d = a_row[col] - b_row[col];
			mse += d*d;
		}
	}
	mse /= a.columns();
	mse /= a.rows();
	return mse;
}

static void test(const CudaMat& a, const CudaMat& b, const CudaMat& e, MatMulMethod method, const char* name) {
	steady_clock::time_point s = steady_clock::now();
	auto c(CudaMat::multiply(a, b, method));
	if (method < CPU)
		c.download();
	auto ms = duration<float, milli>(steady_clock::now() - s);
	auto mse = calc_mse(c, e);
	printf("%s: %.2lf ms, MSE %.3e\n", name, ms.count(), mse);
}

int main() {
	const int N = 1000;

	mt19937 gen(42);
	uniform_real_distribution<float> uni(-1.0, 1.0);

	CudaMat a(N, N);
	CudaMat b(N, N);
	for (int i = 0; i < a.rows()*a.stride(); ++i)
		a.data()[i] = uni(gen);
	for (int i = 0; i < b.rows()*b.stride(); ++i)
		b.data()[i] = uni(gen);
	a.upload();
	b.upload();
	auto baseline = CudaMat::multiply(a, b, GPU_CUBLAS);
	baseline.download();

	struct {
		MatMulMethod kernel;
		const char* name;
	} tests[] = {
		{ CPU_CELL_WISE, "CPU cell wise" },
		{ CPU_LAYER_WISE, "CPU layer wise" },
		{ GPU_CELL_WISE, "GPU cell wise" },
		{ GPU_BLOCK_WISE, "GPU block wise" },
		{ GPU_LAYER_WISE, "GPU layer wise" },
		{ GPU_CUBLAS, "sgemm" },
	};

	for (int i = 0; i < sizeof(tests)/sizeof(*tests); ++i)
		test(a, b, baseline, tests[i].kernel, tests[i].name);

	return 0;
}
