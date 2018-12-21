// Define macros to access elements, assumes column major storage
#define A(i, j) a[(i) + (j) * LDA]
#define B(i, j) b[(i) + (j) * LDB]
#define C(i, j) c[(i) + (j) * LDC]

// Simple base version of multiplication kernel
__kernel void SGEMM_0(  unsigned long M, unsigned long N, unsigned long K,
                        float alpha,
                        __global const float* a, unsigned long LDA,
                        __global const float* b, unsigned long LDB,
                        float beta,
                        __global float* c, unsigned long LDC) {
	// Each thread is resposible for computing an element of the output matrix

	// Get global row and column
	const size_t row = get_global_id(0);
	const size_t col = get_global_id(1);

	if (row < M && col < N) {
		float row_col_dot = 0.f;
		// Compute sum
		for (unsigned long k = 0; k < K; ++k) {
			row_col_dot += A(row, k) * B(k, col); 
		}
		// Store result
		C(row, col) = alpha * row_col_dot + beta * C(row, col);
	}
}

// Macro to access shared memory at given index
// BLOCK_SIZE must be defined when the kernel is compiled
#define SA(i, j) shared_a[(i) + (j) * BLOCK_SIZE]
#define SB(i, j) shared_b[(i) + (j) * BLOCK_SIZE]

// More advacned version of SGEMM that makes use of local memory to accelerate access to matrix data
__kernel void SGEMM_1(  unsigned long M, unsigned long N, unsigned long K,
                        float alpha,
                        __global const float* a, unsigned long LDA,
                        __global const float* b, unsigned long LDB,
                        float beta,
                        __global float* c, unsigned long LDC,
                        __local float* shared_a,
                        __local float* shared_b) {
	// Get thread id in work group and global row and column
	const size_t t_row = get_local_id(0);
	const size_t t_col = get_local_id(1);
	const size_t row = get_global_id(0);
	const size_t col = get_global_id(1);

	

}