// Define macros to access elements, assumes column major storage
#define A(i, j) a[(i) + (j) * LDA]
#define B(i, j) b[(i) + (j) * LDB]
#define C(i, j) c[(i) + (j) * LDC]

// Simple base version of multiplication kernel
__kernel void SGEMM_0(  unsigned int M, unsigned int N, unsigned int K,
                        float alpha,
                        __global const float* a, unsigned int LDA,
                        __global const float* b, unsigned int LDB,
                        float beta,
                        __global float* c, unsigned int LDC) {
	// Each thread is resposible for computing an element of the output matrix

	// Get global row and column
	const size_t row = get_global_id(0);
	const size_t col = get_global_id(1);

	if (row < M && col < N) {
		float row_col_dot = 0.f;
		// Compute sum
		for (unsigned int k = 0; k < K; ++k) {
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

// More advanced version of SGEMM that makes use of local memory to accelerate access to matrix data
__kernel void SGEMM_1(  unsigned int M, unsigned int N, unsigned int K,
                        float alpha,
                        __global const float* a, unsigned int LDA,
                        __global const float* b, unsigned int LDB,
                        float beta,
                        __global float* c, unsigned int LDC,
                        __local float* shared_a,
                        __local float* shared_b) {
	// Again each thread is resposible for computing an element of the output matrix

	// Initialise product of A's row and B's column
	float sum = 0.f;

	// Get local index and work group index
	const size_t local_row = get_local_id(0);
	const size_t local_col = get_local_id(1);
	const size_t wg_row = get_group_id(0);
	const size_t wg_col = get_group_id(1);

	// Compute starting row of A and starting column of B
	const size_t A_start_row = wg_row * BLOCK_SIZE;
	const size_t B_start_col = wg_col * BLOCK_SIZE;

	// Loop over all the tiles 
	const unsigned int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
	for (unsigned int tile = 0; tile < num_tiles; ++tile) {
		// Compute current starting k in sum computation
		const unsigned int start_k = tile * BLOCK_SIZE;
		// Compute position of the element to load in matrix A and B
		const size_t a_row = A_start_row + local_row;
		const size_t a_col = start_k + local_col;
		const size_t b_row = start_k + local_row;
		const size_t b_col = B_start_col + local_col;

		// Test A element
		if (a_row < M && a_col < K) {
			SA(local_row, local_col) = A(a_row, a_col);
		} else {
			SA(local_row, local_col) = 0.f;
		}

		// Test B element
		if (b_row < K && b_col < N) {
			SB(local_row, local_col) = B(b_row, b_col);
		} else {
			SB(local_row, local_col) = 0.f;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Increment sum
		#pragma unroll
		for (unsigned int k = 0; k < BLOCK_SIZE; ++k) {
			sum += SA(local_row, k) * SB(k, local_col);
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	const size_t row = get_global_id(0);
	const size_t col = get_global_id(1);

	// Store sum
	if (row < M && col < N) {
		C(row, col) = alpha * sum + beta * C(row, col);
	}
}
