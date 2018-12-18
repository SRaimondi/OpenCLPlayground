__kernel void SAXPY(__global const TYPE* x, __global TYPE* y, TYPE a, unsigned long size) {
    // Get thread identifier
    const size_t tid = get_global_id(0);

    // Check we are inside the size of the vector
    if (tid < size) {
        // Compute y = a * x + y
        y[tid] += a * x[tid];
    }
}