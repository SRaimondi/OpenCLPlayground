__kernel void SAXPY(__global const float* x, __global float* y, float a, unsigned long size) {
    // Get thread identifier
    const size_t tid = get_global_id(0);

    // Check we are inside the size of the vector
    if (tid < size) {
        // Compute y = a * x + y
        y[tid] += a * x[tid];
    }
}