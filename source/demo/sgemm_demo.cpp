//
// Created by Simon on 2018-12-20.
//

#include "sgemm_demo.hpp"
#include "cl_error.hpp"
#include "cl_utils.hpp"

#include <iostream>
#include <cmath>

namespace demo {

    void SGEMMDEmo(cl_platform_id platform, cl_device_id device, unsigned long matrix_size) {
        // Compute size of data
        const size_t data_size = matrix_size * matrix_size * sizeof(float);

        // Create context
        cl_int err_code = CL_SUCCESS;
        const cl_context_properties context_properties[3] = {
                CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform),
                0};
        cl_context context = clCreateContext(context_properties,
                                             1, &device,
                                             cl::utils::ContextCallback, nullptr,
                                             &err_code);
        CHECK_CL(err_code);

        // Create command queue
        cl_command_queue command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err_code);
        CHECK_CL(err_code);

        // Load program sources
        std::vector<std::string> sources;
        sources.push_back(cl::utils::ReadFile("../kernels/sgemm.cl"));

        // Create program
        cl_program program = cl::utils::CreateProgram(context, device, sources, "-cl-std=CL1.2", true);
        if (!program) {
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
            return;
        }

        // Create kernel
        cl_kernel sgemm_kernel = clCreateKernel(program, "SGEMM_0", &err_code);
        CHECK_CL(err_code);

        // Create some data to feed to the kernel
        cl_mem d_A = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                    data_size,
                                    nullptr,
                                    &err_code);
        CHECK_CL(err_code);

        cl_mem d_B = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                    data_size,
                                    nullptr,
                                    &err_code);
        CHECK_CL(err_code);

        cl_mem d_C = clCreateBuffer(context,
                                    CL_MEM_READ_WRITE,
                                    data_size,
                                    nullptr,
                                    &err_code);
        CHECK_CL(err_code);

        // Map memory regions
        auto h_map_A = reinterpret_cast<float *>(clEnqueueMapBuffer(command_queue,
                                                                    d_A,
                                                                    CL_TRUE,
                                                                    CL_MAP_WRITE,
                                                                    0,
                                                                    data_size,
                                                                    0,
                                                                    nullptr,
                                                                    nullptr,
                                                                    &err_code));
        CHECK_CL(err_code);

        auto h_map_B = reinterpret_cast<float *>(clEnqueueMapBuffer(command_queue,
                                                                    d_B,
                                                                    CL_TRUE,
                                                                    CL_MAP_WRITE,
                                                                    0,
                                                                    data_size,
                                                                    0,
                                                                    nullptr,
                                                                    nullptr,
                                                                    &err_code));
        CHECK_CL(err_code);

        auto h_map_C = reinterpret_cast<float *>(clEnqueueMapBuffer(command_queue,
                                                                    d_C,
                                                                    CL_TRUE,
                                                                    CL_MAP_WRITE,
                                                                    0,
                                                                    data_size,
                                                                    0,
                                                                    nullptr,
                                                                    nullptr,
                                                                    &err_code));
        CHECK_CL(err_code);

        // Fill matrices
        const float c_value = 1.f;
        for (unsigned long j = 0; j < matrix_size; ++j) {
            for (unsigned long i = 0; i < matrix_size; ++i) {
                // Matrix A stores the column index
                h_map_A[j * matrix_size + i] = j;
                // Matrix B stores the row index
                h_map_B[j * matrix_size + i] = i;
                // Matrix C has a uniform value
                h_map_C[j * matrix_size + i] = c_value;
            }
        }

        // We have written the values, unmap the memory
        cl_event unmap_events[3];
        CHECK_CL(clEnqueueUnmapMemObject(command_queue, d_A, h_map_A, 0, nullptr, &unmap_events[0]));
        CHECK_CL(clEnqueueUnmapMemObject(command_queue, d_B, h_map_B, 0, nullptr, &unmap_events[1]));
        CHECK_CL(clEnqueueUnmapMemObject(command_queue, d_C, h_map_C, 0, nullptr, &unmap_events[2]));

        // Setup argument for the kernel
        const float alpha = 1.f;
        const float beta = 1.f;
        // Set matrices sizes
        CHECK_CL(clSetKernelArg(sgemm_kernel, 0, sizeof(unsigned long), &matrix_size));
        CHECK_CL(clSetKernelArg(sgemm_kernel, 1, sizeof(unsigned long), &matrix_size));
        CHECK_CL(clSetKernelArg(sgemm_kernel, 2, sizeof(unsigned long), &matrix_size));
        // Set alpha value for C = alpha * A * B + beta * C
        CHECK_CL(clSetKernelArg(sgemm_kernel, 3, sizeof(float), &alpha));
        CHECK_CL(clSetKernelArg(sgemm_kernel, 4, sizeof(cl_mem), &d_A));
        CHECK_CL(clSetKernelArg(sgemm_kernel, 5, sizeof(unsigned long), &matrix_size));
        CHECK_CL(clSetKernelArg(sgemm_kernel, 6, sizeof(cl_mem), &d_B));
        CHECK_CL(clSetKernelArg(sgemm_kernel, 7, sizeof(unsigned long), &matrix_size));
        CHECK_CL(clSetKernelArg(sgemm_kernel, 8, sizeof(float), &beta));
        CHECK_CL(clSetKernelArg(sgemm_kernel, 9, sizeof(cl_mem), &d_C));
        CHECK_CL(clSetKernelArg(sgemm_kernel, 10, sizeof(unsigned long), &matrix_size));

        // FIXME Setup launch sizes
        std::size_t local_size[2] = {16, 16};
        std::size_t global_size[2];
        global_size[0] = cl::utils::DivideUp(matrix_size, local_size[0]) * local_size[0];
        global_size[1] = cl::utils::DivideUp(matrix_size, local_size[1]) * local_size[1];

        // Launch kernel
        cl_event kernel_event;
        CHECK_CL(clEnqueueNDRangeKernel(command_queue,
                                        sgemm_kernel,
                                        2,
                                        nullptr,
                                        global_size,
                                        local_size,
                                        2,
                                        unmap_events,
                                        &kernel_event));

        // Map result to read on the host
        h_map_C = reinterpret_cast<float *>(clEnqueueMapBuffer(command_queue,
                                                               d_C,
                                                               CL_TRUE,
                                                               CL_MAP_READ,
                                                               0,
                                                               data_size,
                                                               1,
                                                               &kernel_event,
                                                               nullptr,
                                                               &err_code));
        CHECK_CL(err_code);

        // Check result
        float result = 0.f;
        for (unsigned long i = 0; i < matrix_size; ++i) {
            result += i * i;
        }

        bool result_is_correct = true;
        for (unsigned long j = 0; j < matrix_size; ++j) {
            for (unsigned long i = 0; i < matrix_size; ++i) {
                if (std::abs(h_map_C[j * matrix_size + i] - result) > 0.0001f) {
                    result_is_correct = false;
                    break;
                }
            }
        }
        std::cout << "Result is " << (result_is_correct ? "correct" : "incorrect") << "!\n";

        // Unmap
        CHECK_CL(clEnqueueUnmapMemObject(command_queue, d_C, h_map_C, 0, nullptr, nullptr));

        // Wait for all commands to finish
        CHECK_CL(clFinish(command_queue));

        // Cleanup
        clReleaseMemObject(d_C);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_A);
        clReleaseKernel(sgemm_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
    }

}