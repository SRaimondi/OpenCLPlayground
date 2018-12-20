//
// Created by simon on 12/18/2018.
//

#include "saxpy_demo.hpp"
#include "cl_error.hpp"
#include "cl_utils.hpp"

#include <iostream>

namespace demo {

    void SAXPYDemo(cl_platform_id platform, cl_device_id device, unsigned long vector_size) {
        // Compute size of data
        const size_t data_size = vector_size * sizeof(float);

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
        sources.push_back(cl::utils::ReadFile("../kernels/saxpy.cl"));

        // Create program
        std::string compile_op("-cl-std=CL1.2");
        cl_program program = cl::utils::CreateProgram(context, device, sources, compile_op, true);
        if (!program) {
            clReleaseContext(context);
            return;
        }

        // Create kernel
        cl_kernel saxpy_kernel = clCreateKernel(program, "SAXPY", &err_code);
        CHECK_CL(err_code);

        // Create some data to feed to the kernel
        cl_mem d_x = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                    data_size,
                                    nullptr,
                                    &err_code);
        CHECK_CL(err_code);
        cl_mem d_y = clCreateBuffer(context,
                                    CL_MEM_READ_WRITE,
                                    data_size,
                                    nullptr,
                                    &err_code);
        CHECK_CL(err_code);

        // Map memory regions
        auto h_map_x = reinterpret_cast<float *>(clEnqueueMapBuffer(command_queue,
                                                                    d_x,
                                                                    CL_TRUE,
                                                                    CL_MAP_WRITE,
                                                                    0,
                                                                    data_size,
                                                                    0,
                                                                    nullptr,
                                                                    nullptr,
                                                                    &err_code));
        CHECK_CL(err_code);
        auto h_map_y = reinterpret_cast<float *>(clEnqueueMapBuffer(command_queue,
                                                                    d_y,
                                                                    CL_TRUE,
                                                                    CL_MAP_WRITE,
                                                                    0,
                                                                    data_size,
                                                                    0,
                                                                    nullptr,
                                                                    nullptr,
                                                                    &err_code));
        CHECK_CL(err_code);

        // Just some random values that are used in the expression and when we check the result
        const float x_value = 2;
        const float y_value = 3;
        const float a_value = 4;

        // Now that we have our memory mapped, we can write to it
        for (unsigned long i = 0; i < vector_size; ++i) {
            h_map_x[i] = x_value;
            h_map_y[i] = y_value;
        }

        // Now we can unmap the memory
        cl_event unmap_events[2];
        CHECK_CL(clEnqueueUnmapMemObject(command_queue, d_x, h_map_x, 0, nullptr, &unmap_events[0]));
        CHECK_CL(clEnqueueUnmapMemObject(command_queue, d_y, h_map_y, 0, nullptr, &unmap_events[1]));

        // Setup argument for the kernel
        CHECK_CL(clSetKernelArg(saxpy_kernel, 0, sizeof(cl_mem), &d_x));
        CHECK_CL(clSetKernelArg(saxpy_kernel, 1, sizeof(cl_mem), &d_y));
        CHECK_CL(clSetKernelArg(saxpy_kernel, 2, sizeof(float), &a_value));
        CHECK_CL(clSetKernelArg(saxpy_kernel, 3, sizeof(vector_size), &vector_size));

        size_t kernel_wg_size;
        CHECK_CL(clGetKernelWorkGroupInfo(saxpy_kernel,
                                          device,
                                          CL_KERNEL_WORK_GROUP_SIZE,
                                          sizeof(kernel_wg_size),
                                          &kernel_wg_size,
                                          nullptr));
        std::cout << "Work group maximum size: " << kernel_wg_size << "\n";

        size_t kernel_wg_pref_size;
        CHECK_CL(clGetKernelWorkGroupInfo(saxpy_kernel,
                                          device,
                                          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                          sizeof(kernel_wg_pref_size),
                                          &kernel_wg_pref_size,
                                          nullptr));
        std::cout << "Preferred work group size multiple: " << kernel_wg_pref_size << "\n";

        // Launch kernel
        cl_event kernel_event;
        size_t global_size = cl::utils::DivideUp(vector_size, kernel_wg_size) * kernel_wg_size;
        CHECK_CL(clEnqueueNDRangeKernel(command_queue,
                                        saxpy_kernel,
                                        1,
                                        nullptr,
                                        &global_size,
                                        &kernel_wg_size,
                                        2,
                                        unmap_events,
                                        &kernel_event));

        // Read back result to host
        h_map_y = reinterpret_cast<float *>(clEnqueueMapBuffer(command_queue,
                                                               d_y,
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
        bool result_is_correct = true;
        for (unsigned long i = 0; i < vector_size; ++i) {
            if (h_map_y[i] != y_value + a_value * x_value) {
                result_is_correct = false;
                break;
            }
        }
        std::cout << "Result is " << (result_is_correct ? "correct" : "incorrect") << "!\n";

        // Unmap
        CHECK_CL(clEnqueueUnmapMemObject(command_queue, d_y, h_map_y, 0, nullptr, nullptr));

        // Wait for all commands to finish
        CHECK_CL(clFinish(command_queue));

        // Cleanup
        clReleaseMemObject(d_y);
        clReleaseMemObject(d_x);
        clReleaseKernel(saxpy_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
    }

} // demo namespace
