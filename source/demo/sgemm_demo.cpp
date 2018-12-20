//
// Created by Simon on 2018-12-20.
//

#include "sgemm_demo.hpp"
#include "cl_error.hpp"
#include "cl_utils.hpp"

#include <iostream>

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

        // Cleanup;
        clReleaseContext(context);
    }

}