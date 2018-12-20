//
// Created by Simon on 2018-12-20.
//

#ifndef OPENCLPLAYGROUND_SGEMM_DEMO_HPP
#define OPENCLPLAYGROUND_SGEMM_DEMO_HPP

#ifdef __APPLE__

#include <OpenCL/opencl.h>

#else
#include <CL/opencl.h>
#endif

namespace demo {

    // Run the different ggemm kernel depending on the compile flags
    void SGEMMDEmo(cl_platform_id platform, cl_device_id device, unsigned long matrix_size = 4096);

} // demo namespace

#endif //OPENCLPLAYGROUND_SGEMM_DEMO_HPP
