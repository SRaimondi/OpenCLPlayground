//
// Created by simon on 12/18/2018.
//

#ifndef OPENCLPLAYGROUND_SAXPY_DEMO_HPP
#define OPENCLPLAYGROUND_SAXPY_DEMO_HPP

#ifdef __APPLE__

#include <OpenCL/opencl.h>

#else
#include <CL/opencl.h>
#endif

namespace demo
{

// Run the SAXPY kernel on the given device
void SAXPYDemo(cl_platform_id platform, cl_device_id device, unsigned long vector_size = 4096);

} // demo namespace

#endif //OPENCLPLAYGROUND_SAXPY_DEMO_HPP
