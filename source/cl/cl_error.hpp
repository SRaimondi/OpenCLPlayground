//
// Created by simon on 12/18/2018.
//

#ifndef OPENCLPLAYGROUND_CL_ERROR_HPP
#define OPENCLPLAYGROUND_CL_ERROR_HPP

// Local files
#include "config.hpp"

// Standard
#include <string>

// External
#ifdef __APPLE__

#include <OpenCL/opencl.h>

#else
#include <CL/opencl.h>
#endif

namespace cl {

    // Convert an OpenCL error code to string
    std::string ErrorToString(cl_int err_code) CXX_NOEXCEPT;

    // Check error code and prints error message if it's not CL_SUCCESS
    bool CheckStatus(cl_int err_code, int line, const char* file) CXX_NOEXCEPT;

    // Check status macro
#define CHECK_CL(STATUS) cl::CheckStatus(STATUS, __LINE__, __FILE__)

} // cl namespace

#endif //OPENCLPLAYGROUND_CL_ERROR_HPP
