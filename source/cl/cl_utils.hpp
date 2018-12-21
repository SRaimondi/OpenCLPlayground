//
// Created by simon on 12/18/2018.
//

#ifndef OPENCLPLAYGROUND_CL_UTILS_HPP
#define OPENCLPLAYGROUND_CL_UTILS_HPP

#include "config.hpp"

#include <string>
#include <vector>

#ifdef __APPLE__

#include <OpenCL/opencl.h>

#else

#include <CL/opencl.h>

#endif

namespace cl {
    namespace utils {

        // Divide number to the closest next integer number
        CXX_CONSTEXPR size_t DivideUp(size_t a, size_t b) CXX_NOEXCEPT {
            return (a + b - 1) / b;
        }

        // Read whole file into string
        std::string ReadFile(const std::string& file_name);

        // Context callback simple function
        void CL_CALLBACK ContextCallback(const char* errinfo, const void* private_info, size_t cb, void* user_data);

        // Build program with given sources
        cl_program CreateProgram(cl_context context,
                                 cl_device_id device,
                                 const std::vector<std::string>& sources,
                                 const std::string& build_options = "",
                                 bool verbose = false);

    } // utils namespace
} // cl namesapce

#endif //OPENCLPLAYGROUND_CL_UTILS_HPP
