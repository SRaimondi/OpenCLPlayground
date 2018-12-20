//
// Created by simon on 12/18/2018.
//

#include "cl_utils.hpp"
#include "cl_error.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <memory>

namespace cl {
    namespace utils {

        std::string ReadFile(const std::string& file_name) {
            std::ifstream file(file_name);
            if (!file.is_open()) {
                std::cerr << "Could not open file " << file_name << "\n";
                return "";
            }

            std::stringstream buffer;
            buffer << file.rdbuf();

            return buffer.str();
        }

        void ContextCallback(const char* errinfo, const void*, size_t, void*) {
            // Simply write the error to the stderr
            fprintf(stderr, "Context callback: %s\n", errinfo);
        }

        cl_program CreateProgram(cl_context context,
                                 cl_device_id device,
                                 const std::vector<std::string>& sources,
                                 const std::string& build_options,
                                 bool verbose) {
            cl_int err_code = CL_SUCCESS;
            // Setup data for the OpenCL function
            auto strings = std::make_unique<const char* []>(sources.size());
            auto lengths = std::make_unique<size_t[]>(sources.size());

            for (std::size_t source_index = 0; source_index < sources.size(); ++source_index) {
                strings[source_index] = sources[source_index].data();
                lengths[source_index] = sources[source_index].length();
            }

            // Create program with the given sources
            cl_program program = clCreateProgramWithSource(context, static_cast<cl_uint>(sources.size()),
                                                           strings.get(), lengths.get(), &err_code);
            CHECK_CL(err_code);

            // Try to build the program
            const bool build_ok = CHECK_CL(
                    clBuildProgram(program, 1, &device, build_options.c_str(), nullptr, nullptr));

            // If there are errors, get the log and print it
            if (!build_ok || verbose) {
                if (!build_ok) {
                    std::cout << "#### Program build failed ####\n";
                } else {
                    std::cout << "#### Program build log ####\n";
                }
                std::size_t log_size = 0;
                CHECK_CL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
                auto log = std::make_unique<char[]>(log_size);
                CHECK_CL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.get(), nullptr));
                std::cout << log.get() << "\n";

                if (err_code != CL_SUCCESS) {
                    return nullptr;
                }
            }

            return program;
        }

    } // utils namespace
} // cl namespace