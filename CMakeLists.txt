cmake_minimum_required(VERSION 3.7)
project(OpenCLPlayground)

set(CMAKE_CXX_STANDARD 14)

include_directories(source)

set(SOURCE_FILES
        source/main.cpp
        source/cl_error.cpp
        source/cl_error.hpp
        source/config.hpp
        source/cl_utils.cpp
        source/cl_utils.hpp
        source/saxpy_demo.cpp
        source/saxpy_demo.hpp)

add_executable(OpenCLPlayground ${SOURCE_FILES})

target_compile_definitions(OpenCLPlayground PUBLIC -DSAXPY_DOUBLE)

# FIXME OpenCL
target_include_directories(OpenCLPlayground PUBLIC $ENV{CUDA_PATH}/include)
target_link_libraries(OpenCLPlayground PUBLIC $ENV{CUDA_PATH}/lib/x64/OpenCL.lib)