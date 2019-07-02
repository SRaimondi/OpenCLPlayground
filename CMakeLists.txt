cmake_minimum_required(VERSION 3.7)
project(OpenCLPlayground)

set(CMAKE_CXX_STANDARD 14)

if (${CMAKE_HOST_SYSTEM_NAME} MATCHES Darwin)
    find_package(OpenCL REQUIRED)
endif ()

include_directories(source
        source/cl
        source/demo)

set(SOURCE_FILES
        source/main.cpp
        source/cl/cl_error.cpp
        source/cl/cl_error.hpp
        source/config.hpp
        source/cl/cl_utils.cpp
        source/cl/cl_utils.hpp
        source/demo/saxpy_demo.cpp
        source/demo/saxpy_demo.hpp
        source/demo/sgemm_demo.cpp
        source/demo/sgemm_demo.hpp)

add_executable(OpenCLPlayground ${SOURCE_FILES})

#target_compile_definitions(OpenCLPlayground PRIVATE -DSGEMM_BASE)

# FIXME OpenCL
if (${CMAKE_HOST_SYSTEM_NAME} MATCHES Darwin)
    # Silence depracations on OSX
    target_compile_definitions(OpenCLPlayground PRIVATE -DCL_SILENCE_DEPRECATION)
    target_link_libraries(OpenCLPlayground OpenCL::OpenCL)
else ()
    target_include_directories(OpenCLPlayground PUBLIC $ENV{CUDA_PATH}/include)
    target_link_libraries(OpenCLPlayground PUBLIC $ENV{CUDA_PATH}/lib/x64/OpenCL.lib)
endif ()