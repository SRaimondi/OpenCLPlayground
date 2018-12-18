#include "cl_error.hpp"
#include "cl_utils.hpp"
#include "saxpy_demo.hpp"

#include <vector>
#include <iostream>

int main() {
    // Retrieve available platforms
    cl_uint num_platforms;
    CHECK_CL(clGetPlatformIDs(0, nullptr, &num_platforms));
    if (num_platforms == 0) {
        std::cerr << "No available OpenCL platforms\n";
        exit(EXIT_FAILURE);
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    CHECK_CL(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    // If we have more than one platform, select it by index
    std::size_t selected_platform_index = 0;
    if (num_platforms != 1) {
        selected_platform_index = platforms.size();
        std::cout << "Select platform by index: \n";

        char platform_name[1024];
        for (std::size_t p = 0; p < num_platforms; ++p) {
            CHECK_CL(clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr));
            std::cout << "[" << p << "] " << platform_name << "\n";
        }

        do {
            std::cin >> selected_platform_index;
        } while (selected_platform_index >= platforms.size());
    }

    // platform holds the platform we selected
    cl_platform_id platform = platforms[selected_platform_index];

    // Now look for all the devices in the platform
    cl_uint num_devices = 0;
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices));
    if (num_devices == 0) {
        std::cerr << "No available devices in selected platform\n";
        exit(EXIT_FAILURE);
    }
    std::vector<cl_device_id> devices(num_devices);
    CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr));

    // If we have more than one device, select it by index
    std::size_t selected_device_index = 0;
    if (num_devices != 1) {
        selected_device_index = devices.size();
        std::cout << "Select device by index: \n";

        char device_name[1024];
        for (std::size_t d = 0; d < num_devices; ++d) {
            CHECK_CL(clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr));
            std::cout << "[" << d << "] " << device_name << "\n";
        }

        do {
            std::cin >> selected_device_index;
        } while (selected_device_index >= devices.size());
    }

    // device hold the selected device
    cl_device_id device = devices[selected_device_index];

    // Call code from different demos based on the selected one
    demo::SAXPYDemo(platform, device, 1000000);

    // Cleanup
    for (auto& dev : devices) {
        clReleaseDevice(dev);
    }

    return 0;
}