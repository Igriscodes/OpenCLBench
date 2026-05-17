#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

namespace oclbench {

struct DeviceInfo {
    cl::Platform platform;
    cl::Device device;
    std::string name;
    std::string vendor;
    std::string version;
    std::string driver_version;
    cl_device_type type;
    size_t max_compute_units;
    size_t max_work_group_size;
    size_t max_work_item_sizes[3];
    size_t global_mem_size;
    size_t local_mem_size;
    size_t max_clock_frequency;
    
    void print() const;
};

class DeviceManager {
public:
    DeviceManager();
    ~DeviceManager() = default;

    void discoverDevices();
    const std::vector<DeviceInfo>& getDevices() const { return devices_; }
    const DeviceInfo& getDevice(size_t index) const;

private:
    std::vector<DeviceInfo> devices_;
};

}
