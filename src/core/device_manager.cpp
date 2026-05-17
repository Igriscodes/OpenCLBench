#include "device_manager.hpp"

namespace oclbench {

void DeviceInfo::print() const {
    std::cout << "Device: " << name << " (" << vendor << ")" << std::endl;
    std::cout << "  Type: " << (type == CL_DEVICE_TYPE_GPU ? "GPU" : (type == CL_DEVICE_TYPE_CPU ? "CPU" : "Other")) << std::endl;
    std::cout << "  Version: " << version << " (Driver: " << driver_version << ")" << std::endl;
    std::cout << "  Compute Units: " << max_compute_units << " @ " << max_clock_frequency << " MHz" << std::endl;
    std::cout << "  Global Mem: " << global_mem_size / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Local Mem: " << local_mem_size / 1024 << " KB" << std::endl;
    std::cout << "  Max Work Group Size: " << max_work_group_size << std::endl;
}

DeviceManager::DeviceManager() {
    discoverDevices();
}

void DeviceManager::discoverDevices() {
    devices_.clear();
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (const auto& platform : platforms) {
        std::vector<cl::Device> platform_devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);

        for (const auto& device : platform_devices) {
            DeviceInfo info;
            info.platform = platform;
            info.device = device;
            info.name = device.getInfo<CL_DEVICE_NAME>();
            info.vendor = device.getInfo<CL_DEVICE_VENDOR>();
            info.version = device.getInfo<CL_DEVICE_VERSION>();
            info.driver_version = device.getInfo<CL_DRIVER_VERSION>();
            info.type = device.getInfo<CL_DEVICE_TYPE>();
            info.max_compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            info.max_work_group_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
            auto sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
            info.max_work_item_sizes[0] = sizes[0];
            info.max_work_item_sizes[1] = sizes[1];
            info.max_work_item_sizes[2] = sizes[2];
            info.global_mem_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
            info.local_mem_size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
            info.max_clock_frequency = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

            devices_.push_back(info);
        }
    }
}

const DeviceInfo& DeviceManager::getDevice(size_t index) const {
    if (index >= devices_.size()) {
        throw std::out_of_range("Device index out of range");
    }
    return devices_[index];
}

}
