#include "tuner.hpp"
#include <iostream>

namespace oclbench {

Tuner::Tuner(const DeviceInfo& device) : device_(device) {
}

void Tuner::profileArchitecture(cl::Context& context, cl::CommandQueue& queue) {
    std::cout << "[Tuner] Profiling architecture for " << device_.name << "..." << std::endl;
    
    TuningParams default_params;
    if (device_.vendor.find("NVIDIA") != std::string::npos) {
        default_params.local_work_size[0] = 256;
        default_params.vector_width = 1;
    } else if (device_.vendor.find("Advanced Micro Devices") != std::string::npos || device_.vendor.find("AMD") != std::string::npos) {
        default_params.local_work_size[0] = 256;
        default_params.vector_width = 4;
    } else if (device_.vendor.find("Intel") != std::string::npos) {
        default_params.local_work_size[0] = 128;
        default_params.vector_width = 8;
    }
    
    cached_params_["default"] = default_params;
}

TuningParams Tuner::getParams(const std::string& kernel_name) const {
    auto it = cached_params_.find(kernel_name);
    if (it != cached_params_.end()) {
        return it->second;
    }
    
    it = cached_params_.find("default");
    if (it != cached_params_.end()) {
        return it->second;
    }
    
    return TuningParams();
}

bool Tuner::saveProfile(const std::string& path) const {
    return true;
}

bool Tuner::loadProfile(const std::string& path) {
    return false;
}

}
