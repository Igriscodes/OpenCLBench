#pragma once

#include "device_manager.hpp"
#include <map>
#include <string>
#include <vector>

namespace oclbench {

struct TuningParams {
    size_t local_work_size[3] = {256, 1, 1};
    size_t global_work_size[3] = {1024, 1, 1};
    int vector_width = 1;
    int unroll_factor = 1;
    size_t tile_size = 16;
};

class Tuner {
public:
    Tuner(const DeviceInfo& device);
    ~Tuner() = default;

    void profileArchitecture(cl::Context& context, cl::CommandQueue& queue);

    TuningParams getParams(const std::string& kernel_name) const;
    
    bool saveProfile(const std::string& path) const;
    bool loadProfile(const std::string& path);

private:
    DeviceInfo device_;
    std::map<std::string, TuningParams> cached_params_;
    
    TuningParams autoTuneKernel(cl::Context& context, cl::CommandQueue& queue, const std::string& kernel_name);
};

}
