#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <filesystem>

namespace oclbench {

inline std::string loadKernelSource(const std::string& relative_path) {
    std::filesystem::path kernel_path = std::filesystem::path("kernels") / relative_path;
    
    std::ifstream file(kernel_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + kernel_path.string());
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

}
