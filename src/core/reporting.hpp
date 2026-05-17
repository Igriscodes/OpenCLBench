#pragma once

#include "benchmark_runner.hpp"
#include "device_manager.hpp"
#include <string>

namespace oclbench {

class ReportingEngine {
public:
    ReportingEngine(const DeviceInfo& device, const std::map<std::string, BenchmarkMetrics>& results);
    ~ReportingEngine() = default;

    void printTerminalSummary() const;
    void exportJSON(const std::string& path) const;
    void exportCSV(const std::string& path) const;
    void exportMarkdown(const std::string& path) const;

private:
    DeviceInfo device_;
    std::map<std::string, BenchmarkMetrics> results_;
};

}
