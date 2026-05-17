#include "reporting.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>

namespace oclbench {

ReportingEngine::ReportingEngine(const DeviceInfo& device, const std::map<std::string, BenchmarkMetrics>& results)
    : device_(device), results_(results) {}

void ReportingEngine::printTerminalSummary() const {
    std::cout << "\n=========================================================================================================" << std::endl;
    std::cout << " OpenCLBench Report - " << device_.name << std::endl;
    std::cout << "=========================================================================================================" << std::endl;
    std::cout << std::left << std::setw(20) << "Benchmark" 
              << std::right << std::setw(12) << "Avg(ms)" 
              << std::setw(10) << "Min" 
              << std::setw(10) << "Max" 
              << std::setw(10) << "StdDev" 
              << std::setw(12) << "GB/s" 
              << std::setw(12) << "GFLOPS" 
              << std::setw(15) << "Degradation" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------------" << std::endl;

    for (const auto& [name, metrics] : results_) {
        std::cout << std::left << std::setw(20) << name 
                  << std::right << std::fixed << std::setprecision(3) 
                  << std::setw(12) << metrics.avg_time_ms
                  << std::setw(10) << metrics.min_time_ms
                  << std::setw(10) << metrics.max_time_ms
                  << std::setw(10) << metrics.stddev_time_ms
                  << std::setw(12) << metrics.bandwidth_gbps()
                  << std::setw(12) << metrics.throughput_gflops();
        if (metrics.thermal_degradation_pct != 0.0) {
            std::cout << std::setw(14) << std::showpos << metrics.thermal_degradation_pct << "%" << std::noshowpos;
        } else {
            std::cout << std::setw(15) << "N/A";
        }
        std::cout << std::endl;
    }
    std::cout << "=========================================================================================================\n" << std::endl;
}

void ReportingEngine::exportJSON(const std::string& path) const {
    std::ofstream out(path);
    if (!out) return;
    out << "{\n  \"device\": \"" << device_.name << "\",\n  \"results\": {\n";
    bool first = true;
    for (const auto& [name, metrics] : results_) {
        if (!first) out << ",\n";
        out << "    \"" << name << "\": {\n";
        out << "      \"time_ms\": " << metrics.avg_time_ms << ",\n";
        out << "      \"bandwidth_gbps\": " << metrics.bandwidth_gbps() << ",\n";
        out << "      \"throughput_gflops\": " << metrics.throughput_gflops() << "\n    }";
        first = false;
    }
    out << "\n  }\n}\n";
}

void ReportingEngine::exportCSV(const std::string& path) const {}
void ReportingEngine::exportMarkdown(const std::string& path) const {}

}
