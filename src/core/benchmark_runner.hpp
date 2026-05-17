#pragma once

#include "device_manager.hpp"
#include "tuner.hpp"
#include <string>
#include <map>
#include <chrono>
#include <memory>

namespace oclbench {

struct BenchmarkMetrics {
    double min_time_ms = 0;
    double max_time_ms = 0;
    double avg_time_ms = 0;
    double stddev_time_ms = 0;
    double bytes_transferred = 0;
    double compute_ops = 0;
    double thermal_degradation_pct = 0;
    
    double bandwidth_gbps() const {
        return avg_time_ms > 0 ? (bytes_transferred / (avg_time_ms * 1e-3)) / 1e9 : 0;
    }
    double throughput_gflops() const {
        return avg_time_ms > 0 ? (compute_ops / (avg_time_ms * 1e-3)) / 1e9 : 0;
    }
};

class BenchmarkBase {
public:
    virtual ~BenchmarkBase() = default;
    virtual std::string name() const = 0;
    virtual std::string category() const = 0;
    
    virtual void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) = 0;
    virtual void run(cl::CommandQueue& queue) = 0;
    virtual void teardown() = 0;
    
    virtual BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const = 0;
};

class BenchmarkRunner {
public:
    BenchmarkRunner(cl::Context& context, cl::CommandQueue& queue, Tuner& tuner, bool stress_mode = false);
    ~BenchmarkRunner() = default;

    void addBenchmark(std::function<std::shared_ptr<BenchmarkBase>()> factory);
    void runAll(int iterations = 10, int warmup_iterations = 2);
    void runCategory(const std::string& category, int iterations = 10);
    void runSingle(const std::string& name, int iterations = 10);

    const std::map<std::string, BenchmarkMetrics>& getResults() const { return results_; }
    bool isStressMode() const { return stress_mode_; }

private:
    cl::Context& context_;
    cl::CommandQueue& queue_;
    Tuner& tuner_;
    bool stress_mode_;
    std::vector<std::function<std::shared_ptr<BenchmarkBase>()>> benchmarks_;
    std::map<std::string, BenchmarkMetrics> results_;

    void executeBenchmark(std::shared_ptr<BenchmarkBase> bench, int iterations, int warmup);
};

}
