#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <iostream>

namespace oclbench {

class VulkanInteropBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "VulkanInterop"; }
    std::string category() const override { return "interop"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        // Query if cl_khr_external_memory is available
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        std::string exts = devices[0].getInfo<CL_DEVICE_EXTENSIONS>();
        supported_ = (exts.find("cl_khr_external_memory") != std::string::npos);
    }

    void run(cl::CommandQueue& queue) override {
        // Simulating the sync overhead of VK semaphore waiting and signalling
        if (supported_) {
            queue.finish();
        }
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = 1024 * 1024; 
        m.compute_ops = 0; 
        return m;
    }

private:
    bool supported_ = false;
};

std::shared_ptr<BenchmarkBase> createVulkanInteropBenchmark() { return std::make_shared<VulkanInteropBenchmark>(); }

} // namespace oclbench