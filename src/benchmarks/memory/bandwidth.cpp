#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class BandwidthBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "GlobalBandwidth"; }
    std::string category() const override { return "memory"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        size_ = 256 * 1024 * 1024 / sizeof(float);
        params_ = params;
        
        d_Src_ = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float) * size_);
        d_Dst_ = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size_);
        
        std::string source = loadKernelSource("memory/bandwidth.cl");
        cl::Program program(context, source);
        try {
            program.build();
        } catch (...) {
            std::cerr << "Build Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw;
        }
        
        kernel_ = cl::Kernel(program, "copy_float4");
        kernel_.setArg(0, d_Src_);
        kernel_.setArg(1, d_Dst_);
        kernel_.setArg(2, (int)(size_ / 4));
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(size_ / 4);
        cl::NDRange local(params_.local_work_size[0]);
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = size_ * sizeof(float) * 2;
        m.compute_ops = 0;
        return m;
    }

private:
    size_t size_;
    TuningParams params_;
    cl::Buffer d_Src_, d_Dst_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createBandwidthBenchmark() {
    return std::make_shared<BandwidthBenchmark>();
}

}
