#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class FWTBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "FastWalshTransform"; }
    std::string category() const override { return "dsp"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        size_ = 1024 * 1024 * 32;
        params_ = params;
        
        std::vector<float> h_data(size_, 1.0f);
        
        d_Data_ = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * size_, h_data.data());
        
        std::string source = loadKernelSource("dsp/fwt.cl");
        cl::Program program(context, source);
        try {
            program.build();
        } catch (...) {
            std::cerr << "Build Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw;
        }
        
        kernel_ = cl::Kernel(program, "fwt_step");
    }

    void run(cl::CommandQueue& queue) override {
        int log2N = 0;
        for (size_t temp = size_; temp > 1; temp >>= 1) log2N++;
        
        for (int step = 0; step < log2N; ++step) {
            int step_size = 1 << step;
            kernel_.setArg(0, d_Data_);
            kernel_.setArg(1, step_size);
            
            cl::NDRange global(size_ / 2);
            cl::NDRange local(params_.local_work_size[0]);
            queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
        }
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        int log2N = 0;
        for (size_t temp = size_; temp > 1; temp >>= 1) log2N++;
        
        m.bytes_transferred = (size_ / 2) * sizeof(float) * 2 * 2 * log2N;
        m.compute_ops = (size_ / 2) * 2 * log2N;
        return m;
    }

private:
    size_t size_;
    TuningParams params_;
    cl::Buffer d_Data_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createFWTBenchmark() {
    return std::make_shared<FWTBenchmark>();
}

}
