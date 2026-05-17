#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>
#include <cmath>

namespace oclbench {

class BlackScholesBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "BlackScholes"; }
    std::string category() const override { return "compute"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        num_options_ = 1024 * 1024 * 16;
        params_ = params;
        
        std::vector<float> h_S(num_options_, 100.0f);
        std::vector<float> h_X(num_options_, 98.0f);
        std::vector<float> h_T(num_options_, 2.0f);
        
        d_S_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_options_, h_S.data());
        d_X_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_options_, h_X.data());
        d_T_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_options_, h_T.data());
        d_Call_ = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * num_options_);
        d_Put_ = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * num_options_);
        
        std::string source = loadKernelSource("compute/black_scholes.cl");
        cl::Program program(context, source);
        try {
            program.build("-cl-mad-enable -cl-fast-relaxed-math");
        } catch (...) {
            std::cerr << "Build Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw;
        }
        
        kernel_ = cl::Kernel(program, "blackScholes");
        kernel_.setArg(0, d_S_);
        kernel_.setArg(1, d_X_);
        kernel_.setArg(2, d_T_);
        kernel_.setArg(3, d_Call_);
        kernel_.setArg(4, d_Put_);
        kernel_.setArg(5, 0.02f);
        kernel_.setArg(6, 0.30f);
        kernel_.setArg(7, (int)num_options_);
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(num_options_);
        cl::NDRange local(params_.local_work_size[0]);
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {
    }

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = num_options_ * sizeof(float) * 5;
        m.compute_ops = num_options_ * 50;
        return m;
    }

private:
    size_t num_options_;
    TuningParams params_;
    cl::Buffer d_S_, d_X_, d_T_, d_Call_, d_Put_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createBlackScholesBenchmark() {
    return std::make_shared<BlackScholesBenchmark>();
}

}
