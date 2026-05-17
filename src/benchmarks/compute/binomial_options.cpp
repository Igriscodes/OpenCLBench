#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class BinomialOptionsBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "BinomialOptions"; }
    std::string category() const override { return "compute"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        num_options_ = 1024 * 1024; // 1M options
        params_ = params;
        
        std::vector<float> h_S(num_options_, 100.0f);
        std::vector<float> h_X(num_options_, 98.0f);
        std::vector<float> h_T(num_options_, 2.0f);
        
        d_S_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_options_, h_S.data());
        d_X_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_options_, h_X.data());
        d_T_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_options_, h_T.data());
        d_Call_ = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * num_options_);
        
        std::string source = loadKernelSource("compute/binomial_options.cl");
        cl::Program program(context, source);
        program.build("-cl-mad-enable -cl-fast-relaxed-math");
        
        kernel_ = cl::Kernel(program, "binomial_options");
        kernel_.setArg(0, d_S_);
        kernel_.setArg(1, d_X_);
        kernel_.setArg(2, d_T_);
        kernel_.setArg(3, d_Call_);
        kernel_.setArg(4, 0.02f); // risk_free
        kernel_.setArg(5, 0.30f); // volatility
        kernel_.setArg(6, 256); // num_steps
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(num_options_);
        cl::NDRange local(params_.local_work_size[0]);
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = num_options_ * sizeof(float) * 4; // 3 reads, 1 write
        m.compute_ops = num_options_ * 256 * 10; // Approx 10 ops per step
        return m;
    }

private:
    size_t num_options_;
    TuningParams params_;
    cl::Buffer d_S_, d_X_, d_T_, d_Call_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createBinomialOptionsBenchmark() { return std::make_shared<BinomialOptionsBenchmark>(); }

} // namespace oclbench