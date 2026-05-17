#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class RecursiveGaussianBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "RecursiveGaussian"; }
    std::string category() const override { return "image"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        width_ = 2048;
        height_ = 2048;
        params_ = params;
        
        std::vector<float> h_in(width_ * height_, 1.0f);
        
        d_In_ = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * width_ * height_, h_in.data());
        d_Out_ = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * width_ * height_);
        
        std::string source = loadKernelSource("image/recursive_gaussian.cl");
        cl::Program program(context, source);
        program.build("-cl-mad-enable -cl-fast-relaxed-math");
        
        kernel_ = cl::Kernel(program, "recursive_gaussian_transpose");
        kernel_.setArg(0, d_In_);
        kernel_.setArg(1, d_Out_);
        kernel_.setArg(2, (int)width_);
        kernel_.setArg(3, (int)height_);
    }

    void run(cl::CommandQueue& queue) override {
        // Just mocking the transpose pass for metric timing
        cl::NDRange global(width_, height_);
        cl::NDRange local(16, 16);
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = width_ * height_ * sizeof(float) * 2; 
        m.compute_ops = width_ * height_ * 5; 
        return m;
    }

private:
    size_t width_, height_;
    TuningParams params_;
    cl::Buffer d_In_, d_Out_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createRecursiveGaussianBenchmark() { return std::make_shared<RecursiveGaussianBenchmark>(); }

} // namespace oclbench