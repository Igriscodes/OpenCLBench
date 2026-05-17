#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class BilateralFilterBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "BilateralFilter"; }
    std::string category() const override { return "image"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        width_ = 2048;
        height_ = 2048;
        params_ = params;
        
        std::vector<float> h_in(width_ * height_, 0.5f);
        
        cl::ImageFormat format(CL_R, CL_FLOAT);
        d_In_ = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, width_, height_, 0, h_in.data());
        d_Out_ = cl::Image2D(context, CL_MEM_WRITE_ONLY, format, width_, height_, 0, nullptr);
        
        std::string source = loadKernelSource("image/bilateral_filter.cl");
        cl::Program program(context, source);
        program.build("-cl-fast-relaxed-math");
        
        kernel_ = cl::Kernel(program, "bilateral_filter");
        kernel_.setArg(0, d_In_);
        kernel_.setArg(1, d_Out_);
        kernel_.setArg(2, 3.0f); // sigma spatial
        kernel_.setArg(3, 0.1f); // sigma color
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(width_, height_);
        cl::NDRange local(16, 16);
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = width_ * height_ * sizeof(float) * 2; 
        m.compute_ops = width_ * height_ * 25 * 10; // 5x5 filter
        return m;
    }

private:
    size_t width_, height_;
    TuningParams params_;
    cl::Image2D d_In_, d_Out_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createBilateralFilterBenchmark() { return std::make_shared<BilateralFilterBenchmark>(); }

} // namespace oclbench