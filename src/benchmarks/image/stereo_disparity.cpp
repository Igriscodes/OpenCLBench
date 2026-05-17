#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class StereoDisparityBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "StereoDisparity"; }
    std::string category() const override { return "image"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        width_ = 1920;
        height_ = 1080;
        max_disparity_ = 64;
        params_ = params;
        
        std::vector<unsigned char> h_in(width_ * height_, 128); // Grayscale
        
        d_Left_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width_ * height_, h_in.data());
        d_Right_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width_ * height_, h_in.data());
        d_Out_ = cl::Buffer(context, CL_MEM_WRITE_ONLY, width_ * height_);
        
        std::string source = loadKernelSource("image/stereo_disparity.cl");
        cl::Program program(context, source);
        program.build("-cl-fast-relaxed-math");
        
        kernel_ = cl::Kernel(program, "stereo_disparity");
        kernel_.setArg(0, d_Left_);
        kernel_.setArg(1, d_Right_);
        kernel_.setArg(2, d_Out_);
        kernel_.setArg(3, (int)width_);
        kernel_.setArg(4, (int)height_);
        kernel_.setArg(5, (int)max_disparity_);
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(width_, height_);
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, cl::NullRange);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        // Sad block match (7x7) over max_disparity range
        m.bytes_transferred = width_ * height_ * max_disparity_ * 2; 
        m.compute_ops = width_ * height_ * max_disparity_ * 49 * 3; // Block math ops
        return m;
    }

private:
    size_t width_, height_;
    int max_disparity_;
    TuningParams params_;
    cl::Buffer d_Left_, d_Right_, d_Out_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createStereoDisparityBenchmark() { return std::make_shared<StereoDisparityBenchmark>(); }

} // namespace oclbench