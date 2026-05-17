#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class NV12toBGRandResizeBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "NV12toBGRandResize"; }
    std::string category() const override { return "image"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        width_ = 1920;
        height_ = 1080;
        params_ = params;
        
        size_t nv12_size = width_ * height_ + (width_ * height_ / 2);
        std::vector<unsigned char> h_in(nv12_size, 128);
        
        d_In_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nv12_size, h_in.data());
        d_Out_ = cl::Buffer(context, CL_MEM_WRITE_ONLY, width_ * height_ * 3); // BGR
        
        std::string source = loadKernelSource("image/nv12_to_bgr.cl");
        cl::Program program(context, source);
        program.build("-cl-fast-relaxed-math");
        
        kernel_ = cl::Kernel(program, "nv12_to_bgr");
        kernel_.setArg(0, d_In_);
        kernel_.setArg(1, d_Out_);
        kernel_.setArg(2, (int)width_);
        kernel_.setArg(3, (int)height_);
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(width_ / 2, height_ / 2); // 2x2 macroblocks
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, cl::NullRange);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = (width_ * height_ * 1.5) + (width_ * height_ * 3);
        m.compute_ops = width_ * height_ * 15;
        return m;
    }

private:
    size_t width_, height_;
    TuningParams params_;
    cl::Buffer d_In_, d_Out_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createNV12toBGRBenchmark() { return std::make_shared<NV12toBGRandResizeBenchmark>(); }

} // namespace oclbench