#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class BicubicTextureBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "BicubicTexture"; }
    std::string category() const override { return "memory"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        width_ = 4096;
        height_ = 4096;
        params_ = params;
        
        std::vector<float> h_in(width_ * height_, 0.5f); 
        cl::ImageFormat format(CL_R, CL_FLOAT);
        
        d_In_ = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, width_, height_, 0, h_in.data());
        d_Out_ = cl::Image2D(context, CL_MEM_WRITE_ONLY, format, width_ * 2, height_ * 2, 0, nullptr);
        
        std::string source = loadKernelSource("memory/bicubic_texture.cl");
        cl::Program program(context, source);
        program.build("-cl-fast-relaxed-math");
        
        kernel_ = cl::Kernel(program, "bicubic_upscale");
        kernel_.setArg(0, d_In_);
        kernel_.setArg(1, d_Out_);
        kernel_.setArg(2, 2.0f); // Scale factor
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(width_ * 2, height_ * 2); // Upscaled dimensions
        cl::NDRange local(16, 16);
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        // Writes to out, reads 16 samples per out pixel
        m.bytes_transferred = (width_ * 2 * height_ * 2) * sizeof(float) * 17;
        m.compute_ops = (width_ * 2 * height_ * 2) * 50; 
        return m;
    }

private:
    size_t width_, height_;
    TuningParams params_;
    cl::Image2D d_In_, d_Out_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createBicubicTextureBenchmark() { return std::make_shared<BicubicTextureBenchmark>(); }

} // namespace oclbench