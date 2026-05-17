#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class SobelBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "SobelFilter"; }
    std::string category() const override { return "image"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        width_ = 8192;
        height_ = 8192;
        params_ = params;
        
        std::vector<unsigned char> h_in(width_ * height_ * 4, 128);
        
        cl::ImageFormat format(CL_RGBA, CL_UNORM_INT8);
        
        d_In_ = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, width_, height_, 0, h_in.data());
        d_Out_ = cl::Image2D(context, CL_MEM_WRITE_ONLY, format, width_, height_, 0, nullptr);
        
        std::string source = loadKernelSource("image/sobel.cl");
        cl::Program program(context, source);
        try {
            program.build();
        } catch (...) {
            std::cerr << "Build Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw;
        }
        
        kernel_ = cl::Kernel(program, "sobel_filter");
        kernel_.setArg(0, d_In_);
        kernel_.setArg(1, d_Out_);
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(width_, height_);
        
        size_t l_x = 16;
        size_t l_y = 16;
        if (params_.local_work_size[0] < 256) {
             l_x = 8; l_y = 8;
        }
        cl::NDRange local(l_x, l_y);
        
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = width_ * height_ * 4 * 2;
        m.compute_ops = width_ * height_ * 30;
        return m;
    }

private:
    size_t width_, height_;
    TuningParams params_;
    cl::Image2D d_In_, d_Out_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createSobelBenchmark() {
    return std::make_shared<SobelBenchmark>();
}

}
