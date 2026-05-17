#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class MandelbrotBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "Mandelbrot"; }
    std::string category() const override { return "compute"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        width_ = 8192;
        height_ = 8192;
        max_iters_ = 256;
        params_ = params;
        
        d_Out_ = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * width_ * height_);
        
        std::string source = loadKernelSource("compute/mandelbrot.cl");
        cl::Program program(context, source);
        try {
            program.build("-cl-mad-enable -cl-fast-relaxed-math");
        } catch (...) {
            std::cerr << "Build Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw;
        }
        
        kernel_ = cl::Kernel(program, "mandelbrot");
        kernel_.setArg(0, d_Out_);
        kernel_.setArg(1, (int)width_);
        kernel_.setArg(2, (int)height_);
        kernel_.setArg(3, (int)max_iters_);
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(width_, height_);
        
        size_t l_x = 16, l_y = 16;
        if (params_.local_work_size[0] < 256) {
             l_x = 8; l_y = 8;
        }
        cl::NDRange local(l_x, l_y);
        
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = width_ * height_ * sizeof(int); // 1 write per pixel
        // Rough average of 50 iters per pixel * 8 ops per iter
        m.compute_ops = (double)width_ * (double)height_ * 50.0 * 8.0; 
        return m;
    }

private:
    size_t width_, height_;
    int max_iters_;
    TuningParams params_;
    cl::Buffer d_Out_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createMandelbrotBenchmark() {
    return std::make_shared<MandelbrotBenchmark>();
}

} // namespace oclbench