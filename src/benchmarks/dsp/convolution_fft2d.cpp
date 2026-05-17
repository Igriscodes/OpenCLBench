#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class ConvolutionFFT2DBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "ConvolutionFFT2D"; }
    std::string category() const override { return "dsp"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        width_ = 4096;
        height_ = 4096;
        params_ = params;
        
        std::vector<float> h_real(width_ * height_, 1.0f);
        std::vector<float> h_imag(width_ * height_, 0.0f);
        
        d_Real_ = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * width_ * height_, h_real.data());
        d_Imag_ = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * width_ * height_, h_imag.data());
        
        std::string source = loadKernelSource("dsp/convolution_fft2d.cl");
        cl::Program program(context, source);
        program.build("-cl-mad-enable -cl-fast-relaxed-math");
        
        kernel_ = cl::Kernel(program, "fft2d_radix2");
        kernel_.setArg(0, d_Real_);
        kernel_.setArg(1, d_Imag_);
        kernel_.setArg(2, (int)width_);
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(width_ / 2, height_);
        cl::NDRange local(params_.local_work_size[0], 1);
        
        // Emulate log2(N) passes of a row FFT
        int log2N = 12; // 4096 is 2^12
        for (int step = 0; step < log2N; ++step) {
            int step_size = 1 << step;
            kernel_.setArg(3, step_size);
            queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
        }
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        int log2N = 12;
        m.bytes_transferred = (width_ * height_) * sizeof(float) * 2 * 2 * log2N;
        m.compute_ops = (width_ * height_) * 5 * log2N; // Complex math ops
        return m;
    }

private:
    size_t width_, height_;
    TuningParams params_;
    cl::Buffer d_Real_, d_Imag_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createConvolutionFFT2DBenchmark() { return std::make_shared<ConvolutionFFT2DBenchmark>(); }

} // namespace oclbench