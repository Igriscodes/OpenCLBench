#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class HaarWavelet1DBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "HaarWavelet1D"; }
    std::string category() const override { return "dsp"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        size_ = 1024 * 1024 * 32; // 32M elements (must be power of 2)
        params_ = params;
        
        std::vector<float> h_in(size_, 1.0f);
        
        d_In_ = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * size_, h_in.data());
        d_Out_ = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * size_);
        
        std::string source = loadKernelSource("dsp/haar_wavelet_1d.cl");
        cl::Program program(context, source);
        program.build("-cl-mad-enable -cl-fast-relaxed-math");
        
        kernel_ = cl::Kernel(program, "haar_1d_step");
        kernel_.setArg(0, d_In_);
        kernel_.setArg(1, d_Out_);
    }

    void run(cl::CommandQueue& queue) override {
        int current_size = size_;
        while (current_size > 1) {
            cl::NDRange global(current_size / 2);
            
            kernel_.setArg(2, current_size);
            queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, cl::NullRange);
            current_size /= 2;
        }
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = size_ * sizeof(float) * 2; 
        m.compute_ops = size_ * 2; 
        return m;
    }

private:
    size_t size_;
    TuningParams params_;
    cl::Buffer d_In_, d_Out_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createHaarWavelet1DBenchmark() { return std::make_shared<HaarWavelet1DBenchmark>(); }

} // namespace oclbench