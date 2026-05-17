#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class SobolQRNGBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "SobolQRNG"; }
    std::string category() const override { return "compute"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        num_vectors_ = 1024 * 1024 * 4; // 4M vectors (prevent out of memory on 1GB cards)
        num_dimensions_ = 32;
        params_ = params;
        
        std::vector<unsigned int> h_dir(num_dimensions_ * 32, 1); // Mock direction numbers
        
        d_Dir_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * 32 * num_dimensions_, h_dir.data());
        d_Out_ = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * num_vectors_ * num_dimensions_);
        
        std::string source = loadKernelSource("compute/sobol_qrng.cl");
        cl::Program program(context, source);
        program.build("-cl-mad-enable -cl-fast-relaxed-math");
        
        kernel_ = cl::Kernel(program, "sobol_qrng");
        kernel_.setArg(0, d_Dir_);
        kernel_.setArg(1, d_Out_);
        kernel_.setArg(2, (int)num_vectors_);
        kernel_.setArg(3, (int)num_dimensions_);
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(num_vectors_);
        cl::NDRange local(params_.local_work_size[0]);
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = num_vectors_ * num_dimensions_ * sizeof(float);
        m.compute_ops = num_vectors_ * num_dimensions_ * 10; // Bitwise XORs
        return m;
    }

private:
    size_t num_vectors_;
    size_t num_dimensions_;
    TuningParams params_;
    cl::Buffer d_Dir_, d_Out_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createSobolQRNGBenchmark() { return std::make_shared<SobolQRNGBenchmark>(); }

} // namespace oclbench