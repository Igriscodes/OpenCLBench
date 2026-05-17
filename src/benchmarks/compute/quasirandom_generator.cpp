#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class QuasirandomGeneratorBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "QuasirandomGenerator"; }
    std::string category() const override { return "compute"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        num_vectors_ = 1024 * 1024 * 8; // 8M vectors
        params_ = params;
        
        std::vector<unsigned int> h_table(32 * 63, 1); // Mock direction table
        
        d_Table_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * 32 * 63, h_table.data());
        d_Out_ = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * num_vectors_ * 3); // 3D vectors
        
        std::string source = loadKernelSource("compute/quasirandom_generator.cl");
        cl::Program program(context, source);
        program.build("-cl-mad-enable -cl-fast-relaxed-math");
        
        kernel_ = cl::Kernel(program, "quasirandom_generator");
        kernel_.setArg(0, d_Table_);
        kernel_.setArg(1, d_Out_);
        kernel_.setArg(2, (int)num_vectors_);
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(num_vectors_);
        cl::NDRange local(params_.local_work_size[0]);
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = num_vectors_ * 3 * sizeof(float);
        m.compute_ops = num_vectors_ * 63 * 3; // Approx bitwise operations
        return m;
    }

private:
    size_t num_vectors_;
    TuningParams params_;
    cl::Buffer d_Table_, d_Out_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createQuasirandomGeneratorBenchmark() { return std::make_shared<QuasirandomGeneratorBenchmark>(); }

} // namespace oclbench