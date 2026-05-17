#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class VolumeFilteringBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "VolumeFiltering"; }
    std::string category() const override { return "memory"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        dim_ = 256;
        params_ = params;
        
        std::vector<float> h_in(dim_ * dim_ * dim_, 1.0f);
        
        cl::ImageFormat format(CL_R, CL_FLOAT);
        d_In_ = cl::Image3D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, dim_, dim_, dim_, 0, 0, h_in.data());
        d_Out_ = cl::Image3D(context, CL_MEM_WRITE_ONLY, format, dim_, dim_, dim_, 0, 0, nullptr);
        
        std::string source = loadKernelSource("memory/volume_filtering.cl");
        cl::Program program(context, source);
        program.build("-cl-fast-relaxed-math");
        
        kernel_ = cl::Kernel(program, "volume_filter");
        kernel_.setArg(0, d_In_);
        kernel_.setArg(1, d_Out_);
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(dim_, dim_, dim_);
        cl::NDRange local(8, 8, 4);
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = (dim_ * dim_ * dim_) * sizeof(float) * 28; // 27 reads + 1 write
        m.compute_ops = (dim_ * dim_ * dim_) * 27 * 2; 
        return m;
    }

private:
    size_t dim_;
    TuningParams params_;
    cl::Image3D d_In_, d_Out_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createVolumeFilteringBenchmark() { return std::make_shared<VolumeFilteringBenchmark>(); }

} // namespace oclbench