#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class FDTD3dBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "FDTD3d"; }
    std::string category() const override { return "memory"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        dim_x_ = 256;
        dim_y_ = 256;
        dim_z_ = 256;
        params_ = params;
        
        size_t total_elements = dim_x_ * dim_y_ * dim_z_;
        std::vector<float> h_in(total_elements, 1.0f);
        
        d_In_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * total_elements, h_in.data());
        d_Out_ = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * total_elements);
        
        std::string source = loadKernelSource("memory/fdtd3d.cl");
        cl::Program program(context, source);
        try {
            program.build();
        } catch (...) {
            std::cerr << "Build Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw;
        }
        
        kernel_ = cl::Kernel(program, "fdtd3d_step");
        kernel_.setArg(0, d_In_);
        kernel_.setArg(1, d_Out_);
        kernel_.setArg(2, (int)dim_x_);
        kernel_.setArg(3, (int)dim_y_);
        kernel_.setArg(4, (int)dim_z_);
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(dim_x_, dim_y_, dim_z_);
        
        size_t l_x = 8, l_y = 8, l_z = 4; // Typical 3D local size
        cl::NDRange local(l_x, l_y, l_z);
        
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        size_t total_elements = dim_x_ * dim_y_ * dim_z_;
        // 1 write + 7 reads (central + 6 neighbors) per element
        m.bytes_transferred = total_elements * sizeof(float) * 8; 
        m.compute_ops = (double)total_elements * 13.0; // Multiplies and adds in stencil
        return m;
    }

private:
    size_t dim_x_, dim_y_, dim_z_;
    TuningParams params_;
    cl::Buffer d_In_, d_Out_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createFDTD3dBenchmark() {
    return std::make_shared<FDTD3dBenchmark>();
}

} // namespace oclbench