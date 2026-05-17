#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class FluidsBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "Fluids"; }
    std::string category() const override { return "simulation"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        num_particles_ = 65536; // 64K particles (O(N^2) SPH)
        params_ = params;
        
        std::vector<float> h_pos(num_particles_ * 4, 1.0f);
        std::vector<float> h_density(num_particles_, 1000.0f);
        
        d_Pos_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 4 * num_particles_, h_pos.data());
        d_Density_ = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * num_particles_);
        
        std::string source = loadKernelSource("simulation/fluids.cl");
        cl::Program program(context, source);
        program.build("-cl-mad-enable -cl-fast-relaxed-math");
        
        kernel_ = cl::Kernel(program, "sph_density");
        kernel_.setArg(0, d_Pos_);
        kernel_.setArg(1, d_Density_);
        kernel_.setArg(2, 0.1f); // smoothing length
        kernel_.setArg(3, (int)num_particles_);
        kernel_.setArg(4, sizeof(float) * 4 * params_.local_work_size[0], nullptr);
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(num_particles_);
        cl::NDRange local(params_.local_work_size[0]);
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = num_particles_ * sizeof(float) * 5; 
        m.compute_ops = (double)num_particles_ * (double)num_particles_ * 15.0; 
        return m;
    }

private:
    size_t num_particles_;
    TuningParams params_;
    cl::Buffer d_Pos_, d_Density_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createFluidsBenchmark() { return std::make_shared<FluidsBenchmark>(); }

} // namespace oclbench