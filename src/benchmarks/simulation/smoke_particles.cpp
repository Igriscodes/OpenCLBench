#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class SmokeParticlesBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "SmokeParticles"; }
    std::string category() const override { return "simulation"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        num_particles_ = 1024 * 1024; // 1M particles
        params_ = params;
        
        std::vector<float> h_pos(num_particles_ * 4, 1.0f);
        std::vector<float> h_vel(num_particles_ * 4, 0.5f);
        
        d_Pos_ = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * 4 * num_particles_, h_pos.data());
        d_Vel_ = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 4 * num_particles_, h_vel.data());
        
        std::string source = loadKernelSource("simulation/smoke_particles.cl");
        cl::Program program(context, source);
        program.build("-cl-mad-enable -cl-fast-relaxed-math");
        
        kernel_ = cl::Kernel(program, "advect_smoke");
        kernel_.setArg(0, d_Pos_);
        kernel_.setArg(1, d_Vel_);
        kernel_.setArg(2, 0.016f); // dt
        kernel_.setArg(3, (int)num_particles_);
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(num_particles_);
        cl::NDRange local(params_.local_work_size[0]);
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = num_particles_ * sizeof(float) * 8; // Read pos, read vel, write pos
        m.compute_ops = num_particles_ * 15; // Math ops per particle
        return m;
    }

private:
    size_t num_particles_;
    TuningParams params_;
    cl::Buffer d_Pos_, d_Vel_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createSmokeParticlesBenchmark() { return std::make_shared<SmokeParticlesBenchmark>(); }

} // namespace oclbench