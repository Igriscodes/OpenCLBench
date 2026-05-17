#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <vector>

namespace oclbench {

class NBodyBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "NBody"; }
    std::string category() const override { return "simulation"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        num_particles_ = 32768;
        params_ = params;
        
        std::vector<float> h_pos(num_particles_ * 4, 1.0f);
        std::vector<float> h_vel(num_particles_ * 4, 0.0f);
        
        d_Pos_ = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * 4 * num_particles_, h_pos.data());
        d_Vel_ = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * 4 * num_particles_, h_vel.data());
        d_PosNext_ = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * 4 * num_particles_);
        d_VelNext_ = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * 4 * num_particles_);
        
        std::string source = loadKernelSource("simulation/nbody.cl");
        cl::Program program(context, source);
        try {
            program.build("-cl-mad-enable -cl-fast-relaxed-math");
        } catch (...) {
            std::cerr << "Build Log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw;
        }
        
        kernel_ = cl::Kernel(program, "nbody_integration");
        kernel_.setArg(0, d_Pos_);
        kernel_.setArg(1, d_Vel_);
        kernel_.setArg(2, d_PosNext_);
        kernel_.setArg(3, d_VelNext_);
        kernel_.setArg(4, 0.01f);
        kernel_.setArg(5, 0.001f);
        kernel_.setArg(6, (int)num_particles_);
        
        kernel_.setArg(7, sizeof(float) * 4 * params_.local_work_size[0], nullptr);
    }

    void run(cl::CommandQueue& queue) override {
        cl::NDRange global(num_particles_);
        cl::NDRange local(params_.local_work_size[0]);
        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
        
        std::swap(d_Pos_, d_PosNext_);
        std::swap(d_Vel_, d_VelNext_);
        kernel_.setArg(0, d_Pos_);
        kernel_.setArg(1, d_Vel_);
        kernel_.setArg(2, d_PosNext_);
        kernel_.setArg(3, d_VelNext_);
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = num_particles_ * sizeof(float) * 4 * 2;
        m.compute_ops = (double)num_particles_ * (double)num_particles_ * 20.0;
        return m;
    }

private:
    size_t num_particles_;
    TuningParams params_;
    cl::Buffer d_Pos_, d_Vel_, d_PosNext_, d_VelNext_;
    cl::Kernel kernel_;
};

std::shared_ptr<BenchmarkBase> createNBodyBenchmark() {
    return std::make_shared<NBodyBenchmark>();
}

}
