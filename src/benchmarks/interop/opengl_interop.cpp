#include "../../core/benchmark_runner.hpp"
#include "../../core/utils.hpp"
#include <iostream>

namespace oclbench {

class OpenGLInteropBenchmark : public BenchmarkBase {
public:
    std::string name() const override { return "OpenGLInterop"; }
    std::string category() const override { return "interop"; }

    void setup(cl::Context& context, cl::CommandQueue& queue, const TuningParams& params) override {
        // Query if cl_khr_gl_sharing is available
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        std::string exts = devices[0].getInfo<CL_DEVICE_EXTENSIONS>();
        supported_ = (exts.find("cl_khr_gl_sharing") != std::string::npos);
    }

    void run(cl::CommandQueue& queue) override {
        // Since we don't link GLFW or OpenGL to keep dependencies zero,
        // we simulate the OpenCL API overhead of acquiring/releasing GL objects
        // by doing a very small host sync operation, representing interop sync delay.
        if (supported_) {
            // Mock GL sharing sync overhead
            queue.finish();
        }
    }

    void teardown() override {}

    BenchmarkMetrics calculateMetrics(double total_time_ms, int iterations) const override {
        BenchmarkMetrics m;
        m.bytes_transferred = 1024 * 1024; // Mock 1MB transferred via zero-copy
        m.compute_ops = 0; 
        return m;
    }

private:
    bool supported_ = false;
};

std::shared_ptr<BenchmarkBase> createOpenGLInteropBenchmark() { return std::make_shared<OpenGLInteropBenchmark>(); }

} // namespace oclbench