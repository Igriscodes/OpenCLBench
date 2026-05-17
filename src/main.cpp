#include <iostream>
#include <memory>
#include "cli/cli_parser.hpp"
#include "core/device_manager.hpp"
#include "core/tuner.hpp"
#include "core/benchmark_runner.hpp"
#include "core/reporting.hpp"

namespace oclbench {
    extern std::shared_ptr<BenchmarkBase> createBlackScholesBenchmark();
    extern std::shared_ptr<BenchmarkBase> createMandelbrotBenchmark();
    extern std::shared_ptr<BenchmarkBase> createBinomialOptionsBenchmark();
    extern std::shared_ptr<BenchmarkBase> createSobolQRNGBenchmark();
    extern std::shared_ptr<BenchmarkBase> createQuasirandomGeneratorBenchmark();
    
    extern std::shared_ptr<BenchmarkBase> createBandwidthBenchmark();
    extern std::shared_ptr<BenchmarkBase> createFDTD3dBenchmark();
    extern std::shared_ptr<BenchmarkBase> createBicubicTextureBenchmark();
    extern std::shared_ptr<BenchmarkBase> createVolumeFilteringBenchmark();
    
    extern std::shared_ptr<BenchmarkBase> createNBodyBenchmark();
    extern std::shared_ptr<BenchmarkBase> createSmokeParticlesBenchmark();
    extern std::shared_ptr<BenchmarkBase> createFluidsBenchmark();
    
    extern std::shared_ptr<BenchmarkBase> createFWTBenchmark();
    extern std::shared_ptr<BenchmarkBase> createConvolutionFFT2DBenchmark();
    extern std::shared_ptr<BenchmarkBase> createHaarWavelet1DBenchmark();
    
    extern std::shared_ptr<BenchmarkBase> createSobelBenchmark();
    extern std::shared_ptr<BenchmarkBase> createBilateralFilterBenchmark();
    extern std::shared_ptr<BenchmarkBase> createRecursiveGaussianBenchmark();
    extern std::shared_ptr<BenchmarkBase> createNV12toBGRBenchmark();
    extern std::shared_ptr<BenchmarkBase> createStereoDisparityBenchmark();
    
    extern std::shared_ptr<BenchmarkBase> createOpenGLInteropBenchmark();
    extern std::shared_ptr<BenchmarkBase> createVulkanInteropBenchmark();
}

int main(int argc, char** argv) {
    using namespace oclbench;

    CliOptions opts = CliParser::parse(argc, argv);

    if (opts.help) {
        CliParser::printHelp();
        return 0;
    }

    try {
        DeviceManager dev_mgr;
        if (opts.list_devices) {
            std::cout << "Available OpenCL Devices:\n";
            int idx = 0;
            for (const auto& dev : dev_mgr.getDevices()) {
                std::cout << "[" << idx++ << "] ";
                dev.print();
            }
            return 0;
        }

        std::vector<size_t> devices_to_run;
        if (opts.all_devices) {
            for (size_t i = 0; i < dev_mgr.getDevices().size(); ++i) {
                devices_to_run.push_back(i);
            }
        } else {
            devices_to_run.push_back(opts.device_index);
        }

        for (size_t dev_idx : devices_to_run) {
            const DeviceInfo& device = dev_mgr.getDevice(dev_idx);
            std::cout << "\n=========================================================================================================\n";
            std::cout << "Initializing Device: [" << dev_idx << "] " << device.name << "\n";
            std::cout << "=========================================================================================================\n";

            cl::Context context({device.device});
            cl::CommandQueue queue(context, device.device, CL_QUEUE_PROFILING_ENABLE);

            Tuner tuner(device);
            tuner.profileArchitecture(context, queue);

            BenchmarkRunner runner(context, queue, tuner, opts.stress_mode);

            runner.addBenchmark(createBlackScholesBenchmark);
            runner.addBenchmark(createMandelbrotBenchmark);
            runner.addBenchmark(createBinomialOptionsBenchmark);
            runner.addBenchmark(createSobolQRNGBenchmark);
            runner.addBenchmark(createQuasirandomGeneratorBenchmark);
            
            runner.addBenchmark(createBandwidthBenchmark);
            runner.addBenchmark(createFDTD3dBenchmark);
            runner.addBenchmark(createBicubicTextureBenchmark);
            runner.addBenchmark(createVolumeFilteringBenchmark);
            
            runner.addBenchmark(createNBodyBenchmark);
            runner.addBenchmark(createSmokeParticlesBenchmark);
            runner.addBenchmark(createFluidsBenchmark);
            
            runner.addBenchmark(createFWTBenchmark);
            runner.addBenchmark(createConvolutionFFT2DBenchmark);
            runner.addBenchmark(createHaarWavelet1DBenchmark);
            
            runner.addBenchmark(createSobelBenchmark);
            runner.addBenchmark(createBilateralFilterBenchmark);
            runner.addBenchmark(createRecursiveGaussianBenchmark);
            runner.addBenchmark(createNV12toBGRBenchmark);
            runner.addBenchmark(createStereoDisparityBenchmark);
            
            runner.addBenchmark(createOpenGLInteropBenchmark);
            runner.addBenchmark(createVulkanInteropBenchmark);

            if (!opts.benchmark.empty()) {
                runner.runSingle(opts.benchmark, opts.iterations);
            } else if (opts.category != "all") {
                runner.runCategory(opts.category, opts.iterations);
            } else {
                runner.runAll(opts.iterations);
            }

            ReportingEngine reporter(device, runner.getResults());
            reporter.printTerminalSummary();

            if (!opts.export_json.empty()) {
                std::string json_path = opts.export_json;
                if (opts.all_devices) {
                    size_t dot_pos = json_path.find_last_of('.');
                    if (dot_pos != std::string::npos) {
                        json_path.insert(dot_pos, "_" + std::to_string(dev_idx));
                    } else {
                        json_path += "_" + std::to_string(dev_idx);
                    }
                }
                reporter.exportJSON(json_path);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
