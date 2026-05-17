#include "benchmark_runner.hpp"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace oclbench {

BenchmarkRunner::BenchmarkRunner(cl::Context& context, cl::CommandQueue& queue, Tuner& tuner, bool stress_mode)
    : context_(context), queue_(queue), tuner_(tuner), stress_mode_(stress_mode) {
}

void BenchmarkRunner::addBenchmark(std::function<std::shared_ptr<BenchmarkBase>()> factory) {
    benchmarks_.push_back(factory);
}

void BenchmarkRunner::runAll(int iterations, int warmup_iterations) {
    for (auto& factory : benchmarks_) {
        executeBenchmark(factory(), iterations, warmup_iterations);
    }
}

void BenchmarkRunner::runCategory(const std::string& category, int iterations) {
    for (auto& factory : benchmarks_) {
        auto bench = factory();
        if (bench->category() == category) {
            executeBenchmark(bench, iterations, 2);
        }
    }
}

void BenchmarkRunner::runSingle(const std::string& name, int iterations) {
    for (auto& factory : benchmarks_) {
        auto bench = factory();
        if (bench->name() == name) {
            executeBenchmark(bench, iterations, 2);
        }
    }
}

void BenchmarkRunner::executeBenchmark(std::shared_ptr<BenchmarkBase> bench, int iterations, int warmup) {
    std::cout << "[Runner] Running " << bench->name() << " [" << bench->category() << "]..." << std::endl;
    
    try {
        TuningParams params = tuner_.getParams(bench->name());
        bench->setup(context_, queue_, params);

        for (int i = 0; i < warmup; ++i) {
            bench->run(queue_);
        }
        queue_.finish();

        std::vector<double> timings;
        double degradation = 0.0;
        
        if (stress_mode_) {
            std::cout << "         -> [Stress Mode] Running for 30 seconds..." << std::endl;
            auto stress_start = std::chrono::high_resolution_clock::now();
            std::vector<double> initial_timings;
            std::vector<double> final_timings;
            int runs = 0;
            
            while (true) {
                auto start = std::chrono::high_resolution_clock::now();
                bench->run(queue_);
                queue_.finish();
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> diff = end - start;
                
                if (runs < 10) initial_timings.push_back(diff.count());
                
                std::chrono::duration<double> total_elapsed = end - stress_start;
                if (total_elapsed.count() >= 30.0) {
                    final_timings.push_back(diff.count());
                    if (final_timings.size() >= 10 || total_elapsed.count() >= 35.0) break;
                }
                
                timings.push_back(diff.count());
                runs++;
            }
            
            if (initial_timings.size() > 0 && final_timings.size() > 0) {
                double avg_init = std::accumulate(initial_timings.begin(), initial_timings.end(), 0.0) / initial_timings.size();
                double avg_final = std::accumulate(final_timings.begin(), final_timings.end(), 0.0) / final_timings.size();
                degradation = ((avg_final - avg_init) / avg_init) * 100.0;
            }
            iterations = timings.size();
        } else {
            timings.reserve(iterations);
            for (int i = 0; i < iterations; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                bench->run(queue_);
                queue_.finish();
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> diff = end - start;
                timings.push_back(diff.count());
            }
        }

        bench->teardown();

        if (timings.empty()) return;

        double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
        double avg = sum / iterations;
        double min_t = *std::min_element(timings.begin(), timings.end());
        double max_t = *std::max_element(timings.begin(), timings.end());
        
        double sq_sum = std::inner_product(timings.begin(), timings.end(), timings.begin(), 0.0);
        double stddev = std::sqrt(std::max(0.0, sq_sum / iterations - avg * avg));

        BenchmarkMetrics metrics = bench->calculateMetrics(avg, 1);
        metrics.min_time_ms = min_t;
        metrics.max_time_ms = max_t;
        metrics.avg_time_ms = avg;
        metrics.stddev_time_ms = stddev;
        metrics.thermal_degradation_pct = degradation;

        results_[bench->name()] = metrics;

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error in " << bench->name() << ": " << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (const std::exception& err) {
        std::cerr << "Error in " << bench->name() << ": " << err.what() << std::endl;
    }
}

}
