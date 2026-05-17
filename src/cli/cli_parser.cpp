#include "cli_parser.hpp"
#include <iostream>
#include <cstring>

namespace oclbench {

CliOptions CliParser::parse(int argc, char** argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-d") == 0 || std::strcmp(argv[i], "--device") == 0) {
            if (i + 1 < argc) opts.device_index = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-c") == 0 || std::strcmp(argv[i], "--category") == 0) {
            if (i + 1 < argc) opts.category = argv[++i];
        } else if (std::strcmp(argv[i], "-b") == 0 || std::strcmp(argv[i], "--benchmark") == 0) {
            if (i + 1 < argc) opts.benchmark = argv[++i];
        } else if (std::strcmp(argv[i], "-i") == 0 || std::strcmp(argv[i], "--iterations") == 0) {
            if (i + 1 < argc) opts.iterations = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--stress") == 0) {
            opts.stress_mode = true;
        } else if (std::strcmp(argv[i], "--json") == 0) {
            if (i + 1 < argc) opts.export_json = argv[++i];
        } else if (std::strcmp(argv[i], "-l") == 0 || std::strcmp(argv[i], "--list") == 0) {
            opts.list_devices = true;
        } else if (std::strcmp(argv[i], "-a") == 0 || std::strcmp(argv[i], "--all-devices") == 0) {
            opts.all_devices = true;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            opts.help = true;
        }
    }
    return opts;
}

void CliParser::printHelp() {
    std::cout << "OpenCLBench - Professional OpenCL GPU Benchmark Suite\n"
              << "Usage: OpenCLBench [options]\n"
              << "Options:\n"
              << "  -d, --device <index>     Select device index (default: 0)\n"
              << "  -a, --all-devices        Run on all available devices\n"
              << "  -l, --list               List available OpenCL devices\n"
              << "  -c, --category <name>    Run specific category (compute, memory, simulation, dsp, image, interop, all)\n"
              << "  -b, --benchmark <name>   Run specific benchmark\n"
              << "  -i, --iterations <num>   Number of iterations per benchmark (default: 10)\n"
              << "      --stress             Enable stress test mode\n"
              << "      --json <path>        Export results to JSON file\n"
              << "  -h, --help               Print this help message\n";
}

}
