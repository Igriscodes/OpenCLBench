#pragma once

#include <string>
#include <vector>

namespace oclbench {

struct CliOptions {
    int device_index = 0;
    std::string category = "all";
    std::string benchmark = "";
    int iterations = 10;
    bool stress_mode = false;
    std::string export_json = "";
    bool list_devices = false;
    bool all_devices = false;
    bool help = false;
};

class CliParser {
public:
    static CliOptions parse(int argc, char** argv);
    static void printHelp();
};

}
