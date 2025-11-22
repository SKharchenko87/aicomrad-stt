#include "config.h"
#include <string>

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            cfg.port = std::stoi(argv[++i]);
        } else if (arg == "--gpu-layers" && i + 1 < argc) {
            cfg.n_gpu_layers = std::stoi(argv[++i]);
        }
    }
    return cfg;
}
