#ifndef CONFIG_H
#define CONFIG_H

#include <string>

struct Config {
    std::string model_path = "models/ggml-large-v3-turbo.bin";
    int port = 9090;
    int n_gpu_layers = 0;
};

Config parse_args(int argc, char** argv);

#endif
