#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <cstdio>
#include <mutex> // Добавляем мьютекс

#include "httplib.h"
#include "whisper.h"

// --- Конфигурация ---
struct Config {
    std::string model_path = "models/ggml-large-v3-turbo.bin";
    int port = 9090;
    int n_gpu_layers = 0;
};

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

// --- Улучшенная Проверка и Загрузка WAV (без Resampling) ---
// Проверяет, что файл является 16-bit PCM, Mono, 16000 Hz.
std::vector<float> load_wav(const std::string &path, int &n_samples) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open WAV file: " + path);

    char header[44];
    if (!f.read(header, 44)) {
        throw std::runtime_error("Failed to read full 44-byte WAV header.");
    }
    
    // Проверка формата (необходима для стабильности)
    // Байты 24-27: Sample Rate (Частота дискретизации)
    uint32_t sample_rate = *reinterpret_cast<uint32_t*>(&header[24]);
    // Байты 22-23: Number of Channels (Каналы)
    uint16_t n_channels = *reinterpret_cast<uint16_t*>(&header[22]);
    // Байты 34-35: Bits Per Sample (Битность)
    uint16_t bits_per_sample = *reinterpret_cast<uint16_t*>(&header[34]);

    if (sample_rate != 16000) {
        throw std::runtime_error("Invalid audio format: Sample rate must be 16000 Hz, found " + std::to_string(sample_rate) + " Hz.");
    }
    if (n_channels != 1) {
        throw std::runtime_error("Invalid audio format: Must be Mono (1 channel), found " + std::to_string(n_channels) + " channels.");
    }
    if (bits_per_sample != 16) {
        throw std::runtime_error("Invalid audio format: Must be 16-bit PCM, found " + std::to_string(bits_per_sample) + " bits.");
    }

    std::vector<float> audio;
    int16_t sample;

    // Чтение сэмплов (предполагая data chunk сразу после 44 байт)
    while (f.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        // Нормализация в диапазон [-1, 1]
        audio.push_back(sample / 32768.0f);
    }

    n_samples = audio.size();
    if (n_samples == 0) throw std::runtime_error("WAV file is empty or corrupted (0 samples read).");

    return audio;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);
    std::cout << "Model path: " << cfg.model_path << std::endl;
    std::cout << "Port: " << cfg.port << std::endl;
    std::cout << "GPU Layers: " << cfg.n_gpu_layers << " (use >0 for GPU, 0 for CPU)" << std::endl;

    // --- Инициализация Whisper ---
    whisper_context_params cparams = whisper_context_default_params();

    // Инициализируем контекст модели
    struct whisper_context* ctx = whisper_init_from_file_with_params(cfg.model_path.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Failed to load model: " << cfg.model_path << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully." << std::endl;

    // --- МЬЮТЕКС: Защита контекста Whisper ---
    // Так как httplib может использовать несколько потоков, а ctx не потокобезопасен.
    std::mutex whisper_mutex; 

    httplib::Server svr;
    svr.set_payload_max_length(50 * 1024 * 1024); // 50MB

    svr.Post("/file", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.body.empty()) {
            res.status = 400;
            res.set_content("No file uploaded", "text/plain");
            return;
        }

        // Проверка Content-Type (оставлена без изменений)
        if (req.get_header_value("Content-Type").find("audio/") == std::string::npos &&
            req.get_header_value("Content-Type") != "application/octet-stream" &&
            req.get_header_value("Content-Type") != "application/x-www-form-urlencoded") {
            res.status = 415;
            res.set_content("Unsupported Content-Type. Please use audio/* or application/octet-stream.", "text/plain");
            return;
        }

        std::string tmp_file;
        try {
            // Создание и запись временного файла
            std::ostringstream ss;
            ss << "/tmp/tmp_audio_" << std::hex << std::hash<std::string>{}(req.body) << ".wav";
            tmp_file = ss.str();

            { 
                std::ofstream out(tmp_file, std::ios::binary);
                if (!out.is_open()) throw std::runtime_error("Cannot create temporary file: " + tmp_file);
                out.write(req.body.c_str(), req.body.size());
            }

            int n_samples = 0;
            // Загрузка и проверка аудио
            std::vector<float> audio = load_wav(tmp_file, n_samples);

            // --- Настройки распознавания ---
            struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            wparams.print_progress = false;
            wparams.print_special = false;
            wparams.print_realtime = false;
            wparams.language = "ru";

            int n_threads = std::min(4, (int)std::thread::hardware_concurrency());
            wparams.n_threads = n_threads;

            // !!! БЕЗОПАСНОСТЬ ПОТОКОВ: Защищаем вызов whisper_full мьютексом !!!
            std::lock_guard<std::mutex> lock(whisper_mutex);
            
            // Распознаем
            if (whisper_full(ctx, wparams, audio.data(), n_samples) != 0) {
                throw std::runtime_error("Whisper processing failed");
            }

            // Собираем текст
            std::ostringstream text;
            int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const char* seg = whisper_full_get_segment_text(ctx, i);
                text << seg;
            }

            res.set_content(text.str(), "application/json");

        } catch (const std::exception &e) {
            std::cerr << "Processing Error: " << e.what() << std::endl;
            res.status = 400; // Используем 400 Bad Request для ошибок формата
            res.set_content(std::string("Error: ") + e.what(), "text/plain");
        }

        // --- Очистка временного файла ---
        if (!tmp_file.empty()) {
            if (std::remove(tmp_file.c_str()) != 0) {
                 std::cerr << "Warning: Failed to delete temporary file: " << tmp_file << std::endl;
            }
        }
    });

    std::cout << "Server starting on port " << cfg.port << "..." << std::endl;
    if (!svr.listen("0.0.0.0", cfg.port)) {
        std::cerr << "Failed to start server on port " << cfg.port << std::endl;
        return 1;
    }

    whisper_free(ctx);
    return 0;
}
