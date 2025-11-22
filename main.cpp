#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <cstdio>

#include "httplib.h"
#include "whisper.h"
#include "config.h"
#include "audio_processor.h"
#include "whisper_engine.h"
#include "json_utils.h"


int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);
    std::cout << "Model path: " << cfg.model_path << std::endl;
    std::cout << "Port: " << cfg.port << std::endl;
    std::cout << "GPU Layers: " << cfg.n_gpu_layers << " (use >0 for GPU, 0 for CPU)" << std::endl;

    // Инициализация Whisper движка
    WhisperEngine whisper_engine;
    if (!whisper_engine.initialize(cfg.model_path)) {
        std::cerr << "Failed to load model: " << cfg.model_path << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully." << std::endl;

    httplib::Server svr;
    svr.set_payload_max_length(50 * 1024 * 1024);

    // Endpoint для распознавания
    svr.Post("/file", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.body.empty()) {
            res.status = 400;
            res.set_content(JsonUtils::build_error_response("No file uploaded"), "application/json");
            return;
        }

        if (req.get_header_value("Content-Type").find("audio/") == std::string::npos &&
            req.get_header_value("Content-Type") != "application/octet-stream" &&
            req.get_header_value("Content-Type") != "application/x-www-form-urlencoded") {
            res.status = 415;
            res.set_content(JsonUtils::build_error_response("Unsupported Content-Type. Please use audio/* or application/octet-stream."), "application/json");
            return;
        }

        std::string tmp_file;
        try {
            // Создание временного файла
            std::ostringstream ss;
            ss << "/tmp/tmp_audio_" << std::hex << std::hash<std::string>{}(req.body) << ".wav";
            tmp_file = ss.str();

            { 
                std::ofstream out(tmp_file, std::ios::binary);
                if (!out.is_open()) throw std::runtime_error("Cannot create temporary file: " + tmp_file);
                out.write(req.body.c_str(), req.body.size());
            }

            // Загрузка и обработка аудио
            int n_samples = 0;
            std::vector<float> audio = AudioProcessor::load_wav(tmp_file, n_samples);

            // Распознавание
            TranscriptionResult result = whisper_engine.transcribe(audio);

            // Формирование ответа
            std::string json_response = JsonUtils::build_success_response(
                result.text, result.segments, result.confidences, result.overall_confidence
            );
            res.set_content(json_response, "application/json");

        } catch (const std::exception &e) {
            std::cerr << "Processing Error: " << e.what() << std::endl;
            res.status = 400;
            res.set_content(JsonUtils::build_error_response(e.what()), "application/json");
        }

        // Очистка временного файла
        if (!tmp_file.empty()) {
            if (std::remove(tmp_file.c_str()) != 0) {
                 std::cerr << "Warning: Failed to delete temporary file: " << tmp_file << std::endl;
            }
        }
    });

    // Health check endpoint
    svr.Get("/health", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content(JsonUtils::build_health_response(), "application/json");
    });

    std::cout << "Server starting on port " << cfg.port << "..." << std::endl;
    if (!svr.listen("0.0.0.0", cfg.port)) {
        std::cerr << "Failed to start server on port " << cfg.port << std::endl;
        return 1;
    }

    return 0;
}
