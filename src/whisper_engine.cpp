#include "whisper_engine.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <thread>

WhisperEngine::WhisperEngine() : ctx(nullptr) {}

WhisperEngine::~WhisperEngine() {
    if (ctx) {
        whisper_free(ctx);
    }
}

bool WhisperEngine::initialize(const std::string& model_path) {
    whisper_context_params cparams = whisper_context_default_params();
    ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    return ctx != nullptr;
}

float WhisperEngine::calculate_segment_confidence(int segment_index) const {
    float segment_confidence = 0.0f;
    int n_tokens = whisper_full_n_tokens(ctx, segment_index);
    if (n_tokens > 0) {
        float total_confidence = 0.0f;
        for (int j = 0; j < n_tokens; ++j) {
            whisper_token_data token_data = whisper_full_get_token_data(ctx, segment_index, j);
            total_confidence += token_data.p;
        }
        segment_confidence = total_confidence / n_tokens;
    }
    return segment_confidence;
}

float WhisperEngine::calculate_overall_confidence() const {
    float total_confidence = 0.0f;
    int total_tokens = 0;
    int n_segments = whisper_full_n_segments(ctx);
    
    for (int i = 0; i < n_segments; ++i) {
        int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            whisper_token_data token_data = whisper_full_get_token_data(ctx, i, j);
            total_confidence += token_data.p;
            total_tokens++;
        }
    }
    return total_tokens > 0 ? total_confidence / total_tokens : 0.0f;
}

std::string WhisperEngine::clean_text(const std::string& text) const {
    std::string cleaned = text;
    if (!cleaned.empty()) {
        size_t start = cleaned.find_first_not_of(" \t\n\r");
        size_t end = cleaned.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            cleaned = cleaned.substr(start, end - start + 1);
        }
    }
    return cleaned;
}

TranscriptionResult WhisperEngine::transcribe(const std::vector<float>& audio_data) {
    if (!ctx) {
        throw std::runtime_error("Whisper engine not initialized");
    }

    std::lock_guard<std::mutex> lock(mutex);

    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress = false;
    wparams.print_special = false;
    wparams.print_realtime = false;
    wparams.language = "ru";
    
    int n_threads = std::min(4, (int)std::thread::hardware_concurrency());
    wparams.n_threads = n_threads;

    if (whisper_full(ctx, wparams, audio_data.data(), audio_data.size()) != 0) {
        throw std::runtime_error("Whisper processing failed");
    }

    TranscriptionResult result;
    
    // Полный текст
    std::ostringstream full_text;
    int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char* seg = whisper_full_get_segment_text(ctx, i);
        full_text << seg;
    }
    result.text = clean_text(full_text.str());
    
    // Сегменты и confidence
    for (int i = 0; i < n_segments; ++i) {
        const char* seg_text = whisper_full_get_segment_text(ctx, i);
        std::string cleaned_segment = clean_text(seg_text);
        if (!cleaned_segment.empty()) {
            result.segments.push_back(cleaned_segment);
            result.confidences.push_back(calculate_segment_confidence(i));
        }
    }
    
    result.overall_confidence = calculate_overall_confidence();
    
    return result;
}
