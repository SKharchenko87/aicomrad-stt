#ifndef WHISPER_ENGINE_H
#define WHISPER_ENGINE_H

#include <string>
#include <vector>
#include <mutex>
#include "whisper.h"

struct TranscriptionResult {
    std::string text;
    std::vector<std::string> segments;
    std::vector<float> confidences;
    float overall_confidence;
};

class WhisperEngine {
public:
    WhisperEngine();
    ~WhisperEngine();
    
    bool initialize(const std::string& model_path);
    TranscriptionResult transcribe(const std::vector<float>& audio_data);
    bool is_initialized() const { return ctx != nullptr; }
    
private:
    whisper_context* ctx;
    std::mutex mutex;
    
    float calculate_segment_confidence(int segment_index) const;
    float calculate_overall_confidence() const;
    std::string clean_text(const std::string& text) const;
};

#endif
