#include "audio_processor.h"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstdint>

void AudioProcessor::validate_wav_header(const char header[44]) {
    uint32_t sample_rate = *reinterpret_cast<const uint32_t*>(&header[24]);
    uint16_t n_channels = *reinterpret_cast<const uint16_t*>(&header[22]);
    uint16_t bits_per_sample = *reinterpret_cast<const uint16_t*>(&header[34]);

    if (sample_rate != 16000) {
        throw std::runtime_error("Invalid audio format: Sample rate must be 16000 Hz, found " + std::to_string(sample_rate) + " Hz.");
    }
    if (n_channels != 1) {
        throw std::runtime_error("Invalid audio format: Must be Mono (1 channel), found " + std::to_string(n_channels) + " channels.");
    }
    if (bits_per_sample != 16) {
        throw std::runtime_error("Invalid audio format: Must be 16-bit PCM, found " + std::to_string(bits_per_sample) + " bits.");
    }
}

std::vector<float> AudioProcessor::load_wav(const std::string &path, int &n_samples) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open WAV file: " + path);

    char header[44];
    if (!f.read(header, 44)) {
        throw std::runtime_error("Failed to read full 44-byte WAV header.");
    }
    
    validate_wav_header(header);

    std::vector<float> audio;
    int16_t sample;

    while (f.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        audio.push_back(sample / 32768.0f);
    }

    n_samples = audio.size();
    if (n_samples == 0) throw std::runtime_error("WAV file is empty or corrupted (0 samples read).");

    return audio;
}

bool AudioProcessor::validate_audio_format(const std::vector<float>& audio, int sample_rate) {
    return !audio.empty();
}
