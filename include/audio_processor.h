#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H

#include <vector>
#include <string>

class AudioProcessor {
public:
    static std::vector<float> load_wav(const std::string &path, int &n_samples);
    static bool validate_audio_format(const std::vector<float>& audio, int sample_rate = 16000);
    
private:
    static void validate_wav_header(const char header[44]);
};

#endif
