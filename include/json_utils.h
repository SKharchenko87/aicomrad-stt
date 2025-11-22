#ifndef JSON_UTILS_H
#define JSON_UTILS_H

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

namespace JsonUtils {
    std::string escape_json(const std::string& s);
    std::string build_success_response(const std::string& text, 
                                      const std::vector<std::string>& segments,
                                      const std::vector<float>& confidences,
                                      float overall_confidence);
    std::string build_error_response(const std::string& error_message);
    std::string build_health_response();
}

#endif
