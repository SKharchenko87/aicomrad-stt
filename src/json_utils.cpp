#include "json_utils.h"
#include <string>
#include <vector>

std::string JsonUtils::escape_json(const std::string& s) {
    std::ostringstream o;
    for (auto c = s.cbegin(); c != s.cend(); c++) {
        switch (*c) {
        case '"': o << "\\\""; break;
        case '\\': o << "\\\\"; break;
        case '\b': o << "\\b"; break;
        case '\f': o << "\\f"; break;
        case '\n': o << "\\n"; break;
        case '\r': o << "\\r"; break;
        case '\t': o << "\\t"; break;
        default:
            if ('\x00' <= *c && *c <= '\x1f') {
                o << "\\u"
                  << std::hex << std::setw(4) << std::setfill('0') << (int)*c;
            } else {
                o << *c;
            }
        }
    }
    return o.str();
}

std::string JsonUtils::build_success_response(const std::string& text, 
                                             const std::vector<std::string>& segments,
                                             const std::vector<float>& confidences,
                                             float overall_confidence) {
    std::ostringstream json_response;
    json_response << "{";
    json_response << "\"text\": \"" << escape_json(text) << "\",";
    
    json_response << "\"segments\": [";
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i > 0) json_response << ",";
        json_response << "{";
        json_response << "\"id\": " << i << ",";
        json_response << "\"text\": \"" << escape_json(segments[i]) << "\",";
        json_response << "\"confidence\": " << confidences[i];
        json_response << "}";
    }
    json_response << "],";
    
    json_response << "\"confidence\": " << overall_confidence;
    json_response << "}";
    
    return json_response.str();
}

std::string JsonUtils::build_error_response(const std::string& error_message) {
    return "{\"error\": \"" + escape_json(error_message) + "\"}";
}

std::string JsonUtils::build_health_response() {
    return "{\"status\": \"ok\"}";
}
