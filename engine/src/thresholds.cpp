#include "fd/thresholds.hpp"

#include <fstream>
#include <stdexcept>

namespace fd {

std::string_view decision_name(Decision d) noexcept {
    switch (d) {
        case Decision::Approve: return "approve";
        case Decision::Review: return "review";
        case Decision::Block: return "block";
    }
    return "unknown";
}

Decision Thresholds::decide(double score) const noexcept {
    if (score >= block) return Decision::Block;
    if (score >= review) return Decision::Review;
    return Decision::Approve;
}

namespace {
std::string trim(const std::string& s) {
    const auto a = s.find_first_not_of(" \t\r");
    if (a == std::string::npos) return "";
    const auto b = s.find_last_not_of(" \t\r");
    return s.substr(a, b - a + 1);
}

double to_number(const std::string& key, const std::string& value) {
    std::size_t used = 0;
    double v = 0.0;
    try {
        v = std::stod(value, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used == 0 || used != value.size()) {
        throw std::runtime_error("thresholds: '" + key + "' is not a number: " + value);
    }
    return v;
}
}  // namespace

Thresholds load_thresholds(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open thresholds file: " + path.string());
    Thresholds t;
    bool has_review = false;
    bool has_block = false;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) throw std::runtime_error("thresholds: bad line: " + line);
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (key == "review") {
            t.review = to_number(key, value);
            has_review = true;
        } else if (key == "block") {
            t.block = to_number(key, value);
            has_block = true;
        } else if (key == "model_version") {
            t.model_version = value;
        }
    }
    if (!has_review || !has_block) {
        throw std::runtime_error("thresholds file needs both review= and block=");
    }
    if (!(0.0 <= t.review && t.review <= t.block && t.block <= 1.0)) {
        throw std::runtime_error("thresholds must satisfy 0 <= review <= block <= 1");
    }
    return t;
}

}  // namespace fd
