#pragma once

// Turns a fraud score into a decision. Thresholds come from models/thresholds.ini, written by
// ml/train.py after choosing them on the validation set.

#include <filesystem>
#include <string>
#include <string_view>

namespace fd {

enum class Decision { Approve, Review, Block };

std::string_view decision_name(Decision d) noexcept;

struct Thresholds {
    double review = 0.5;
    double block = 0.9;
    std::string model_version = "unknown";

    // score >= block -> Block; score >= review -> Review; otherwise Approve.
    Decision decide(double score) const noexcept;
};

// Reads key=value lines (# comments allowed). Throws std::runtime_error if the file is
// missing, a value isn't a number, or the thresholds are not 0 <= review <= block <= 1.
Thresholds load_thresholds(const std::filesystem::path& path);

}  // namespace fd
