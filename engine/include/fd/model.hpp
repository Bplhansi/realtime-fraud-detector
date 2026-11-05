#pragma once

// Runs the ONNX model exported by ml/export_onnx.py: one feature row in, P(fraud) out.
// ONNX Runtime is hidden behind a pointer so other files don't need its headers, and the
// engine still builds (without model scoring) when configured with -DFD_WITH_ONNX=OFF.

#include <filesystem>
#include <memory>

#include "fd/features.hpp"

namespace fd {

// False if the engine was built without ONNX Runtime.
bool model_support_available() noexcept;

class Model {
public:
    // Throws std::runtime_error if the file is missing, isn't a valid model, or its input is
    // not [N, kNumFeatures] float32 with a "probabilities" [N, 2] output.
    explicit Model(const std::filesystem::path& onnx_path);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Probability of fraud, 0 to 1. Features are converted to float32, as in Python.
    double score(const FeatureRow& row);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace fd
