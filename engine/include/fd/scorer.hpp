#pragma once

// The whole live path for one transaction: features from history -> model score -> decision,
// then the transaction is added to history.

#include <filesystem>
#include <memory>
#include <optional>

#include "fd/features.hpp"
#include "fd/model.hpp"
#include "fd/thresholds.hpp"
#include "fd/transaction.hpp"

namespace fd {

struct ScoreResult {
    double score = 0.0;
    Decision decision = Decision::Approve;
    FeatureRow features{};
};

class Scorer {
public:
    // Loads the ONNX model and thresholds. Throws std::runtime_error on any problem.
    Scorer(const std::filesystem::path& model_path, const std::filesystem::path& thresholds_path);
    // For tests: an already-loaded model (or none, which scores everything 0) and thresholds.
    Scorer(std::unique_ptr<Model> model, Thresholds thresholds);

    ScoreResult score(const Transaction& tx);
    // Adds history without scoring (to warm up the store with past transactions).
    void remember(const Transaction& tx) { store_.update(tx); }
    // Forgets all history (used between benchmark repeats).
    void reset_history() { store_ = FeatureStore{}; }

    const Thresholds& thresholds() const noexcept { return thresholds_; }
    const FeatureStore& store() const noexcept { return store_; }

private:
    std::unique_ptr<Model> model_;
    Thresholds thresholds_;
    FeatureStore store_;
};

}  // namespace fd
