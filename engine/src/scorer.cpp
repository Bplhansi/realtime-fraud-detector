#include "fd/scorer.hpp"

#include <utility>

namespace fd {

Scorer::Scorer(const std::filesystem::path& model_path,
               const std::filesystem::path& thresholds_path)
    : model_(std::make_unique<Model>(model_path)), thresholds_(load_thresholds(thresholds_path)) {}

Scorer::Scorer(std::unique_ptr<Model> model, Thresholds thresholds)
    : model_(std::move(model)), thresholds_(std::move(thresholds)) {}

ScoreResult Scorer::score(const Transaction& tx) {
    ScoreResult r;
    r.features = store_.process(tx);
    r.score = model_ ? model_->score(r.features) : 0.0;
    r.decision = thresholds_.decide(r.score);
    return r;
}

}  // namespace fd
