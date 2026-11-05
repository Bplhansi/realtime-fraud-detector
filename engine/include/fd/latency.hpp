#pragma once

// Collects per-transaction scoring times and reports percentiles.

#include <cstdint>
#include <vector>

namespace fd {

struct LatencySummary {
    std::size_t count = 0;
    double p50_ns = 0.0;
    double p99_ns = 0.0;
    double max_ns = 0.0;
    double mean_ns = 0.0;
};

class LatencyRecorder {
public:
    void reserve(std::size_t n) { samples_.reserve(n); }
    void add(std::int64_t ns) { samples_.push_back(ns); }
    std::size_t size() const noexcept { return samples_.size(); }
    // Nearest-rank percentiles. An empty recorder gives all zeros.
    LatencySummary summary() const;

private:
    std::vector<std::int64_t> samples_;
};

// Nearest-rank percentile (p in 0..100) of a SORTED vector. Empty -> 0.
double percentile_sorted(const std::vector<std::int64_t>& sorted, double p);

}  // namespace fd
