#include "fd/latency.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace fd {

double percentile_sorted(const std::vector<std::int64_t>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    const double clamped = std::clamp(p, 0.0, 100.0);
    auto rank = static_cast<std::size_t>(std::ceil(clamped / 100.0 * double(sorted.size())));
    rank = std::max<std::size_t>(rank, 1);
    return static_cast<double>(sorted[rank - 1]);
}

LatencySummary LatencyRecorder::summary() const {
    LatencySummary s;
    if (samples_.empty()) return s;
    std::vector<std::int64_t> sorted = samples_;
    std::sort(sorted.begin(), sorted.end());
    s.count = sorted.size();
    s.p50_ns = percentile_sorted(sorted, 50.0);
    s.p99_ns = percentile_sorted(sorted, 99.0);
    s.max_ns = static_cast<double>(sorted.back());
    s.mean_ns = std::accumulate(sorted.begin(), sorted.end(), 0.0) / double(sorted.size());
    return s;
}

}  // namespace fd
