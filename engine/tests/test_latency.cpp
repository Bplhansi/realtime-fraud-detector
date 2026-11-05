#include <gtest/gtest.h>

#include "fd/latency.hpp"

using namespace fd;

TEST(Latency, EmptyIsZero) {
    LatencyRecorder r;
    const auto s = r.summary();
    EXPECT_EQ(s.count, 0u);
    EXPECT_EQ(s.p99_ns, 0.0);
}

TEST(Latency, NearestRankPercentiles) {
    LatencyRecorder r;
    for (int i = 100; i >= 1; --i) r.add(i);  // 1..100, added out of order
    const auto s = r.summary();
    EXPECT_EQ(s.count, 100u);
    EXPECT_EQ(s.p50_ns, 50.0);
    EXPECT_EQ(s.p99_ns, 99.0);
    EXPECT_EQ(s.max_ns, 100.0);
    EXPECT_DOUBLE_EQ(s.mean_ns, 50.5);
}

TEST(Latency, SingleSample) {
    EXPECT_EQ(percentile_sorted({42}, 0.0), 42.0);
    EXPECT_EQ(percentile_sorted({42}, 100.0), 42.0);
    EXPECT_EQ(percentile_sorted({}, 50.0), 0.0);
}
