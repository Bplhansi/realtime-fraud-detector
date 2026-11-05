// The C++ feature store must compute exactly what Python's ml/features.py computes.
#include <gtest/gtest.h>

#include <cmath>

#include "fd/features.hpp"
#include "test_helpers.hpp"

using namespace fd;

TEST(Parity, FeaturesMatchPythonOnThousandsOfTransactions) {
    const auto txs = read_transactions(test::parity_file("parity.tx.bin"));
    const auto expected = test::read_binary<double>(test::parity_file("parity.features.bin"));
    ASSERT_GT(txs.size(), 1000u);
    ASSERT_EQ(expected.size(), txs.size() * kNumFeatures);

    FeatureStore store;
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < txs.size(); ++i) {
        const FeatureRow row = store.process(txs[i]);
        for (std::size_t j = 0; j < kNumFeatures; ++j) {
            const double want = expected[i * kNumFeatures + j];
            const double tol = 1e-9 * std::max(1.0, std::abs(want));
            if (std::abs(row[j] - want) > tol && mismatches++ < 10) {
                ADD_FAILURE() << "transaction " << i << " feature " << kFeatureNames[j]
                              << ": C++ " << row[j] << " vs Python " << want;
            }
        }
    }
    EXPECT_EQ(mismatches, 0u);
}
