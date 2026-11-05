#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <stdexcept>

#include "fd/features.hpp"
#include "test_helpers.hpp"

using namespace fd;
using test::tx;

namespace {
std::size_t idx(std::string_view name) {
    for (std::size_t i = 0; i < kNumFeatures; ++i) {
        if (kFeatureNames[i] == name) return i;
    }
    throw std::invalid_argument("no feature " + std::string(name));
}
}  // namespace

TEST(Features, NamesMatchTheSharedList) { EXPECT_NO_THROW(check_feature_names(FD_FEATURES_TXT)); }

TEST(Features, WrongListIsRejected) {
    const auto path = test::tmp_path("bad_features.txt");
    std::ofstream(path) << "# comment\namount_log\n";
    EXPECT_THROW(check_feature_names(path), std::runtime_error);
    EXPECT_THROW(check_feature_names("missing.txt"), std::runtime_error);
}

TEST(Features, BrandNewAccounts) {
    FeatureStore store;
    const auto f = store.compute(tx(10, TxType::Transfer, 100.0, 1, 2));
    EXPECT_DOUBLE_EQ(f[idx("amount_log")], std::log1p(100.0));
    EXPECT_EQ(f[idx("amount_vs_sender_avg")], 1.0);
    EXPECT_EQ(f[idx("sender_is_new")], 1.0);
    EXPECT_EQ(f[idx("receiver_is_new")], 1.0);
    EXPECT_EQ(f[idx("hours_since_sender_last")], -1.0);
    EXPECT_EQ(f[idx("hours_since_sender_first_seen")], -1.0);
    EXPECT_EQ(f[idx("type_TRANSFER")], 1.0);
    EXPECT_EQ(f[idx("type_PAYMENT")], 0.0);
    EXPECT_EQ(f[idx("receiver_is_merchant")], 0.0);
    EXPECT_EQ(store.accounts(), 0u) << "compute() must not change the store";
}

TEST(Features, UsesOnlyEarlierHistory) {
    FeatureStore store;
    store.process(tx(1, TxType::Payment, 100.0, 1, 50 | kMerchantBit));
    store.process(tx(5, TxType::Payment, 300.0, 1, 50 | kMerchantBit));
    const auto f = store.process(tx(5, TxType::Transfer, 1000.0, 1, 2));
    EXPECT_DOUBLE_EQ(f[idx("amount_vs_sender_avg")], 1000.0 / 200.0);
    EXPECT_EQ(f[idx("sender_tx_1h")], 1.0);   // the step-5 payment
    EXPECT_EQ(f[idx("sender_tx_24h")], 2.0);
    EXPECT_EQ(f[idx("hours_since_sender_last")], 0.0);
    EXPECT_EQ(f[idx("hours_since_sender_first_seen")], 4.0);
    EXPECT_EQ(f[idx("pair_seen_before")], 0.0);
    EXPECT_EQ(f[idx("sender_is_new")], 0.0);
    // Now the pair is known.
    const auto g = store.compute(tx(6, TxType::Transfer, 10.0, 1, 2));
    EXPECT_EQ(g[idx("pair_seen_before")], 1.0);
    EXPECT_EQ(g[idx("receiver_is_new")], 0.0);
}

TEST(Features, OldActivityLeavesTheWindows) {
    FeatureStore store;
    store.process(tx(1, TxType::Payment, 10.0, 1, 2));
    const auto f = store.compute(tx(25, TxType::Payment, 10.0, 1, 3));
    EXPECT_EQ(f[idx("sender_tx_24h")], 0.0);  // step 1 is not > 25 - 24
    const auto g = store.compute(tx(24, TxType::Payment, 10.0, 1, 3));
    EXPECT_EQ(g[idx("sender_tx_24h")], 1.0);
    EXPECT_EQ(g[idx("sender_tx_1h")], 0.0);
}

TEST(Features, MuleCashOut) {
    FeatureStore store;
    const std::uint64_t mule = 900;
    store.process(tx(10, TxType::Transfer, 400.0, 1, mule));
    store.process(tx(11, TxType::Transfer, 600.0, 2, mule));
    store.process(tx(11, TxType::Transfer, 50.0, 1, mule));
    const auto f = store.compute(tx(12, TxType::CashOut, 1050.0, mule, 77 | kMerchantBit));
    EXPECT_DOUBLE_EQ(f[idx("sender_inflow_24h_ratio")], 1050.0 / 1050.0);
    EXPECT_EQ(f[idx("hours_since_sender_first_seen")], 2.0);
    EXPECT_EQ(f[idx("sender_is_new")], 1.0);  // the mule has never paid anyone
    EXPECT_EQ(f[idx("receiver_is_merchant")], 1.0);
    // The receiver side: the mule had two distinct senders.
    const auto g = store.compute(tx(12, TxType::Transfer, 5.0, 3, mule));
    EXPECT_EQ(g[idx("receiver_distinct_senders_24h")], 2.0);
}

TEST(Features, HistoryIsBoundedTo64) {
    FeatureStore store;
    for (int i = 0; i < 100; ++i) store.process(tx(1, TxType::Payment, 1.0, 1, 1000 + i));
    const auto f = store.compute(tx(1, TxType::Payment, 1.0, 1, 1000));
    EXPECT_EQ(f[idx("sender_tx_1h")], 64.0) << "only the last 64 payments are remembered";
    EXPECT_EQ(f[idx("pair_seen_before")], 0.0) << "receiver 1000 fell out of the ring buffer";
    EXPECT_DOUBLE_EQ(f[idx("amount_vs_sender_avg")], 1.0) << "the running average uses all";
}
