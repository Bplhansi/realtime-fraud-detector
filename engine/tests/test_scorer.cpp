#include <gtest/gtest.h>

#include <memory>

#include "fd/scorer.hpp"
#include "test_helpers.hpp"

using namespace fd;
using test::tx;

TEST(Scorer, WithoutModelApprovesAndRemembers) {
    Scorer s(nullptr, Thresholds{0.5, 0.9, "none"});
    const auto r = s.score(tx(1, TxType::Payment, 10.0, 1, 2));
    EXPECT_EQ(r.decision, Decision::Approve);
    EXPECT_EQ(s.store().accounts(), 2u);
    s.remember(tx(2, TxType::Payment, 10.0, 3, 4));
    EXPECT_EQ(s.store().accounts(), 4u);
    s.reset_history();
    EXPECT_EQ(s.store().accounts(), 0u);
}

TEST(Scorer, LoadsModelAndThresholdsFiles) {
    if (!model_support_available()) GTEST_SKIP() << "built without ONNX Runtime";
    Scorer s(test::parity_file("parity.onnx"), test::parity_file("parity.thresholds.ini"));
    EXPECT_EQ(s.thresholds().model_version, "parity");
    const auto txs = read_transactions(test::parity_file("parity.tx.bin"));
    std::size_t flagged = 0;
    for (const auto& t : txs) {
        const auto r = s.score(t);
        EXPECT_EQ(r.decision, s.thresholds().decide(r.score));
        flagged += r.decision != Decision::Approve;
    }
    EXPECT_GT(flagged, 0u);
    EXPECT_LT(flagged, txs.size());
}
