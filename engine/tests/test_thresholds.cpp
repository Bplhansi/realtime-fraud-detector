#include <gtest/gtest.h>

#include <fstream>
#include <stdexcept>

#include "fd/thresholds.hpp"
#include "test_helpers.hpp"

using namespace fd;

TEST(Thresholds, Decides) {
    Thresholds t{0.3, 0.8, "v1"};
    EXPECT_EQ(t.decide(0.0), Decision::Approve);
    EXPECT_EQ(t.decide(0.29), Decision::Approve);
    EXPECT_EQ(t.decide(0.3), Decision::Review);
    EXPECT_EQ(t.decide(0.79), Decision::Review);
    EXPECT_EQ(t.decide(0.8), Decision::Block);
    EXPECT_EQ(t.decide(1.0), Decision::Block);
    EXPECT_EQ(decision_name(Decision::Review), "review");
}

TEST(Thresholds, LoadsFile) {
    const auto path = test::tmp_path("ok.ini");
    std::ofstream(path) << "# comment\nmodel_version = v7\r\nreview=0.25\nblock=0.9\n";
    const auto t = load_thresholds(path);
    EXPECT_DOUBLE_EQ(t.review, 0.25);
    EXPECT_DOUBLE_EQ(t.block, 0.9);
    EXPECT_EQ(t.model_version, "v7");
}

TEST(Thresholds, RejectsBadFiles) {
    EXPECT_THROW(load_thresholds("missing.ini"), std::runtime_error);
    const auto path = test::tmp_path("bad.ini");
    std::ofstream(path) << "review=abc\nblock=0.9\n";
    EXPECT_THROW(load_thresholds(path), std::runtime_error);
    std::ofstream(path) << "review=0.95\nblock=0.9\n";  // review above block
    EXPECT_THROW(load_thresholds(path), std::runtime_error);
    std::ofstream(path) << "review=0.5\n";  // no block
    EXPECT_THROW(load_thresholds(path), std::runtime_error);
    std::ofstream(path) << "review=0.5 extra\nblock=0.9\n";
    EXPECT_THROW(load_thresholds(path), std::runtime_error);
}
