// The C++ engine must give the same fraud scores as scikit-learn in Python.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <stdexcept>

#include "fd/model.hpp"
#include "test_helpers.hpp"

using namespace fd;

TEST(Model, MissingFileThrows) {
    if (!model_support_available()) GTEST_SKIP() << "built without ONNX Runtime";
    EXPECT_THROW(Model("no-such-model.onnx"), std::runtime_error);
}

TEST(Model, GarbageFileThrows) {
    if (!model_support_available()) GTEST_SKIP() << "built without ONNX Runtime";
    const auto path = test::tmp_path("garbage.onnx");
    std::ofstream(path) << "not a model";
    EXPECT_THROW(Model{path}, std::runtime_error);
}

TEST(Model, ScoresMatchPython) {
    if (!model_support_available()) GTEST_SKIP() << "built without ONNX Runtime";
    Model model(test::parity_file("parity.onnx"));
    const auto rows = test::read_binary<double>(test::parity_file("parity.features.bin"));
    const auto want = test::read_binary<double>(test::parity_file("parity.scores.bin"));
    ASSERT_EQ(rows.size(), want.size() * kNumFeatures);
    double worst = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        FeatureRow row{};
        for (std::size_t j = 0; j < kNumFeatures; ++j) row[j] = rows[i * kNumFeatures + j];
        const double got = model.score(row);
        ASSERT_GE(got, 0.0);
        ASSERT_LE(got, 1.0);
        worst = std::max(worst, std::abs(got - want[i]));
    }
    EXPECT_LT(worst, 1e-5) << "largest C++ vs Python score difference";
}
