#include "fd/model.hpp"

#include <stdexcept>
#include <string>

#ifdef FD_WITH_ONNX
#include <onnxruntime_cxx_api.h>

#include <array>
#endif

namespace fd {

#ifdef FD_WITH_ONNX

bool model_support_available() noexcept { return true; }

struct Model::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "fraud-engine"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<float, kNumFeatures> input{};
    std::array<std::int64_t, 2> shape{1, static_cast<std::int64_t>(kNumFeatures)};
    const char* input_name = "features";
    const char* output_name = "probabilities";

    explicit Impl(const std::filesystem::path& path) {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("model file not found: " + path.string());
        }
        options.SetIntraOpNumThreads(1);  // one transaction at a time: threads only add overhead
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        try {
            session = Ort::Session(env, path.c_str(), options);
        } catch (const Ort::Exception& e) {
            throw std::runtime_error("cannot load model " + path.string() + ": " + e.what());
        }
        check_shapes();
    }

    void check_shapes() {
        if (session.GetInputCount() != 1) throw std::runtime_error("model must have one input");
        Ort::AllocatorWithDefaultOptions alloc;
        if (std::string(session.GetInputNameAllocated(0, alloc).get()) != input_name) {
            throw std::runtime_error("model input must be called 'features'");
        }
        const auto in_shape =
            session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (in_shape.size() != 2 || in_shape[1] != static_cast<std::int64_t>(kNumFeatures)) {
            throw std::runtime_error("model expects a different number of features than the "
                                     "engine computes (" + std::to_string(kNumFeatures) + ")");
        }
        bool found = false;
        for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
            if (std::string(session.GetOutputNameAllocated(i, alloc).get()) == output_name) {
                found = true;
            }
        }
        if (!found) throw std::runtime_error("model has no 'probabilities' output");
    }

    double run(const FeatureRow& row) {
        for (std::size_t i = 0; i < kNumFeatures; ++i) input[i] = static_cast<float>(row[i]);
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory, input.data(), input.size(), shape.data(), shape.size());
        auto out = session.Run(Ort::RunOptions{nullptr}, &input_name, &tensor, 1,
                               &output_name, 1);
        const float* probs = out[0].GetTensorData<float>();
        return static_cast<double>(probs[1]);
    }
};

Model::Model(const std::filesystem::path& onnx_path) : impl_(std::make_unique<Impl>(onnx_path)) {}
Model::~Model() = default;
double Model::score(const FeatureRow& row) { return impl_->run(row); }

#else  // built without ONNX Runtime

bool model_support_available() noexcept { return false; }
struct Model::Impl {};
Model::Model(const std::filesystem::path&) {
    throw std::runtime_error("engine was built without ONNX Runtime (FD_WITH_ONNX=OFF)");
}
Model::~Model() = default;
double Model::score(const FeatureRow&) { return 0.0; }

#endif

}  // namespace fd
