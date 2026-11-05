// Python bindings (pybind11): lets the FastAPI service call the C++ engine directly.
//
//   import fraud_engine
//   eng = fraud_engine.Engine("models/fraud_model.onnx", "models/thresholds.ini")
//   eng.remember_file("data/transactions.tx.bin")       # optional warm-up with past history
//   eng.score(step=700, type="TRANSFER", amount=950.0, orig=123, dest=456)
//   -> {"score": 0.93, "decision": "review", "features": [...], "latency_ns": 41000}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include "fd/scorer.hpp"
#include "fd/transaction.hpp"

namespace py = pybind11;

namespace {

class PyEngine {
public:
    PyEngine(const std::string& model, const std::string& thresholds) : scorer_(model, thresholds) {}

    py::dict score(int step, const std::string& type, double amount, std::uint64_t orig,
                   std::uint64_t dest) {
        if (!std::isfinite(amount) || amount < 0.0) {
            throw std::invalid_argument("amount must be a finite number >= 0");
        }
        fd::Transaction tx;
        tx.step = step;
        tx.type = static_cast<std::uint8_t>(fd::parse_tx_type(type));
        tx.amount = amount;
        tx.orig = orig;
        tx.dest = dest;
        const auto t0 = std::chrono::steady_clock::now();
        const fd::ScoreResult r = scorer_.score(tx);
        const auto t1 = std::chrono::steady_clock::now();
        py::dict out;
        out["score"] = r.score;
        out["decision"] = std::string(fd::decision_name(r.decision));
        out["features"] = std::vector<double>(r.features.begin(), r.features.end());
        out["latency_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        return out;
    }

    std::size_t remember_file(const std::string& path) {
        const auto txs = fd::read_transactions(path);
        for (const auto& tx : txs) scorer_.remember(tx);
        return txs.size();
    }

    std::string model_version() const { return scorer_.thresholds().model_version; }
    double review() const { return scorer_.thresholds().review; }
    double block() const { return scorer_.thresholds().block; }
    std::size_t accounts() const { return scorer_.store().accounts(); }

private:
    fd::Scorer scorer_;
};

}  // namespace

PYBIND11_MODULE(fraud_engine, m) {
    m.doc() = "C++ fraud scoring engine";
    m.def("feature_names", [] {
        return std::vector<std::string>(fd::kFeatureNames.begin(), fd::kFeatureNames.end());
    });
    py::class_<PyEngine>(m, "Engine")
        .def(py::init<const std::string&, const std::string&>(), py::arg("model"),
             py::arg("thresholds"))
        .def("score", &PyEngine::score, py::arg("step"), py::arg("type"), py::arg("amount"),
             py::arg("orig"), py::arg("dest"))
        .def("remember_file", &PyEngine::remember_file, py::arg("path"))
        .def_property_readonly("model_version", &PyEngine::model_version)
        .def_property_readonly("review", &PyEngine::review)
        .def_property_readonly("block", &PyEngine::block)
        .def_property_readonly("accounts", &PyEngine::accounts);
}
