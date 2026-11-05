// fd_replay: replays a transaction file through the engine in time order and reports speed.
//
//   fd_replay <transactions.tx.bin> [--model fraud_model.onnx --thresholds thresholds.ini]
//             [--from-step N] [--repeat N] [--scores out.csv] [--features out.bin]
//
// Every transaction updates the feature store. Only those after --from-step (e.g. the test
// period) are timed, counted and written to --scores. Without --model only features are timed.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "fd/latency.hpp"
#include "fd/scorer.hpp"
#include "fd/transaction.hpp"

#ifndef FD_BUILD_CONFIG
#define FD_BUILD_CONFIG "unknown"
#endif
#ifndef FD_BUILD_OPT
#define FD_BUILD_OPT ""
#endif

namespace {

struct Args {
    std::string tx_file;
    std::string model;
    std::string thresholds;
    std::string scores_out;
    std::string features_out;
    int from_step = 0;
    int repeat = 1;
};

constexpr const char* kUsage =
    "usage: fd_replay <transactions.tx.bin> [--model m.onnx --thresholds t.ini] "
    "[--from-step N] [--repeat N] [--scores out.csv] [--features out.bin]\n";

int parse_int(std::string_view name, const std::string& value, int min) {
    std::size_t used = 0;
    int v = 0;
    try {
        v = std::stoi(value, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used == 0 || used != value.size() || v < min) {
        throw std::invalid_argument("error: " + std::string(name) + " needs a whole number >= " +
                                    std::to_string(min) + ", got '" + value + "'");
    }
    return v;
}

Args parse(int argc, char** argv) {
    Args a;
    std::vector<std::string> rest(argv + 1, argv + argc);
    for (std::size_t i = 0; i < rest.size(); ++i) {
        const std::string& arg = rest[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= rest.size()) throw std::invalid_argument("error: " + arg + " needs a value");
            return rest[++i];
        };
        if (arg == "--model") a.model = value();
        else if (arg == "--thresholds") a.thresholds = value();
        else if (arg == "--scores") a.scores_out = value();
        else if (arg == "--features") a.features_out = value();
        else if (arg == "--from-step") a.from_step = parse_int(arg, value(), 0);
        else if (arg == "--repeat") a.repeat = parse_int(arg, value(), 1);
        else if (!arg.empty() && arg[0] == '-') throw std::invalid_argument("error: unknown option " + arg);
        else if (a.tx_file.empty()) a.tx_file = arg;
        else throw std::invalid_argument("error: more than one transaction file given");
    }
    if (a.tx_file.empty()) throw std::invalid_argument("error: no transaction file given");
    if (a.model.empty() != a.thresholds.empty()) {
        throw std::invalid_argument("error: --model and --thresholds go together");
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n" << kUsage;
        return 2;
    }

    try {
        const auto txs = fd::read_transactions(args.tx_file);
        std::unique_ptr<fd::Scorer> scorer;
        if (!args.model.empty()) {
            scorer = std::make_unique<fd::Scorer>(args.model, args.thresholds);
        } else {
            scorer = std::make_unique<fd::Scorer>(nullptr, fd::Thresholds{});
        }

        std::ofstream scores_file;
        if (!args.scores_out.empty()) {
            scores_file.open(args.scores_out);
            if (!scores_file) throw std::runtime_error("cannot write " + args.scores_out);
            scores_file << "tx_id,score,decision\n";
            scores_file.precision(9);
        }
        std::ofstream features_file;
        if (!args.features_out.empty()) {
            features_file.open(args.features_out, std::ios::binary);
            if (!features_file) throw std::runtime_error("cannot write " + args.features_out);
        }

        std::vector<double> throughputs;
        fd::LatencyRecorder latency;  // pooled over all runs
        latency.reserve(txs.size() * static_cast<std::size_t>(args.repeat));
        std::size_t counts[3] = {0, 0, 0};
        std::size_t timed = 0, fraud = 0, fraud_flagged = 0, flagged = 0;
        for (int run = 0; run < args.repeat; ++run) {
            scorer->reset_history();
            const bool first = run == 0;
            const auto run_start = std::chrono::steady_clock::now();
            for (const auto& tx : txs) {
                const auto t0 = std::chrono::steady_clock::now();
                const fd::ScoreResult r = scorer->score(tx);
                const auto t1 = std::chrono::steady_clock::now();
                if (first && features_file) {
                    features_file.write(reinterpret_cast<const char*>(r.features.data()),
                                        sizeof(double) * fd::kNumFeatures);
                }
                if (tx.step <= args.from_step) continue;
                latency.add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                if (!first) continue;
                ++timed;
                ++counts[static_cast<int>(r.decision)];
                const bool is_flagged = r.decision != fd::Decision::Approve;
                flagged += is_flagged;
                fraud += tx.is_fraud;
                fraud_flagged += (tx.is_fraud && is_flagged);
                if (scores_file) {
                    scores_file << tx.tx_id << ',' << r.score << ','
                                << fd::decision_name(r.decision) << '\n';
                }
            }
            const double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - run_start).count();
            throughputs.push_back(double(txs.size()) / secs);
        }

        const auto s = latency.summary();
        double best = 0.0, sum = 0.0;
        for (double t : throughputs) { best = std::max(best, t); sum += t; }
        std::printf("build: %s %s\n", FD_BUILD_CONFIG, FD_BUILD_OPT);
        std::printf("model: %s\n", args.model.empty() ? "none (features only)" : args.model.c_str());
        std::printf("transactions: %zu replayed, %zu timed (step > %d), %d run(s)\n",
                    txs.size(), timed, args.from_step, args.repeat);
        std::printf("throughput: %.0f tx/s (mean of runs), best %.0f tx/s\n",
                    sum / double(throughputs.size()), best);
        std::printf("latency p50: %.0f ns, latency p99: %.0f ns, max: %.0f ns, mean: %.0f ns\n",
                    s.p50_ns, s.p99_ns, s.max_ns, s.mean_ns);
        if (!args.model.empty()) {
            std::printf("decisions: approve %zu, review %zu, block %zu (model %s)\n", counts[0],
                        counts[1], counts[2], scorer->thresholds().model_version.c_str());
            if (fraud > 0) {
                std::printf("fraud flagged: %zu of %zu (recall %.3f), precision %.3f\n",
                            fraud_flagged, fraud, double(fraud_flagged) / double(fraud),
                            flagged ? double(fraud_flagged) / double(flagged) : 0.0);
            }
        }
        if (!args.scores_out.empty()) std::printf("scores written to: %s\n", args.scores_out.c_str());
        if (!args.features_out.empty()) std::printf("features written to: %s\n", args.features_out.c_str());
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
