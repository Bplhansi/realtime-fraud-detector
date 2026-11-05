#pragma once

// The feature store: remembers recent activity per account and turns each new transaction
// into the model's input row. Must give the same numbers as Python's ml/features.py; the
// feature list and order are in config/features.txt (checked by the tests).

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <unordered_map>

#include "fd/ring_buffer.hpp"
#include "fd/transaction.hpp"

namespace fd {

inline constexpr std::size_t kNumFeatures = 17;
inline constexpr std::size_t kHistory = 64;  // remembered transactions per account and direction
inline constexpr std::int32_t kWindow1h = 1;  // in steps (hours)
inline constexpr std::int32_t kWindow24h = 24;

extern const std::array<std::string_view, kNumFeatures> kFeatureNames;

using FeatureRow = std::array<double, kNumFeatures>;

// One remembered payment: who was on the other side, when, and how much.
struct Flow {
    std::uint64_t other = 0;
    std::int32_t step = 0;
    double amount = 0.0;
};

struct AccountState {
    RingBuffer<Flow, kHistory> out;  // payments this account sent
    RingBuffer<Flow, kHistory> in;   // payments this account received
    double out_total = 0.0;
    std::uint32_t out_count = 0;
    std::int32_t first_seen = -1;
    std::int32_t last_out_step = -1;
};

class FeatureStore {
public:
    // Features from history only; does not change the store.
    FeatureRow compute(const Transaction& tx) const;
    // Adds the transaction to both accounts' history.
    void update(const Transaction& tx);
    // compute() then update(): the normal path for a live transaction.
    FeatureRow process(const Transaction& tx) {
        FeatureRow row = compute(tx);
        update(tx);
        return row;
    }

    std::size_t accounts() const noexcept { return accounts_.size(); }
    const AccountState* find(std::uint64_t account) const;

private:
    std::unordered_map<std::uint64_t, AccountState> accounts_;
};

// Throws std::runtime_error unless the names in the file (comments and blank lines skipped)
// are exactly kFeatureNames, in order.
void check_feature_names(const std::filesystem::path& features_txt);

}  // namespace fd
