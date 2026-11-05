#include "fd/features.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fd {

const std::array<std::string_view, kNumFeatures> kFeatureNames = {
    "amount_log",
    "amount_vs_sender_avg",
    "sender_tx_1h",
    "sender_tx_24h",
    "hours_since_sender_last",
    "sender_is_new",
    "pair_seen_before",
    "receiver_distinct_senders_24h",
    "receiver_is_new",
    "sender_inflow_24h_ratio",
    "hours_since_sender_first_seen",
    "receiver_is_merchant",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
};

const AccountState* FeatureStore::find(std::uint64_t account) const {
    const auto it = accounts_.find(account);
    return it == accounts_.end() ? nullptr : &it->second;
}

FeatureRow FeatureStore::compute(const Transaction& tx) const {
    const AccountState* s = find(tx.orig);
    const AccountState* r = find(tx.dest);
    const std::int32_t step = tx.step;
    FeatureRow f{};

    f[0] = std::log1p(tx.amount);
    f[1] = (s != nullptr && s->out_count > 0)
               ? tx.amount / (s->out_total / static_cast<double>(s->out_count))
               : 1.0;

    int n1 = 0;
    int n24 = 0;
    bool seen_pair = false;
    double inflow = 0.0;
    if (s != nullptr) {
        s->out.for_each([&](const Flow& e) {
            if (e.step > step - kWindow1h) ++n1;
            if (e.step > step - kWindow24h) ++n24;
            if (e.other == tx.dest) seen_pair = true;
        });
        s->in.for_each([&](const Flow& e) {
            if (e.step > step - kWindow24h) inflow += e.amount;
        });
    }
    f[2] = n1;
    f[3] = n24;
    f[4] = (s != nullptr && s->last_out_step >= 0) ? double(step - s->last_out_step) : -1.0;
    f[5] = (s == nullptr || s->out_count == 0) ? 1.0 : 0.0;
    f[6] = seen_pair ? 1.0 : 0.0;

    // Distinct senders: at most 64 entries, so a small linear search beats a hash set.
    std::array<std::uint64_t, kHistory> senders{};
    std::size_t n_senders = 0;
    if (r != nullptr) {
        r->in.for_each([&](const Flow& e) {
            if (e.step <= step - kWindow24h) return;
            for (std::size_t i = 0; i < n_senders; ++i) {
                if (senders[i] == e.other) return;
            }
            senders[n_senders++] = e.other;
        });
    }
    f[7] = double(n_senders);
    f[8] = r == nullptr ? 1.0 : 0.0;
    f[9] = inflow > 0.0 ? tx.amount / inflow : 0.0;
    f[10] = (s != nullptr && s->first_seen >= 0) ? double(step - s->first_seen) : -1.0;
    f[11] = is_merchant(tx.dest) ? 1.0 : 0.0;
    for (int t = 0; t < kNumTxTypes; ++t) f[12 + t] = (tx.type == t) ? 1.0 : 0.0;
    return f;
}

void FeatureStore::update(const Transaction& tx) {
    AccountState& s = accounts_[tx.orig];
    s.out.push({tx.dest, tx.step, tx.amount});
    s.out_total += tx.amount;
    ++s.out_count;
    s.last_out_step = tx.step;
    if (s.first_seen < 0) s.first_seen = tx.step;

    AccountState& r = accounts_[tx.dest];  // may rehash: don't use `s` after this line
    r.in.push({tx.orig, tx.step, tx.amount});
    if (r.first_seen < 0) r.first_seen = tx.step;
}

void check_feature_names(const std::filesystem::path& features_txt) {
    std::ifstream in(features_txt);
    if (!in) throw std::runtime_error("cannot open feature list: " + features_txt.string());
    std::vector<std::string> names;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        names.push_back(line);
    }
    if (names.size() != kNumFeatures) {
        throw std::runtime_error("feature list has " + std::to_string(names.size()) +
                                 " names; the engine computes " + std::to_string(kNumFeatures));
    }
    for (std::size_t i = 0; i < kNumFeatures; ++i) {
        if (names[i] != kFeatureNames[i]) {
            throw std::runtime_error("feature " + std::to_string(i) + " is '" + names[i] +
                                     "' in the list but '" + std::string(kFeatureNames[i]) +
                                     "' in the engine");
        }
    }
}

}  // namespace fd
