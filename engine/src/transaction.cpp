#include "fd/transaction.hpp"

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fd {

namespace {
constexpr std::array<std::string_view, kNumTxTypes> kTypeNames = {
    "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"};
}

TxType parse_tx_type(std::string_view name) {
    for (std::size_t i = 0; i < kTypeNames.size(); ++i) {
        if (kTypeNames[i] == name) return static_cast<TxType>(i);
    }
    throw std::invalid_argument("unknown transaction type: " + std::string(name));
}

std::string_view tx_type_name(TxType type) noexcept {
    const auto i = static_cast<std::size_t>(type);
    return i < kTypeNames.size() ? kTypeNames[i] : std::string_view("UNKNOWN");
}

std::vector<Transaction> read_transactions(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot read transaction file: " + path.string());
    const auto bytes = static_cast<std::size_t>(in.tellg());
    if (bytes % sizeof(Transaction) != 0) {
        throw std::runtime_error("transaction file size is not a multiple of 40 bytes: " +
                                 path.string());
    }
    std::vector<Transaction> txs(bytes / sizeof(Transaction));
    in.seekg(0);
    if (!txs.empty() &&
        !in.read(reinterpret_cast<char*>(txs.data()), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("failed reading transaction file: " + path.string());
    }
    for (const auto& tx : txs) {
        if (tx.type >= kNumTxTypes) {
            throw std::runtime_error("bad transaction type " + std::to_string(tx.type) +
                                     " in " + path.string());
        }
    }
    return txs;
}

}  // namespace fd
