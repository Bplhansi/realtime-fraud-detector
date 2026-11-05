#pragma once

// One transaction as stored in the binary files written by ml/data.py (40 bytes,
// little-endian). Account ids: customers are their PaySim digits; merchants also have
// kMerchantBit set.

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace fd {

enum class TxType : std::uint8_t { CashIn = 0, CashOut = 1, Debit = 2, Payment = 3, Transfer = 4 };
inline constexpr int kNumTxTypes = 5;
inline constexpr std::uint64_t kMerchantBit = std::uint64_t{1} << 62;

struct Transaction {
    std::uint64_t orig = 0;
    std::uint64_t dest = 0;
    double amount = 0.0;
    std::uint64_t tx_id = 0;
    std::int32_t step = 0;  // PaySim time step: 1 step = 1 hour
    std::uint8_t type = 0;  // TxType
    std::uint8_t is_fraud = 0;
    std::uint16_t pad = 0;
};
static_assert(sizeof(Transaction) == 40, "Transaction must match ml/data.py TX_DTYPE");

inline bool is_merchant(std::uint64_t account) noexcept { return (account & kMerchantBit) != 0; }

// "TRANSFER" -> TxType::Transfer. Throws std::invalid_argument for unknown names.
TxType parse_tx_type(std::string_view name);
std::string_view tx_type_name(TxType type) noexcept;

// Reads a whole .tx.bin file. Throws std::runtime_error if it can't be read or its size is
// not a multiple of 40 bytes, or a record has an unknown type.
std::vector<Transaction> read_transactions(const std::filesystem::path& path);

}  // namespace fd
