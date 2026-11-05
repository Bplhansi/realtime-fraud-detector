#include <gtest/gtest.h>

#include <fstream>
#include <stdexcept>

#include "fd/transaction.hpp"
#include "test_helpers.hpp"

using namespace fd;

TEST(Transaction, ParsesAndNamesTypes) {
    EXPECT_EQ(parse_tx_type("CASH_IN"), TxType::CashIn);
    EXPECT_EQ(parse_tx_type("TRANSFER"), TxType::Transfer);
    EXPECT_EQ(tx_type_name(TxType::CashOut), "CASH_OUT");
    EXPECT_THROW(parse_tx_type("transfer"), std::invalid_argument);
    EXPECT_THROW(parse_tx_type(""), std::invalid_argument);
}

TEST(Transaction, MerchantBit) {
    EXPECT_FALSE(is_merchant(1231006815));
    EXPECT_TRUE(is_merchant(1979787155 | kMerchantBit));
}

TEST(Transaction, ReadsWhatItWrites) {
    const auto path = test::tmp_path("roundtrip.tx.bin");
    std::vector<Transaction> txs = {test::tx(1, TxType::Payment, 9.5, 11, 22 | kMerchantBit),
                                    test::tx(2, TxType::Transfer, 1000.0, 11, 33)};
    txs[1].is_fraud = 1;
    txs[1].tx_id = 7;
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(txs.data()), sizeof(Transaction) * txs.size());
    }
    const auto back = read_transactions(path);
    ASSERT_EQ(back.size(), 2u);
    EXPECT_EQ(back[1].amount, 1000.0);
    EXPECT_EQ(back[1].tx_id, 7u);
    EXPECT_EQ(back[1].is_fraud, 1);
    EXPECT_EQ(back[0].dest, 22 | kMerchantBit);
}

TEST(Transaction, RejectsMissingFile) {
    EXPECT_THROW(read_transactions("no-such-file.tx.bin"), std::runtime_error);
}

TEST(Transaction, RejectsTruncatedFile) {
    const auto path = test::tmp_path("truncated.tx.bin");
    std::ofstream(path, std::ios::binary) << "only 21 bytes, broken";
    EXPECT_THROW(read_transactions(path), std::runtime_error);
}

TEST(Transaction, RejectsUnknownType) {
    const auto path = test::tmp_path("badtype.tx.bin");
    Transaction t = test::tx(1, TxType::Payment, 1.0, 1, 2);
    t.type = 9;
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(&t), sizeof t);
    EXPECT_THROW(read_transactions(path), std::runtime_error);
}

TEST(Transaction, ReadsPythonFile) {
    const auto txs = read_transactions(test::parity_file("parity.tx.bin"));
    ASSERT_GT(txs.size(), 1000u);
    for (std::size_t i = 1; i < txs.size(); ++i) {
        ASSERT_LE(txs[i - 1].step, txs[i].step) << "Python writes transactions in time order";
    }
}
