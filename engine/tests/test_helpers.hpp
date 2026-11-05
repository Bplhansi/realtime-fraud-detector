#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fd/transaction.hpp"

namespace fd::test {

inline Transaction tx(std::int32_t step, TxType type, double amount, std::uint64_t orig,
                      std::uint64_t dest) {
    Transaction t;
    t.step = step;
    t.type = static_cast<std::uint8_t>(type);
    t.amount = amount;
    t.orig = orig;
    t.dest = dest;
    return t;
}

inline std::filesystem::path tmp_path(const std::string& name) {
    std::filesystem::path dir(FD_TEST_TMP);
    std::filesystem::create_directories(dir);
    return dir / name;
}

template <typename T>
std::vector<T> read_binary(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    const auto bytes = static_cast<std::size_t>(in.tellg());
    std::vector<T> out(bytes / sizeof(T));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    return out;
}

inline std::filesystem::path parity_file(const std::string& name) {
    return std::filesystem::path(FD_PARITY_DIR) / name;
}

}  // namespace fd::test
