#pragma once

// A fixed-size buffer that keeps the newest N items. When full, a push overwrites the oldest.
// Memory per account stays constant however many transactions it makes.

#include <array>
#include <cstddef>
#include <stdexcept>

namespace fd {

template <typename T, std::size_t N>
class RingBuffer {
    static_assert(N > 0, "RingBuffer needs a capacity of at least 1");

public:
    void push(const T& item) noexcept {
        data_[(start_ + size_) % N] = item;
        if (size_ < N) {
            ++size_;
        } else {
            start_ = (start_ + 1) % N;
        }
    }

    // i = 0 is the oldest item kept, size() - 1 the newest.
    const T& operator[](std::size_t i) const noexcept { return data_[(start_ + i) % N]; }

    const T& at(std::size_t i) const {
        if (i >= size_) throw std::out_of_range("RingBuffer::at");
        return (*this)[i];
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    static constexpr std::size_t capacity() noexcept { return N; }

    // Calls fn(item) from oldest to newest (the same order Python's deque uses).
    template <typename Fn>
    void for_each(Fn&& fn) const {
        for (std::size_t i = 0; i < size_; ++i) fn((*this)[i]);
    }

private:
    std::array<T, N> data_{};
    std::size_t start_ = 0;
    std::size_t size_ = 0;
};

}  // namespace fd
