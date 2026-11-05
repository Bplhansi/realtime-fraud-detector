#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "fd/ring_buffer.hpp"

using fd::RingBuffer;

TEST(RingBuffer, StartsEmpty) {
    RingBuffer<int, 3> rb;
    EXPECT_TRUE(rb.empty());
    EXPECT_EQ(rb.size(), 0u);
    EXPECT_EQ(rb.capacity(), 3u);
    EXPECT_THROW((void)rb.at(0), std::out_of_range);
}

TEST(RingBuffer, KeepsItemsInOrderUntilFull) {
    RingBuffer<int, 3> rb;
    rb.push(1);
    rb.push(2);
    ASSERT_EQ(rb.size(), 2u);
    EXPECT_EQ(rb[0], 1);
    EXPECT_EQ(rb[1], 2);
}

TEST(RingBuffer, OverwritesOldestWhenFull) {
    RingBuffer<int, 3> rb;
    for (int i = 1; i <= 5; ++i) rb.push(i);
    ASSERT_EQ(rb.size(), 3u);
    EXPECT_EQ(rb.at(0), 3);
    EXPECT_EQ(rb.at(1), 4);
    EXPECT_EQ(rb.at(2), 5);
    EXPECT_THROW((void)rb.at(3), std::out_of_range);
}

TEST(RingBuffer, ForEachGoesOldestToNewest) {
    RingBuffer<int, 4> rb;
    for (int i = 0; i < 10; ++i) rb.push(i);
    std::vector<int> seen;
    rb.for_each([&](int v) { seen.push_back(v); });
    EXPECT_EQ(seen, (std::vector<int>{6, 7, 8, 9}));
}

TEST(RingBuffer, CapacityOneKeepsNewest) {
    RingBuffer<int, 1> rb;
    rb.push(7);
    rb.push(8);
    ASSERT_EQ(rb.size(), 1u);
    EXPECT_EQ(rb[0], 8);
}
