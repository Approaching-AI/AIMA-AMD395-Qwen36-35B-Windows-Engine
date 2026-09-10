#include "native/providers/moe_accumulator/sm121_group16_modulo.h"
#include "native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"

#include <array>
#include <cstdio>
#include <cstdint>

namespace {
using namespace qrt_sm121_group16;

bool check(const std::array<int64_t, 16>& products, int64_t accumulator) {
    int64_t reference = accumulator;
    uint32_t modulo_sum = static_cast<uint32_t>(accumulator);
    for (int64_t product : products) {
        reference += product;
        modulo_sum += static_cast<uint32_t>(product);
    }
    const auto result = decode_modulo_sum(modulo_sum, products[0] < 0);
    const uint64_t magnitude = static_cast<uint64_t>(
        reference < 0 ? -reference : reference
    );
    if (result.magnitude != magnitude || result.negative != (reference < 0)) {
        std::fprintf(stderr, "sum=%lld modulo=%u magnitude=%u negative=%d\n",
                     static_cast<long long>(reference), modulo_sum,
                     result.magnitude, result.negative);
        return false;
    }
    return true;
}

bool check_final_group_endpoint(qrt_q1_moe_hawkeye::Value value) {
    const auto result = qrt_q1_moe_hawkeye::group_sum<26, -133>(&value, 1u);
    const auto optimized = qrt_sm121_group16::finish_accumulator(value);
    if (result.significand != optimized.significand ||
        result.exponent != optimized.exponent || result.negative != optimized.negative) {
        std::fprintf(stderr, "one-value group endpoint changed\n");
        return false;
    }
    return true;
}

uint32_t next(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
}  // namespace

int main() {
    using namespace qrt_sm121_group16;
    std::array<int64_t, 16> products{};
    if (!check(products, 0)) return 1;
    if (!check_final_group_endpoint({0u, -133, false})) return 6;
    if (!check_final_group_endpoint({0u, -133, true})) return 6;
    for (uint32_t significand : {1u, 2u, 0x007fffffu, 0x00800000u, 0x00ffffffu}) {
        for (bool negative : {false, true}) {
            if (!check_final_group_endpoint({significand, -126, negative})) return 6;
        }
    }
    // Exercise both signed-overflow directions and every mixed-sign position.
    for (int sign : {-1, 1}) {
        products.fill(sign * static_cast<int64_t>(kMaxAlignedProduct));
        for (int64_t accumulator : {
                 -static_cast<int64_t>(kMaxAlignedAccumulator), int64_t{0},
                 static_cast<int64_t>(kMaxAlignedAccumulator)}) {
            if (!check(products, accumulator)) return 2;
            for (size_t lane = 0; lane < products.size(); ++lane) {
                const int64_t saved = products[lane];
                products[lane] = -saved;
                if (!check(products, accumulator)) return 3;
                products[lane] = 0;
                if (!check(products, accumulator)) return 4;
                products[lane] = saved;
            }
        }
    }
    uint32_t state = 0x39507169u;
    for (unsigned int group = 0; group < 1000000u; ++group) {
        for (size_t lane = 0; lane < products.size(); ++lane) {
            const uint32_t bits = next(state);
            // Include dense same-sign sums near the ambiguous interval,
            // cancellation, zeros and arbitrary exponent-aligned magnitudes.
            const uint32_t magnitude = group % 4u == 0u
                ? kMaxAlignedProduct - (bits & 0xffu)
                : bits % (kMaxAlignedProduct + 1u);
            const bool negative = group % 4u == 0u
                ? (group & 4u) != 0u : (bits & 1u) != 0u;
            products[lane] = negative ? -static_cast<int64_t>(magnitude)
                                      : static_cast<int64_t>(magnitude);
        }
        const uint32_t bits = next(state);
        const int64_t magnitude = bits % (kMaxAlignedAccumulator + 1u);
        if (!check(products, bits & 1u ? -magnitude : magnitude)) return 5;
        const int16_t exponent = static_cast<int16_t>(static_cast<int>(bits % 638u) - 126);
        const uint32_t fraction = next(state) & 0x007fffffu;
        const uint32_t significand = exponent == -126 && fraction != 0u
            ? fraction : fraction | 0x00800000u;
        if (!check_final_group_endpoint({significand, exponent, (bits & 1u) != 0u}))
            return 6;
    }
    std::puts("sm121_group16_modulo=pass groups=1000000 signed_overflow_edges=pass");
    return 0;
}
