#include "../../native/providers/moe_accumulator/sm121_half_f32_carry.h"
#include <cstdio>
#include <cstring>

namespace original = qrt_q1_moe_hawkeye;
namespace half = qrt_sm121_scaled_half_products;
namespace fast = qrt_sm121_half_f32_carry;

int main() {
    const uint16_t edges[] = {0, 0x8000, 1, 0x7f, 0x80, 0x807f, 0x3f80,
        0xbf80, 0x3fff, 0xbfff, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffff};
    unsigned state = 0x3958192u;
    auto random = [&] { state ^= state << 13u; state ^= state >> 17u; state ^= state << 5u; return state; };
    original::Value previous{0u, -133, false};
    unsigned accepted = 0u, rejected = 0u, accepted_zero = 0u, accepted_partial = 0u;
    for (unsigned group = 0u; group < 500000u; ++group) {
        uint16_t a[16], b[16];
        original::Value values[17];
        const unsigned base_a = 1u + random() % 254u, base_b = 1u + random() % 254u;
        const unsigned spread = 1u + group % 30u;
        for (unsigned i = 0u; i < 16u; ++i) {
            if (group < 65536u) { a[i] = uint16_t(group + i * 4093u); b[i] = edges[i]; }
            else {
                const unsigned ea = group % 3u ? 90u + group % 55u + random() % spread : base_a;
                const unsigned eb = group % 3u ? 92u + group % 49u + random() % spread : base_b;
                a[i] = uint16_t((random() & 0x807fu) | (ea << 7u));
                b[i] = uint16_t((random() & 0x807fu) | (eb << 7u));
                if (group % 11u == 0u && i % 3u == 0u) a[i] &= 0x8000u;
                if (group % 13u == 0u && i % 3u == 1u) b[i] &= 0x8000u;
                if (group % 17u == 0u) a[i] &= 0x8000u;
            }
            values[i + 1u] = original::multiply_bf16(a[i], b[i], -133);
        }
        const unsigned exponent = 1u + random() % 254u;
        const uint32_t raw = (random() & 0x807fffffu) | (exponent << 23u);
        values[0] = {0x800000u | (raw & 0x7fffffu), int16_t(int(exponent) - 127), bool(raw >> 31u)};
        if (group % 4u == 0u) values[0] = {0u, -133, false};
        if (group % 4u == 1u && previous.significand >= 0x800000u &&
            previous.exponent >= -126 && previous.exponent <= 127) values[0] = previous;
        previous = original::group_sum<26, -133>(values, 17u);
        const auto left = half::prepare(a), right = half::prepare(b);
        const auto left_before = left, right_before = right;
        const float carry = original::value_to_float(values[0]);
        float output = 123.0f;
        const bool ok = fast::accumulate(carry, left, right, &output);
        const uint32_t actual = qrt_sm121_f32_carry::bits(output);
        const uint32_t expected = qrt_sm121_f32_carry::bits(original::value_to_float(previous));
        if ((ok && actual != expected) || (!ok && output != 123.0f) ||
            std::memcmp(&left, &left_before, sizeof(left)) || std::memcmp(&right, &right_before, sizeof(right))) {
            std::printf("half_f32_failure group=%u accepted=%u expected=%08x actual=%08x\n", group, ok, expected, actual);
            return 1;
        }
        accepted += ok; rejected += !ok;
        accepted_zero += ok && !(actual & 0x7fffffffu);
        const unsigned active = (left.control & right.control) >> 16u;
        accepted_partial += ok && active && active != 65535u;
    }
    uint16_t input[16];
    for (auto& x : input) x = 0x3f80;
    const auto row = half::prepare(input);
    const uint32_t invalid_carries[] = {1u, 0x007fffffu, 0x80000001u, 0x7f800000u, 0xff800000u, 0x7fc12345u};
    for (uint32_t bits : invalid_carries) {
        float output = 123.0f;
        if (fast::accumulate(qrt_sm121_float_alignment::from_bits(bits), row, row, &output) || output != 123.0f) return 2;
    }
    std::printf("{\"ordered_groups\":500000,\"accepted\":%u,\"fallback\":%u,\"accepted_zero\":%u,\"accepted_partial\":%u,\"raw_bit_mismatches\":0,\"immutable_inputs\":true}\n",
        accepted, rejected, accepted_zero, accepted_partial);
    return !accepted || !rejected || !accepted_zero || !accepted_partial;
}
