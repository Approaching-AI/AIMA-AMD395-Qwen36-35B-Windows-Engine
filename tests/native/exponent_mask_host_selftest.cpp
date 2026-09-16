#include "../../native/providers/moe_accumulator/sm121_exponent_mask.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace mask = qrt_sm121_exponent_mask;
namespace alignment = qrt_sm121_float_alignment;
namespace f32 = qrt_sm121_f32_carry;
void require(bool value) { if (!value) throw std::runtime_error("exponent mask host comparison"); }
int exponent(uint16_t x) { return (x & 0x7fffu) ? int((x >> 7u) & 255u) - 127 : -512; }
void check_pack(const uint16_t* words, const uint32_t* packed) {
    int maximum = -512;
    uint16_t positions[3]{};
    for (unsigned i = 0u; i < 16u; ++i) maximum = std::max(maximum, exponent(words[i]));
    for (unsigned i = 0u; i < 16u; ++i)
        for (unsigned rank = 0u; rank < 3u; ++rank)
            if ((words[i] & 0x7fffu) && exponent(words[i]) == maximum - int(rank)) positions[rank] |= uint16_t(1u << i);
    for (unsigned i = 0u; i < 16u; ++i) {
        require(uint16_t(packed[i] >> 16u) == words[i]);
        require(uint16_t(packed[i]) == (i == 0u ? uint16_t(maximum) : i < 4u ? positions[i - 1u] : uint16_t(exponent(words[i]))));
    }
}
template<bool CarryAware>
void check_group(const uint16_t* left, const uint16_t* right, const uint32_t* a,
    const uint32_t* b, float carry, float* result, bool* accepted) {
    alignment::Group reference, candidate;
    for (unsigned i = 0u; i < 16u; ++i) reference.set(i, left[i], right[i]);
    mask::group<1u, 16u, CarryAware>(candidate, a, b, carry);
    const uint32_t absolute = f32::bits(carry) & 0x7fffffffu;
    const int ce = absolute ? int(absolute >> 23u) - 127 : -133;
    require(candidate.maximum == std::max(reference.maximum, CarryAware ? ce : -133));
    require(candidate.first_negative == reference.first_negative);
    for (unsigned i = 0u; i < 16u; ++i) require(f32::bits(candidate.products[i]) == f32::bits(reference.products[i]));
    float expected = alignment::from_bits(0x4f654321u);
    *result = expected;
    const bool ok = f32::accumulate<0u>(carry, reference, &expected);
    *accepted = f32::accumulate<0u>(carry, candidate, result);
    require(*accepted == ok && f32::bits(expected) == f32::bits(*result));
    if (!ok) require(f32::bits(*result) == 0x4f654321u);
}

int main() try {
    uint32_t random = 0x31ee8719u;
    auto next = [&] { random ^= random << 13u; random ^= random >> 17u; random ^= random << 5u; return random; };
    unsigned certificates = 0u, misses = 0u, checks = 0u, groups = 0u, declines = 0u;
    for (unsigned raw = 0u; raw < 65536u; raw += 16u) {
        uint16_t words[16], saved[16]; uint32_t packed[16];
        for (unsigned i = 0u; i < 16u; ++i) words[i] = uint16_t(raw + i);
        std::memcpy(saved, words, sizeof(words)); mask::pack(words, packed); check_pack(words, packed);
        require(!std::memcmp(saved, words, sizeof(words)));
    }
    // Explicit distinct top-rank intersections and a miss below rank two.
    for (unsigned delta = 0u; delta < 4u; ++delta) {
        uint16_t a[16], b[16]; uint32_t ap[16], bp[16];
        for (unsigned i = 0u; i < 16u; ++i) { a[i] = uint16_t(110u << 7u); b[i] = uint16_t(110u << 7u); }
        a[0] = uint16_t(127u << 7u); b[0] = uint16_t((127u - delta) << 7u); b[1] = uint16_t(127u << 7u);
        mask::pack(a, ap); mask::pack(b, bp);
        int value = 0x321;
        const bool ok = mask::maximum(mask::metadata<1u>(ap), mask::metadata<1u>(bp), -133, &value);
        require(ok == (delta < 3u)); require(value == (ok ? -int(delta) : 0x321));
    }
    for (unsigned sample = 0u; sample < 65536u; ++sample) {
        uint16_t a[16], b[16], saved_a[16], saved_b[16]; uint32_t ap[16], bp[16], strided[256]{};
        int product_maximum = -133;
        for (unsigned i = 0u; i < 16u; ++i) {
            a[i] = uint16_t((next() & 0x807fu) | ((64u + next() % 127u) << 7u));
            b[i] = uint16_t((next() & 0x807fu) | ((64u + next() % 127u) << 7u));
            if (sample % 7u == 0u || next() % 7u == 0u) a[i] &= 0x8000u;
            if (sample % 11u == 0u || next() % 11u == 0u) b[i] &= 0x8000u;
            if ((a[i] & 0x7fffu) && (b[i] & 0x7fffu)) product_maximum = std::max(product_maximum, exponent(a[i]) + exponent(b[i]));
        }
        std::memcpy(saved_a, a, sizeof(a)); std::memcpy(saved_b, b, sizeof(b));
        mask::pack(a, ap); mask::pack(b, bp); check_pack(a, ap); check_pack(b, bp);
        for (unsigned i = 0u; i < 16u; ++i) strided[i * 16u] = bp[i];
        for (int ce : {-133, -126, 0, 63, 127}) {
            int value = 0x321;
            const bool ok = mask::maximum(mask::metadata<1u>(ap), mask::metadata<16u>(strided), ce, &value);
            require(value == (ok ? std::max(ce, product_maximum) : 0x321));
            certificates += ok; misses += !ok; ++checks;
            const float carry = ce == -133 ? 0.0f : alignment::from_bits((uint32_t(ce + 127) << 23u) | (sample & 1u ? 0x80000000u : 0u));
            float result; bool accepted;
            check_group<false>(a, b, ap, strided, carry, &result, &accepted); declines += !accepted;
            check_group<true>(a, b, ap, strided, carry, &result, &accepted); declines += !accepted;
        }
        require(!std::memcmp(saved_a, a, sizeof(a)) && !std::memcmp(saved_b, b, sizeof(b)));
        check_pack(a, ap); check_pack(b, bp);
    }
    for (unsigned dot = 0u; dot < 16384u; ++dot) {
        float carry = 0.0f;
        for (unsigned g = 0u; g < 16u; ++g) {
            uint16_t a[16], b[16]; uint32_t ap[16], bp[16], strided[256]{};
            for (unsigned i = 0u; i < 16u; ++i) {
                a[i] = uint16_t((next() & 0x807fu) | ((115u + next() % 25u) << 7u));
                b[i] = uint16_t((next() & 0x807fu) | ((115u + next() % 25u) << 7u));
                if (next() % 11u == 0u) a[i] &= 0x8000u;
                if (next() % 13u == 0u) b[i] &= 0x8000u;
            }
            mask::pack(a, ap); mask::pack(b, bp);
            for (unsigned i = 0u; i < 16u; ++i) strided[i * 16u] = bp[i];
            const float expected = qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(carry, a, b, 16u);
            float result0, result1; bool ok0, ok1;
            check_group<false>(a, b, ap, strided, carry, &result0, &ok0);
            check_group<true>(a, b, ap, strided, carry, &result1, &ok1);
            require(ok0 && ok1 && f32::bits(result0) == f32::bits(expected) && f32::bits(result1) == f32::bits(expected));
            carry = expected; ++groups;
        }
    }
    require(certificates && misses && declines);
    std::printf("{\"kind\":\"exponent_mask_host\",\"bf16_encodings\":65536,\"certificate_checks\":%u,\"certificates\":%u,\"misses\":%u,\"canonical_groups\":%u,\"declined_accumulations\":%u,\"raw_mismatches\":0,\"immutable_inputs\":true,\"declined_output_unchanged\":true,\"inference_acceptance\":false}\n", checks, certificates, misses, groups, declines);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
