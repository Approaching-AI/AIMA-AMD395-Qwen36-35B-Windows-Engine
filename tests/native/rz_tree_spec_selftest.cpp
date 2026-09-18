#include <cassert>
#include <cfenv>
#include <cstdio>
#include <vector>
#include "sm121_rz_tree_spec.h"
#pragma STDC FENV_ACCESS ON
namespace spec = qrt_sm121_rz_tree;
uint32_t state = 0x3958192u;
uint32_t random_word() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
int main() {
    constexpr unsigned count = 1048576u;
    std::vector<float> a(count), b(count), expected(count), actual(count);
    unsigned large_gap = 0, opposing = 0, zeros = 0;
    assert(std::fesetround(FE_TONEAREST) == 0);
    for (unsigned i = 0; i < count; ++i) {
        uint32_t x = (random_word() & 0x807fffffu) | ((51u + random_word() % 149u) << 23u);
        uint32_t y = (random_word() & 0x807fffffu) | ((51u + random_word() % 149u) << 23u);
        if (i % 11u == 0u) y = x ^ 0x80000000u;
        if (i % 13u == 0u) x &= 0x80000000u;
        if (i % 17u == 0u) y &= 0x80000000u;
        if (i % 19u == 0u) { x &= 0xff800000u; y = x - 1u; }
        // Keep the admitted finite/normal-or-zero domain after edge shaping.
        if (((y >> 23u) & 255u) == 255u) y = 0u;
        a[i] = qrt_sm121_float_alignment::from_bits(x);
        b[i] = qrt_sm121_float_alignment::from_bits(y);
        expected[i] = spec::add_spec(a[i], b[i]);
        large_gap += std::abs(int((x >> 23u) & 255u) - int((y >> 23u) & 255u)) > 29;
        opposing += (x ^ y) >> 31u;
        zeros += !(x & 0x7fffffffu) || !(y & 0x7fffffffu);
    }
    assert(std::fesetround(FE_TOWARDZERO) == 0);
    for (unsigned i = 0; i < count; ++i) { volatile float left = a[i], right = b[i]; actual[i] = left + right; }
    assert(std::fesetround(FE_TONEAREST) == 0);
    for (unsigned i = 0; i < count; ++i) {
        if (qrt_sm121_f32_carry::bits(actual[i]) != qrt_sm121_f32_carry::bits(expected[i])) {
            std::fprintf(stderr, "i=%u a=%08x b=%08x expected=%08x actual=%08x\n", i,
                qrt_sm121_f32_carry::bits(a[i]), qrt_sm121_f32_carry::bits(b[i]),
                qrt_sm121_f32_carry::bits(expected[i]), qrt_sm121_f32_carry::bits(actual[i]));
            return 1;
        }
    }
    for (unsigned word = 0u; word < 65536u; ++word) {
        const unsigned e = (word >> 7u) & 255u;
        assert(spec::eligible(uint16_t(word)) == (!(word & 0x7fffu) || (e >= 96u && e <= 158u)));
    }
    assert(large_gap > 500000u && opposing > 400000u && zeros > 100000u);
    std::printf("{\"kind\":\"rz_tree_host_spec\",\"hardware_rz_additions\":%u,\"large_gap\":%u,\"opposing_signs\":%u,\"zero_inputs\":%u,\"eligibility_encodings\":65536,\"raw_bit_mismatches\":0,\"fp_mode_restored\":true,\"gb10_acceptance\":false}\n", count, large_gap, opposing, zeros);
}
