#include "../../native/providers/moe_accumulator/sm121_matrix_remainder_group.h"
#include "narrow_half_cases.h"
#include <cassert>
#include <cstdio>
#include <vector>

namespace matrix = qrt_sm121_compact_matrix_group;
namespace remainder_group = qrt_sm121_matrix_remainder_group;
namespace original = qrt_q1_moe_hawkeye;
int signed_word(uint16_t x) { return x & 0x8000u ? int(x) - 65536 : int(x); }

int main() {
    uint64_t groups = 0, original_admitted = 0, additional_admitted = 0, rejected = 0;
    const unsigned widths[] = {16u,32u,256u,2048u,4096u,8192u};
    for (unsigned width : widths) for (unsigned row = 0; row < 256u; ++row) {
        std::vector<uint16_t> a(width), b(width);
        std::vector<matrix::Row> pa(width/16u), pb(width/16u);
        bool narrow = true;
        for (unsigned g = 0; g < width/16u; ++g) {
            for (unsigned i = 0; i < 16u; ++i) {
                const auto values = qrt_narrow_half_cases::input(row,g,i);
                a[g*16u+i] = values.x; b[g*16u+i] = values.y;
                narrow &= qrt_sm121_narrow_f32_carry::eligible(values.x) &&
                    qrt_sm121_narrow_f32_carry::eligible(values.y);
            }
            pa[g] = matrix::prepare(a.data()+g*16u);
            pb[g] = matrix::prepare(b.data()+g*16u);
        }
        if (!narrow) continue;
        original::Value carry{0u,-133,false};
        float carried = 0.0f;
        for (unsigned g = 0; g < width/16u; ++g) {
            original::Value terms[17]; terms[0] = carry;
            int64_t mathematical = 0;
            for (unsigned i = 0; i < 16u; ++i) {
                terms[i+1u] = original::multiply_bf16(a[g*16u+i],b[g*16u+i],-133);
                mathematical += int64_t(signed_word(matrix::compact::word(pa[g].encoded,i))) *
                    signed_word(matrix::compact::word(pb[g].encoded,i));
            }
            const auto expected = original::group_sum<26,-133>(terms,17u);
            const float expected_float = original::value_to_float(expected);
            const auto sentinel = matrix::f32::alignment::from_bits(0x4f395819u);
            float control = sentinel, result = sentinel;
            const bool old_admitted = matrix::accumulate(carried,pa[g],pb[g],mathematical,&control);
            const bool admitted = remainder_group::accumulate(carried,pa[g],pb[g],mathematical,&result);
            assert(!old_admitted || admitted);
            if (admitted) {
                assert(matrix::f32::bits(result) == matrix::f32::bits(expected_float));
                if (old_admitted) {
                    assert(matrix::f32::bits(result) == matrix::f32::bits(control));
                    ++original_admitted;
                } else ++additional_admitted;
            } else {
                assert(matrix::f32::bits(result) == 0x4f395819u);
                qrt_sm121_float_alignment::Group products;
                for (unsigned i = 0; i < 16u; ++i) products.set(i,a[g*16u+i],b[g*16u+i]);
                result = qrt_sm121_narrow_f32_carry::accumulate(carried,products);
                assert(matrix::f32::bits(result) == matrix::f32::bits(expected_float));
                ++rejected;
            }
            carry = expected; carried = result; ++groups;
        }
    }
    assert(original_admitted && additional_admitted && rejected);
    std::printf("{\"kind\":\"matrix_remainder_group_host\",\"ordered_groups\":%llu,\"original_matrix_admitted\":%llu,\"additional_remainder_admitted\":%llu,\"original_narrow_fallback\":%llu,\"raw_carry_mismatches\":0,\"all_original_admissions_preserved\":true,\"rejected_output_unchanged\":true,\"widths_through8192\":true}\n",
        (unsigned long long)groups,(unsigned long long)original_admitted,
        (unsigned long long)additional_admitted,(unsigned long long)rejected);
}
