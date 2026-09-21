#include "../../native/providers/moe_accumulator/sm121_partial_matrix_group.h"
#include "narrow_half_cases.h"
#include <cassert>
#include <cstdio>
#include <vector>

namespace candidate = qrt_sm121_partial_matrix_group;
namespace matrix = qrt_sm121_compact_matrix_group;
namespace original = qrt_q1_moe_hawkeye;
int signed_word(uint16_t x) { return x & 0x8000u ? int(x) - 65536 : int(x); }

struct Counts {
    uint64_t words = 0, groups = 0, original_admitted = 0;
    uint64_t remainder_admitted = 0, partial_admitted = 0, rejected = 0;
    uint64_t sparse_products = 0, domain_rejected_rows = 0;
};

candidate::Row prepare(const uint16_t* values, Counts& count) {
    const auto row = candidate::prepare(values);
    bool eligible = true;
    for (unsigned i = 0; i < 16u; ++i) {
        assert(candidate::original(row, i) == values[i]);
        eligible &= qrt_sm121_narrow_f32_carry::eligible(values[i]);
        const uint16_t coefficient = uint16_t(candidate::coefficient_pair(row, i/2u) >> ((i&1u)*16u));
        if (!matrix::compact::unit(row.common.encoded) || row.exceptions & (1u << i))
            assert(coefficient == 0u);
        else assert(coefficient == matrix::compact::word(row.common.encoded, i));
        ++count.words;
    }
    if (!eligible) { assert(!matrix::compact::unit(row.common.encoded)); ++count.domain_rejected_rows; }
    return row;
}

float group(float carried, original::Value& carry, const uint16_t* a,
    const uint16_t* b, Counts& count) {
    const auto pa = prepare(a,count), pb = prepare(b,count);
    const auto old_a = matrix::prepare(a), old_b = matrix::prepare(b);
    original::Value terms[17]; terms[0] = carry;
    int64_t mathematical = 0, old_mathematical = 0;
    for (unsigned i = 0; i < 16u; ++i) {
        terms[i+1u] = original::multiply_bf16(a[i],b[i],-133);
        mathematical += int64_t(signed_word(uint16_t(candidate::coefficient_pair(pa,i/2u) >> ((i&1u)*16u)))) *
            signed_word(uint16_t(candidate::coefficient_pair(pb,i/2u) >> ((i&1u)*16u)));
        old_mathematical += int64_t(signed_word(matrix::compact::word(old_a.encoded,i))) *
            signed_word(matrix::compact::word(old_b.encoded,i));
    }
    const auto expected = original::group_sum<26,-133>(terms,17u);
    const auto expected_bits = matrix::f32::bits(original::value_to_float(expected));
    const auto sentinel = matrix::f32::alignment::from_bits(0x4f395819u);
    float old = sentinel, rem = sentinel, value = sentinel;
    const bool admitted_old = matrix::accumulate(carried,old_a,old_b,old_mathematical,&old);
    const bool admitted_remainder = qrt_sm121_matrix_remainder_group::accumulate(
        carried,old_a,old_b,old_mathematical,&rem);
    const bool admitted = candidate::accumulate(carried,pa,pb,mathematical,&value);
    assert(!admitted_old || admitted_remainder);
    assert(!admitted_remainder || admitted);
    if (admitted) {
        if (matrix::f32::bits(value) != expected_bits) {
            std::fprintf(stderr,"group=%llu got=%08x expected=%08x exceptions=%04x/%04x\n",
                (unsigned long long)count.groups,matrix::f32::bits(value),expected_bits,pa.exceptions,pb.exceptions);
            std::abort();
        }
        if (admitted_old) { assert(matrix::f32::bits(old) == expected_bits); ++count.original_admitted; }
        else if (admitted_remainder) { assert(matrix::f32::bits(rem) == expected_bits); ++count.remainder_admitted; }
        else {
            ++count.partial_admitted;
            count.sparse_products += unsigned(__builtin_popcount((pa.exceptions | pb.exceptions) &
                matrix::compact::nonzero(pa.common.encoded) & matrix::compact::nonzero(pb.common.encoded)));
        }
    } else {
        assert(matrix::f32::bits(value) == 0x4f395819u);
        qrt_sm121_float_alignment::Group raw;
        for (unsigned i = 0; i < 16u; ++i) raw.set(i,a[i],b[i]);
        value = qrt_sm121_narrow_f32_carry::accumulate(carried,raw);
        assert(matrix::f32::bits(value) == expected_bits);
        ++count.rejected;
    }
    carry = expected; ++count.groups;
    return value;
}

int main() {
    Counts count;
    const unsigned widths[] = {16u,32u,256u,2048u,4096u,8192u};
    for (unsigned width : widths) for (unsigned row = 0; row < 256u; ++row) {
        std::vector<uint16_t> a(width), b(width);
        bool narrow = true;
        for (unsigned g = 0; g < width/16u; ++g) for (unsigned i = 0; i < 16u; ++i) {
            const auto values = qrt_narrow_half_cases::input(row,g,i);
            a[g*16u+i] = values.x; b[g*16u+i] = values.y;
            narrow &= qrt_sm121_narrow_f32_carry::eligible(values.x) &&
                qrt_sm121_narrow_f32_carry::eligible(values.y);
        }
        if (!narrow) continue;
        original::Value carry{0u,-133,false}; float carried = 0.0f;
        for (unsigned g = 0; g < width/16u; ++g)
            carried = group(carried,carry,a.data()+g*16u,b.data()+g*16u,count);
    }
    // Every possible BF16 word must survive preparation, including all NaNs,
    // infinities, subnormals and both zeros. Qualified words additionally enter
    // independent original-group comparisons at every exception slot.
    for (unsigned word = 0u; word < 65536u; ++word) {
        const unsigned lane = word & 15u;
        uint16_t a[16],b[16];
        for (unsigned i = 0u; i < 16u; ++i) {
            a[i] = uint16_t((i&1u ? 0x8000u : 0u) | ((i%3u ? 130u : 159u) << 7u) | (i*7u));
            b[i] = uint16_t((i&2u ? 0x8000u : 0u) | ((i%5u ? 95u : 127u) << 7u) | (i*3u));
        }
        a[lane] = uint16_t(word); b[15u-lane] = uint16_t(word ^ 0x8000u);
        prepare(a,count); prepare(b,count);
        if (qrt_sm121_narrow_f32_carry::eligible(uint16_t(word))) {
            original::Value carry{0u,-133,false};
            group(0.0f,carry,a,b,count);
        } else {
            const auto pa = candidate::prepare(a), pb = candidate::prepare(b);
            float output = matrix::f32::alignment::from_bits(0x4f395819u);
            assert(!candidate::accumulate(0.0f,pa,pb,0,&output));
            assert(matrix::f32::bits(output) == 0x4f395819u);
        }
    }
    assert(count.original_admitted && count.remainder_admitted && count.partial_admitted && count.rejected);
    assert(count.domain_rejected_rows && count.sparse_products);
    std::printf("{\"kind\":\"partial_matrix_group_host\",\"ordered_groups\":%llu,\"lossless_words\":%llu,\"original_matrix_admitted\":%llu,\"additional_remainder_admitted\":%llu,\"additional_partial_admitted\":%llu,\"partial_scalar_products\":%llu,\"original_narrow_fallback\":%llu,\"domain_rejected_rows\":%llu,\"raw_carry_mismatches\":0,\"all_original_admissions_preserved\":true,\"rejected_output_unchanged\":true,\"all_bf16_words_prepared\":true,\"widths_through8192\":true}\n",
        (unsigned long long)count.groups,(unsigned long long)count.words,
        (unsigned long long)count.original_admitted,(unsigned long long)count.remainder_admitted,
        (unsigned long long)count.partial_admitted,(unsigned long long)count.sparse_products,
        (unsigned long long)count.rejected,(unsigned long long)count.domain_rejected_rows);
    return 0;
}
