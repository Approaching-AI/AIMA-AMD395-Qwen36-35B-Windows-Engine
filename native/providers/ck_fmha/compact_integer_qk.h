#pragma once
#include "blackwell_attention.h"
#include "../moe_accumulator/sm121_compact_integer_dot4.h"

// Isolated four-score exact QK with52-byte lossless signed16 rows. Every
// K16 keeps its original width26 carry and complete original-group fallback.
namespace qrt_compact_integer_qk {
using namespace qrt_blackwell_attention;
using Row = qrt_sm121_compact_integer_dot4::Row;
using Value = qrt_q1_moe_hawkeye::Value;
using qrt_sm121_compact_integer_dot4::prepare;
template<IntegerRowKind Kind>
__global__ void prepare_rows(const uint16_t* input, Row* output,
    unsigned tokens, unsigned start, unsigned count) {
    static_assert(Kind == IntegerRowKind::Query || Kind == IntegerRowKind::Key);
    const size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= integer_row_count(Kind, tokens, count)) return;
    uint16_t words[16];
    for (unsigned i = 0u; i < 16u; ++i)
        words[i] = input[integer_row_input_index(Kind, row, i, tokens, start, count)];
    output[row] = prepare(words);
}
template<unsigned Window>
__global__ void scores(const Row* query, const Row* key, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    static_assert(Window == 64u || Window == 128u);
    constexpr unsigned groups = Window / 16u, rows = 32u, columns = 32u;
    constexpr unsigned words = sizeof(Row) / 4u, threads = 256u;
    __shared__ Row left[groups][rows], right[groups][columns];
    const unsigned head = blockIdx.y, kv_head = head / 8u;
    const unsigned query_tile = blockIdx.z * rows, key_tile = blockIdx.x * columns;
    const unsigned qr = threadIdx.x / 16u, kc = threadIdx.x % 16u;
    const unsigned last_query = query_start + min(query_tile + rows, query_count) - 1u;
    if (key_tile > last_query) {
#pragma unroll
        for (unsigned q = 0u; q < 2u; ++q) {
#pragma unroll
            for (unsigned k = 0u; k < 2u; ++k) {
                const unsigned r = query_tile + qr + q * 16u, c = key_tile + kc + k * 16u;
                if (r < query_count && c < stride)
                    output[(size_t(r) * 16u + head) * stride + c] = -INFINITY;
            }
        }
        return;
    }
    Value integer[2][2];
    bool active[2][2];
#pragma unroll
    for (unsigned q = 0u; q < 2u; ++q) {
#pragma unroll
        for (unsigned k = 0u; k < 2u; ++k) {
            integer[q][k] = {0u, qrt_blackwell_attention::kBlackwellZeroExponent, false};
            const unsigned r = query_tile + qr + q * 16u, c = key_tile + kc + k * 16u;
            active[q][k] = r < query_count && c < stride && c <= query_start + r;
        }
    }
    for (unsigned start = 0u; start < 16u; start += groups) {
        for (unsigned item = threadIdx.x; item < groups * (rows + columns) * words; item += threads) {
            const unsigned index = item / words, word = item % words;
            const unsigned group = index / (rows + columns), local = index % (rows + columns);
            const bool is_query = local < rows;
            const unsigned r = is_query ? query_tile + local : key_tile + local - rows;
            uint32_t value = 0u;
            if (r < (is_query ? query_count : stride)) {
                const Row* source = is_query
                    ? query + (size_t(r) * 16u + head) * 16u + start + group
                    : key + (size_t(kv_head) * 16u + start + group) * key_stride + r;
                __builtin_memcpy(&value, reinterpret_cast<const unsigned char*>(source) + word * 4u, 4u);
            }
            Row* target = is_query ? &left[group][local] : &right[group][local - rows];
            __builtin_memcpy(reinterpret_cast<unsigned char*>(target) + word * 4u, &value, 4u);
        }
        __syncthreads();
#pragma unroll 1
        for (unsigned group = 0u; group < groups; ++group) {
#pragma unroll
            for (unsigned q = 0u; q < 2u; ++q) {
#pragma unroll
                for (unsigned k = 0u; k < 2u; ++k) if (active[q][k]) {
                    integer[q][k] = qrt_sm121_compact_integer_dot4::accumulate(
                        integer[q][k], left[group][qr + q * 16u], right[group][kc + k * 16u]);
                }
            }
        }
        // Inactive cells retain their whole-CTA barrier duties.
        __syncthreads();
    }
#pragma unroll
    for (unsigned q = 0u; q < 2u; ++q) {
#pragma unroll
        for (unsigned k = 0u; k < 2u; ++k) {
            const unsigned r = query_tile + qr + q * 16u, c = key_tile + kc + k * 16u;
            if (r < query_count && c < stride) {
                float result = -INFINITY;
                if (active[q][k]) {
                    result = qrt_q1_moe_hawkeye::value_to_float(
                        qrt_sm121_group16::finish_accumulator(integer[q][k])) * qrt_blackwell_attention::kExactScale;
                }
                output[(size_t(r) * 16u + head) * stride + c] = result;
            }
        }
    }
}
} // namespace qrt_compact_integer_qk
