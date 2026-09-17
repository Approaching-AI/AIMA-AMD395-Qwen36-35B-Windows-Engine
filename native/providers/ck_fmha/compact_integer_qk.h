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
__device__ __forceinline__ void store_score(Value carry, bool active, float* output,
    unsigned row, unsigned column, unsigned head, unsigned count, unsigned stride) {
    if (row < count && column < stride)
        output[(size_t(row) * 16u + head) * stride + column] = active
            ? qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry)) * kExactScale
            : -INFINITY;
}

template<unsigned Window>
__global__ __launch_bounds__(256) void scores(const Row* query, const Row* key, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    static_assert(Window == 64u || Window == 128u);
    constexpr unsigned groups = Window / 16u, rows = 32u, columns = 32u;
    constexpr unsigned words = sizeof(Row) / 4u, threads = 256u;
    __shared__ Row left[groups][rows], right[groups][columns];
    const unsigned head = blockIdx.y, kv_head = head / 8u;
    const unsigned query_tile = blockIdx.z * rows, key_tile = blockIdx.x * columns;
    const unsigned qr = threadIdx.x / 16u, kc = threadIdx.x % 16u;
    const unsigned row0 = query_tile + qr, row1 = row0 + 16u;
    const unsigned column0 = key_tile + kc, column1 = column0 + 16u;
    const unsigned last_query = query_start + min(query_tile + rows, query_count) - 1u;
    const Value zero{0u, kBlackwellZeroExponent, false};
    if (key_tile > last_query) {
        store_score(zero, false, output, row0, column0, head, query_count, stride);
        store_score(zero, false, output, row0, column1, head, query_count, stride);
        store_score(zero, false, output, row1, column0, head, query_count, stride);
        store_score(zero, false, output, row1, column1, head, query_count, stride);
        return;
    }
    // Explicit independent SSA values avoid the compiler's private-array
    // lowering of carry[2][2] into32 bytes of LDS per maximum workgroup lane.
    Value carry00 = zero, carry01 = zero, carry10 = zero, carry11 = zero;
    const bool active00 = row0 < query_count && column0 < stride && column0 <= query_start + row0;
    const bool active01 = row0 < query_count && column1 < stride && column1 <= query_start + row0;
    const bool active10 = row1 < query_count && column0 < stride && column0 <= query_start + row1;
    const bool active11 = row1 < query_count && column1 < stride && column1 <= query_start + row1;
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
            if (active00) carry00 = qrt_sm121_compact_integer_dot4::accumulate(carry00, left[group][qr], right[group][kc]);
            if (active01) carry01 = qrt_sm121_compact_integer_dot4::accumulate(carry01, left[group][qr], right[group][kc + 16u]);
            if (active10) carry10 = qrt_sm121_compact_integer_dot4::accumulate(carry10, left[group][qr + 16u], right[group][kc]);
            if (active11) carry11 = qrt_sm121_compact_integer_dot4::accumulate(carry11, left[group][qr + 16u], right[group][kc + 16u]);
        }
        // Inactive cells retain their whole-CTA barrier duties.
        __syncthreads();
    }
    store_score(carry00, active00, output, row0, column0, head, query_count, stride);
    store_score(carry01, active01, output, row0, column1, head, query_count, stride);
    store_score(carry10, active10, output, row1, column0, head, query_count, stride);
    store_score(carry11, active11, output, row1, column1, head, query_count, stride);
}
} // namespace qrt_compact_integer_qk
