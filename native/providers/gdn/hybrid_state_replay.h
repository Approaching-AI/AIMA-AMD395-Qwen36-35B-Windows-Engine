#pragma once
#include "separate_state_replay.h"
#include "absolute_dot_bound.h"
#include "../moe_accumulator/sm121_product_f32_carry.h"

// Isolated third-stage experiment: fast -> retry -> retained replay. It is
// not wired into the runtime. Retry only receipt-zero CTAs, preserve global
// recurrent state on rejection, and publish receipt two after full success.
namespace qrt_fla_hybrid_state {
namespace separate = qrt_fla_separate_state;
namespace scalar = qrt_fla_blackwell_scalar;
namespace norm = qrt_fla_absolute_dot;
namespace consumer = qrt_fla_consumer_interval;
namespace product = qrt_sm121_product_f32_carry;
using scalar::from_bf16; using scalar::to_bf16; using scalar::exponential;
using scalar::pack; using scalar::threads;

template<unsigned Width, unsigned Columns>
__device__ __forceinline__ bool product_dot(const uint32_t* left,
    const uint32_t (&right)[Width / 2u][Columns], unsigned column, float* output) {
    static_assert(Width == 64u || Width == 128u);
    float carry = 0.0f;
    for (unsigned base = 0u; base < Width; base += 16u) {
        product::Group group;
#pragma unroll
        for (unsigned i = 0u; i < 16u; i += 2u) {
            const uint32_t a = left[(base + i) / 2u], b = right[(base + i) / 2u][column];
            group.set(i, uint16_t(a), uint16_t(b));
            group.set(i + 1u, uint16_t(a >> 16u), uint16_t(b >> 16u));
        }
        float next;
        if (!product::accumulate(carry, group, &next)) return false;
        carry = next;
    }
    *output = carry;
    return true;
}

template<unsigned Columns>
__global__ void retry_kernel(const uint16_t* k, const uint16_t* u, const uint16_t* w,
    const float* g, uint16_t* h, uint16_t* v_new, float* state, unsigned count,
    const unsigned char* table, unsigned* completed) {
    static_assert(Columns == 4u || Columns == 8u);
    const unsigned receipt = blockIdx.y * (128u / Columns) + blockIdx.x;
    if (completed[receipt]) return;
    __shared__ float current[Columns][128], decay[64], segment_decay;
    __shared__ uint32_t rounded[64][Columns], residual[32][Columns];
    __shared__ uint16_t residual_words[64][Columns];
    union Operands { uint32_t weights[64][65]; uint32_t keys[128][33]; };
    __shared__ Operands operands;
    __shared__ unsigned weight_ok[64], state_ok[Columns], accepted, all_narrow;
    __shared__ norm::Norm weight_norm[64], state_norm[Columns];
    const unsigned tid = threadIdx.x, head = blockIdx.y, first_column = blockIdx.x * Columns;
    for (unsigned cell = tid; cell < Columns * 128u; cell += threads)
        current[cell / 128u][cell % 128u] = state[(head * 128u + first_column + cell / 128u) * 128u + cell % 128u];
    __syncthreads();
    for (unsigned offset = 0u; offset < count; offset += 64u) {
        const unsigned valid = min(64u, count - offset);
        if (tid < 64u) weight_ok[tid] = 1u;
        if (tid < Columns) state_ok[tid] = 1u;
        if (!tid) { accepted = 1u; all_narrow = 1u; segment_decay = exponential(g[size_t(offset + valid - 1u) * 32u + head], table); }
        if (tid < valid) decay[tid] = exponential(g[size_t(offset + valid - 1u) * 32u + head] - g[size_t(offset + tid) * 32u + head], table);
        __syncthreads();
        for (unsigned cell = tid; cell < 64u * Columns; cell += threads) {
            const unsigned pair = cell / Columns, column = cell % Columns;
            const uint16_t a = to_bf16(current[column][pair * 2u]), b = to_bf16(current[column][pair * 2u + 1u]);
            rounded[pair][column] = pack(a, b);
            const size_t index = size_t(offset / 64u) * 524288u + (head * 128u + first_column + column) * 128u + pair * 2u;
            h[index] = a; h[index + 1u] = b;
            if (!separate::admitted(a, b)) { atomicAnd(&state_ok[column], 0u); atomicAnd(&all_narrow, 0u); }
        }
        for (unsigned cell = tid; cell < 64u * 64u; cell += threads) {
            const unsigned row = cell / 64u, pair = cell % 64u;
            uint16_t a = 0u, b = 0u;
            if (row < valid) { const size_t index = (size_t(offset + row) * 32u + head) * 128u + pair * 2u; a = w[index]; b = w[index + 1u]; }
            operands.weights[row][pair] = pack(a, b);
            if (!separate::admitted(a, b)) { atomicAnd(&weight_ok[row], 0u); atomicAnd(&all_narrow, 0u); }
        }
        __syncthreads();
        // Norms are needed only when a W/H operand leaves the narrow domain.
        // A certificate must fix both BF16 residual consumers. It never
        // substitutes a bound for the unrounded recurrent state update.
        if (!all_narrow) {
            if (tid < valid) {
                norm::Row row;
                for (unsigned i = 0u; i < 64u; ++i) {
                    const uint32_t value = operands.weights[tid][i];
                    norm::include(row, uint16_t(value)); norm::include(row, uint16_t(value >> 16u));
                }
                weight_norm[tid] = norm::finish(row);
            }
            if (tid < Columns) {
                norm::Row row;
                for (unsigned i = 0u; i < 64u; ++i) {
                    const uint32_t value = rounded[i][tid];
                    norm::include(row, uint16_t(value)); norm::include(row, uint16_t(value >> 16u));
                }
                state_norm[tid] = norm::finish(row);
            }
        }
        __syncthreads();
        unsigned local_ok = 1u;
        for (unsigned cell = tid; cell < 64u * Columns; cell += threads) {
            const unsigned row = cell / Columns, column = cell % Columns;
            uint16_t value = 0u;
            if (row < valid) {
                const size_t index = (size_t(offset + row) * 32u + head) * 128u + first_column + column;
                if (weight_ok[row] && state_ok[column]) {
                    const float sum = separate::narrow_dot<128u, Columns>(operands.weights[row], rounded, column);
                    const float difference = from_bf16(u[index]) - sum;
                    v_new[index] = to_bf16(difference); value = to_bf16(difference * decay[row]);
                } else {
                    uint16_t updated;
                    if (consumer::residual(norm::enclose(weight_norm[row], state_norm[column]), from_bf16(u[index]), decay[row], &updated, &value))
                        v_new[index] = updated;
                    else local_ok = 0u;
                }
            }
            residual_words[row][column] = value;
        }
        if (!local_ok) atomicAnd(&accepted, 0u);
        __syncthreads();
        if (!accepted) return;
        for (unsigned cell = tid; cell < 128u * 32u; cell += threads) {
            const unsigned feature = cell / 32u, pair = cell % 32u;
            uint16_t a = 0u, b = 0u;
            if (pair * 2u < valid) a = k[(size_t(offset + pair * 2u) * 16u + head / 2u) * 128u + feature];
            if (pair * 2u + 1u < valid) b = k[(size_t(offset + pair * 2u + 1u) * 16u + head / 2u) * 128u + feature];
            operands.keys[feature][pair] = pack(a, b);
        }
        for (unsigned cell = tid; cell < 32u * Columns; cell += threads) {
            const unsigned pair = cell / Columns, column = cell % Columns;
            residual[pair][column] = pack(residual_words[pair * 2u][column], residual_words[pair * 2u + 1u][column]);
        }
        __syncthreads();
        local_ok = 1u;
        for (unsigned cell = tid; cell < 128u * Columns; cell += threads) {
            const unsigned feature = cell / Columns, column = cell % Columns;
            float sum;
            if (product_dot<64u, Columns>(operands.keys[feature], residual, column, &sum))
                current[column][feature] = fmaf(current[column][feature], segment_decay, sum);
            else local_ok = 0u;
        }
        if (!local_ok) atomicAnd(&accepted, 0u);
        __syncthreads();
        if (!accepted) return;
    }
    for (unsigned cell = tid; cell < Columns * 128u; cell += threads)
        state[(head * 128u + first_column + cell / 128u) * 128u + cell % 128u] = current[cell / 128u][cell % 128u];
    __syncthreads();
    if (!tid) completed[receipt] = 2u;
}
}  // namespace qrt_fla_hybrid_state
