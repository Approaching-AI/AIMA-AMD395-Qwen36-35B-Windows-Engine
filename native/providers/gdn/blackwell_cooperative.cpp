#include "blackwell_cooperative.h"
#include "blackwell_accumulator.h"
#include "sm121_exp2_table.h"
#include "../moe_accumulator/sm121_subgroup.h"

namespace qrt_fla_blackwell_cooperative {
namespace {
using qrt_fla_blackwell::from_bf16;
using qrt_fla_blackwell::to_bf16;
constexpr unsigned threads = 256u, lanes = 4u, groups = threads / lanes;
constexpr unsigned tile_columns = 8u, state_columns = 4u;
__device__ __forceinline__ float exponential(float x, const unsigned char* table) {
    return qrt_sm121_exp2::evaluate(table, x * 1.4426950408889634074f);
}
__device__ __forceinline__ float dot(const uint16_t* a, const uint16_t* b, unsigned count) {
    return qrt_sm121_subgroup::dot<lanes>(a, b, count);
}

// K-contiguous shared rows serve four BF16 products per subgroup lane. Each
// CTA captures all of its V columns before publishing U, preserving U=V.
__global__ void wu_kernel(const uint16_t* k, const uint16_t* v, const uint16_t* beta,
                          const uint16_t* inverse, const float* g, uint16_t* w,
                          uint16_t* u, unsigned count, const unsigned char* table) {
    __shared__ uint16_t keys[tile_columns][64], values[tile_columns][64];
    __shared__ uint16_t inverse_rows[64][64];
    const unsigned offset = blockIdx.z * 64u, head = blockIdx.y;
    const unsigned columns = blockIdx.x * tile_columns, thread = threadIdx.x;
    const unsigned valid = count - offset < 64u ? count - offset : 64u;
    for (unsigned cell = thread; cell < 64u * tile_columns; cell += threads) {
        const unsigned token = cell / tile_columns, column = cell % tile_columns;
        uint16_t kb = 0u, vb = 0u;
        if (token < valid) {
            const size_t position = offset + token;
            const float b = from_bf16(beta[position * 32u + head]);
            const uint16_t scaled = to_bf16(from_bf16(k[(position * 16u + head / 2u) * 128u + columns + column]) * b);
            kb = to_bf16(from_bf16(scaled) * exponential(g[position * 32u + head], table));
            vb = to_bf16(from_bf16(v[(position * 32u + head) * 128u + columns + column]) * b);
        }
        keys[column][token] = kb;
        values[column][token] = vb;
    }
    for (unsigned cell = thread; cell < 64u * 64u; cell += threads) {
        const unsigned token = cell / 64u, reduction = cell % 64u;
        inverse_rows[token][reduction] = token < valid && reduction < valid
            ? inverse[((size_t(offset + token) * 32u + head) * 64u) + reduction] : 0u;
    }
    __syncthreads();
    for (unsigned cell = thread / lanes; cell < valid * tile_columns; cell += groups) {
        const unsigned token = cell / tile_columns, column = cell % tile_columns;
        const float sw = dot(inverse_rows[token], keys[column], 64u);
        const float su = dot(inverse_rows[token], values[column], 64u);
        if ((thread & (lanes - 1u)) == 0u) {
            const size_t index = (size_t(offset + token) * 32u + head) * 128u + columns + column;
            w[index] = to_bf16(sw);
            u[index] = to_bf16(su);
        }
    }
}

__global__ void scores_kernel(const uint16_t* q, const uint16_t* k, const float* g,
                              uint16_t* scores, unsigned count, const unsigned char* table) {
    const unsigned offset = blockIdx.z * 64u, head = blockIdx.y;
    const unsigned valid = count - offset < 64u ? count - offset : 64u;
    const unsigned cell = blockIdx.x * groups + threadIdx.x / lanes;
    const unsigned token = cell / 64u, source = cell % 64u, lane = threadIdx.x & (lanes - 1u);
    const size_t index = (size_t(offset + token) * 32u + head) * 64u + source;
    // Tail rows are outside the logical allocation and must not be stored.
    if (token >= valid) return;
    if (source > token) { if (!lane) scores[index] = 0u; return; }
    const float sum = dot(q + (size_t(offset + token) * 16u + head / 2u) * 128u,
                          k + (size_t(offset + source) * 16u + head / 2u) * 128u, 128u);
    if (!lane) scores[index] = to_bf16(sum * exponential(
        g[size_t(offset + token) * 32u + head] - g[size_t(offset + source) * 32u + head], table));
}

// One CTA reuses eight value/checkpoint columns across all query rows of a
// chunk. Both reductions remain K-contiguous and their final FMA is unchanged.
__global__ void output_kernel(const uint16_t* q, const uint16_t* v, const uint16_t* h,
                              const float* g, const uint16_t* scores, float* output,
                              unsigned count, const unsigned char* table) {
    __shared__ uint16_t values[tile_columns][64], checkpoint[tile_columns][128];
    const unsigned offset = blockIdx.z * 64u, head = blockIdx.y;
    const unsigned columns = blockIdx.x * tile_columns, thread = threadIdx.x;
    const unsigned valid = count - offset < 64u ? count - offset : 64u;
    for (unsigned cell = thread; cell < 64u * tile_columns; cell += threads) {
        const unsigned source = cell / tile_columns, column = cell % tile_columns;
        values[column][source] = source < valid
            ? v[(size_t(offset + source) * 32u + head) * 128u + columns + column] : 0u;
    }
    for (unsigned cell = thread; cell < tile_columns * 128u; cell += threads)
        checkpoint[cell / 128u][cell % 128u] = h[size_t(blockIdx.z) * 524288u +
            (head * 128u + columns + cell / 128u) * 128u + cell % 128u];
    __syncthreads();
    for (unsigned cell = thread / lanes; cell < valid * tile_columns; cell += groups) {
        const unsigned token = cell / tile_columns, column = cell % tile_columns;
        const size_t position = offset + token;
        const float old = dot(q + (position * 16u + head / 2u) * 128u, checkpoint[column], 128u);
        const float local = dot(scores + (position * 32u + head) * 64u, values[column], 64u);
        if ((thread & (lanes - 1u)) == 0u) {
            constexpr float scale = 0.08838834764831845f;
            const float prior = old * exponential(g[position * 32u + head], table);
            output[(position * 32u + head) * 128u + columns + column] =
                from_bf16(to_bf16(fmaf(local, scale, prior * scale)));
        }
    }
}

// Retain four complete state rows through each bounded segment. A shared
// key transpose makes every K64 update contiguous; all checkpoints and final
// FP32 cells keep the original ownership and layout.
__global__ void state_kernel(const uint16_t* k, const uint16_t* u,
                             const uint16_t* w, const float* g, uint16_t* h,
                             uint16_t* v_new, float* state, unsigned count,
                             const unsigned char* table) {
    __shared__ float current[state_columns][128];
    __shared__ uint16_t rounded[state_columns][128], residual[state_columns][64];
    __shared__ uint16_t keys[128][64];
    const unsigned head = blockIdx.y, columns = blockIdx.x * state_columns;
    const unsigned thread = threadIdx.x, lane = thread & (lanes - 1u);
    for (unsigned cell = thread; cell < state_columns * 128u; cell += threads)
        current[cell / 128u][cell % 128u] = state[
            (head * 128u + columns + cell / 128u) * 128u + cell % 128u];
    __syncthreads();
    for (unsigned offset = 0u; offset < count; offset += 64u) {
        const unsigned valid = count - offset < 64u ? count - offset : 64u;
        for (unsigned cell = thread; cell < state_columns * 128u; cell += threads) {
            const uint16_t value = to_bf16(current[cell / 128u][cell % 128u]);
            rounded[cell / 128u][cell % 128u] = value;
            h[size_t(offset / 64u) * 524288u +
                (head * 128u + columns + cell / 128u) * 128u + cell % 128u] = value;
        }
        for (unsigned cell = thread; cell < 64u * 128u; cell += threads) {
            const unsigned token = cell / 128u, key = cell % 128u;
            keys[key][token] = token < valid
                ? k[(size_t(offset + token) * 16u + head / 2u) * 128u + key] : 0u;
        }
        __syncthreads();
        for (unsigned cell = thread / lanes; cell < 64u * state_columns; cell += groups) {
            const unsigned token = cell / state_columns, column = cell % state_columns;
            if (token >= valid) { if (!lane) residual[column][token] = 0u; continue; }
            const size_t position = offset + token;
            const float sum = dot(w + (position * 32u + head) * 128u, rounded[column], 128u);
            if (!lane) {
                const size_t index = (position * 32u + head) * 128u + columns + column;
                const float value = from_bf16(u[index]) - sum;
                v_new[index] = to_bf16(value);
                residual[column][token] = to_bf16(value * exponential(
                    g[size_t(offset + valid - 1u) * 32u + head] - g[position * 32u + head], table));
            }
        }
        __syncthreads();
        for (unsigned cell = thread / lanes; cell < state_columns * 128u; cell += groups) {
            const unsigned column = cell / 128u, key = cell % 128u;
            const float sum = dot(keys[key], residual[column], 64u);
            if (!lane) current[column][key] = fmaf(current[column][key],
                exponential(g[size_t(offset + valid - 1u) * 32u + head], table), sum);
        }
        __syncthreads();
    }
    for (unsigned cell = thread; cell < state_columns * 128u; cell += threads)
        state[(head * 128u + columns + cell / 128u) * 128u + cell % 128u] =
            current[cell / 128u][cell % 128u];
}
}

hipError_t wu(const uint16_t* k, const uint16_t* v, const uint16_t* beta,
              const uint16_t* inverse, const float* g, uint16_t* w, uint16_t* u,
              unsigned count, const unsigned char* table, hipStream_t stream) {
    hipLaunchKernelGGL(wu_kernel, dim3(128u / tile_columns, 32u, (count + 63u) / 64u),
        dim3(threads), 0u, stream, k, v, beta, inverse, g, w, u, count, table);
    return hipGetLastError();
}
hipError_t scores(const uint16_t* q, const uint16_t* k, const float* g, uint16_t* result,
                  unsigned count, const unsigned char* table, hipStream_t stream) {
    hipLaunchKernelGGL(scores_kernel, dim3(64u * 64u / groups, 32u, (count + 63u) / 64u),
        dim3(threads), 0u, stream, q, k, g, result, count, table);
    return hipGetLastError();
}
hipError_t output(const uint16_t* q, const uint16_t* v, const uint16_t* h, const float* g,
                  const uint16_t* scores, float* result, unsigned count,
                  const unsigned char* table, hipStream_t stream) {
    hipLaunchKernelGGL(output_kernel, dim3(128u / tile_columns, 32u, (count + 63u) / 64u),
        dim3(threads), 0u, stream, q, v, h, g, scores, result, count, table);
    return hipGetLastError();
}
hipError_t state(const uint16_t* k, const uint16_t* u, const uint16_t* w, const float* g,
                 uint16_t* h, uint16_t* v_new, float* state, unsigned count,
                 const unsigned char* table, hipStream_t stream) {
    hipLaunchKernelGGL(state_kernel, dim3(128u / state_columns, 32u), dim3(threads),
        0u, stream, k, u, w, g, h, v_new, state, count, table);
    return hipGetLastError();
}
}
