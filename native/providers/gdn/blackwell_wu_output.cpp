#include "blackwell_wu_output.h"
#include "blackwell_state.h"
#include "blackwell_accumulator.h"
#include "sm121_exp2_table.h"

namespace qrt_fla_blackwell_aux {
namespace {
using namespace qrt_fla_blackwell;
__device__ __forceinline__ float exponential(float x, const unsigned char* table) {
    return qrt_sm121_exp2::evaluate(table, x * 1.4426950408889634074f);
}
__device__ __forceinline__ float finish(qrt_q1_moe_hawkeye::Value sum) {
    sum = qrt_q1_moe_hawkeye::group_sum<26, kZeroExponent>(&sum, 1);
    return qrt_q1_moe_hawkeye::value_to_float(sum);
}

// One CTA owns all 64 rows of an eight-column W/U tile. Shared capture of
// every V operand completes before any U store, making the production U=V
// alias safe across CTAs: no other CTA reads or writes these V columns.
__global__ void wu_kernel(const uint16_t* k, const uint16_t* v, const uint16_t* beta,
                          const uint16_t* inverse, const float* g, uint16_t* w,
                          uint16_t* u, unsigned count, const unsigned char* table) {
    __shared__ uint16_t keys[64u * 8u], values[64u * 8u];
    const unsigned head = blockIdx.y, column_base = blockIdx.x * 8u;
    const unsigned lane = threadIdx.x % kGroup, group = threadIdx.x / kGroup;
    for (unsigned i = threadIdx.x; i < 64u * 8u; i += kThreads) {
        const unsigned token = i / 8u, column = column_base + i % 8u;
        uint16_t kb = 0, vb = 0;
        if (token < count) {
            const float b = from_bf16(beta[token * 32u + head]);
            const uint16_t k_beta = to_bf16(from_bf16(k[(token * 16u + head / 2u) * 128u + column]) * b);
            kb = to_bf16(from_bf16(k_beta) * exponential(g[token * 32u + head], table));
            vb = to_bf16(from_bf16(v[(token * 32u + head) * 128u + column]) * b);
        }
        keys[i] = kb; values[i] = vb;
    }
    __syncthreads();
    for (unsigned cell = group; cell < count * 8u; cell += kThreads / kGroup) {
        const unsigned token = cell / 8u, column = cell % 8u;
        qrt_q1_moe_hawkeye::Value sw{0u, kZeroExponent, false}, su{0u, kZeroExponent, false};
        for (unsigned base = 0; base < 64u; base += kGroup) {
            const unsigned reduction = base + lane;
            const uint16_t a = reduction < count ? inverse[(token * 32u + head) * 64u + reduction] : 0;
            sw = accumulate(sw, a, keys[reduction * 8u + column], lane);
            su = accumulate(su, a, values[reduction * 8u + column], lane);
        }
        if (lane == 0) {
            const unsigned index = (token * 32u + head) * 128u + column_base + column;
            w[index] = to_bf16(finish(sw)); u[index] = to_bf16(finish(su));
        }
    }
}

__global__ void score_kernel(const uint16_t* q, const uint16_t* k, const float* g,
                             uint16_t* scores, unsigned count, const unsigned char* table) {
    const unsigned cell = blockIdx.x * (kThreads / kGroup) + threadIdx.x / kGroup;
    const unsigned token = cell / 64u, source = cell % 64u, head = blockIdx.y, lane = threadIdx.x % kGroup;
    if (token >= 64u) return;
    const unsigned index = (token * 32u + head) * 64u + source;
    if (token >= count || source > token) { if (lane == 0) scores[index] = 0; return; }
    qrt_q1_moe_hawkeye::Value sum{0u, kZeroExponent, false};
    for (unsigned base = 0; base < 128u; base += kGroup) {
        const unsigned d = base + lane;
        sum = accumulate(sum, q[(token * 16u + head / 2u) * 128u + d],
                         k[(source * 16u + head / 2u) * 128u + d], lane);
    }
    if (lane == 0) scores[index] = to_bf16(finish(sum) * exponential(g[token * 32u + head] - g[source * 32u + head], table));
}

__global__ void output_kernel(const uint16_t* q, const uint16_t* v, const uint16_t* h,
                              const float* g, const uint16_t* scores, float* output,
                              unsigned count, const unsigned char* table) {
    const unsigned cell = blockIdx.x * (kThreads / kGroup) + threadIdx.x / kGroup;
    const unsigned token = cell / 128u, column = cell % 128u, head = blockIdx.y, lane = threadIdx.x % kGroup;
    if (token >= count) return;
    qrt_q1_moe_hawkeye::Value old{0u, kZeroExponent, false}, local{0u, kZeroExponent, false};
    for (unsigned base = 0; base < 128u; base += kGroup) {
        const unsigned key = base + lane;
        old = accumulate(old, q[(token * 16u + head / 2u) * 128u + key], h[(head * 128u + column) * 128u + key], lane);
    }
    for (unsigned base = 0; base < 64u; base += kGroup) {
        const unsigned source = base + lane;
        const uint16_t value = source < count ? v[(source * 32u + head) * 128u + column] : 0;
        local = accumulate(local, scores[(token * 32u + head) * 64u + source], value, lane);
    }
    if (lane == 0) {
        constexpr float scale = 0.08838834764831845f;
        const float prior = finish(old) * exponential(g[token * 32u + head], table);
        // The reference rounds prior * scale before fusing the local term.
        // Reversing these operands changes BF16 cells near cancellation.
        output[(token * 32u + head) * 128u + column] = from_bf16(to_bf16(fmaf(finish(local), scale, prior * scale)));
    }
}
}

hipError_t recompute_wu(const uint16_t* k, const uint16_t* v, const uint16_t* beta,
                         const uint16_t* inverse, const float* g, uint16_t* w,
                         uint16_t* u, unsigned count, hipStream_t stream) {
    const auto* table = qrt_fla_blackwell_state::exp2_table_device();
    if (!table || !valid_wu(k, v, beta, inverse, g, w, u, count)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(wu_kernel, dim3(16u, 32u), dim3(256u), 0, stream, k, v, beta, inverse, g, w, u, count, table);
    return hipGetLastError();
}
hipError_t output_scores(const uint16_t* q, const uint16_t* k, const float* g,
                          uint16_t* scores, unsigned count, hipStream_t stream) {
    const auto* table = qrt_fla_blackwell_state::exp2_table_device();
    if (!table || !q || !k || !g || !scores || !count || count > 64 || scores == q || scores == k || static_cast<void*>(scores) == g) return hipErrorInvalidValue;
    hipLaunchKernelGGL(score_kernel, dim3(256u, 32u), dim3(256u), 0, stream, q, k, g, scores, count, table);
    return hipGetLastError();
}
hipError_t output_values(const uint16_t* q, const uint16_t* v, const uint16_t* h,
                          const float* g, const uint16_t* scores, float* output,
                          unsigned count, hipStream_t stream) {
    const auto* table = qrt_fla_blackwell_state::exp2_table_device();
    if (!table || !q || !v || !h || !g || !scores || !output || !count || count > 64) return hipErrorInvalidValue;
    const void* inputs[] = {q, v, h, g, scores};
    for (const void* input : inputs) if (static_cast<void*>(output) == input) return hipErrorInvalidValue;
    hipLaunchKernelGGL(output_kernel, dim3(512u, 32u), dim3(256u), 0, stream, q, v, h, g, scores, output, count, table);
    return hipGetLastError();
}
}
