#include "blackwell_state.h"
#include "blackwell_accumulator.h"

namespace qrt_fla_blackwell_state {
namespace {
using namespace qrt_fla_blackwell;

__device__ __forceinline__ float gate_exp(float value) {
    const float scaled = value * 1.4426950408889634074f;
    float result;
    asm("v_exp_f32 %0, %1" : "=v"(result) : "v"(scaled));
    return result;
}
__device__ __forceinline__ float finish(qrt_q1_moe_hawkeye::Value accumulator) {
    accumulator = qrt_q1_moe_hawkeye::group_sum<26, kZeroExponent>(&accumulator, 1);
    return qrt_q1_moe_hawkeye::value_to_float(accumulator);
}

// A subgroup owns one (token,head,value) projection. Only token zero writes
// the BF16 chunk checkpoint; other tokens independently round the F32 seed.
// No subgroup consumes another subgroup's checkpoint output.
__global__ void project_kernel(const uint16_t* w, const uint16_t* u, const float* g,
                               const float* initial, uint16_t* h, uint16_t* v_new,
                               uint16_t* residual, unsigned count) {
    const unsigned cell = blockIdx.x * (kThreads / kGroup) + threadIdx.x / kGroup;
    const unsigned token = cell / 128u, v = cell % 128u, head = blockIdx.y, lane = threadIdx.x % kGroup;
    if (token >= 64u || head >= 32u) return;
    const unsigned output = (token * 32u + head) * 128u + v;
    if (token >= count) { if (lane == 0) residual[output] = 0; return; }
    qrt_q1_moe_hawkeye::Value accumulator{0u, kZeroExponent, false};
    for (unsigned base = 0; base < 128u; base += kGroup) {
        const unsigned key = base + lane, state_index = (head * 128u + v) * 128u + key;
        const uint16_t state = to_bf16(initial[state_index]);
        if (token == 0) h[state_index] = state;
        accumulator = accumulate(accumulator, w[(token * 32u + head) * 128u + key], state, lane);
    }
    if (lane == 0) {
        const float current = from_bf16(u[output]) - finish(accumulator);
        v_new[output] = to_bf16(current);
        const float gate = gate_exp(g[(count - 1u) * 32u + head] - g[token * 32u + head]);
        residual[output] = to_bf16(current * gate);
    }
}

// A subgroup owns one final (head,value,key) state cell. The K64 dot starts
// at zero, then an explicit IEEE FMA applies decay to the untouched F32 seed.
__global__ void update_kernel(const uint16_t* k, const uint16_t* residual, const float* g,
                              const float* initial, float* final, unsigned count) {
    const unsigned cell = blockIdx.x * (kThreads / kGroup) + threadIdx.x / kGroup;
    const unsigned v = cell / 128u, key = cell % 128u, head = blockIdx.y, lane = threadIdx.x % kGroup;
    if (v >= 128u || head >= 32u) return;
    qrt_q1_moe_hawkeye::Value accumulator{0u, kZeroExponent, false};
    for (unsigned base = 0; base < 64u; base += kGroup) {
        const unsigned token = base + lane;
        const uint16_t left = token < count ? k[(token * 16u + head / 2u) * 128u + key] : 0;
        const uint16_t right = token < count ? residual[(token * 32u + head) * 128u + v] : 0;
        accumulator = accumulate(accumulator, left, right, lane);
    }
    if (lane == 0) {
        const unsigned index = (head * 128u + v) * 128u + key;
        final[index] = fmaf(initial[index], gate_exp(g[(count - 1u) * 32u + head]), finish(accumulator));
    }
}
}

hipError_t project(const uint16_t* w, const uint16_t* u, const float* g,
                   const float* initial, uint16_t* h, uint16_t* v_new,
                   uint16_t* residual, unsigned count, hipStream_t stream) {
    if (!valid_project(w, u, g, initial, h, v_new, residual, count)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(project_kernel, dim3(512u, 32u), dim3(256u), 0, stream,
                       w, u, g, initial, h, v_new, residual, count);
    return hipGetLastError();
}
hipError_t update(const uint16_t* k, const uint16_t* residual, const float* g,
                  const float* initial, float* final, unsigned count, hipStream_t stream) {
    if (!valid_update(k, residual, g, initial, final, count)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(update_kernel, dim3(1024u, 32u), dim3(256u), 0, stream,
                       k, residual, g, initial, final, count);
    return hipGetLastError();
}
}
