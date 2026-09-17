#pragma once
#include <hip/hip_runtime.h>
#include "blackwell_accumulator.h"
#include "linear_input_contract.h"

// Isolated complete convolution -> normalized Q/K and compact V candidate.
// Every tap retains its BF16 product round, then the original ascending FP32
// addition. Q/K retain eight contiguous dimensions per lane and XOR8,4,2,1.
// The input span includes the causal halo; outputs describe only [first,end).
// No production caller selects this component yet.
namespace qrt_linear_input_preparation {
using qrt_fla_blackwell::from_bf16;
using qrt_fla_blackwell::to_bf16;
// Convolution uses the whole provider's plain RNE bit conversion. It differs
// from the FLA NaN-preserving conversion, which remains in normalization.
__device__ __forceinline__ uint16_t projection_round(float value) {
    const uint32_t bits = __float_as_uint(value);
    return uint16_t((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

__device__ __forceinline__ float convolution(const Inputs& in, unsigned token, unsigned feature) {
    float sum = 0.0f;
#pragma unroll
    for (unsigned tap = 0u; tap < 4u; ++tap) {
        if (token + tap >= 3u) {
            const uint16_t x = projection_round(in.projected[size_t(token + tap - 3u) * features + feature]);
            const float product = from_bf16(projection_round(from_bf16(x) * from_bf16(in.weights[feature * 4u + tap])));
            sum += product;
        }
    }
    return from_bf16(qrt_sm121_silu::evaluate(in.silu, sum));
}

template<bool Capture>
__global__ void qk_kernel(Inputs in, Outputs out, unsigned first, unsigned count) {
    const unsigned row = blockIdx.x * 16u + threadIdx.x / 16u;
    const unsigned lane = threadIdx.x % 16u;
    if (row >= count * 16u) return;
    const unsigned local = row / 16u, head = row % 16u, token = first + local;
    float q[8], k[8];
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) {
        const unsigned feature = head * 128u + lane * 8u + i;
        q[i] = convolution(in, token, feature);
        k[i] = convolution(in, token, feature + qk_features);
        if constexpr (Capture) {
            out.raw[size_t(local) * features + feature] = q[i];
            out.raw[size_t(local) * features + feature + qk_features] = k[i];
        }
    }
    float qs = fmaf(q[0], q[0], q[1] * q[1]);
    float ks = fmaf(k[0], k[0], k[1] * k[1]);
#pragma unroll
    for (unsigned i = 2u; i < 8u; ++i) { qs = fmaf(q[i], q[i], qs); ks = fmaf(k[i], k[i], ks); }
#pragma unroll
    for (unsigned delta = 8u; delta; delta >>= 1u) {
        qs += __shfl_xor(qs, delta, 16); ks += __shfl_xor(ks, delta, 16);
    }
    const float qr = qrt_sm121_rsqrt::evaluate(in.rsqrt, qs + 1.0e-6f);
    const float kr = qrt_sm121_rsqrt::evaluate(in.rsqrt, ks + 1.0e-6f);
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) {
        const size_t offset = size_t(row) * 128u + lane * 8u + i;
        out.q[offset] = to_bf16(q[i] * qr); out.k[offset] = to_bf16(k[i] * kr);
    }
}

template<bool Capture>
__global__ void v_kernel(Inputs in, Outputs out, unsigned first, unsigned count) {
    const unsigned feature = blockIdx.x * threads + threadIdx.x;
    const unsigned start = blockIdx.y * 4u;
    // Four adjacent tokens reuse one weight tuple and the three-token halo.
    float weights[4], history[3];
#pragma unroll
    for (unsigned i = 0u; i < 4u; ++i) weights[i] = from_bf16(in.weights[(feature + 4096u) * 4u + i]);
#pragma unroll
    for (unsigned i = 0u; i < 3u; ++i) history[i] = first + start + i >= 3u
        ? from_bf16(projection_round(in.projected[size_t(first + start + i - 3u) * features + feature + 4096u])) : 0.0f;
#pragma unroll
    for (unsigned step = 0u; step < 4u; ++step) if (start + step < count) {
        const unsigned token = first + start + step;
        const float current = from_bf16(projection_round(in.projected[size_t(token) * features + feature + 4096u]));
        float sum = 0.0f;
        // Missing taps are skipped, including special weights and signed zero.
#pragma unroll
        for (unsigned tap = 0u; tap < 3u; ++tap)
            if (token + tap >= 3u) sum += from_bf16(projection_round(history[tap] * weights[tap]));
        sum += from_bf16(projection_round(current * weights[3]));
        const uint16_t value = qrt_sm121_silu::evaluate(in.silu, sum);
        out.v[size_t(start + step) * v_features + feature] = value;
        if constexpr (Capture) out.raw[size_t(start + step) * features + feature + 4096u] = from_bf16(value);
        history[0] = history[1]; history[1] = history[2]; history[2] = current;
    }
}

inline hipError_t launch(const Inputs& in, const Outputs& out, unsigned first, unsigned count, hipStream_t stream) {
    if (!valid(in, out, first, count)) return hipErrorInvalidValue;
#define QRT_LINEAR_INPUT_LAUNCH(capture) \
    hipLaunchKernelGGL(qk_kernel<capture>, dim3(count), dim3(threads), 0u, stream, in, out, first, count); \
    { const auto status = hipGetLastError(); if (status != hipSuccess) return status; } \
    hipLaunchKernelGGL(v_kernel<capture>, dim3(v_features / threads, (count + 3u) / 4u), dim3(threads), 0u, stream, in, out, first, count)
    if (out.raw) { QRT_LINEAR_INPUT_LAUNCH(true); } else { QRT_LINEAR_INPUT_LAUNCH(false); }
#undef QRT_LINEAR_INPUT_LAUNCH
    return hipGetLastError();
}
} // namespace qrt_linear_input_preparation
