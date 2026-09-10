#ifndef QRT_SM121_SHARED_GATE_H
#define QRT_SM121_SHARED_GATE_H

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_SHARED_GATE_HD __host__ __device__
#else
#define QRT_SHARED_GATE_HD
#endif

namespace qrt_sm121_shared_gate {

constexpr unsigned lanes = 16;
constexpr unsigned hidden = 2048;

QRT_SHARED_GATE_HD inline float value(uint16_t bf16) {
    const uint32_t bits = static_cast<uint32_t>(bf16) << 16u;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// Original cuBLAS gemvx: each lane folds columns lane, lane+16, ... in FP32.
// The caller then reduces lane partials at offsets 8, 4, 2, 1, in that order.
// An adjacent tree or an FP64 dot can change the BF16 midpoint decision.
QRT_SHARED_GATE_HD inline float lane_dot(
    const uint16_t *input, const uint16_t *weight, unsigned lane
) {
    float sum = 0.0f;
    for (unsigned column = lane; column < hidden; column += lanes) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
        sum = __fmaf_rn(value(input[column]), value(weight[column]), sum);
#else
        sum = std::fma(value(input[column]), value(weight[column]), sum);
#endif
    }
    return sum;
}

}  // namespace qrt_sm121_shared_gate

#undef QRT_SHARED_GATE_HD
#endif
