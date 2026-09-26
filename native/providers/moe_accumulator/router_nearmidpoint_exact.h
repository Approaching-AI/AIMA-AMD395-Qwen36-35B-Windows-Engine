#ifndef QRT_ROUTER_NEARMIDPOINT_EXACT_H
#define QRT_ROUTER_NEARMIDPOINT_EXACT_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_ROUTER_HOST_DEVICE __host__ __device__
#else
#define QRT_ROUTER_HOST_DEVICE
#endif

namespace qrt_router_nearmidpoint {

struct Decision {
    uint16_t baseline;
    uint16_t selected;
    double exact_sum;
    double midpoint_distance;
    bool recomputed;
};

QRT_ROUTER_HOST_DEVICE inline uint32_t float_bits(float value) {
    union { float f; uint32_t u; } bits;
    bits.f = value;
    return bits.u;
}

QRT_ROUTER_HOST_DEVICE inline float bf16_float(uint16_t value) {
    union { float f; uint32_t u; } bits;
    bits.u = static_cast<uint32_t>(value) << 16u;
    return bits.f;
}

QRT_ROUTER_HOST_DEVICE inline uint16_t round_bf16(float value) {
    const uint32_t bits = float_bits(value);
    if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
        return static_cast<uint16_t>(bits >> 16u);
    }
    return static_cast<uint16_t>(
        (bits + UINT32_C(0x7fff) + ((bits >> 16u) & 1u)) >> 16u
    );
}

QRT_ROUTER_HOST_DEVICE inline bool finite_bf16(uint16_t value) {
    return (value & UINT16_C(0x7f80)) != UINT16_C(0x7f80);
}

// A narrow, optional repair for accumulation ambiguity at a BF16 endpoint.
// The caller limits it to a small row; this helper revisits only a Hawkeye
// F32 result within 64 F32 units of a BF16 midpoint.
QRT_ROUTER_HOST_DEVICE inline Decision select(
    const uint16_t *input,
    const uint16_t *weight,
    size_t elements,
    float hawkeye,
    uint32_t maximum_midpoint_distance = 64u,
    double maximum_exact_margin = 1.0e-7
) {
    const uint16_t baseline = round_bf16(hawkeye);
    Decision result{baseline, baseline, 0.0, 0.0, false};
    const uint32_t bits = float_bits(hawkeye);
    if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
        return result;
    }
    const uint32_t low = bits & UINT32_C(0xffff);
    const uint32_t midpoint_distance = low >= UINT32_C(0x8000)
        ? low - UINT32_C(0x8000)
        : UINT32_C(0x8000) - low;
    if (midpoint_distance > maximum_midpoint_distance) {
        return result;
    }

    double sum = 0.0;
    for (size_t index = 0u; index < elements; ++index) {
        sum += static_cast<double>(bf16_float(input[index])) *
               static_cast<double>(bf16_float(weight[index]));
    }
    result.exact_sum = sum;
    result.recomputed = true;
    uint16_t chosen = round_bf16(static_cast<float>(sum));
    double chosen_distance = fabs(sum - static_cast<double>(bf16_float(chosen)));
    for (int offset = -1; offset <= 1; offset += 2) {
        const int next = static_cast<int>(chosen) + offset;
        if (next < 0 || next > 65535 || !finite_bf16(static_cast<uint16_t>(next))) {
            continue;
        }
        const double distance = fabs(sum - static_cast<double>(bf16_float(static_cast<uint16_t>(next))));
        if (distance < chosen_distance ||
            (distance == chosen_distance && (next & 1) == 0)) {
            chosen = static_cast<uint16_t>(next);
            chosen_distance = distance;
        }
    }

    double margin = 1.0e300;
    for (int offset = -1; offset <= 1; offset += 2) {
        const int next = static_cast<int>(chosen) + offset;
        if (next < 0 || next > 65535 || !finite_bf16(static_cast<uint16_t>(next))) {
            continue;
        }
        const double midpoint = 0.5 * (
            static_cast<double>(bf16_float(chosen)) +
            static_cast<double>(bf16_float(static_cast<uint16_t>(next)))
        );
        const double distance = fabs(sum - midpoint);
        if (distance < margin) {
            margin = distance;
        }
    }
    result.midpoint_distance = margin;
    if (margin <= maximum_exact_margin) {
        result.selected = chosen;
    }
    return result;
}

}  // namespace qrt_router_nearmidpoint

#undef QRT_ROUTER_HOST_DEVICE

#endif
