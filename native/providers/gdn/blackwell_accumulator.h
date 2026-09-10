#ifndef QRT_FLA_BLACKWELL_ACCUMULATOR_H
#define QRT_FLA_BLACKWELL_ACCUMULATOR_H

#include "../moe_accumulator/sm121_wave16.h"

// Reuse the characterized Blackwell K16/26-bit accumulator. Each 16-lane
// subgroup owns one dot cell; lane zero carries its K16 groups without
// splitting the K128 reduction into separately rounded K64 partials.
namespace qrt_fla_blackwell {
constexpr unsigned int kGroup = 16;
constexpr int16_t kZeroExponent = -133;
constexpr unsigned int kThreads = 256;
constexpr unsigned int kChunk = 64;

using qrt_sm121_wave16::accumulate;

__device__ __forceinline__ float from_bf16(uint16_t bits) {
    union { uint32_t u; float f; } value;
    value.u = uint32_t(bits) << 16;
    return value.f;
}

__device__ __forceinline__ uint16_t to_bf16(float input) {
    union { float f; uint32_t u; } value;
    value.f = input;
    if ((value.u & 0x7fffffffU) > 0x7f800000U) return uint16_t((value.u | 0x00400000U) >> 16);
    return uint16_t((value.u + 0x7fffU + ((value.u >> 16) & 1U)) >> 16);
}


}  // namespace qrt_fla_blackwell
#endif
