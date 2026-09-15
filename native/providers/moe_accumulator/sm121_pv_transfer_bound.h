#pragma once
#include "sm121_coarse_projection_bound.h"

#if defined(__HIPCC__)
#define QRT_PV_TRANSFER_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PV_TRANSFER_INLINE inline
#endif

// Component envelope for independently evaluated PV chunks. It retains the
// existing conditional 2^-19 zero-C WMMA assumption and finite operand domain.
// The canonical comparison includes every K16 alignment/FP32 endpoint and
// every original K32 rescale, even though the producer starts its chunk at zero.
namespace qrt_sm121_pv_transfer_bound {
namespace coarse = qrt_sm121_coarse_projection_bound;
namespace scalar = qrt_sm121_pv_bound;
using State = coarse::State;
QRT_PV_TRANSFER_INLINE float product(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    float result;
    asm("v_mul_f32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
#else
    volatile float result = a * b; return result;
#endif
}
QRT_PV_TRANSFER_INLINE bool alpha_valid(float alpha) {
    return scalar::bits(alpha) <= 0x3f800000u;
}
template<unsigned Groups>
QRT_PV_TRANSFER_INLINE State advance(State before, float entry_scaled,
    float partial, float absolute, bool alpha_ok, bool entry_erased) {
    static_assert(Groups == 8u || Groups == 16u || Groups == 32u);
    State result{entry_scaled + partial, scalar::infinity()};
    if (!alpha_ok || !scalar::finite(before.center) ||
        !scalar::finite(before.error) || before.error < 0.0f ||
        !scalar::finite(entry_scaled) || !scalar::finite(partial) ||
        !scalar::finite(absolute) || absolute < 0.0f) return result;
    constexpr unsigned tiles = Groups / 2u;
    constexpr float epsilon = 0x1p-19f, floor = float(Groups) * 0x1p-118f;
    // The unweighted positive sum bounds every partial prefix because all
    // alphas are in [0,1]. Allow both zero-C error and explicit FP32 additions.
    constexpr float inflation = 1.0f / (1.0f - epsilon - float(Groups) * 0x1p-23f);
    const float positive = scalar::upper(scalar::upper(absolute * inflation) + floor);
    const float native_magnitude = scalar::upper(positive * inflation);
    const float native_error = scalar::upper(scalar::upper(positive * epsilon) +
        scalar::upper(float(Groups + tiles) * coarse::unit(native_magnitude, 23u)) + floor);
    const float entry_magnitude = scalar::upper(scalar::absolute(before.center) + before.error);
    const float magnitude = scalar::upper(entry_magnitude + positive);
    // Signed alignment of 16 products plus the carry loses at most17 units;
    // its FP32 endpoint loses at most4 more. A rescale contributes one FP32
    // unit. The entry-only rescale chain also rounds and is counted separately.
    const float canonical_error = scalar::upper(
        scalar::upper(float(21u * Groups) * coarse::unit(magnitude, 25u)) +
        scalar::upper(float(tiles) * coarse::unit(magnitude, 23u)) + floor);
    const float entry_error = scalar::upper(float(tiles) *
        coarse::unit(scalar::absolute(before.center), 23u) + floor);
    const float addition_error = coarse::unit(
        scalar::upper(scalar::absolute(entry_scaled) + scalar::absolute(partial)), 23u);
    const float inherited = entry_erased ? 0.0f : before.error;
    result.error = scalar::upper(scalar::upper(inherited + native_error) +
        scalar::upper(scalar::upper(canonical_error + entry_error) + addition_error));
    return result;
}
} // namespace qrt_sm121_pv_transfer_bound
#undef QRT_PV_TRANSFER_INLINE
