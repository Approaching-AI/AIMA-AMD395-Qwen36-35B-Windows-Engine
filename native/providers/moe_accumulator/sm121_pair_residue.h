#pragma once
#include <cstdint>

#if defined(__HIPCC__)
#define QRT_PAIR_RESIDUE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PAIR_RESIDUE_INLINE inline
#endif

// Isolated arithmetic candidate. Runtime dispatch does not use this header.
// Each core integer is a signed BF16 significand shifted by at most seven.
// It is exactly representable in FP16 and its magnitude is at most32640.
// A zero-C FP16 DOT2 estimate and exact low-byte integer residue identify
// the full integer sum only under the explicit |estimate-exact|<128 condition.
// Native tests characterize that condition; they do not prove an ISA bound.
namespace qrt_sm121_pair_residue {
constexpr int32_t maximum_core = 255 * 128;
constexpr int32_t maximum_sum = 2 * maximum_core * maximum_core;
static_assert(maximum_sum == 2130739200 && maximum_sum + 1024 < INT32_MAX);
struct Pair { uint32_t half, bytes; };

QRT_PAIR_RESIDUE_INLINE uint16_t encode(unsigned mantissa, unsigned shift, bool negative) {
    return uint16_t((negative ? 0x8000u : 0u) | ((22u + shift) << 10u) | ((mantissa - 128u) << 3u));
}
QRT_PAIR_RESIDUE_INLINE Pair prepare(unsigned m0, unsigned s0, bool n0,
    unsigned m1, unsigned s1, bool n1) {
    const int32_t a = n0 ? -int32_t(m0 << s0) : int32_t(m0 << s0);
    const int32_t b = n1 ? -int32_t(m1 << s1) : int32_t(m1 << s1);
    return {uint32_t(encode(m0,s0,n0)) | (uint32_t(encode(m1,s1,n1)) << 16u),
        (uint32_t(a) & 255u) | ((uint32_t(b) & 255u) << 8u)};
}
QRT_PAIR_RESIDUE_INLINE bool recover(float estimate, uint32_t residue, int32_t* output) {
    if (!output || !(estimate > -float(maximum_sum + 512) && estimate < float(maximum_sum + 512))) return false;
    const int32_t truncated = int32_t(estimate);
    const uint32_t offset = (residue - uint32_t(truncated)) & 255u;
    if (offset == 128u && estimate == float(truncated)) return false;
    int32_t result = truncated + int32_t(offset);
    if (offset > 128u || (offset == 128u && estimate < float(truncated))) result -= 256;
    if (result < -maximum_sum || result > maximum_sum) return false;
    *output = result;
    return true;
}
QRT_PAIR_RESIDUE_INLINE uint32_t aligned(int32_t mathematical, Pair left, Pair right, unsigned shift) {
    // Caller has shift in[0,8]. Each operand's signed low byte preserves
    // product residues modulo2^shift. Convert each residue separately to
    // a signed remainder before shifting the exactly divisible numerator.
    const uint32_t mask = (1u << shift) - 1u;
    const unsigned r0 = ((left.bytes & 255u) * (right.bytes & 255u)) & mask;
    const unsigned r1 = (((left.bytes >> 8u) & 255u) * ((right.bytes >> 8u) & 255u)) & mask;
    const uint32_t signs = left.half ^ right.half;
    const int32_t d0 = int32_t(r0) - ((r0 && (signs & 0x8000u)) ? int32_t(1u << shift) : 0);
    const int32_t d1 = int32_t(r1) - ((r1 && (signs & 0x80000000u)) ? int32_t(1u << shift) : 0);
    const int32_t numerator = mathematical - d0 - d1;
    static_assert((-2 >> 1) == -1, "requires arithmetic signed shift");
    return uint32_t(numerator >> shift);
}
#if defined(__HIPCC__)
__device__ __forceinline__ float estimate(Pair left, Pair right) {
    float value;
    asm("v_dot2_f32_f16 %0, %1, %2, 0" : "=v"(value) : "v"(left.half), "v"(right.half));
    return value;
}
__device__ __forceinline__ uint32_t residue(Pair left, Pair right) {
    uint32_t value;
    asm("v_dot4_i32_iu8 %0, %1, %2, 0" : "=v"(value) : "v"(left.bytes), "v"(right.bytes));
    return value & 255u;
}
#endif
} // namespace qrt_sm121_pair_residue
#undef QRT_PAIR_RESIDUE_INLINE
