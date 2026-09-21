#pragma once
#include "sm121_narrow_f32_carry.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PRODUCT_CARRY_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PRODUCT_CARRY_INLINE inline
#endif

// Isolated component; no runtime dispatch. Admit the maximum product exponent
// of each ordered K16 group instead of requiring both operands in [-32,32].
// The carry and modulo reduction remain those of the narrow implementation.
namespace qrt_sm121_product_f32_carry {
namespace alignment = qrt_sm121_float_alignment;
namespace narrow = qrt_sm121_narrow_f32_carry;

struct Group {
    alignment::Group aligned{};
    int normal_maximum = -254;
    int subnormal_maximum = -254;
    bool finite = true;
    bool nonzero = false;

    // Every slot must be set once before accumulation. A nonzero product
    // involving a subnormal operand is omitted only if admitted() proves
    // that the original integer alignment also truncates it to zero.
    QRT_PRODUCT_CARRY_INLINE void set(unsigned i, uint16_t left, uint16_t right) {
        const unsigned a = (left >> 7u) & 255u, b = (right >> 7u) & 255u;
        finite = finite && a != 255u && b != 255u;
        if (!i) aligned.first_negative = ((left ^ right) & 0x8000u) != 0u;
        aligned.products[i] = 0.0f;
        if (!(left & 0x7fffu) || !(right & 0x7fffu)) return;
        nonzero = true;
        if (a && b && a != 255u && b != 255u) {
            const int exponent = int(a) + int(b) - 254;
            normal_maximum = exponent > normal_maximum ? exponent : normal_maximum;
            aligned.maximum = exponent > aligned.maximum ? exponent : aligned.maximum;
            aligned.products[i] = alignment::from_bits(uint32_t(left) << 16u) *
                                  alignment::from_bits(uint32_t(right) << 16u);
        } else {
            const int exponent = int(a ? a : 1u) + int(b ? b : 1u) - 254;
            subnormal_maximum = exponent > subnormal_maximum ? exponent : subnormal_maximum;
        }
    }

    QRT_PRODUCT_CARRY_INLINE bool admitted() const {
        // An operand pair has magnitude < 4*2^exponent, including a
        // subnormal operand when its effective exponent is taken as -126.
        // A gap of 27 therefore gives |product*2^(25-maximum)| < 1.
        // The original shifted integer product is zero even without FTZ.
        return finite && (!nonzero || (normal_maximum >= -64 &&
            normal_maximum <= 64 && subnormal_maximum <= normal_maximum - 27));
    }
};

QRT_PRODUCT_CARRY_INLINE bool accumulate(float carry, const Group& group, float* output) {
    const uint32_t absolute = qrt_sm121_f32_carry::bits(carry) & 0x7fffffffu;
    const int exponent = int(absolute >> 23u) - 127;
    if (!group.admitted() || (absolute && (exponent < -89 || exponent > 74))) return false;
    // For a dot starting at zero, a nonzero group has maximum >= -64,
    // hence its smallest aligned quantum is 2^-89. All-zero groups preserve
    // the endpoint, and <=256 finite products keep absolute growth <2^74.
    // A normal product that flushes during multiplication is smaller than
    // this quantum and also contributes zero in the original alignment.
    *output = narrow::accumulate(carry, group.aligned);
    return true;
}

template<unsigned Width>
QRT_PRODUCT_CARRY_INLINE bool dot(const uint16_t* left, const uint16_t* right, float* output) {
    static_assert(Width == 16u || Width == 64u || Width == 128u || Width == 256u);
    float carry = 0.0f;
    for (unsigned base = 0u; base < Width; base += 16u) {
        Group group;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (unsigned i = 0u; i < 16u; ++i) group.set(i, left[base + i], right[base + i]);
        float next;
        if (!accumulate(carry, group, &next)) return false;
        carry = next;
    }
    *output = carry;
    return true;
}
}  // namespace qrt_sm121_product_f32_carry
#undef QRT_PRODUCT_CARRY_INLINE
