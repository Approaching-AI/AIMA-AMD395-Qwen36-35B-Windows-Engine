#ifndef QRT_SM121_PAIRED_PRODUCTS_H
#define QRT_SM121_PAIRED_PRODUCTS_H
#include <cstdint>
#ifndef QRT_SM121_PAIRED_PRODUCTS
#define QRT_SM121_PAIRED_PRODUCTS 0
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PAIRED_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PAIRED_INLINE inline
#endif

namespace qrt_sm121_paired_products {
struct Products { uint32_t low, high; };

// Two BF16 operands occupy the low/high halfwords of each argument. Exponent
// and significand preparation proceeds independently in both halfwords. The
// integer packed multiply is exact: each unsigned product is at most 65025.
// The result matches pack_product(multiply_bf16(..., -133)), including zero,
// subnormal, sign and exceptional input encodings used by that reference.
QRT_PAIRED_INLINE Products multiply(uint32_t left, uint32_t right) {
    const uint32_t le = (left >> 7u) & 0x00ff00ffu;
    const uint32_t re = (right >> 7u) & 0x00ff00ffu;
    const uint32_t ln = ((le + 0x00ff00ffu) & 0x01000100u) >> 8u;
    const uint32_t rn = ((re + 0x00ff00ffu) & 0x01000100u) >> 8u;
    const uint32_t ls = (left & 0x007f007fu) | (ln << 7u);
    const uint32_t rs = (right & 0x007f007fu) | (rn << 7u);
    uint32_t significands;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    asm("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(significands) : "v"(ls), "v"(rs));
#else
    significands = ((ls & 65535u) * (rs & 65535u)) |
        (((ls >> 16u) * (rs >> 16u)) << 16u);
#endif
    // The masked add cannot carry across a halfword, even for 65535 products.
    const uint32_t nonzero = ((significands | ((significands & 0x7fff7fffu) + 0x7fff7fffu)) &
        0x80008000u) >> 15u;
    const uint32_t mask = nonzero * 65535u;
    const uint32_t exponent_sum = le + re + 0x00020002u - ln - rn;
    const uint32_t exponents = (exponent_sum & mask) | (0x00790079u & ~mask);
    const uint32_t signs = left ^ right;
    return {(significands & 65535u) | (exponents << 16u) | ((signs & 0x8000u) << 16u),
            (significands >> 16u) | (exponents & 0x01ff0000u) | (signs & 0x80000000u)};
}
} // namespace qrt_sm121_paired_products
#undef QRT_PAIRED_INLINE
#endif
