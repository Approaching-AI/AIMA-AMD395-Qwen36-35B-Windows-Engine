#ifndef QRT_SM121_DOT_CERTIFICATE_H
#define QRT_SM121_DOT_CERTIFICATE_H
#include "q1_moe_hawkeye_bf16_accumulator.h"
#ifndef QRT_SM121_CERTIFIED_DOT_TILES
#define QRT_SM121_CERTIFIED_DOT_TILES 0
#endif
static_assert(QRT_SM121_CERTIFIED_DOT_TILES == 0 || QRT_SM121_CERTIFIED_DOT_TILES == 1);
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_DOT_CERT_INLINE __host__ __device__ __forceinline__
#else
#define QRT_DOT_CERT_INLINE inline
#endif
namespace qrt_sm121_dot_certificate {
struct Stats { unsigned attempted = 0u, accepted = 0u; };
QRT_DOT_CERT_INLINE bool canonical_normal(qrt_q1_moe_hawkeye::Value carry) {
    return carry.exponent >= -126 && (carry.significand - 0x00800000u) < 0x00800000u;
}
// With unchanged carry sign/exponent, trunc(4*s+p)/4 is exactly
// s+floor(p/4) for positive s, or s+ceil(p/4) for negative s. A product-only
// K16 sum fits signed32. Unsigned prefix arithmetic also remains defined
// after a rejected prefix; no later prefix can reinstate a failed certificate.
QRT_DOT_CERT_INLINE bool advance(uint32_t& signed_significand, uint32_t product_sum, bool negative) {
    uint32_t increment = (product_sum >> 2u) | ((product_sum & 0x80000000u) ? 0xc0000000u : 0u);
    if (negative && (product_sum & 3u)) ++increment;
    signed_significand += increment;
    const uint32_t magnitude = negative ? 0u - signed_significand : signed_significand;
    return (magnitude - 0x00800000u) < 0x00800000u;
}
}
#undef QRT_DOT_CERT_INLINE
#endif
