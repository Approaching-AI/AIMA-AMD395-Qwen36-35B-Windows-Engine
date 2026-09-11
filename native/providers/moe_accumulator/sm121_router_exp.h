#ifndef QRT_SM121_ROUTER_EXP_H
#define QRT_SM121_ROUTER_EXP_H
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_ROUTER_HD __host__ __device__
#else
#define QRT_ROUTER_HD
#endif
namespace qrt_sm121_router {
QRT_ROUTER_HD inline uint32_t bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
QRT_ROUTER_HD inline float value(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
QRT_ROUTER_HD inline float fma(float a, float b, float c) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}
// CUDA 13.1 expf range reduction and SM121 MUFU.EX2 fraction lookup. This is
// the same arithmetic used by the qualified prefill MoE router. The table
// covers every binary32 fraction, independent of model weights or tokens.
QRT_ROUTER_HD inline float exp(float x, const uint32_t *fraction) {
    float rf = fma(x, value(0x3bbb989du), 0.5f);
    rf = rf < 0.0f ? 0.0f : (rf > 1.0f ? 1.0f : rf);
    uint32_t offset = 0;
    if (rf >= 1.0f) offset = 252;
    else if (rf > 0.0f) {
        const uint32_t u = bits(rf);
        const int shift = 23 - (int((u >> 23u) & 255u) - 127);
        const uint64_t product = uint64_t((u & 0x7fffffu) | 0x800000u) * 252u;
        offset = shift >= 64 ? 0u : uint32_t(product >> shift);
    }
    const float biased = 12582913.0f + float(offset);
    const float integer = biased - 12583039.0f;
    float reduced = fma(x, value(0x3fb8aa3bu), -integer);
    reduced = fma(x, value(0x32a57060u), reduced);
    if (reduced < 0.0f) return 0.0f;
    uint32_t index = 0;
    if (reduced > 0.0f) {
        const uint32_t u = bits(reduced), sig = (u & 0x7fffffu) | 0x800000u;
        const int exponent = int((u >> 23u) & 255u) - 127;
        if (exponent >= 0) index = (sig << exponent) & 0x7fffffu;
        else if (exponent >= -23) index = sig >> -exponent;
    }
    return value(fraction[index]) * value(bits(biased) << 23u);
}
}  // namespace qrt_sm121_router
#undef QRT_ROUTER_HD
#endif
