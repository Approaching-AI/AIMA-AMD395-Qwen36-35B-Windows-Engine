#ifndef QRT_SM121_ATTENTION_RCP_H
#define QRT_SM121_ATTENTION_RCP_H
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_RCP_HD __host__ __device__
#else
#define QRT_RCP_HD
#endif
namespace qrt_sm121_attention_rcp {
constexpr size_t entries = 1u << 23u;
constexpr size_t table_bytes = 32u + entries;
constexpr unsigned char sha256[32] = {
    0xd2,0xe5,0x57,0x54,0x3f,0x6b,0xc5,0x1f,0x51,0x41,0xba,0x64,0x14,0x00,0x0c,0xd8,
    0xed,0x89,0x2e,0x2e,0x91,0x5e,0xda,0x19,0x24,0x5c,0x3c,0xae,0x22,0xc1,0x6b,0x39};
QRT_RCP_HD inline uint32_t bits(float x) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __float_as_uint(x);
#else
    uint32_t y; std::memcpy(&y, &x, 4); return y;
#endif
}
QRT_RCP_HD inline float value(uint32_t x) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __uint_as_float(x);
#else
    float y; std::memcpy(&y, &x, 4); return y;
#endif
}
// Call only with the SHA-validated model-independent artifact. All 8,388,608
// mantissas and exponents 0..18 were exhaustively checked against SM121.
QRT_RCP_HD inline float evaluate(const unsigned char* table, float denominator) {
    const uint32_t input = bits(denominator);
    const int exponent = int((input >> 23u) & 255u) - 127;
    if (!table || (input >> 31u) || exponent < 0 || exponent > 18)
        return value(0x7fc00000u);
    const uint32_t mantissa = input & 0x7fffffu;
    volatile float rn = 1.0f / value(0x3f800000u | mantissa);
    const int32_t delta = reinterpret_cast<const int8_t*>(table + 32u)[mantissa];
    const uint32_t result = bits(rn) + delta - (uint32_t(exponent) << 23u);
    return value(result);
}
inline bool valid_layout(const unsigned char* table, size_t bytes) {
    if (!table || bytes != table_bytes || std::memcmp(table, "QRCPTB01", 8)) return false;
    const uint32_t expected[] = {1u, uint32_t(entries), 0u, 18u, 1u, 0u};
    if (std::memcmp(table + 8u, expected, sizeof(expected))) return false;
    const auto* delta = reinterpret_cast<const int8_t*>(table + 32u);
    for (size_t i = 0; i < entries; ++i) if (delta[i] < -1 || delta[i] > 1) return false;
    return true;
}
}
#undef QRT_RCP_HD
#endif
