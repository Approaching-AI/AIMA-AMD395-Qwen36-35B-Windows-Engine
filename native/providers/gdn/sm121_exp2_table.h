#ifndef QRT_SM121_EXP2_TABLE_H
#define QRT_SM121_EXP2_TABLE_H
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_EXP2_HD __host__ __device__
#else
#define QRT_EXP2_HD
#endif

namespace qrt_sm121_exp2 {
constexpr uint32_t begin = 0x2f800000u, end = 0x43180000u;
constexpr uint32_t pages = (end - begin) / 256u;
constexpr uint64_t payload_bytes = 172901632u;
constexpr uint64_t payload_start = 48u + uint64_t(pages) * 8u;
constexpr uint64_t table_bytes = payload_start + payload_bytes;
constexpr unsigned char sha256[32] = {
    0xf4,0x90,0x94,0x0d,0xf2,0xbd,0x80,0x42,0x11,0x59,0xa9,0x64,0x24,0xc3,0xe9,0x22,
    0x33,0x0b,0x7c,0xa1,0x20,0xd5,0xae,0x7b,0x62,0x9a,0x97,0x3b,0x91,0x83,0x73,0x0b};

QRT_EXP2_HD inline uint32_t bits(float value) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __float_as_uint(value);
#else
    uint32_t result; std::memcpy(&result, &value, 4); return result;
#endif
}
QRT_EXP2_HD inline float value(uint32_t bits) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float result; std::memcpy(&result, &bits, 4); return result;
#endif
}
// Only the validated, SHA-bound artifact may reach this lookup. The table
// enumerates every nonpositive float, independent of any model or prompt.
// Parallel prefix scans can round equal negative sums in opposite directions.
// Their tiny positive differences must retain exp2(x) == 1 instead of becoming
// NaNs that contaminate the subsequent triangular inverse.
QRT_EXP2_HD inline float evaluate(const unsigned char* table, float argument) {
    const uint32_t input = bits(argument), magnitude = input & 0x7fffffffu;
    if (magnitude < begin) return 1.0f;
    if (magnitude > 0x7f800000u || !(input >> 31u)) return value(0x7fc00000u);
    if (magnitude >= end) return 0.0f;
    const uint32_t relative = magnitude - begin;
    const auto* entry = reinterpret_cast<const uint32_t*>(table + 48u) + (relative >> 8u) * 2u;
    const uint32_t base = entry[0], tag = entry[1], kind = tag >> 30u;
    const auto* payload = table + payload_start + (tag & 0x3fffffffu);
    const uint32_t cell = relative & 255u;
    const uint32_t delta = kind == 0u ? 0u : kind == 1u ? payload[cell]
        : kind == 2u ? reinterpret_cast<const uint16_t*>(payload)[cell]
        : reinterpret_cast<const uint32_t*>(payload)[cell];
    return value(base + delta);
}
inline bool valid_layout(const unsigned char* data, size_t bytes) {
    if (!data || bytes != table_bytes || std::memcmp(data, "QEX2TBL1", 8) != 0) return false;
    uint32_t header[8]; uint64_t payload;
    std::memcpy(header, data + 8, sizeof(header)); std::memcpy(&payload, data + 40, sizeof(payload));
    const uint32_t expected[] = {1,8,begin,end,0x3f800000u,0,pages,1};
    if (std::memcmp(header, expected, sizeof(header)) != 0 || payload != payload_bytes) return false;
    for (uint32_t page = 0; page < pages; ++page) {
        uint32_t pair[2]; std::memcpy(pair, data + 48u + uint64_t(page) * 8u, 8);
        const uint32_t kind = pair[1] >> 30u, offset = pair[1] & 0x3fffffffu;
        const uint32_t widths[] = {0,1,2,4};
        if (pair[0] > 0x3f800000u || (!kind && offset) ||
            uint64_t(offset) + 256u * widths[kind] > payload_bytes || (kind && offset % widths[kind])) return false;
    }
    return true;
}
}
#undef QRT_EXP2_HD
#endif
