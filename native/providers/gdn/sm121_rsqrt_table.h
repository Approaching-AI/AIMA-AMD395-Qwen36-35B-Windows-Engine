#ifndef QRT_SM121_RSQRT_TABLE_H
#define QRT_SM121_RSQRT_TABLE_H
#include "sm121_exp2_table.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_RSQRT_HD __host__ __device__
#else
#define QRT_RSQRT_HD
#endif

namespace qrt_sm121_rsqrt {
constexpr uint32_t begin = 0x3f800000u, end = 0x40800000u;
constexpr uint32_t pages = (end - begin) / 256u;
constexpr uint64_t payload_bytes = 16777472u;
constexpr uint64_t payload_start = 48u + uint64_t(pages) * 8u;
constexpr uint64_t table_bytes = payload_start + payload_bytes;
constexpr unsigned char sha256[32] = {
    0xca,0x02,0x30,0xa8,0xba,0xe9,0xbd,0x10,0x1a,0xc3,0x68,0xf8,0xa7,0xc3,0x40,0x07,
    0xcd,0xa6,0x37,0xdf,0x65,0x13,0xdb,0xe4,0x71,0x42,0x53,0xc3,0x6b,0x94,0x08,0x50};

// The [1,4) table and this exponent rule were checked against all 2,139,095,041
// nonnegative FP32 encodings through +infinity on SM121. No model inputs.
QRT_RSQRT_HD inline float evaluate(const unsigned char* table, float argument) {
    const uint32_t input = qrt_sm121_exp2::bits(argument);
    if (input > 0x7f800000u) return qrt_sm121_exp2::value(0x7fc00000u);
    if (input < 0x00800000u) return qrt_sm121_exp2::value(0x7f800000u);
    if (input == 0x7f800000u) return 0.0f;
    const int exponent = static_cast<int>(input >> 23u) - 127;
    const int scale = exponent >= 0 ? exponent / 2 : -((-exponent + 1) / 2);
    const uint32_t relative = (input & 0x7fffffu) | (uint32_t(exponent & 1) << 23u);
    const auto* entry = reinterpret_cast<const uint32_t*>(table + 48u) + (relative >> 8u) * 2u;
    const uint32_t base = entry[0], tag = entry[1], kind = tag >> 30u;
    const auto* payload = table + payload_start + (tag & 0x3fffffffu);
    const uint32_t cell = relative & 255u;
    const uint32_t delta = kind == 0u ? 0u : kind == 1u ? payload[cell]
        : kind == 2u ? reinterpret_cast<const uint16_t*>(payload)[cell]
        : reinterpret_cast<const uint32_t*>(payload)[cell];
    return qrt_sm121_exp2::value(static_cast<uint32_t>(static_cast<int32_t>(base + delta) - scale * 0x800000));
}
inline bool valid_layout(const unsigned char* data, size_t bytes) {
    if (!data || bytes != table_bytes || std::memcmp(data, "QRSQTBL1", 8) != 0) return false;
    uint32_t header[8]; uint64_t payload;
    std::memcpy(header, data + 8, sizeof(header)); std::memcpy(&payload, data + 40, sizeof(payload));
    const uint32_t expected[] = {1,8,begin,end,0x7f800000u,0,pages,1};
    if (std::memcmp(header, expected, sizeof(header)) != 0 || payload != payload_bytes) return false;
    for (uint32_t page = 0; page < pages; ++page) {
        uint32_t pair[2]; std::memcpy(pair, data + 48u + uint64_t(page) * 8u, 8);
        const uint32_t kind = pair[1] >> 30u, offset = pair[1] & 0x3fffffffu;
        const uint32_t widths[] = {0,1,2,4};
        if (pair[0] < 0x3f000000u || pair[0] > 0x3f800000u || (!kind && offset) ||
            uint64_t(offset) + 256u * widths[kind] > payload_bytes || (kind && offset % widths[kind])) return false;
    }
    return true;
}
}
#undef QRT_RSQRT_HD
#endif
