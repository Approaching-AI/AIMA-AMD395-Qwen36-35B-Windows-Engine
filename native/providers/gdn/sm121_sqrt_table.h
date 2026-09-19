#ifndef QRT_SM121_SQRT_TABLE_H
#define QRT_SM121_SQRT_TABLE_H
#include "sm121_exp2_table.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_SQRT_HD __host__ __device__
#else
#define QRT_SQRT_HD
#endif
namespace qrt_sm121_sqrt {
constexpr uint32_t begin = 0x3f800000u, end = 0x40800000u;
constexpr size_t header_bytes = 32u, pages = 65536u;
constexpr size_t payload_start = header_bytes + pages * sizeof(uint32_t);
constexpr size_t table_bytes = payload_start + 16777216u;
constexpr unsigned char sha256[32] = {
    0x4f,0x40,0xec,0x04,0x65,0x6a,0x43,0x94,0x88,0x13,0xe1,0x88,0xf9,0x14,0xe7,0xe2,
    0xf7,0x8d,0x8f,0x6b,0x09,0xb4,0x5e,0x64,0x70,0x27,0x97,0x8a,0xa5,0x20,0x19,0x1a};

// Every nonnegative FP32 encoding through +infinity was checked on SM121.
// Subnormal inputs follow sqrt.approx.ftz; no model or prompt supplied the table.
QRT_SQRT_HD inline float evaluate(const unsigned char* table, float argument) {
    const uint32_t input = qrt_sm121_exp2::bits(argument);
    if (!table || input > 0x7f800000u) return qrt_sm121_exp2::value(0x7fc00000u);
    if (input < 0x00800000u) return 0.0f;
    if (input == 0x7f800000u) return argument;
    const int exponent = int(input >> 23u) - 127;
    const int scale = exponent >= 0 ? exponent / 2 : -((-exponent + 1) / 2);
    const uint32_t relative = (input & 0x7fffffu) | (uint32_t(exponent & 1) << 23u);
    const auto* bases = reinterpret_cast<const uint32_t*>(table + header_bytes);
    const uint32_t normalized = bases[relative >> 8u] + table[payload_start + relative];
    return qrt_sm121_exp2::value(uint32_t(int32_t(normalized) + scale * 0x800000));
}
inline bool valid_layout(const unsigned char* data, size_t bytes) {
    if (!data || bytes != table_bytes || std::memcmp(data, "QSQRTB01", 8)) return false;
    const uint32_t expected[] = {1u, begin, end, 256u, uint32_t(pages), 16777216u};
    if (std::memcmp(data + 8u, expected, sizeof(expected))) return false;
    const auto* bases = reinterpret_cast<const uint32_t*>(data + header_bytes);
    for (size_t page = 0; page < pages; ++page) {
        const size_t offset = payload_start + page * 256u;
        if (bases[page] < 0x3f800000u || bases[page] >= 0x40000000u || data[offset] ||
            bases[page] + data[offset + 255u] > 0x40000000u) return false;
    }
    return true;
}
} // namespace qrt_sm121_sqrt
#undef QRT_SQRT_HD
#endif
