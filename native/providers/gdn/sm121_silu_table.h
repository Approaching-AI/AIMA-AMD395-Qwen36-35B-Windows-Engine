#ifndef QRT_SM121_SILU_TABLE_H
#define QRT_SM121_SILU_TABLE_H
#include "sm121_exp2_table.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_SILU_HD __host__ __device__
#else
#define QRT_SILU_HD
#endif

namespace qrt_sm121_silu {
constexpr uint32_t directory_count = 65537u, transition_count = 64304u;
constexpr uint64_t key_start = 64u + uint64_t(directory_count) * 4u;
constexpr uint64_t value_start = key_start + uint64_t(transition_count) * 4u;
constexpr uint64_t table_bytes = value_start + uint64_t(transition_count) * 2u;
constexpr unsigned char sha256[32] = {
    0x67,0x3f,0x8d,0xd1,0x28,0x07,0x00,0x57,0x8c,0x1e,0x87,0x43,0xaf,0xd2,0xe3,0xb4,
    0xda,0x13,0x4b,0x1f,0xbd,0x46,0x3c,0x89,0x05,0x27,0xe1,0xc4,0xd9,0xf7,0x96,0xb8};

// Every finite FP32 input and every local output transition were enumerated
// and independently verified on SM121. No model/prompt values create entries.
QRT_SILU_HD inline uint16_t evaluate(const unsigned char *table, float argument) {
    const uint32_t raw = qrt_sm121_exp2::bits(argument);
    if ((raw & 0x7fffffffu) >= 0x7f800000u) return 0x7fc0u;
    const auto *directory = reinterpret_cast<const uint32_t *>(table + 64u);
    const auto *keys = reinterpret_cast<const uint32_t *>(table + key_start);
    const auto *values = reinterpret_cast<const uint16_t *>(table + value_start);
    const uint32_t page = raw >> 16u;
    uint32_t lower = directory[page], upper = directory[page + 1u] + 1u;
    while (lower < upper) {
        const uint32_t middle = (lower + upper) / 2u;
        if (keys[middle] <= raw) lower = middle + 1u;
        else upper = middle;
    }
    return values[lower - 1u];
}

inline bool valid_layout(const unsigned char *data, size_t bytes) {
    if (!data || bytes != table_bytes || std::memcmp(data, "QSLUTB1\0", 8)) return false;
    uint32_t fields[6]; uint64_t spans[4];
    std::memcpy(fields, data + 8, sizeof(fields));
    std::memcpy(spans, data + 32, sizeof(spans));
    const uint32_t expected_fields[] = {1u,16u,directory_count,transition_count,0x7f800000u,0u};
    const uint64_t expected_spans[] = {key_start,value_start,table_bytes,UINT64_C(4278190080)};
    if (std::memcmp(fields, expected_fields, sizeof(fields)) ||
        std::memcmp(spans, expected_spans, sizeof(spans))) return false;
    const auto *directory = reinterpret_cast<const uint32_t *>(data + 64u);
    const auto *keys = reinterpret_cast<const uint32_t *>(data + key_start);
    const auto *values = reinterpret_cast<const uint16_t *>(data + value_start);
    if (keys[0] != 0u) return false;
    for (uint32_t i = 0; i < transition_count; ++i) {
        if ((i && keys[i] <= keys[i - 1u]) || (keys[i] & 0x7fffffffu) >= 0x7f800000u ||
            (values[i] & 0x7fffu) > 0x7f80u) return false;
    }
    uint32_t floor = 0;
    for (uint32_t page = 0; page < directory_count; ++page) {
        const uint64_t boundary = uint64_t(page) << 16u;
        while (floor + 1u < transition_count && keys[floor + 1u] <= boundary) ++floor;
        if (directory[page] != floor) return false;
    }
    return true;
}
} // namespace qrt_sm121_silu
#undef QRT_SILU_HD
#endif
