#ifndef QRT_SM121_EXP2_INTERPOLATED_H
#define QRT_SM121_EXP2_INTERPOLATED_H
#include "sm121_exp2_table.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_EX2_INTERP_INLINE __host__ __device__ __forceinline__
#else
#define QRT_EX2_INTERP_INLINE inline
#endif

// Lossless re-encoding of the SHA-bound SM121 negative exp2 table. The
// integer chord predicts a bit pattern; a packed residual restores every
// original FP32 bit. No floating interpolation or transcendental substitute
// is used. The sentinel supplies the next page's first value at the endpoint.
namespace qrt_sm121_exp2_interpolated {
constexpr uint32_t begin = qrt_sm121_exp2::begin, end = qrt_sm121_exp2::end;
constexpr uint32_t pages = qrt_sm121_exp2::pages;
constexpr uint64_t header_bytes = 96u;
constexpr uint64_t payload_start = header_bytes + uint64_t(pages + 1u) * 8u;
constexpr uint64_t payload_bytes = 28636608u;
constexpr uint64_t table_bytes = payload_start + payload_bytes;
constexpr int32_t maximum_page_slope = 22699;
constexpr unsigned char sha256[32] = {
    0xb5,0x75,0x2a,0x88,0x48,0x8d,0x8e,0x63,0x82,0xcc,0x99,0xa8,0x97,0xef,0x82,0x2e,
    0x97,0x75,0x8e,0x0a,0x78,0x29,0x30,0xc1,0x74,0x95,0x4a,0x52,0x99,0x12,0x6e,0xec};

QRT_EX2_INTERP_INLINE int32_t floor_div256(int32_t number) {
    // Sign-extend the shifted unsigned word without a negative signed shift.
    return int32_t(uint32_t(number) >> 8u) - (number < 0 ? 0x01000000 : 0);
}
QRT_EX2_INTERP_INLINE uint32_t decode(const unsigned char* table, uint32_t relative) {
    const auto* entry = reinterpret_cast<const uint32_t*>(table + header_bytes) + (relative >> 8u) * 2u;
    const uint32_t base = entry[0], tag = entry[1], kind = tag >> 29u;
    if (kind == 0u) return base;
    const unsigned char* payload = table + payload_start + (tag & 0x1fffffffu);
    const uint32_t cell = relative & 255u;
    if (kind == 7u) return reinterpret_cast<const uint32_t*>(payload)[cell];
    const int32_t slope = int32_t(entry[2]) - int32_t(base);
    const int32_t prediction = int32_t(base) + floor_div256(slope * int32_t(cell));
    int32_t residual;
    if (kind <= 2u) residual = int32_t((payload[cell >> 3u] >> (cell & 7u)) & 1u) - (kind == 2u ? 1 : 0);
    else if (kind == 3u) residual = int32_t((payload[cell >> 2u] >> ((cell & 3u) * 2u)) & 3u) - 1;
    else if (kind == 4u) residual = int32_t((payload[cell >> 1u] >> ((cell & 1u) * 4u)) & 15u) - 7;
    else if (kind == 5u) residual = int32_t(payload[cell]) - 127;
    else residual = int32_t(reinterpret_cast<const uint16_t*>(payload)[cell]) - 32767;
    return uint32_t(prediction + residual);
}
QRT_EX2_INTERP_INLINE float evaluate(const unsigned char* table, float argument) {
    const uint32_t input = qrt_sm121_exp2::bits(argument), magnitude = input & 0x7fffffffu;
    if (magnitude < begin || (!(input >> 31u) && magnitude < qrt_sm121_exp2::positive_one_end)) return 1.0f;
    if (magnitude > 0x7f800000u || !(input >> 31u)) return qrt_sm121_exp2::value(0x7fc00000u);
    if (magnitude >= end) return 0.0f;
    return qrt_sm121_exp2::value(decode(table, magnitude - begin));
}
inline bool valid_layout(const unsigned char* data, size_t bytes) {
    if (!data || bytes != table_bytes || std::memcmp(data, "QEX2IPL1", 8)) return false;
    const uint32_t expected[] = {1u, 8u, begin, end, qrt_sm121_exp2::positive_one_end, pages, 8u, 3u};
    if (std::memcmp(data + 8u, expected, sizeof(expected)) ||
        std::memcmp(data + 48u, qrt_sm121_exp2::sha256, 32u)) return false;
    uint64_t stored_payload; std::memcpy(&stored_payload, data + 40u, 8u);
    if (stored_payload != payload_bytes) return false;
    for (unsigned i = 80u; i < header_bytes; ++i) if (data[i] != 0u) return false;
    for (uint32_t page = 0u; page <= pages; ++page) {
        uint32_t entry[2]; std::memcpy(entry, data + header_bytes + uint64_t(page) * 8u, 8u);
        const unsigned kind = entry[1] >> 29u, offset = entry[1] & 0x1fffffffu;
        constexpr unsigned widths[] = {0u, 1u, 1u, 2u, 4u, 8u, 16u, 32u};
        if (entry[0] > 0x3f800000u || (kind == 0u && offset != 0u) || offset % 32u ||
            uint64_t(offset) + 32u * widths[kind] > payload_bytes) return false;
        if (page == pages) { if (entry[0] || entry[1]) return false; }
        else {
            uint32_t next; std::memcpy(&next, data + header_bytes + uint64_t(page + 1u) * 8u, 4u);
            const int64_t slope = int64_t(next) - entry[0];
            if (next > 0x3f800000u || slope < -maximum_page_slope || slope > maximum_page_slope ||
                (kind == 0u && slope != 0)) return false;
        }
    }
    return true;
}
}  // namespace qrt_sm121_exp2_interpolated
#undef QRT_EX2_INTERP_INLINE
#endif
