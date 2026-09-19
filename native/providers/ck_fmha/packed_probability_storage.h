#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PACKED_P_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PACKED_P_INLINE inline
#endif

// Keep the original four-byte score row stride. An even lane publishes both
// BF16 probabilities into one earlier, already-consumed score object. Each
// object has one writer; memcpy changes its representation without type punning
// or floating-point arithmetic. Replay reads individual two-byte payloads.
namespace qrt_packed_probability_storage {
QRT_PACKED_P_INLINE void store_pair(float* scores, size_t row, unsigned even_key,
    unsigned stride, uint16_t first, uint16_t second) {
    const uint16_t pair[2] = {first, second};
    __builtin_memcpy(scores + row * stride + even_key / 2u, pair, sizeof(pair));
}
QRT_PACKED_P_INLINE uint16_t load(const float* scores, size_t row, unsigned key,
    unsigned stride) {
    uint16_t value;
    const auto* bytes = reinterpret_cast<const unsigned char*>(scores);
    __builtin_memcpy(&value, bytes + row * stride * sizeof(float) + key * sizeof(value), sizeof(value));
    return value;
}
} // namespace qrt_packed_probability_storage
#undef QRT_PACKED_P_INLINE
