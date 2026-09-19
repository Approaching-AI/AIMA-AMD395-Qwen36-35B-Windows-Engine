#pragma once
#include <cstdint>
#include "long_attention_layout.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_INPLACE_P_INLINE __host__ __device__ __forceinline__
#else
#define QRT_INPLACE_P_INLINE inline
#endif

// Component-only storage policy. A score's sole probability producer replaces
// that same FP32 cell after reading it. A normal FP32 tag carries the original
// BF16 payload without arithmetic or a cross-cell/cross-row write. Selected PV
// replay runs after production and reads the payload through the same FP32 type.
namespace qrt_inplace_probability_storage {
constexpr uint32_t tag = 0x3f800000u;
QRT_INPLACE_P_INLINE float encode(uint16_t probability) {
    const uint32_t bits = tag | probability;
    float value; __builtin_memcpy(&value, &bits, sizeof(value)); return value;
}
QRT_INPLACE_P_INLINE uint16_t decode(float value) {
    uint32_t bits; __builtin_memcpy(&bits, &value, sizeof(bits)); return uint16_t(bits);
}
template<bool Inplace>
QRT_INPLACE_P_INLINE void store(uint16_t* values, size_t index, uint16_t probability) {
    if constexpr (Inplace) reinterpret_cast<float*>(values)[index] = encode(probability);
    else values[index] = probability;
}
template<bool Inplace>
QRT_INPLACE_P_INLINE uint16_t load(const uint16_t* values, size_t index) {
    if constexpr (Inplace) return decode(reinterpret_cast<const float*>(values)[index]);
    else return values[index];
}
constexpr qrt_long_attention_layout::Layout layout(unsigned queries, unsigned stride) {
    const auto original = qrt_long_attention_layout::layout(queries, stride);
    if (!original.elements) return {};
    const size_t saved = size_t(queries) * 16u * stride / 2u;
    return {0u, original.scales - saved, original.errors - saved,
        original.indices - saved, original.count - saved, original.elements - saved};
}
} // namespace qrt_inplace_probability_storage
#undef QRT_INPLACE_P_INLINE
