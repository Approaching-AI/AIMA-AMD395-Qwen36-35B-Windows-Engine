#ifndef QRT_SCALED_L2_REFERENCE_H
#define QRT_SCALED_L2_REFERENCE_H
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace scaled_l2_test {
inline long double bf16(uint16_t raw) {
    const unsigned e = (raw >> 7u) & 255u;
    if (e == 255u) return std::numeric_limits<long double>::infinity();
    return std::ldexp(static_cast<long double>((raw & 127u) | (e ? 128u : 0u)),
                      int(e ? e : 1u) - 134);
}
inline long double norm(const uint16_t *row, unsigned columns) {
    long double sum = 0;
    for (unsigned k = 0; k < columns; ++k) {
        const long double v = bf16(row[k]);
        sum += v * v;
    }
    return std::sqrt(sum);
}
inline std::vector<uint16_t> fixture(unsigned rows, unsigned columns) {
    std::vector<uint16_t> result(size_t(rows) * columns);
    uint32_t state = 395u;
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned k = 0; k < columns; ++k) {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            const unsigned e = r < 256u ? r : (state % 255u);
            uint16_t raw = uint16_t((e << 7u) | (k & 127u) | ((k & 128u) << 8u));
            if (r == 256u) raw = 0u;
            if (r == 257u) raw = k == 0u ? 1u : 0x8000u;
            if (r == 258u) raw = (k & 1u) ? 0x3f80u : 1u;
            result[size_t(r) * columns + k] = raw;
        }
    }
    return result;
}
} // namespace scaled_l2_test
#endif
