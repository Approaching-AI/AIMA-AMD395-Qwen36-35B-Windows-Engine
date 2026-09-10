#ifndef QRT_FLA_BLACKWELL_WU_OUTPUT_H
#define QRT_FLA_BLACKWELL_WU_OUTPUT_H
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace qrt_fla_blackwell_aux {
inline bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    const auto av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
    return av <= bv ? bv - av < an : av - bv < bn;
}
// One chunk. W and U may not alias any input except that U may equal V.
// The W/U kernel captures each complete V column tile before writing it.
inline bool valid_wu(const uint16_t* k, const uint16_t* v, const uint16_t* beta,
                     const uint16_t* inverse, const float* g, uint16_t* w,
                     uint16_t* u, unsigned count) {
    if (!k || !v || !beta || !inverse || !g || !w || !u || !count || count > 64 || w == u) return false;
    const size_t output_bytes = size_t(count) * 4096u * 2u;
    if (overlaps(w, output_bytes, u, output_bytes)) return false;
    const void* inputs[] = {k, v, beta, inverse, g};
    const size_t sizes[] = {size_t(count) * 2048u * 2u, output_bytes, size_t(count) * 32u * 2u,
                            size_t(count) * 2048u * 2u, size_t(count) * 32u * 4u};
    for (unsigned i = 0; i < 5u; ++i) {
        if (overlaps(w, output_bytes, inputs[i], sizes[i])) return false;
        if (overlaps(u, output_bytes, inputs[i], sizes[i]) && !(i == 1u && u == v)) return false;
    }
    return true;
}
hipError_t recompute_wu(const uint16_t* k, const uint16_t* v, const uint16_t* beta,
                         const uint16_t* inverse, const float* g, uint16_t* w,
                         uint16_t* u, unsigned count, hipStream_t stream);
hipError_t output_scores(const uint16_t* q, const uint16_t* k, const float* g,
                          uint16_t* scores, unsigned count, hipStream_t stream);
hipError_t output_values(const uint16_t* q, const uint16_t* v, const uint16_t* h,
                          const float* g, const uint16_t* scores, float* output,
                          unsigned count, hipStream_t stream);
}
#endif
