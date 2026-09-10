#ifndef QRT_FLA_BLACKWELL_INVERSE_H
#define QRT_FLA_BLACKWELL_INVERSE_H
#include "blackwell_wu_output.h"
namespace qrt_fla_blackwell_inverse {
inline bool valid_solve(const float* a, uint16_t* inverse, unsigned tokens) {
    if (!a || !inverse || !tokens || tokens > 8192u || reinterpret_cast<uintptr_t>(a) % alignof(float) ||
        reinterpret_cast<uintptr_t>(inverse) % alignof(uint16_t)) return false;
    return !qrt_fla_blackwell_aux::overlaps(a, size_t(tokens) * 2048u * 4u, inverse, size_t(tokens) * 2048u * 2u);
}
hipError_t solve(const float* a, uint16_t* inverse, unsigned tokens, hipStream_t stream);
}
#endif
