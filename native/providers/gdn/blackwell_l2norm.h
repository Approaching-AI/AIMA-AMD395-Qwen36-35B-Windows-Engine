#ifndef QRT_FLA_BLACKWELL_L2NORM_H
#define QRT_FLA_BLACKWELL_L2NORM_H
#include "blackwell_wu_output.h"

namespace qrt_fla_blackwell_norm {
inline bool valid_normalize(const float* raw, uint16_t* q, uint16_t* k, unsigned tokens) {
    if (!raw || !q || !k || !tokens || tokens > 8192u) return false;
    if (reinterpret_cast<uintptr_t>(raw) % alignof(float) ||
        reinterpret_cast<uintptr_t>(q) % alignof(uint16_t) || reinterpret_cast<uintptr_t>(k) % alignof(uint16_t)) return false;
    const size_t input_bytes = size_t(tokens) * 8192u * 4u, output_bytes = size_t(tokens) * 2048u * 2u;
    using qrt_fla_blackwell_aux::overlaps;
    return !overlaps(q, output_bytes, k, output_bytes) &&
           !overlaps(raw, input_bytes, q, output_bytes) && !overlaps(raw, input_bytes, k, output_bytes);
}
hipError_t prepare_table();
void release_table();
uint64_t table_storage_bytes();
const unsigned char* table_device();
hipError_t normalize(const float* raw, uint16_t* q, uint16_t* k, unsigned tokens, hipStream_t stream);
}
#endif
