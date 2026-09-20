#pragma once
#include "sm121_q1_math.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MTP_GATE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MTP_GATE_INLINE inline
#endif
namespace qrt_sm121_mtp {
// Torch materializes BF16 sigmoid before multiplying its BF16 context.
// The caller must bind the complete original BF16 sigmoid-domain table.
QRT_MTP_GATE_INLINE uint16_t gated_context(uint16_t context, uint16_t gate,
                                         const uint16_t* sigmoid_table) {
    using namespace qrt_sm121_q1;
    return bf16(multiply(widen(context), widen(sigmoid_table[gate])));
}
} // namespace qrt_sm121_mtp
#undef QRT_MTP_GATE_INLINE
