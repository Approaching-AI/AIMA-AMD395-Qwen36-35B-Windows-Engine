#pragma once
#include "packed_probability_pipeline.h"
#include "long_probability_storage_policy.h"

namespace qrt_long_probability_pipeline {
inline int launch(unsigned mode, const qrt_long_narrow_qk::Workspace& qk,
    const qrt_native_exp2_workspace::Workspace& exp_owner,
    const uint16_t* query, const uint16_t* transposed_key, const uint16_t* value,
    const uint16_t* transposed_value, float* output, unsigned start, unsigned count,
    unsigned output_start, unsigned key_stride, const unsigned char* exp,
    const unsigned char* rcp, float* scratch, size_t scratch_elements, hipStream_t stream,
    qrt_blackwell_attention::SplitCompletionObserver* observer = nullptr, bool final_bound = false) {
    if (mode > 2u) return int(hipErrorInvalidValue);
    const auto implementation = mode == 0u ? qrt_long_attention_pipeline::launch
        : mode == 1u ? qrt_inplace_probability_pipeline::launch : qrt_packed_probability_pipeline::launch;
    return implementation(qk, exp_owner, query, transposed_key, value, transposed_value,
        output, start, count, output_start, key_stride, exp, rcp, scratch, scratch_elements,
        stream, observer, final_bound);
}
} // namespace qrt_long_probability_pipeline
