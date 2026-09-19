#pragma once
#include "prepared_decoded_qk.h"
#include "prepared_decoded_qk_range_workspace.h"

// A component range owns only the Q rows it will consume. K metadata covers
// the complete history. Original Q/K BF16 inputs remain the fallback authority.
namespace qrt_prepared_decoded_qk_range {
__global__ void scores(const uint16_t* query, const uint16_t* transposed_key,
    const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags, float* output,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride, unsigned query_origin) {
    qrt_prepared_decoded_qk::scores_body<128u, true>(query, transposed_key, packed_query,
        packed_key, query_flags, key_flags, output, start, count, stride, key_stride, query_origin);
}

inline int prepare_workspace_from_query_origin(const uint16_t* query, const uint16_t* key,
    uint16_t* transposed, const Workspace& workspace, hipStream_t stream,
    unsigned query_origin) {
    if (!query || !key || !transposed || !valid(workspace) ||
        query_origin > workspace.query_start) return int(hipErrorInvalidValue);
    auto* q = workspace.words;
    auto* k = q + query_words;
    auto* qflags = k + key_words(workspace.key_capacity);
    auto* kflags = qflags + query_flag_words;
    const size_t query_offset = size_t(workspace.query_start - query_origin) * qrt_blackwell_attention::kQueryHeads * qrt_blackwell_attention::kHeadDim;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_prepared_decoded_qk::prepare<false>),
        dim3(workspace.query_count * qrt_blackwell_attention::kQueryHeads),
        dim3(qrt_blackwell_attention::kHeadDim), 0u, stream,
        query + query_offset, q, qflags, nullptr, workspace.query_count);
    auto status = hipGetLastError();
    if (status != hipSuccess) return int(status);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_prepared_decoded_qk::prepare<true>),
        dim3(workspace.key_tokens * qrt_blackwell_attention::kKvHeads),
        dim3(qrt_blackwell_attention::kHeadDim), 0u, stream,
        key, k, kflags, transposed, workspace.key_tokens);
    return int(hipGetLastError());
}

inline int prepare_workspace(const uint16_t* query, const uint16_t* key,
    uint16_t* transposed, const Workspace& workspace, hipStream_t stream) {
    return prepare_workspace_from_query_origin(query,key,transposed,workspace,stream,0u);
}

inline int launch_workspace(const void* state, const uint16_t* query,
    const uint16_t* transposed_key, float* output, hipStream_t stream,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    if (!state || !query || !transposed_key || !output) return int(hipErrorInvalidValue);
    const auto& workspace = *static_cast<const Workspace*>(state);
    if (!valid(workspace) || !count || count > 128u || start < workspace.query_start ||
        start - workspace.query_start >= workspace.query_count ||
        count > workspace.query_count - (start - workspace.query_start) ||
        stride != start + count || key_stride != workspace.key_tokens)
        return int(hipErrorInvalidValue);
    const auto* q = workspace.words;
    const auto* k = q + query_words;
    const auto* qflags = k + key_words(workspace.key_capacity);
    const auto* kflags = qflags + query_flag_words;
    hipLaunchKernelGGL(scores,
        dim3((stride + 15u) / 16u, qrt_blackwell_attention::kQueryHeads, (count + 15u) / 16u),
        dim3(qrt_blackwell_attention::kThreads), 0u, stream,
        query, transposed_key, q, k, qflags, kflags, output, start, count, stride, key_stride, workspace.query_start);
    return int(hipGetLastError());
}
} // namespace qrt_prepared_decoded_qk_range
