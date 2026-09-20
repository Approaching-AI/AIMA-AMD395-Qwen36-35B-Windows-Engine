#pragma once
#include <hip/hip_runtime.h>
#include "sm121_mtp_gate_math.h"
namespace qrt_sm121_mtp {
__global__ void gate_contexts(const uint16_t* context, const uint16_t* gate,
    const uint16_t* sigmoid_table, uint16_t* output, unsigned int elements) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) output[i] = gated_context(context[i], gate[i], sigmoid_table);
}
// Query normalization/rotation publishes a contiguous gate vector separately
// from the interleaved Q/gate projection. Context may be replaced in place.
inline hipError_t launch_gate(const uint16_t* context, const uint16_t* gate,
    const uint16_t* sigmoid_table, uint16_t* output, unsigned int rows, hipStream_t stream = nullptr) {
    if (!context || !gate || !sigmoid_table || !output || output == gate || output == sigmoid_table ||
        !rows || rows > 8192u)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(gate_contexts, dim3(rows * 16u), dim3(256u), 0u, stream,
                      context, gate, sigmoid_table, output, rows * 4096u);
    return hipGetLastError();
}
} // namespace qrt_sm121_mtp
