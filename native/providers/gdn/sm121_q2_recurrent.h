#pragma once
#include <hip/hip_runtime.h>
#include "sm121_q2_recurrent_layout.h"

namespace qrt_sm121_q2 {
namespace recurrent_detail {
__global__ void kernel(RecurrentViews view, RecurrentTables tables) {
    const unsigned head = blockIdx.x, value = threadIdx.x;
    const unsigned stride = view.key_major ? 128u : 1u;
    const size_t offset = state_offset(head, value, view.key_major);
    __shared__ RecurrentHead prepared;
    for (unsigned row = 0; row < scheduled_rows; ++row) {
        const uint16_t* conv = view.convolution + row * 8192u;
        if (!value) prepare_head(prepared, conv, view.a[row * 32u + head],
                                  view.b[row * 32u + head], head, tables);
        __syncthreads();
        const float* before = row ? view.staged_states : view.initial_state;
        float* after = view.staged_states + row * state_elements;
        view.staged_core[row * core_elements + head * 128u + value] = recurrent_value(
            before + offset, after + offset, stride, prepared,
            conv[4096u + head * 128u + value], value);
        // Finish reading the shared head and writing the entire first state
        // before row one reuses either. The resident initial state is read-only.
        __syncthreads();
    }
}
} // namespace recurrent_detail

// Allocation-free asynchronous producer. Both outcomes remain private until
// the enclosing target transaction samples and selects its accepted extent.
// All borrowed storage must remain live until completion, including failures.
inline hipError_t launch_recurrent(const RecurrentViews& view, const RecurrentTables& tables,
                                    hipStream_t stream = nullptr) {
    if (!valid_recurrent_views(view, tables)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(recurrent_detail::kernel, dim3(32u), dim3(128u), 0u, stream, view, tables);
    return hipGetLastError();
}
} // namespace qrt_sm121_q2
