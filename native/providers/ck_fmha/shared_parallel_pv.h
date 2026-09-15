#ifndef QRT_SHARED_PARALLEL_PV_H
#define QRT_SHARED_PARALLEL_PV_H
#include "parallel_pv_partials.h"

namespace qrt_parallel_pv {
// Four independent K16 producers publish8 KiB in LDS. The same128 threads
// then own two output cells each and reduce those four groups in order. This
// keeps the separately rounded proxy from the global-partials experiment
// without writing every partial to device memory. Both barriers are uniform,
// including incomplete query tiles and the final two-group batch.
constexpr unsigned shared_groups = 4u;
constexpr size_t shared_bytes = shared_groups * 16u * 16u * sizeof(Partial);
static_assert(shared_bytes == 8192u);
__global__ void shared_kernel(const uint16_t* value, const uint16_t* probability,
    const float* scales, float* output, float* errors, unsigned query_start,
    unsigned query_count, unsigned output_start, unsigned stride,
    const unsigned char* reciprocal_table, float* raw_accumulator, float* raw_denominator) {
    __shared__ Partial partials[shared_groups][256u];
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u;
    const unsigned head = blockIdx.y, column_start = blockIdx.x * 16u;
    const unsigned query_offset = blockIdx.z * 16u;
    const unsigned count = query_count - query_offset < 16u ? query_count - query_offset : 16u;
    const unsigned last = query_start + query_offset + count;
    const unsigned tile_stride = (stride + 31u) / 32u;
    float accumulator[2]{}, state[2]{};
    for (unsigned base = 0u; base < groups(last); base += shared_groups) {
        const unsigned group = base + wave, query = query_offset + lane % 16u;
        const unsigned tokens = query_start + query + 1u;
        original::NativeOperandRow left{}, right{};
#pragma unroll
        for (unsigned i = 0u; i < 16u; ++i) {
            const unsigned key = group * 16u + i;
            left.original[i] = lane % 16u < count && key < tokens
                ? probability[(size_t(query) * heads + head) * stride + key] : 0u;
            right.original[i] = key < last
                ? value[(size_t(key) * 2u + head / 8u) * dimensions + column_start + lane % 16u] : 0u;
        }
        const original::MantissaF32x8 zero{};
        const auto dot = original::blackwell_native_mma(left, right, zero);
        const auto absolute = original::blackwell_native_mma<true>(left, right, zero);
#pragma unroll
        for (unsigned element = 0u; element < 8u; ++element)
            partials[wave][(2u * element + lane / 16u) * 16u + lane % 16u] = {dot[element], absolute[element]};
        __syncthreads();
#pragma unroll
        for (unsigned item = 0u; item < 2u; ++item) {
            const unsigned cell = threadIdx.x + item * 128u;
            const unsigned local_query = cell / 16u, row = (query_offset + local_query) * heads + head;
            const unsigned end = groups(query_start + query_offset + local_query + 1u);
            if (local_query < count) {
#pragma unroll
                for (unsigned g = 0u; g < shared_groups; ++g) if (base + g < end) {
                    if (!(g & 1u)) {
                        const float alpha = scales[size_t(row) * (tile_stride + 1u) + (base + g) / 2u];
                        state[item] = deferred::rescale(state[item], accumulator[item], alpha);
                        accumulator[item] = deferred::multiply(accumulator[item], alpha);
                    }
                    const auto part = partials[g][cell];
                    state[item] = deferred::group(state[item], accumulator[item], part.absolute);
                    accumulator[item] = deferred::add(accumulator[item], part.value);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned item = 0u; item < 2u; ++item) {
        const unsigned cell = threadIdx.x + item * 128u, local_query = cell / 16u;
        if (local_query < count) {
            const unsigned row = (query_offset + local_query) * heads + head, column = column_start + cell % 16u;
            const float denominator = scales[size_t(row) * (tile_stride + 1u) + tile_stride];
            const float reciprocal = qrt_sm121_attention_rcp::evaluate(reciprocal_table, denominator);
            const size_t index = (size_t(output_start) * heads + row) * dimensions + column;
            output[index] = deferred::multiply(accumulator[item], reciprocal);
            errors[size_t(row) * dimensions + column] = bound::finish(
                deferred::finalize(state[item], groups(query_start + query_offset + local_query + 1u)), accumulator[item], reciprocal);
            if (raw_accumulator) raw_accumulator[index] = accumulator[item];
            if (raw_denominator && !column) raw_denominator[size_t(output_start) * heads + row] = denominator;
        }
    }
}
inline int launch_shared(const uint16_t* value, const uint16_t* probability,
    const float* scales, float* output, float* errors, unsigned query_start,
    unsigned query_count, unsigned output_start, unsigned stride,
    const unsigned char* reciprocal_table, hipStream_t stream,
    float* raw_accumulator = nullptr, float* raw_denominator = nullptr) {
    if (!value || !probability || !scales || !output || !errors || !reciprocal_table ||
        !valid(query_start, query_count, output_start, stride)) return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(shared_kernel, dim3(16u, heads, (query_count + 15u) / 16u), dim3(128u),
        0u, stream, value, probability, scales, output, errors, query_start, query_count,
        output_start, stride, reciprocal_table, raw_accumulator, raw_denominator);
    return int(hipGetLastError());
}
} // namespace qrt_parallel_pv
#endif
