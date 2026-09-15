#ifndef QRT_PARALLEL_PV_PARTIALS_H
#define QRT_PARALLEL_PV_PARTIALS_H
#include "parallel_pv_plan.h"
#include "blackwell_attention.h"

// Component experiment. QK, online probabilities, alpha values, reciprocal,
// BF16 admission and canonical exact replay remain owned by the caller.
namespace qrt_parallel_pv {
namespace original = qrt_blackwell_attention;
namespace bound = qrt_sm121_pv_bound;
namespace deferred = qrt_sm121_pv_final_bound;

// Independent K16 dots remove the matrix instruction's loop-carried C
// dependency. Each wave writes a complete16x16 tile. No carried accumulator
// or reference output is an input to this producer.
__global__ void produce(const uint16_t* value, const uint16_t* probability,
    Partial* partials, unsigned query_start, unsigned query_offset,
    unsigned query_count, unsigned stride) {
    constexpr unsigned waves = 4u;
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u;
    const unsigned group = blockIdx.z * waves + wave, head = blockIdx.y;
    const unsigned last = query_start + query_offset + query_count;
    if (group >= groups(last)) return; // Uniform within every WMMA wave.
    const unsigned local_query = lane % 16u, row = query_offset + local_query;
    const unsigned tokens = query_start + row + 1u;
    const unsigned column = blockIdx.x * 16u + lane % 16u;
    const unsigned kv = head / 8u;
    original::NativeOperandRow left{}, right{};
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) {
        const unsigned key = group * 16u + i;
        left.original[i] = local_query < query_count && key < tokens
            ? probability[(size_t(row) * heads + head) * stride + key] : 0u;
        right.original[i] = key < last
            ? value[(size_t(key) * 2u + kv) * dimensions + column] : 0u;
    }
    const original::MantissaF32x8 zero{};
    const auto dot = original::blackwell_native_mma(left, right, zero);
    const auto absolute = original::blackwell_native_mma<true>(left, right, zero);
#pragma unroll
    for (unsigned element = 0u; element < 8u; ++element) {
        const unsigned query = 2u * element + lane / 16u;
        if (query < query_count) {
            const size_t index = ((size_t(group) * tile_queries + query) * heads + head) * dimensions + column;
            partials[index] = {dot[element], absolute[element]};
        }
    }
}

// The proxy changes: a zero-C WMMA is followed by one explicit FP32 add.
// Its17-add gamma allowance plus the canonical26-bit alignment/truncation
// allowance is below32*2^-24. The established2^-19 envelope therefore still
// covers this proxy; no coefficient or BF16 acceptance interval is tightened.
// Absolute dots, subnormal floors and exceptional-value rejection are shared
// with the original bound. Native full-cell and GB10 comparisons are required
// before this route can enter product dispatch.
__global__ void reduce(const Partial* partials, const float* scales,
    float* output, float* errors, unsigned query_start, unsigned query_offset,
    unsigned query_count, unsigned output_start, unsigned stride,
    const unsigned char* reciprocal_table, float* raw_accumulator,
    float* raw_denominator) {
    const unsigned cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= query_count * heads * dimensions) return;
    const unsigned local_query = cell / (heads * dimensions);
    const unsigned feature = cell % (heads * dimensions), row = query_offset * heads + cell / dimensions;
    const unsigned tokens = query_start + query_offset + local_query + 1u;
    const unsigned tile_stride = (stride + 31u) / 32u, count = groups(tokens);
    float accumulator = 0.0f, state = 0.0f;
    for (unsigned group = 0u; group < count; ++group) {
        if (!(group & 1u)) {
            const float alpha = scales[size_t(row) * (tile_stride + 1u) + group / 2u];
            state = deferred::rescale(state, accumulator, alpha);
            accumulator = deferred::multiply(accumulator, alpha);
        }
        const auto part = partials[(size_t(group) * tile_queries + local_query) * heads * dimensions + feature];
        state = deferred::group(state, accumulator, part.absolute);
        accumulator = deferred::add(accumulator, part.value);
    }
    const float denominator = scales[size_t(row) * (tile_stride + 1u) + tile_stride];
    const float reciprocal = qrt_sm121_attention_rcp::evaluate(reciprocal_table, denominator);
    const size_t index = size_t(output_start + query_offset) * heads * dimensions + cell;
    output[index] = deferred::multiply(accumulator, reciprocal);
    errors[size_t(query_offset) * heads * dimensions + cell] =
        bound::finish(deferred::finalize(state, count), accumulator, reciprocal);
    if (raw_accumulator) raw_accumulator[index] = accumulator;
    if (raw_denominator && !(cell % dimensions))
        raw_denominator[size_t(output_start + query_offset) * heads + cell / dimensions] = denominator;
}

inline int launch(const uint16_t* value, const uint16_t* probability,
    const float* scales, float* output, float* errors, unsigned query_start,
    unsigned query_count, unsigned output_start, unsigned stride,
    const unsigned char* reciprocal_table, Partial* partials, size_t capacity,
    hipStream_t stream, float* raw_accumulator = nullptr, float* raw_denominator = nullptr) {
    // The caller owns buffers until stream completion, including after a
    // failed later launch. Submission errors stop all dependent work here.
    if (!value || !probability || !scales || !output || !errors || !reciprocal_table || !partials ||
        !valid(query_start, query_count, output_start, stride) ||
        capacity < partial_count(query_start, query_count)) return int(hipErrorInvalidValue);
    for (unsigned offset = 0u; offset < query_count; offset += tile_queries) {
        const unsigned count = query_count - offset < tile_queries ? query_count - offset : tile_queries;
        const unsigned group_count = groups(query_start + offset + count);
        hipLaunchKernelGGL(produce, dim3(16u, heads, (group_count + 3u) / 4u),
            dim3(128u), 0u, stream, value, probability, partials,
            query_start, offset, count, stride);
        const auto producer_status = hipGetLastError();
        if (producer_status != hipSuccess) return int(producer_status);
        hipLaunchKernelGGL(reduce, dim3(count * heads * dimensions / 256u),
            dim3(256u), 0u, stream, partials, scales, output, errors,
            query_start, offset, count, output_start, stride, reciprocal_table,
            raw_accumulator, raw_denominator);
        const auto reducer_status = hipGetLastError();
        if (reducer_status != hipSuccess) return int(reducer_status);
    }
    return int(hipSuccess);
}
} // namespace qrt_parallel_pv
#endif
