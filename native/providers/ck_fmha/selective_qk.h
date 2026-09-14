#ifndef QRT_SELECTIVE_QK_H
#define QRT_SELECTIVE_QK_H

#include "blackwell_attention.h"

// Route under numerical validation. The strict denominator repair remains a
// component diagnostic; an opt-in product trial isolates the probability path
// with an approximate denominator. Full baseline scores never select work.
namespace qrt_selective_qk {
using namespace qrt_blackwell_attention;
namespace bound = qrt_sm121_pv_bound;

__global__ void native_scores(const uint16_t* query, const uint16_t* transposed_key,
    float* scores, float* errors, unsigned query_start, unsigned query_count,
    unsigned score_stride, unsigned key_stride) {
    __shared__ NativeOperandRow left[16], right[kIntegerMatrixColumns];
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u, head = blockIdx.y;
    const unsigned query_tile = blockIdx.z * 16u, key_tile = blockIdx.x * kIntegerMatrixColumns;
    const unsigned kv_head = head / (kQueryHeads / kKvHeads);
    const unsigned last_query = query_start + min(query_tile + 16u, query_count) - 1u;
    MantissaF32x8 accumulator{}, envelope{};
    if (key_tile <= last_query) {
        for (unsigned base = 0u; base < kHeadDim; base += 16u) {
            if (wave == 0u && lane < 16u) {
                const unsigned row = query_tile + lane;
#pragma unroll
                for (unsigned i = 0u; i < 16u; ++i)
                    left[lane].original[i] = row < query_count
                        ? query[(size_t(query_start + row) * kQueryHeads + head) * kHeadDim + base + i] : 0u;
            }
            if (lane < 16u) {
                const unsigned row = wave * 16u + lane, key = key_tile + row;
#pragma unroll
                for (unsigned i = 0u; i < 16u; ++i)
                    right[row].original[i] = key < score_stride
                        ? transposed_key[(size_t(kv_head) * kHeadDim + base + i) * key_stride + key] : 0u;
            }
            __syncthreads();
            const auto next = blackwell_native_mma(left[lane % 16u], right[wave * 16u + lane % 16u], accumulator);
            const auto absolute_dot = blackwell_native_mma<true>(left[lane % 16u], right[wave * 16u + lane % 16u], MantissaF32x8{});
#pragma unroll
            for (unsigned element = 0u; element < 8u; ++element)
                envelope[element] = bound::group(envelope[element], accumulator[element], absolute_dot[element]);
            accumulator = next;
            __syncthreads();
        }
    }
#pragma unroll
    for (unsigned element = 0u; element < 8u; ++element) {
        const unsigned row = query_tile + 2u * element + lane / 16u, key = key_tile + wave * 16u + lane % 16u;
        if (row < query_count && key < score_stride) {
            const size_t cell = (size_t(row) * kQueryHeads + head) * score_stride + key;
            const bool live = key <= query_start + row;
            scores[cell] = live ? accumulator[element] * kExactScale : -INFINITY;
            // Scaling by this normal power of two is exact on normal scores;
            // finish also encloses scale rounding and the subnormal floor.
            errors[cell] = live ? bound::finish(envelope[element], accumulator[element], kExactScale) : 0.0f;
        }
    }
}

struct Interval { float low, high; };
__device__ __forceinline__ Interval score_interval(float score, float error) {
    if (!bound::finite(score) || !bound::finite(error) || error < 0.0f)
        return {-INFINITY, INFINITY};
    if (error == 0.0f) return {score, score};
    return {bound::next(score - error, false), bound::next(score + error, true)};
}
__device__ __forceinline__ float wave_maximum(float value) {
    for (unsigned mask = 16u; mask; mask >>= 1u) value = fmaxf(value, __shfl_xor(value, mask, 32u));
    return value;
}
__device__ __forceinline__ float wave_sum(float value) {
    constexpr unsigned order[] = {1u, 4u, 2u, 16u, 8u};
#pragma unroll
    for (unsigned step = 0u; step < 5u; ++step) value += __shfl_xor(value, order[step], 32u);
    return value;
}
__device__ __forceinline__ void collect(unsigned index, bool selected, unsigned* indices, unsigned* count) {
    const unsigned mask = __ballot(selected), lane = threadIdx.x % 32u;
    unsigned begin = lane == 0u && mask ? atomicAdd(count, unsigned(__popc(mask))) : 0u;
    begin = __shfl(begin, 0u, 32u);
    if (selected) indices[begin + unsigned(__popc(mask & ((uint32_t(1u) << lane) - 1u)))] = index;
}

// Every possible maximum of a K32 prefix is recomputed. A cell whose upper
// endpoint is below a proven prefix lower bound cannot produce that maximum.
// Include ties and exceptional envelopes. One compact slot per live cell is
// sufficient even if every bound is uncertain.
__global__ void collect_maxima(const float* scores, const float* errors,
    unsigned start, unsigned stride, unsigned* indices, unsigned* count) {
    const unsigned lane = threadIdx.x, row = blockIdx.y * kQueryHeads + blockIdx.x;
    const unsigned tokens = start + blockIdx.y + 1u;
    float prefix_lower = -INFINITY;
    for (unsigned tile = 0u; tile < (tokens + 31u) / 32u; ++tile) {
        const unsigned key = tile * 32u + lane, cell = row * stride + key;
        const Interval interval = key < tokens ? score_interval(scores[cell], errors[cell]) : Interval{-INFINITY, -INFINITY};
        prefix_lower = wave_maximum(fmaxf(prefix_lower, interval.low));
        collect(cell, key < tokens && interval.high >= prefix_lower, indices, count);
    }
}

// Each subgroup replays one selected original K256 dot. No exact-score
// reference is read, and the device-resident count controls bounded work.
__global__ void repair_scores(const uint16_t* query, const uint16_t* key,
    float* scores, float* errors, unsigned start, unsigned stride,
    const unsigned* indices, const unsigned* count) {
    constexpr unsigned lanes = 16u;
    const unsigned lane = threadIdx.x % lanes;
    const unsigned step = gridDim.x * blockDim.x / lanes;
    for (unsigned slot = (blockIdx.x * blockDim.x + threadIdx.x) / lanes; slot < *count; slot += step) {
        const unsigned cell = indices[slot], row = cell / stride, key_token = cell % stride;
        const unsigned head = row % kQueryHeads, kv_head = head / (kQueryHeads / kKvHeads);
        const size_t query_base = (size_t(start + row / kQueryHeads) * kQueryHeads + head) * kHeadDim;
        const size_t key_base = (size_t(key_token) * kKvHeads + kv_head) * kHeadDim;
        qrt_q1_moe_hawkeye::Value carry{0u, kBlackwellZeroExponent, false};
        for (unsigned base = 0u; base < kHeadDim; base += lanes)
            carry = qrt_sm121_wave16::accumulate(carry, query[query_base + base + lane], key[key_base + base + lane], lane);
        if (lane == 0u) {
            scores[cell] = qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry)) * kExactScale;
            errors[cell] = 0.0f;
        }
    }
}

// Exhaustive inspection of all328728576 interior inputs in the SHA-bound
// compact exp2 table establishes monotonicity; its exterior is constant1/0.
// FP32 subtraction and multiplication by positive log2(e) are monotone too.
// Cap score intervals at the exact prefix maximum before table evaluation.
__device__ __forceinline__ Interval probability_interval(float score, float error,
    float maximum, const unsigned char* table) {
    const auto interval = score_interval(score, error);
    return {blackwell_attention_exp(fminf(interval.low, maximum) - maximum, table),
            blackwell_attention_exp(fminf(interval.high, maximum) - maximum, table)};
}

__global__ void collect_probabilities(const float* scores, const float* errors,
    float* maxima, unsigned start, unsigned stride, const unsigned char* table,
    unsigned* indices, unsigned* count) {
    const unsigned lane = threadIdx.x, row = blockIdx.y * kQueryHeads + blockIdx.x;
    const unsigned tokens = start + blockIdx.y + 1u, tiles = (stride + 31u) / 32u;
    float maximum = -INFINITY;
    for (unsigned tile = 0u; tile < (tokens + 31u) / 32u; ++tile) {
        const unsigned key = tile * 32u + lane, cell = row * stride + key;
        const float exact = key < tokens && errors[cell] == 0.0f ? scores[cell] : -INFINITY;
        maximum = wave_maximum(fmaxf(maximum, exact));
        if (lane == 0u) maxima[size_t(row) * tiles + tile] = maximum;
        const auto p = key < tokens ? probability_interval(scores[cell], errors[cell], maximum, table) : Interval{0.0f, 0.0f};
        collect(cell, key < tokens && errors[cell] != 0.0f &&
            (!isfinite(p.low) || !isfinite(p.high) || f32_to_bf16(p.low) != f32_to_bf16(p.high)), indices, count);
    }
}

// Probability repair leaves maxima unchanged. Propagate lower/upper positive
// sums through the exact vLLM butterfly and ordered denominator recurrence.
// Floating monotonicity encloses their rounding without a per-step inflation.
__global__ void probabilities_and_denominators(const float* scores, const float* errors,
    const float* maxima, uint16_t* probabilities, float* scales, float* denominators,
    unsigned start, unsigned stride, const unsigned char* table) {
    const unsigned lane = threadIdx.x, row = blockIdx.y * kQueryHeads + blockIdx.x;
    const unsigned tokens = start + blockIdx.y + 1u, tiles = (stride + 31u) / 32u;
    float maximum = -INFINITY, center = 1.0f, low = 1.0f, high = 1.0f;
    for (unsigned tile = 0u; tile < (tokens + 31u) / 32u; ++tile) {
        const unsigned key = tile * 32u + lane, cell = row * stride + key;
        const float next = maxima[size_t(row) * tiles + tile];
        const float alpha = blackwell_attention_exp(maximum - next, table);
        const float value = key < tokens ? blackwell_attention_exp(fminf(scores[cell], next) - next, table) : 0.0f;
        if (key < stride) probabilities[cell] = f32_to_bf16(value);
        const float c = wave_sum(value);
        if (denominators) {
            const auto p = key < tokens ? probability_interval(scores[cell], errors[cell], next, table) : Interval{0.0f, 0.0f};
            const float a = wave_sum(p.low), b = wave_sum(p.high);
            low = low * alpha + a; high = high * alpha + b;
        }
        center = center * alpha + c;
        if (lane == 0u) scales[size_t(row) * (tiles + 1u) + tile] = alpha;
        maximum = next;
    }
    if (lane == 0u) {
        scales[size_t(row) * (tiles + 1u) + tiles] = center;
        if (denominators) {
            denominators[size_t(row) * 2u] = low; denominators[size_t(row) * 2u + 1u] = high;
        }
    }
}

// The reciprocal artifact permits delta in[-1,+1] around rounded1/m. Thus
// its possible nonmonotonic excursion is at most two positive FP32 bit steps
// for any exponent in the validated range. Enlarge both endpoint evaluations.
__device__ __forceinline__ Interval reciprocal_interval(Interval den, const unsigned char* table) {
    if (!(den.low >= 1.0f && den.low <= den.high && den.high < 0x1p19f)) return {0.0f, INFINITY};
    if (bound::bits(den.low) == bound::bits(den.high)) {
        const float exact = qrt_sm121_attention_rcp::evaluate(table, den.low);
        return {exact, exact};
    }
    const uint32_t low = bound::bits(qrt_sm121_attention_rcp::evaluate(table, den.high));
    const uint32_t high = bound::bits(qrt_sm121_attention_rcp::evaluate(table, den.low));
    return {bound::value(low - 2u), bound::value(high + 2u)};
}

// Optional precision pass for rows whose output certificate is unresolved.
// A work threshold chooses large interval contributors; it does not admit
// results. The recomputed denominator must pass the output certificate again.
// FullRows is the final correctness fallback and includes all remaining scores.
template<bool FullRows>
__global__ void collect_denominator_refinement(const float* scores, const float* errors,
    const float* maxima, const float* scales, const unsigned* needed_rows, const float* row_budgets,
    unsigned start, unsigned stride, const unsigned char* table, unsigned* indices, unsigned* count) {
    const unsigned lane = threadIdx.x, row = blockIdx.y * kQueryHeads + blockIdx.x;
    if (!needed_rows[row]) return;
    const unsigned tokens = start + blockIdx.y + 1u, tiles = (stride + 31u) / 32u;
    float threshold = 0.0f;
    if constexpr (!FullRows) {
        const float budget = row_budgets ? row_budgets[row]
            : scales[size_t(row) * (tiles + 1u) + tiles] * 0x1p-24f;
        threshold = budget / float(tokens);
    }
    for (unsigned tile = 0u; tile < (tokens + 31u) / 32u; ++tile) {
        const unsigned key = tile * 32u + lane, cell = row * stride + key;
        bool selected = key < tokens && errors[cell] != 0.0f;
        if constexpr (!FullRows) {
            if (selected) {
                const auto p = probability_interval(scores[cell], errors[cell], maxima[size_t(row) * tiles + tile], table);
                selected = !(p.high - p.low <= threshold);
            }
        }
        collect(cell, selected, indices, count);
    }
}

inline size_t probability_scratch_elements(unsigned queries, unsigned stride) {
    if (!queries || queries > 128u || stride < queries || stride > 8192u) return 0u;
    const size_t rows = size_t(queries) * kQueryHeads;
    return rows * stride * 2u + rows * ((stride + 31u) / 32u) + 1u;
}

// Experimental floating route: probabilities and alphas use certified score
// repair; the unrounded denominator remains approximate. This helper makes no
// canonical-output claim. Product acceptance requires the full GB10 token
// boundary. All native-PV arithmetic/envelopes and exact-PV replay are shared
// with the existing qualified path; no full exact-score buffer is consulted.
inline int launch_probability_attention(const uint16_t* q, const uint16_t* k,
    const uint16_t* v, float* output, hipStream_t stream, unsigned start, unsigned queries,
    unsigned output_start, const unsigned char* exp2, const unsigned char* reciprocal,
    unsigned layout, float* scratch, size_t scratch_elements, float* work, size_t work_elements,
    const uint16_t* transposed_key, unsigned key_stride, const uint16_t* transposed_value,
    unsigned value_stride, bool final_pv_bound, bool direct_pv_operands,
    float* raw_accumulator = nullptr, float* raw_denominator = nullptr) {
    if (!q || !k || !v || !output || !exp2 || !reciprocal || !scratch || !work || !transposed_key ||
        !queries || queries > 128u || start >= 8192u || queries > 8192u - start ||
        output_start >= kSplitMaxTokens || queries > kSplitMaxTokens - output_start || (layout != 22u && layout != 24u))
        return int(hipErrorInvalidValue);
    const unsigned stride = start + queries;
    if (key_stride < stride || key_stride > 8192u ||
        (transposed_value ? value_stride < stride || value_stride > kSplitMaxTokens : value_stride != 0u) ||
        scratch_elements < split_scratch_elements(queries, stride, layout) ||
        work_elements < probability_scratch_elements(queries, stride)) return int(hipErrorInvalidValue);
    const size_t rows = size_t(queries) * kQueryHeads, cells = rows * stride;
    auto* score_errors = work;
    auto* score_indices = reinterpret_cast<unsigned*>(score_errors + cells);
    auto* maxima = reinterpret_cast<float*>(score_indices + cells);
    auto* score_count = reinterpret_cast<unsigned*>(maxima + rows * ((stride + 31u) / 32u));
    auto* probabilities = reinterpret_cast<uint16_t*>(scratch + cells);
    auto* scales = reinterpret_cast<float*>(probabilities + cells);
    hipLaunchKernelGGL(native_scores,
        dim3((stride + kIntegerMatrixColumns - 1u) / kIntegerMatrixColumns, kQueryHeads, (queries + 15u) / 16u),
        dim3(kThreads), 0u, stream, q, transposed_key, scratch, score_errors, start, queries, stride, key_stride);
    auto status = hipGetLastError(); if (status != hipSuccess) return int(status);
    for (unsigned pass = 0u; pass < 2u; ++pass) {
        status = hipMemsetAsync(score_count, 0, sizeof(unsigned), stream); if (status != hipSuccess) return int(status);
        if (pass == 0u) {
            hipLaunchKernelGGL(collect_maxima, dim3(kQueryHeads, queries), dim3(32u), 0u, stream,
                scratch, score_errors, start, stride, score_indices, score_count);
        } else {
            hipLaunchKernelGGL(collect_probabilities, dim3(kQueryHeads, queries), dim3(32u), 0u, stream,
                scratch, score_errors, maxima, start, stride, exp2, score_indices, score_count);
        }
        status = hipGetLastError(); if (status != hipSuccess) return int(status);
        hipLaunchKernelGGL(repair_scores, dim3(256u), dim3(kThreads), 0u, stream,
            q, k, scratch, score_errors, start, stride, score_indices, score_count);
        status = hipGetLastError(); if (status != hipSuccess) return int(status);
    }
    hipLaunchKernelGGL(probabilities_and_denominators, dim3(kQueryHeads, queries), dim3(32u), 0u, stream,
        scratch, score_errors, maxima, probabilities, scales, nullptr, start, stride, exp2);
    status = hipGetLastError(); if (status != hipSuccess) return int(status);
    auto* errors = scales + rows * ((stride + 31u) / 32u + 1u);
    if (direct_pv_operands && final_pv_bound) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_mantissa_value_kernel<true, false, true, true, true>),
            dim3(kHeadDim / kIntegerMatrixColumns, kQueryHeads, (queries + 15u) / 16u), dim3(kThreads), 0u, stream,
            v, probabilities, scales, output, start, queries, output_start, stride,
            reciprocal, raw_accumulator, raw_denominator, nullptr, nullptr, errors);
    } else if (direct_pv_operands) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_mantissa_value_kernel<true, false, true, false, true>),
            dim3(kHeadDim / kIntegerMatrixColumns, kQueryHeads, (queries + 15u) / 16u), dim3(kThreads), 0u, stream,
            v, probabilities, scales, output, start, queries, output_start, stride,
            reciprocal, raw_accumulator, raw_denominator, nullptr, nullptr, errors);
    } else if (final_pv_bound) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_mantissa_value_kernel<true, false, true, true>),
            dim3(kHeadDim / kIntegerMatrixColumns, kQueryHeads, (queries + 15u) / 16u), dim3(kThreads), 0u, stream,
            v, probabilities, scales, output, start, queries, output_start, stride,
            reciprocal, raw_accumulator, raw_denominator, nullptr, nullptr, errors);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_mantissa_value_kernel<true, false, true>),
            dim3(kHeadDim / kIntegerMatrixColumns, kQueryHeads, (queries + 15u) / 16u), dim3(kThreads), 0u, stream,
            v, probabilities, scales, output, start, queries, output_start, stride,
            reciprocal, raw_accumulator, raw_denominator, nullptr, nullptr, errors);
    }
    status = hipGetLastError(); if (status != hipSuccess) return int(status);
    auto* indices = reinterpret_cast<unsigned*>(errors + rows * kHeadDim);
    auto* count = indices + rows * kHeadDim;
    return launch_compacted_pv_replay(v, probabilities, scales, output, start, queries, output_start,
        stride, reciprocal, raw_accumulator, raw_denominator, errors, indices, count, stream, nullptr,
        transposed_value, value_stride);
}
} // namespace qrt_selective_qk
#endif
