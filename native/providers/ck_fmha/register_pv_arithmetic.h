#pragma once
#include "blackwell_attention.h"

// Isolated paired experiment. These copies preserve the original arithmetic,
// masks and launch ownership. Explicit FP32 register multiplication replaces
// only the volatile K32 rounding barrier. Optional per-row reciprocals are
// computed with the original validated table/function, once per query/head.
// Neither this header nor its copies are used by runtime dispatch.
namespace qrt_register_pv {
using namespace qrt_blackwell_attention;
template<bool RegisterRescale, bool SharedReciprocal,
         bool NativeMma = true, bool Prepacked = false, bool BoundError = true,
         bool FinalBound = true, bool DirectOperands = true>
__global__ void native_value(
    const uint16_t* value, const uint16_t* probabilities, const float* scales,
    float* output, unsigned int query_start, unsigned int query_count, unsigned int output_start,
    unsigned int score_stride, const unsigned char* rcp_table,
    float* raw_accumulator, float* raw_denominator,
    const IntegerOperandRow* prepared_probability, const IntegerOperandRow* prepared_value,
    float* error_bounds, const float* row_reciprocals) {
    static_assert(!BoundError || NativeMma);
    static_assert(!FinalBound || BoundError);
    static_assert(!DirectOperands || (NativeMma && !Prepacked));
    using OperandRow = std::conditional_t<NativeMma, NativeOperandRow, IntegerOperandRow>;
    __shared__ OperandRow left[16], right[kIntegerMatrixColumns];
    const unsigned int lane = threadIdx.x % 32u, wave = threadIdx.x / 32u, head = blockIdx.y;
    const unsigned int column_tile = blockIdx.x * kIntegerMatrixColumns;
    const unsigned int query_tile = blockIdx.z * 16u, kv_head = head / (kQueryHeads / kKvHeads);
    const unsigned int tile_stride = (score_stride + kExactTileTokens - 1u) / kExactTileTokens;
    // Complete both K16 groups of the final online K32 tile, including zeros.
    const unsigned int last_tokens = query_start + min(query_tile + 16u, query_count);
    const unsigned int end = ((last_tokens + 31u) / 32u) * 32u;
    MantissaF32x8 accumulator{}, errors{};
    for (unsigned int base = 0u; base < end; base += 16u) {
        NativeOperandRow direct_left{}, direct_right{};
        if constexpr (DirectOperands) {
            // Each WMMA wave reads its own original operands into registers.
            // Duplicate query loads replace CTA-wide shared publication and
            // retirement barriers at every K16. Values retain token-major
            // coalescing across columns. Matrix instructions, K32 rescaling,
            // error arithmetic, masks and endpoint stores are unchanged.
            const unsigned row = query_tile + lane % 16u;
            const unsigned tokens = query_start + row + 1u;
            const unsigned column = column_tile + wave * 16u + lane % 16u;
#pragma unroll
            for (unsigned i = 0u; i < 16u; ++i) {
                const unsigned key = base + i;
                direct_left.original[i] = row < query_count && key < tokens
                    ? probabilities[(size_t(row) * kQueryHeads + head) * score_stride + key] : 0u;
                direct_right.original[i] = key < last_tokens
                    ? value[(size_t(key) * kKvHeads + kv_head) * kHeadDim + column] : 0u;
            }
        } else {
        if (wave == 0u && lane < 16u) {
            const unsigned int row = query_tile + lane, tokens = query_start + row + 1u;
            if constexpr (Prepacked) {
                left[lane] = row < query_count
                    ? prepared_probability[(size_t(row) * kQueryHeads + head) *
                        ((score_stride + 31u) / 32u * 2u) + base / 16u]
                    : IntegerOperandRow{};
            } else {
#pragma unroll
            for (unsigned int i = 0u; i < 16u; ++i) {
                const unsigned int key = base + i;
                left[lane].original[i] = row < query_count && key < tokens
                    ? probabilities[(static_cast<size_t>(row) * kQueryHeads + head) * score_stride + key] : 0u;
            }
            if constexpr (!NativeMma) blackwell_prepare_integer_row(left[lane]);
            }
        }
        if (lane < 16u) {
            const unsigned int row = wave * 16u + lane;
            if constexpr (Prepacked) {
                right[row] = prepared_value[(size_t(base / 16u) * kKvHeads + kv_head) * kHeadDim + column_tile + row];
            } else {
#pragma unroll
            for (unsigned int i = 0u; i < 16u; ++i) {
                const unsigned int key = base + i;
                right[row].original[i] = key < last_tokens
                    ? value[(static_cast<size_t>(key) * kKvHeads + kv_head) * kHeadDim + column_tile + row] : 0u;
            }
            if constexpr (!NativeMma) blackwell_prepare_integer_row(right[row]);
            }
        }
        __syncthreads();
        }
        if constexpr (NativeMma) {
            if (base % 32u == 0u) {
#pragma unroll
                for (unsigned element = 0u; element < 8u; ++element) {
                    const unsigned row = query_tile + 2u * element + lane / 16u;
                    const unsigned tokens = query_start + row + 1u;
                    if (row < query_count && base / 32u < (tokens + 31u) / 32u) {
                        const float alpha = scales[(static_cast<size_t>(row) * kQueryHeads + head) *
                            (tile_stride + 1u) + base / 32u];
                        if constexpr (BoundError) {
                            if constexpr (FinalBound)
                                errors[element] = qrt_sm121_pv_final_bound::rescale(errors[element], accumulator[element], alpha);
                            else
                                errors[element] = qrt_sm121_pv_bound::rescale(errors[element], accumulator[element], alpha);
                        }
                        if constexpr (RegisterRescale)
                            accumulator[element] = qrt_sm121_pv_final_bound::multiply(accumulator[element], alpha);
                        else {
                            volatile float rounded = accumulator[element] * alpha;
                            accumulator[element] = rounded;
                        }
                    }
                }
            }
            const auto& matrix_left = DirectOperands ? direct_left : left[lane % 16u];
            const auto& matrix_right = DirectOperands ? direct_right : right[wave * 16u + lane % 16u];
            const auto next = blackwell_native_mma(matrix_left, matrix_right, accumulator);
            MantissaF32x8 magnitudes{};
            if constexpr (BoundError)
                magnitudes = blackwell_native_mma<true>(matrix_left, matrix_right, MantissaF32x8{});
#pragma unroll
            for (unsigned element = 0u; element < 8u; ++element) {
                const unsigned row = query_tile + 2u * element + lane / 16u;
                const unsigned tokens = query_start + row + 1u;
                if (row < query_count && base / 32u < (tokens + 31u) / 32u) {
                    if constexpr (BoundError) {
                        if constexpr (FinalBound)
                            errors[element] = qrt_sm121_pv_final_bound::group(errors[element], accumulator[element], magnitudes[element]);
                        else
                            errors[element] = qrt_sm121_pv_bound::group(errors[element], accumulator[element], magnitudes[element]);
                    }
                    accumulator[element] = next[element];
                }
            }
        } else {
        const auto matrix = blackwell_integer_prepared_products(left[lane % 16u], right[wave * 16u + lane % 16u]);
#pragma unroll
        for (unsigned int element = 0u; element < 8u; ++element) {
            const unsigned int local_row = 2u * element + lane / 16u, row = query_tile + local_row;
            const unsigned int tokens = query_start + row + 1u;
            if (row < query_count && base / 32u < (tokens + 31u) / 32u) {
                if (base % 32u == 0u) {
                    const float alpha = scales[(static_cast<size_t>(row) * kQueryHeads + head) *
                        (tile_stride + 1u) + base / 32u];
                    if constexpr (RegisterRescale)
                        accumulator[element] = qrt_sm121_pv_final_bound::multiply(accumulator[element], alpha);
                    else {
                        volatile float rounded = accumulator[element] * alpha;
                        accumulator[element] = rounded;
                    }
                }
                const int32_t partials[4] = {matrix.value[0][element], matrix.value[1][element],
                    matrix.value[2][element], matrix.value[3][element]};
                accumulator[element] = blackwell_integer_accumulate(accumulator[element],
                    left[local_row], right[wave * 16u + lane % 16u], partials,
                    base < tokens ? min(16u, tokens - base) : 0u);
            }
        }
        }
        if constexpr (!DirectOperands) __syncthreads();
    }
#pragma unroll
    for (unsigned int element = 0u; element < 8u; ++element) {
        const unsigned int row = query_tile + 2u * element + lane / 16u;
        const unsigned int column = column_tile + wave * 16u + lane % 16u;
        if (row < query_count) {
            const float denominator = scales[(static_cast<size_t>(row) * kQueryHeads + head) *
                (tile_stride + 1u) + tile_stride];
            const size_t index = (static_cast<size_t>(output_start + row) * kQueryHeads + head) * kHeadDim + column;
            output[index] = SharedReciprocal ? accumulator[element] * row_reciprocals[row * kQueryHeads + head]
                : rcp_table ? accumulator[element] * qrt_sm121_attention_rcp::evaluate(rcp_table, denominator)
                                     : accumulator[element] / denominator;
            if constexpr (BoundError) {
                if constexpr (FinalBound)
                    errors[element] = qrt_sm121_pv_final_bound::finalize(errors[element],
                        ((query_start + row + 32u) / 32u) * 2u);
                error_bounds[(size_t(row) * kQueryHeads + head) * kHeadDim + column] =
                    qrt_sm121_pv_bound::finish(errors[element], accumulator[element],
                        SharedReciprocal ? row_reciprocals[row * kQueryHeads + head]
                                         : qrt_sm121_attention_rcp::evaluate(rcp_table, denominator));
            }
            if (raw_accumulator) raw_accumulator[index] = accumulator[element];
            if (raw_denominator && column == 0u)
                raw_denominator[static_cast<size_t>(output_start + row) * kQueryHeads + head] = denominator;
        }
    }
}

template<bool RegisterRescale, bool SharedReciprocal, bool TransposedValue = true, bool AllCells = false>
__global__ void exact_replay(
    const uint16_t* value, const uint16_t* probabilities, const float* scales,
    float* output, unsigned query_start, unsigned output_start, unsigned score_stride,
    const unsigned char* rcp_table, float* raw_accumulator, float* raw_denominator,
    const unsigned* indices, const unsigned* count,
    const uint16_t* transposed_value, unsigned value_stride,
    unsigned all_cells, const float* row_reciprocals) {
    constexpr unsigned lanes = 4u, items = kBlackwellMmaGroup / lanes;
    const unsigned lane = threadIdx.x & (lanes - 1u);
    const unsigned stride = gridDim.x * blockDim.x / lanes;
    for (unsigned slot = (blockIdx.x * blockDim.x + threadIdx.x) / lanes;
         slot < (AllCells ? all_cells : *count); slot += stride) {
        const unsigned cell = AllCells ? slot : indices[slot], column = cell % kHeadDim;
        const unsigned row = cell / kHeadDim, head = row % kQueryHeads;
        const unsigned query = row / kQueryHeads, kv_head = head / (kQueryHeads / kKvHeads);
        const unsigned tokens = query_start + query + 1u;
        const unsigned tile_stride = (score_stride + kExactTileTokens - 1u) / kExactTileTokens;
        const unsigned tile_count = (tokens + kExactTileTokens - 1u) / kExactTileTokens;
        float accumulator = 0.0f;
        for (unsigned tile = 0u; tile < tile_count; ++tile) {
            const float alpha = scales[size_t(row) * (tile_stride + 1u) + tile];
            float scaled;
            if constexpr (RegisterRescale) scaled = qrt_sm121_pv_final_bound::multiply(accumulator, alpha);
            else { volatile float rounded = accumulator * alpha; scaled = rounded; }
            auto partial = qrt_q1_moe_hawkeye::value_from_float(scaled, kBlackwellZeroExponent);
            for (unsigned begin = 0u; begin < kExactTileTokens; begin += kBlackwellMmaGroup) {
                uint32_t products[items];
#pragma unroll
                for (unsigned item = 0u; item < items; ++item) {
                    const unsigned key = tile * kExactTileTokens + begin + lane * items + item;
                    const uint16_t p = key < tokens ? probabilities[size_t(row) * score_stride + key] : 0u;
                    const uint16_t v = key < tokens ? (TransposedValue
                        ? transposed_value[(size_t(kv_head) * kHeadDim + column) * value_stride + key]
                        : value[(size_t(key) * kKvHeads + kv_head) * kHeadDim + column]) : 0u;
                    products[item] = qrt_sm121_group16::pack_product(
                        qrt_q1_moe_hawkeye::multiply_bf16(p, v, kBlackwellZeroExponent));
                }
                partial = qrt_sm121_subgroup::accumulate_products<lanes>(partial, products);
                partial = qrt_sm121_group16::finish_accumulator(partial);
                partial = qrt_q1_moe_hawkeye::value_from_float(
                    qrt_q1_moe_hawkeye::value_to_float(partial), kBlackwellZeroExponent);
            }
            accumulator = qrt_q1_moe_hawkeye::value_to_float(partial);
        }
        if (lane == 0u) {
            const float denominator = scales[size_t(row) * (tile_stride + 1u) + tile_stride];
            const size_t output_index = size_t(output_start) * kQueryHeads * kHeadDim + cell;
            output[output_index] = SharedReciprocal ? accumulator * row_reciprocals[row]
                : rcp_table
                ? accumulator * qrt_sm121_attention_rcp::evaluate(rcp_table, denominator) : accumulator / denominator;
            if (raw_accumulator) raw_accumulator[output_index] = accumulator;
            if (raw_denominator && column == 0u)
                raw_denominator[size_t(output_start) * kQueryHeads + row] = denominator;
        }
    }
}

__global__ void prepare_reciprocals(const float* scales,float* reciprocals,
    unsigned queries,unsigned stride,const unsigned char* table) {
    const unsigned row=blockIdx.x*blockDim.x+threadIdx.x, tiles=(stride+31u)/32u;
    if(row<queries*kQueryHeads)
        reciprocals[row]=qrt_sm121_attention_rcp::evaluate(table,scales[size_t(row)*(tiles+1u)+tiles]);
}

template<bool RegisterRescale,bool SharedReciprocal>
inline int launch(const uint16_t* value,const uint16_t* probability,const float* scales,
    float* output,float* error,unsigned start,unsigned queries,unsigned output_start,
    unsigned stride,const unsigned char* rcp,const uint16_t* transposed_value,unsigned value_stride,
    unsigned* indices,unsigned* count,size_t index_capacity,float* reciprocals,size_t reciprocal_capacity,
    hipStream_t stream,float* raw_accumulator=nullptr,float* raw_denominator=nullptr) {
    const unsigned cells=queries*kQueryHeads*kHeadDim;
    if(!value||!probability||!scales||!output||!error||!rcp||!transposed_value||!indices||!count||
        !queries||queries>128u||!stride||stride>8192u||start>=stride||queries!=stride-start||
        value_stride<stride||value_stride>8192u||output_start>=qrt_sm121_attention_capacity::kTokens||
        queries>qrt_sm121_attention_capacity::kTokens-output_start||index_capacity<cells||
        (SharedReciprocal&&(!reciprocals||reciprocal_capacity<size_t(queries)*kQueryHeads)))
        return int(hipErrorInvalidValue);
    auto status=hipMemsetAsync(count,0,sizeof(unsigned),stream);if(status!=hipSuccess)return int(status);
    if constexpr(SharedReciprocal) {
        hipLaunchKernelGGL(prepare_reciprocals,dim3((queries*kQueryHeads+255u)/256u),dim3(256u),0u,stream,
            scales,reciprocals,queries,stride,rcp);
        status=hipGetLastError();if(status!=hipSuccess)return int(status);
    }
    hipLaunchKernelGGL((native_value<RegisterRescale,SharedReciprocal>),
        dim3(kHeadDim/kIntegerMatrixColumns,kQueryHeads,(queries+15u)/16u),dim3(kThreads),0u,stream,
        value,probability,scales,output,start,queries,output_start,stride,rcp,
        raw_accumulator,raw_denominator,nullptr,nullptr,error,reciprocals);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(blackwell_collect_pv_replay_kernel,dim3((cells+255u)/256u),dim3(256u),0u,stream,
        output,error,output_start,cells,indices,count);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    const unsigned requested=(cells+63u)/64u,blocks=requested<1024u?requested:1024u;
    hipLaunchKernelGGL((exact_replay<RegisterRescale,SharedReciprocal>),dim3(blocks),dim3(kThreads),0u,stream,
        value,probability,scales,output,start,output_start,stride,rcp,raw_accumulator,raw_denominator,
        indices,count,transposed_value,value_stride,0u,reciprocals);
    return int(hipGetLastError());
}
} // namespace qrt_register_pv
