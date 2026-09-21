#pragma once
#include <hip/hip_runtime.h>
#include "sm121_partial_matrix_group.h"
#include "sm121_subgroup.h"

// Isolated projection replay. One wave owns a 16x16 tile, keeping the eight
// ordered K16 carries belonging to each lane in registers. The original
// candidate bitmap remains authoritative; unsupported whole rows use a
// separate original four-lane replay. No model dispatch selects this provider.
namespace qrt_sm121_partial_wave_projection {
namespace partial = qrt_sm121_partial_matrix_group;
namespace group = qrt_sm121_compact_matrix_group;
namespace original = qrt_q1_moe_hawkeye;
using Row = partial::Row;
using I4 = int __attribute__((ext_vector_type(4)));
using I8 = int __attribute__((ext_vector_type(8)));
using F8 = float __attribute__((ext_vector_type(8)));

// Both operands are group-major so adjacent wave lanes load adjacent rows.
// A flag certifies every original operand in the complete row, including
// groups that the bitmap might otherwise allow a kernel to skip.
__global__ __launch_bounds__(128) void prepare(const uint16_t* raw, Row* packed,
    unsigned* flags, unsigned rows, unsigned width) {
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    unsigned invalid = 0u;
    for (unsigned g = threadIdx.x; g < width/16u; g += 128u) {
        uint16_t words[16];
#pragma unroll
        for (unsigned i = 0u; i < 16u; ++i)
            words[i] = raw[size_t(row)*width+g*16u+i];
        const Row value = partial::prepare(words);
        invalid |= !group::compact::unit(value.common.encoded);
        packed[size_t(g)*rows+row] = value;
    }
    __shared__ unsigned rejected[4];
    const unsigned mask = __ballot(invalid != 0u);
    if (!(threadIdx.x&31u)) rejected[threadIdx.x/32u] = mask;
    __syncthreads();
    if (!threadIdx.x) flags[row] = !(rejected[0]|rejected[1]|rejected[2]|rejected[3]);
}

__device__ __forceinline__ Row broadcast(const Row& row, unsigned source) {
    Row result;
    result.common.encoded.control = __shfl(row.common.encoded.control,source);
    result.exceptions = __shfl(row.exceptions,source);
#pragma unroll
    for (unsigned i = 0u; i < 4u; ++i)
        result.common.encoded.exponents[i] = __shfl(row.common.encoded.exponents[i],source);
#pragma unroll
    for (unsigned i = 0u; i < 2u; ++i)
        result.common.trailing[i] = __shfl(row.common.trailing[i],source);
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i)
        result.common.encoded.pairs[i] = __shfl(row.common.encoded.pairs[i],source);
    return result;
}
__device__ __forceinline__ void digits(const Row& row, I4& high, I4& low) {
#pragma unroll
    for (unsigned i = 0u; i < 4u; ++i) {
        const uint32_t a = partial::coefficient_pair(row,2u*i);
        const uint32_t b = partial::coefficient_pair(row,2u*i+1u);
        low[i] = int((a&0x00ff00ffu)|((b&0x00ff00ffu)<<8u));
        high[i] = int(((a>>8u)&0x00ff00ffu)|(b&0xff00ff00u));
    }
}
__device__ __forceinline__ float raw_group(float carry, const Row& a, const Row& b) {
    qrt_sm121_float_alignment::Group raw;
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i)
        raw.set(i,partial::original(a,i),partial::original(b,i));
    return qrt_sm121_narrow_f32_carry::accumulate(carry,raw);
}
__device__ __forceinline__ float raw_group(float carry,const uint32_t* a,const uint32_t* b) {
    qrt_sm121_float_alignment::Group raw;
#pragma unroll
    for (unsigned i=0u;i<16u;++i)
        raw.set(i,uint16_t(a[i/2u]>>((i&1u)*16u)),uint16_t(b[i/2u]>>((i&1u)*16u)));
    return qrt_sm121_narrow_f32_carry::accumulate(carry,raw);
}

template<bool Trace, bool ForceOriginal>
__global__ __launch_bounds__(128) void replay(const Row* weights, const Row* inputs,
    const unsigned* weight_flags, const unsigned* input_flags, const unsigned* bitmap,
    float* output, uint32_t* trace, unsigned rows, unsigned tokens, unsigned width) {
    const unsigned lane = threadIdx.x%32u, wave = threadIdx.x/32u;
    const unsigned tile = blockIdx.x*4u+wave, row_tiles = (rows+15u)/16u;
    const unsigned first_row = (tile%row_tiles)*16u, first_token = (tile/row_tiles)*16u;
    if (first_token >= tokens) return; // Uniform within each independent wave.
    const unsigned token = first_token+lane%16u;
    unsigned selected = 0u;
#pragma unroll
    for (unsigned item = 0u; item < 8u; ++item) {
        const unsigned row = first_row+item*2u+lane/16u;
        if (token < tokens && row < rows && input_flags[token] && weight_flags[row]) {
            const size_t cell = size_t(token)*rows+row;
            selected |= unsigned((bitmap[cell/32u]>>(cell&31u))&1u)<<item;
        }
    }
    if (!__ballot(selected != 0u)) return;
    F8 carries{};
#pragma unroll 1
    for (unsigned g = 0u; g < width/16u; ++g) {
        const Row a = token < tokens ? inputs[size_t(g)*tokens+token] : Row{};
        const unsigned source_row = first_row+lane%16u;
        const Row b = source_row < rows ? weights[size_t(g)*rows+source_row] : Row{};
        if constexpr (ForceOriginal) {
            uint32_t left[8],source[8];
#pragma unroll
            for (unsigned pair=0u;pair<8u;++pair) {
                left[pair]=uint32_t(partial::original(a,2u*pair))|
                    (uint32_t(partial::original(a,2u*pair+1u))<<16u);
                source[pair]=uint32_t(partial::original(b,2u*pair))|
                    (uint32_t(partial::original(b,2u*pair+1u))<<16u);
            }
#pragma unroll
            for (unsigned item=0u;item<8u;++item) {
                uint32_t right[8];
#pragma unroll
                for (unsigned pair=0u;pair<8u;++pair)
                    right[pair]=__shfl(source[pair],item*2u+lane/16u);
                if (selected&(1u<<item)) {
                    carries[item]=raw_group(carries[item],left,right);
                    if constexpr (Trace) {
                        const unsigned row=first_row+item*2u+lane/16u;
                        trace[(size_t(token)*rows+row)*(width/16u)+g]=group::f32::bits(carries[item]);
                    }
                }
            }
            continue;
        }
        I8 hh{},hl{},lh{},ll{};
        if constexpr (!ForceOriginal) {
            I4 ah{},al{},bh{},bl{};digits(a,ah,al);digits(b,bh,bl);
            const I8 zero{};
            hh = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,bh,true,ah,zero,false);
            hl = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,bh,false,al,zero,false);
            lh = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,bl,true,ah,zero,false);
            ll = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,bl,false,al,zero,false);
        }
#pragma unroll
        for (unsigned item = 0u; item < 8u; ++item) {
            // Broadcast before any candidate divergence: unselected lanes
            // remain suppliers for every selected cell in this wave.
            const Row right = broadcast(b,item*2u+lane/16u);
            if (selected&(1u<<item)) {
                bool accepted = false;
                if constexpr (!ForceOriginal) {
                    const int64_t sum = int64_t(hh[item])*65536+
                        (int64_t(hl[item])+lh[item])*256+ll[item];
                    float updated;
                    accepted = partial::accumulate(carries[item],a,right,sum,&updated);
                    if (accepted) carries[item] = updated;
                }
                if (!accepted) carries[item] = raw_group(carries[item],a,right);
                if constexpr (Trace) {
                    const unsigned row = first_row+item*2u+lane/16u;
                    trace[(size_t(token)*rows+row)*(width/16u)+g] = group::f32::bits(carries[item]);
                }
            }
        }
    }
#pragma unroll
    for (unsigned item = 0u; item < 8u; ++item) if (selected&(1u<<item))
        output[size_t(token)*rows+first_row+item*2u+lane/16u] = carries[item];
}

template<bool Trace>
__global__ void unsupported(const uint16_t* weights, const uint16_t* inputs,
    const unsigned* weight_flags, const unsigned* input_flags, const unsigned* indices,
    float* output, uint32_t* trace, unsigned rows, unsigned width, unsigned count) {
    const unsigned slot = (blockIdx.x*blockDim.x+threadIdx.x)/4u;
    if (slot >= count) return;
    const unsigned cell = indices[slot],row = cell%rows,token = cell/rows,lane = threadIdx.x&3u;
    if (weight_flags[row] && input_flags[token]) return;
    const auto* left = inputs+size_t(token)*width;
    const auto* right = weights+size_t(row)*width;
    if constexpr (!Trace) {
        const float value = qrt_sm121_subgroup::dot<4u>(left,right,width);
        if (!lane) output[cell] = value;
    } else {
        original::Value carry{0u,-133,false};
        for (unsigned g = 0u; g < width/16u; ++g) {
            uint32_t products[4];
#pragma unroll
            for (unsigned i = 0u; i < 4u; ++i) products[i] = qrt_sm121_group16::pack_product(
                original::multiply_bf16(left[g*16u+lane*4u+i],right[g*16u+lane*4u+i],-133));
            carry = qrt_sm121_subgroup::accumulate_products<4u>(carry,products);
            if (!lane) trace[size_t(cell)*(width/16u)+g] = group::f32::bits(original::value_to_float(carry));
        }
        if (!lane) output[cell] = original::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
    }
}

inline bool valid_shape(unsigned rows,unsigned tokens,unsigned width) {
    return rows && rows<=16384u && tokens && tokens<=8192u && width && width<=8192u && !(width%16u);
}
inline hipError_t encode(const uint16_t* raw,size_t raw_words,Row* packed,size_t packed_rows,
    unsigned* flags,size_t flag_words,unsigned rows,unsigned width,hipStream_t stream) {
    if (!raw || !packed || !flags || !valid_shape(rows,1u,width) ||
        raw_words<size_t(rows)*width || packed_rows<size_t(rows)*(width/16u) || flag_words<rows)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(prepare,dim3(rows),dim3(128u),0u,stream,raw,packed,flags,rows,width);
    return hipGetLastError();
}
template<bool ForceOriginal=false>
inline hipError_t launch(const uint16_t* weights,const uint16_t* inputs,
    const Row* packed_weights,const Row* packed_inputs,const unsigned* weight_flags,
    const unsigned* input_flags,const unsigned* bitmap,size_t bitmap_words,
    const unsigned* indices,unsigned count,float* output,size_t output_cells,
    unsigned rows,unsigned tokens,unsigned width,hipStream_t stream,
    uint32_t* trace=nullptr,size_t trace_words=0u) {
    const size_t cells=size_t(rows)*tokens;
    if (!valid_shape(rows,tokens,width) || !weights || !inputs || !packed_weights || !packed_inputs ||
        !weight_flags || !input_flags || !bitmap || !indices || !output || count>cells ||
        bitmap_words<(cells+31u)/32u || output_cells<cells ||
        (trace && trace_words<cells*(width/16u)) || (!trace && trace_words)) return hipErrorInvalidValue;
    if (!count) return hipSuccess;
    const unsigned tiles=((rows+15u)/16u)*((tokens+15u)/16u);
    const dim3 grid((tiles+3u)/4u);
    if (trace) {
        hipLaunchKernelGGL((replay<true,ForceOriginal>),grid,dim3(128u),0u,stream,
            packed_weights,packed_inputs,weight_flags,input_flags,bitmap,output,trace,rows,tokens,width);
    } else {
        hipLaunchKernelGGL((replay<false,ForceOriginal>),grid,dim3(128u),0u,stream,
            packed_weights,packed_inputs,weight_flags,input_flags,bitmap,output,trace,rows,tokens,width);
    }
    auto status=hipGetLastError();if (status!=hipSuccess) return status;
    if (trace) {
        hipLaunchKernelGGL((unsupported<true>),dim3((count*4u+255u)/256u),dim3(256u),0u,stream,
            weights,inputs,weight_flags,input_flags,indices,output,trace,rows,width,count);
    } else {
        hipLaunchKernelGGL((unsupported<false>),dim3((count*4u+255u)/256u),dim3(256u),0u,stream,
            weights,inputs,weight_flags,input_flags,indices,output,trace,rows,width,count);
    }
    return hipGetLastError();
}
} // namespace qrt_sm121_partial_wave_projection
