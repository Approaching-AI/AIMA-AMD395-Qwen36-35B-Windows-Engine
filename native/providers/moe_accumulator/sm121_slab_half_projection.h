#pragma once
#include "sm121_staged_half_projection.h"
#include "sm121_slab_half_layout.h"

// Component candidate. The selected indices and every ordered K16 arithmetic
// operation stay unchanged. Only the two operand address maps are replaced.
namespace qrt_sm121_slab_half_projection {
namespace layout = qrt_sm121_slab_half_layout;
namespace staged = qrt_sm121_staged_half_projection;
using Stats = staged::Stats;

template<unsigned Rows>
__global__ void prepare_rows(const uint16_t* input, uint32_t* output,
    unsigned rows, unsigned width) {
    const size_t record = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t count = layout::records<Rows>(rows,width);
    if (record >= count) return;
    const unsigned row = layout::row<Rows>(width,record), group = layout::group<Rows>(width,record);
    staged::Row packed{};
    packed.control = uint16_t(-15);
    if (row < rows && group < width/16u)
        packed = staged::half::prepare(input+size_t(row)*width+group*16u);
#pragma unroll
    for (unsigned i=0u;i<8u;++i) output[record*8u+i] = packed.pairs[i];
    output[count*8u+record] = packed.control;
}
template<unsigned Rows>
inline hipError_t prepare(const uint16_t* input, size_t input_words,
    uint32_t* output, size_t output_words, unsigned rows, unsigned width, hipStream_t stream) {
    const size_t count = layout::records<Rows>(rows,width);
    if (!input || !output || !count || input_words < size_t(rows)*width || output_words < count*9u)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prepare_rows<Rows>),dim3(unsigned((count+255u)/256u)),
        dim3(256u),0u,stream,input,output,rows,width);
    return hipGetLastError();
}

// The caller validates extents and row indices. Consecutive iterations advance
// one whole row slab; the two staged records remain adjacent in each plane.
template<unsigned LeftRows,unsigned RightRows,bool Audit=false>
__device__ __forceinline__ float dot(const uint32_t* left, const uint32_t* right,
    unsigned left_rows, unsigned right_rows, unsigned left_row, unsigned right_row,
    unsigned width, uint32_t* trace=nullptr, Stats* statistics=nullptr) {
    const unsigned lane=threadIdx.x&3u,groups=width/16u;
    const uint32_t* left_controls=left+layout::records<LeftRows>(left_rows,width)*8u;
    const uint32_t* right_controls=right+layout::records<RightRows>(right_rows,width)*8u;
    size_t li=layout::offset<LeftRows>(left_rows,width,left_row,0u);
    size_t ri=layout::offset<RightRows>(right_rows,width,right_row,0u);
    staged::Value carry{0u,-133,false}; Stats counts;
#pragma unroll 1
    for (unsigned base=0u;base<groups;base+=2u) {
        staged::LaneOperands operands[2];
        const unsigned count=groups-base<2u?groups-base:2u;
#pragma unroll
        for (unsigned i=0u;i<2u;++i) if (i<count) {
            __builtin_memcpy(operands[i].left,left+(li+i)*8u+lane*2u,8u);
            __builtin_memcpy(operands[i].right,right+(ri+i)*8u+lane*2u,8u);
            operands[i].left_control=left_controls[li+i];
            operands[i].right_control=right_controls[ri+i];
        }
#pragma unroll
        for (unsigned i=0u;i<2u;++i) if (i<count) {
            bool transformed;
            carry=staged::accumulate(carry,operands[i],Audit?&transformed:nullptr);
            if constexpr(Audit) {
                counts.transformed+=transformed; counts.original+=!transformed;
                if (!lane && trace) {
                    trace[3u*(base+i)]=carry.significand;
                    trace[3u*(base+i)+1u]=uint32_t(int32_t(carry.exponent));
                    trace[3u*(base+i)+2u]=unsigned(carry.negative);
                }
            }
        }
        li+=LeftRows*2u; ri+=RightRows*2u;
    }
    if constexpr(Audit) if (!lane && statistics) *statistics=counts;
    return lane?0.0f:qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
} // namespace qrt_sm121_slab_half_projection
