#pragma once
#include "sm121_staged_half_projection.h"
#include "sm121_tiled_projection.h"

// Component only. Reuse the original candidate bitmap and lossless operands.
// Each subgroup owns one ascending K16 recurrence; dense tiles split into
// deterministic groups of 64 candidates instead of reserving multiple carries.
namespace qrt_sm121_cooperative_half_projection {
namespace staged=qrt_sm121_staged_half_projection;
using Row=staged::Row;
using Value=staged::Value;
constexpr unsigned threads=256u,outputs=64u;

__host__ __device__ inline unsigned row_mask(const unsigned* mask,unsigned rows,
    unsigned tokens,unsigned row,unsigned token,unsigned tile_rows) {
    if(row>=rows || token>=tokens)return 0u;
    const size_t cell=size_t(token)*rows+row,cells=size_t(rows)*tokens;
    const unsigned shift=unsigned(cell&31u),valid=rows-row<tile_rows?rows-row:tile_rows;
    unsigned bits=mask[cell/32u]>>shift;
    if(shift && cell/32u+1u<(cells+31u)/32u)bits|=mask[cell/32u+1u]<<(32u-shift);
    return valid==32u?bits:bits&((1u<<valid)-1u);
}

template<unsigned Rows,unsigned KGroups,bool Trace>
__global__ __launch_bounds__(256) void replay_kernel(const Row* weights,const Row* inputs,
    const unsigned* mask,float* output,uint32_t* trace,unsigned rows,unsigned tokens,unsigned width) {
    static_assert(Rows==16u || Rows==32u);
    static_assert(KGroups==8u || KGroups==16u);
    constexpr unsigned Tokens=16u,parts=Rows*Tokens/outputs;
    __shared__ unsigned count,candidates[outputs];
    __shared__ Row operands[(Rows+Tokens)*KGroups];
    const unsigned tid=threadIdx.x,slot=tid/4u,lane=tid&3u;
    const unsigned row_tiles=(rows+Rows-1u)/Rows,tile=blockIdx.x/parts;
    const unsigned first_row=(tile%row_tiles)*Rows,first_token=(tile/row_tiles)*Tokens;
    if(!tid){
        const unsigned begin=(blockIdx.x%parts)*outputs;unsigned seen=0u,written=0u;
        for(unsigned token=0u;token<Tokens && written<outputs;++token){
            unsigned bits=row_mask(mask,rows,tokens,first_row,first_token+token,Rows);
            const unsigned n=__popc(bits);
            if(seen+n<=begin){seen+=n;continue;}
            while(bits && written<outputs){
                const unsigned row=unsigned(__ffs(int(bits))-1);bits&=bits-1u;
                if(seen++>=begin)candidates[written++]=token*Rows+row;
            }
        }
        count=written;
    }
    __syncthreads();
    if(!count)return;
    const bool active=slot<count;
    const unsigned cell=active?candidates[slot]:0u,local_row=cell%Rows,local_token=cell/Rows;
    const unsigned total_groups=width/16u;Value carry{0u,-133,false};
    for(unsigned base=0u;base<total_groups;base+=KGroups){
        // Consecutive lanes load consecutive words within each row's K slab.
        // Padded rows/groups use the canonical zero encoding; no source read.
        for(unsigned word=tid;word<(Rows+Tokens)*KGroups*9u;word+=threads){
            const unsigned record=word/9u,index=word%9u,side=record/KGroups;
            const unsigned group=base+record%KGroups;
            const bool weight=side<Rows;const unsigned row=weight?first_row+side:first_token+side-Rows;
            uint32_t value=index==8u?unsigned(uint16_t(-15)):0u;
            if(group<total_groups && row<(weight?rows:tokens)){
                const Row* source=(weight?weights:inputs)+size_t(row)*total_groups+group;
                __builtin_memcpy(&value,reinterpret_cast<const unsigned char*>(source)+index*4u,4u);
            }
            __builtin_memcpy(reinterpret_cast<unsigned char*>(operands+record)+index*4u,&value,4u);
        }
        __syncthreads();
        if(active){
            const unsigned groups=total_groups-base<KGroups?total_groups-base:KGroups;
            for(unsigned k=0u;k<groups;k+=2u){
                staged::LaneOperands pair[2];
                const unsigned count=groups-k<2u?groups-k:2u;
#pragma unroll
                for(unsigned i=0u;i<2u;++i)if(i<count)
                    pair[i]=staged::load(operands[(Rows+local_token)*KGroups+k+i],operands[local_row*KGroups+k+i]);
#pragma unroll
                for(unsigned i=0u;i<2u;++i)if(i<count){
                    carry=staged::accumulate(carry,pair[i]);
                    if constexpr(Trace)if(!lane){
                        const size_t index=(size_t(first_token+local_token)*rows+first_row+local_row)*total_groups+base+k+i;
                        trace[index*3u]=carry.significand;trace[index*3u+1u]=uint32_t(int32_t(carry.exponent));trace[index*3u+2u]=unsigned(carry.negative);
                    }
                }
            }
        }
        __syncthreads();
    }
    if(active && !lane)output[size_t(first_token+local_token)*rows+first_row+local_row]=
        qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}

template<unsigned Rows,unsigned KGroups>
inline hipError_t dispatch(const Row* weights,const Row* inputs,const unsigned* mask,
    float* output,uint32_t* trace,unsigned rows,unsigned tokens,unsigned width,hipStream_t stream){
    const unsigned blocks=((rows+Rows-1u)/Rows)*((tokens+15u)/16u)*(Rows*16u/outputs);
    if(trace){
        hipLaunchKernelGGL(HIP_KERNEL_NAME(replay_kernel<Rows,KGroups,true>),dim3(blocks),dim3(threads),0u,stream,
            weights,inputs,mask,output,trace,rows,tokens,width);
    }else{
        hipLaunchKernelGGL(HIP_KERNEL_NAME(replay_kernel<Rows,KGroups,false>),dim3(blocks),dim3(threads),0u,stream,
            weights,inputs,mask,output,trace,rows,tokens,width);
    }
    return hipGetLastError();
}

inline hipError_t launch(const Row* weights,size_t weight_records,const Row* inputs,size_t input_records,
    const unsigned* mask,size_t mask_words,float* output,size_t output_cells,unsigned rows,unsigned tokens,
    unsigned width,unsigned variant,hipStream_t stream,uint32_t* trace=nullptr,size_t trace_words=0u){
    const size_t cells=size_t(rows)*tokens;
    if(!weights || !inputs || !mask || !output || !rows || rows>16384u || !tokens || tokens>8192u ||
        !width || width>8192u || width%16u || variant>2u || weight_records<size_t(rows)*(width/16u) ||
        input_records<size_t(tokens)*(width/16u) || mask_words<(cells+31u)/32u || output_cells<cells ||
        (trace && trace_words<cells*(width/16u)*3u) || (!trace && trace_words))return hipErrorInvalidValue;
    if(variant==0u)return dispatch<16u,8u>(weights,inputs,mask,output,trace,rows,tokens,width,stream);
    if(variant==1u)return dispatch<32u,8u>(weights,inputs,mask,output,trace,rows,tokens,width,stream);
    return dispatch<32u,16u>(weights,inputs,mask,output,trace,rows,tokens,width,stream);
}
} // namespace qrt_sm121_cooperative_half_projection
