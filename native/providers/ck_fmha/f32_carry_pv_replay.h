#ifndef QRT_F32_CARRY_PV_REPLAY_H
#define QRT_F32_CARRY_PV_REPLAY_H
#include <algorithm>
#include "blackwell_attention.h"
#include "../moe_accumulator/sm121_decoded_bf16.h"
#include "../moe_accumulator/sm121_f32_carry.h"

// Isolated component. Production dispatch and its candidate selection remain
// unchanged. Every rejected cell restarts the complete original REGISTER PV.
namespace qrt_f32_carry_pv {
namespace decoded = qrt_sm121_decoded_bf16;
namespace fast = qrt_sm121_f32_carry;
constexpr unsigned heads=16u, features=512u, dimensions=256u, threads=256u;
inline unsigned pitch(unsigned tokens) { return (tokens+31u)&~31u; }

struct Workspace {
    uint32_t *probability, *value;
    unsigned *probability_flags, *value_flags, *fallback, *stats;
    size_t probability_words, value_words, probability_flag_words,
        value_flag_words, fallback_words, stats_words;
};

template<bool Probability, bool Store>
__global__ void prepare_rows(const uint16_t* input,uint32_t* output,unsigned* flags,
    unsigned start,unsigned stride,unsigned padded) {
    __shared__ unsigned valid;
    if(!threadIdx.x)valid=1u;
    __syncthreads();
    const unsigned row=blockIdx.x,count=Probability?start+row/heads+1u:stride;
    unsigned eligible=1u;
    for(unsigned k=threadIdx.x;k<padded;k+=threads){
        const uint16_t word=k<count?input[size_t(row)*stride+k]:0u;
        eligible&=unsigned(fast::alignment::eligible(word));
        if constexpr(Store)output[size_t(row)*padded+k]=decoded::pack(word);
    }
    for(unsigned delta=16u;delta;delta>>=1u)eligible&=__shfl_xor(eligible,delta,32u);
    if(!(threadIdx.x&31u)&&!eligible)atomicAnd(&valid,0u);
    __syncthreads();
    if(!threadIdx.x)flags[row]=valid;
}

template<bool Store>
inline int prepare_value(const uint16_t* transposed,unsigned tokens,
    const Workspace& w,hipStream_t stream) {
    if(!transposed||!tokens||tokens>8192u||!w.value_flags||w.value_flag_words<features||
        (Store&&(!w.value||w.value_words<size_t(features)*pitch(tokens))))
        return int(hipErrorInvalidValue);
    hipLaunchKernelGGL((prepare_rows<false,Store>),dim3(features),dim3(threads),0u,stream,
        transposed,w.value,w.value_flags,0u,tokens,pitch(tokens));
    return int(hipGetLastError());
}

template<bool Packed>
__global__ void replay(const uint16_t* p,const uint16_t* tv,const float* scales,
    float* output,float* raw_accumulator,float* raw_denominator,
    unsigned start,unsigned output_start,unsigned stride,unsigned value_stride,
    const unsigned char* rcp,const unsigned* indices,const unsigned* count,Workspace w) {
    constexpr unsigned lanes=4u,items=4u;
    const unsigned lane=threadIdx.x&3u,step=gridDim.x*blockDim.x/lanes;
    const unsigned pp=(stride+31u)&~31u,vp=(value_stride+31u)&~31u;
    const unsigned tiles=(stride+31u)/32u;
    for(unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/lanes;slot<*count;slot+=step){
        const unsigned cell=indices[slot],column=cell%dimensions,row=cell/dimensions;
        const unsigned feature=(row%heads/8u)*dimensions+column,tokens=start+row/heads+1u;
        unsigned rejection=(!w.probability_flags[row]||!w.value_flags[feature])?1u:0u;
        float carry=0.0f;
        for(unsigned tile=0u;tile<(tokens+31u)/32u&&!rejection;++tile){
            carry=qrt_sm121_pv_final_bound::multiply(carry,scales[size_t(row)*(tiles+1u)+tile]);
            const unsigned absolute=fast::bits(carry)&0x7fffffffu;
            // A normal endpoint after every K16 is the fast representation's
            // contract. Rescaling may introduce a subnormal or nonfinite carry.
            if(absolute&&(absolute<0x00800000u||absolute>=0x7f800000u)){rejection=2u;break;}
#pragma unroll
            for(unsigned group=0u;group<2u;++group){
                float products[items];uint32_t first_sign=0u;
                const unsigned carried=fast::bits(carry)&0x7fffffffu;
                int maximum=carried?int(carried>>23u)-127:-133;
#pragma unroll
                for(unsigned i=0u;i<items;++i){
                    const unsigned k=tile*32u+group*16u+lane*items+i;
                    uint32_t left,right;
                    if constexpr(Packed){
                        left=w.probability[size_t(row)*pp+k];
                        right=k<tokens?w.value[size_t(feature)*vp+k]:decoded::pack(0u);
                    }else{
                        left=decoded::pack(k<tokens?p[size_t(row)*stride+k]:0u);
                        right=decoded::pack(k<tokens?tv[size_t(feature)*value_stride+k]:0u);
                    }
                    products[i]=fast::alignment::from_bits(left&0xffff0000u)*
                        fast::alignment::from_bits(right&0xffff0000u);
                    const int exponent=int(int16_t(left))+int(int16_t(right));
                    maximum=exponent>maximum?exponent:maximum;
                    if(!i)first_sign=(left^right)&0x80000000u;
                }
                maximum=qrt_sm121_lane_reduce::maximum<lanes>(maximum);
                if(maximum==-133&&!carried){carry=0.0f;continue;}
                if(maximum < -101 || maximum > 127){rejection=2u;break;}
                const float scale=fast::alignment::from_bits(uint32_t(152-maximum)<<23u);
                uint32_t modulo=0u;
#pragma unroll
                for(unsigned i=0u;i<items;++i)modulo+=uint32_t(int32_t(products[i]*scale));
                modulo=qrt_sm121_lane_reduce::sum<lanes>(modulo);
                modulo+=uint32_t(int32_t(carry*scale));
                // In the decoder's overlap interval all terms have one sign.
                const auto sum=qrt_sm121_group16::decode_modulo_sum(modulo,first_sign!=0u);
                if(!fast::normalize<0u>(sum.magnitude,sum.negative,maximum,&carry)){rejection=2u;break;}
            }
        }
        if(!lane){
            if(rejection){
                w.fallback[atomicAdd(w.stats,1u)]=cell;
                atomicAdd(w.stats+rejection,1u);
            }else{
                const float denominator=scales[size_t(row)*(tiles+1u)+tiles];
                const size_t index=size_t(output_start)*heads*dimensions+cell;
                output[index]=rcp?carry*qrt_sm121_attention_rcp::evaluate(rcp,denominator):carry/denominator;
                if(raw_accumulator)raw_accumulator[index]=carry;
                if(raw_denominator&&!column)raw_denominator[size_t(output_start)*heads+row]=denominator;
            }
        }
    }
}

// The original queue is immutable. Only selected cells can enter fallback.
// V preparation belongs to the caller's per-attention owner; P preparation,
// count reset, fast replay and complete original fallback belong to this call.
template<bool Packed>
inline int launch(const uint16_t* value,const uint16_t* probability,const float* scales,
    float* output,unsigned start,unsigned queries,unsigned output_start,unsigned stride,
    const unsigned char* rcp,float* raw_accumulator,float* raw_denominator,
    const unsigned* indices,const unsigned* count,const uint16_t* tv,unsigned value_stride,
    const Workspace& w,hipStream_t stream) {
    if(!value||!probability||!scales||!output||!indices||!count||!tv||
        !queries||queries>128u||stride>8192u||start>=stride||queries>stride-start||
        value_stride<stride||value_stride>8192u||
        output_start>=qrt_sm121_attention_capacity::kTokens||
        queries>qrt_sm121_attention_capacity::kTokens-output_start||
        !w.probability_flags||w.probability_flag_words<size_t(queries)*heads||
        !w.value_flags||w.value_flag_words<features||!w.fallback||
        w.fallback_words<size_t(queries)*heads*dimensions||!w.stats||w.stats_words<3u||
        (Packed&&(!w.probability||w.probability_words<size_t(queries)*heads*pitch(stride)||
            !w.value||w.value_words<size_t(features)*pitch(value_stride))))
        return int(hipErrorInvalidValue);
    auto status=hipMemsetAsync(w.stats,0,3u*sizeof(unsigned),stream);
    if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL((prepare_rows<true,Packed>),dim3(queries*heads),dim3(threads),0u,stream,
        probability,w.probability,w.probability_flags,start,stride,pitch(stride));
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    const unsigned blocks=std::min(1024u,(queries*heads*dimensions+63u)/64u);
    hipLaunchKernelGGL((replay<Packed>),dim3(blocks),dim3(threads),0u,stream,
        probability,tv,scales,output,raw_accumulator,raw_denominator,start,output_start,
        stride,value_stride,rcp,indices,count,w);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL((qrt_blackwell_attention::blackwell_compacted_pv_replay_kernel<true,false,true>),
        dim3(blocks),dim3(threads),0u,stream,value,probability,scales,output,start,output_start,
        stride,rcp,raw_accumulator,raw_denominator,w.fallback,w.stats,tv,value_stride,0u);
    return int(hipGetLastError());
}
} // namespace qrt_f32_carry_pv
#endif
