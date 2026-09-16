#pragma once
#include "deferred_qk_fallback.h"
#include "../moe_accumulator/sm121_dyadic_carry_scan.h"

// Isolated parallel replacement for the ordered K16 carry chain. Each lane
// owns one K16 group. Approximate prefix sums predict signs/exponents only;
// an exact dyadic-function scan computes all carries and validates every
// prediction. Any failed domain or classification defers the complete dot.
namespace qrt_dyadic_scan_qk {
namespace exact=qrt_sm121_dyadic_carry_scan;
namespace decoded=qrt_sm121_decoded_bf16;
template<unsigned Groups>
__device__ __forceinline__ exact::Function shuffle_up(exact::Function f,unsigned step) {
    const uint32_t blo=__shfl_up(uint32_t(f.b),step,Groups),bhi=__shfl_up(uint32_t(f.b>>32u),step,Groups);
    const uint32_t clo=__shfl_up(uint32_t(f.c),step,Groups),chi=__shfl_up(uint32_t(f.c>>32u),step,Groups);
    return {uint64_t(blo)|(uint64_t(bhi)<<32u),uint64_t(clo)|(uint64_t(chi)<<32u),__shfl_up(f.s,step,Groups)};
}
template<unsigned Groups>
__device__ __forceinline__ bool all(bool valid) {
    unsigned value=valid;
#pragma unroll
    for(unsigned step=1u;step<Groups;step*=2u)value&=__shfl_xor(value,step,Groups);
    return value!=0u;
}
template<unsigned Groups>
__global__ void scores(const uint32_t* packed_query,const uint32_t* packed_key,
    const unsigned* query_flags,const unsigned* key_flags,float* output,
    unsigned query_start,unsigned query_count,unsigned stride,unsigned key_stride) {
    static_assert(Groups==4u || Groups==8u || Groups==16u);
    constexpr unsigned Rows=4u,Keys=256u/Groups/Rows,Window=Groups*16u;
    __shared__ uint32_t qvalues[Rows*Window],kvalues[Window*Keys];
    const unsigned group=threadIdx.x%Groups,dot=threadIdx.x/Groups;
    const unsigned qr=dot/Keys,kc=dot%Keys;
    const unsigned head=blockIdx.y,kv_head=head/8u;
    const unsigned query_tile=blockIdx.z*Rows,key_tile=blockIdx.x*Keys;
    const unsigned row=query_tile+qr,key=key_tile+kc;
    const bool live=row<query_count && key<stride;
    const bool active=live && key<=query_start+row;
    const size_t cell=(size_t(row)*16u+head)*stride+key;
    const unsigned last_query=query_start+min(query_tile+Rows,query_count)-1u;
    if(key_tile>last_query){if(!group&&live)output[cell]=-INFINITY;return;}
    bool fallback=active && (!query_flags[(query_start+row)*16u+head] || !key_flags[key*2u+kv_head]);
    float carry=0.0f;
    for(unsigned window=0u;window<256u;window+=Window) {
        for(unsigned i=threadIdx.x;i<Rows*Window;i+=256u) {
            const unsigned r=i/Window,c=i%Window;
            qvalues[i]=query_tile+r<query_count
                ?packed_query[(size_t(query_start+query_tile+r)*16u+head)*256u+window+c]:decoded::pack(0u);
        }
        for(unsigned i=threadIdx.x;i<Window*Keys;i+=256u) {
            const unsigned r=i/Keys,c=i%Keys;
            kvalues[i]=key_tile+c<stride
                ?packed_key[(size_t(kv_head)*256u+window+r)*key_stride+key_tile+c]:decoded::pack(0u);
        }
        __syncthreads();
        if(active && !fallback) {
            qrt_sm121_float_alignment::Group products;
            float partial=0.0f;
#pragma unroll
            for(unsigned i=0u;i<16u;++i) {
                decoded::set_packed(products,i,qvalues[qr*Window+group*16u+i],kvalues[(group*16u+i)*Keys+kc]);
                partial+=products.products[i];
            }
#pragma unroll
            for(unsigned step=1u;step<Groups;step*=2u) {
                const float prior=__shfl_up(partial,step,Groups);
                if(group>=step)partial=prior+partial;
            }
            const float predicted_float=carry+partial;
            const float previous=__shfl_up(predicted_float,1u,Groups);
            const auto incoming=qrt_q1_moe_hawkeye::value_from_float(group?previous:carry,-133);
            const auto predicted=qrt_q1_moe_hawkeye::value_from_float(predicted_float,-133);
            const int maximum=max(products.maximum,int(incoming.exponent));
            int base=maximum-25;
            if(incoming.significand)base=min(base,int(incoming.exponent)-23);
            if(predicted.significand)base=min(base,int(predicted.exponent)-23);
#pragma unroll
            for(unsigned step=1u;step<Groups;step*=2u)base=min(base,__shfl_xor(base,step,Groups));
            bool valid=exact::regular(incoming)&&exact::regular(predicted)&&
                maximum>=-101&&maximum<=127&&base>=-149&&maximum-25-base<=30;
            if(all<Groups>(valid)) {
                const float scale=qrt_sm121_float_alignment::from_bits(uint32_t(152-maximum)<<23u);
                uint32_t modulo=0u;
#pragma unroll
                for(unsigned i=0u;i<16u;++i)modulo+=uint32_t(int32_t(products.products[i]*scale));
                const int64_t sum=modulo>0x7fffffffu?-int64_t(0u-modulo):int64_t(modulo);
                exact::Function function;
                valid=exact::make_step(incoming,predicted,maximum,sum,base,&function);
#pragma unroll
                for(unsigned step=1u;step<Groups;step*=2u) {
                    const auto prior=shuffle_up<Groups>(function,step);
                    if(group>=step)function=exact::compose(prior,function);
                }
                uint64_t initial=0u;
                valid=exact::to_integer(qrt_q1_moe_hawkeye::value_from_float(carry,-133),base,&initial)&&valid;
                exact::Value result{0u,-133,false};
                valid=exact::from_integer(exact::evaluate(function,initial),base,&result)&&valid;
                valid=exact::same_class(result,predicted)&&valid;
                fallback=!all<Groups>(valid);
                const float endpoint=qrt_q1_moe_hawkeye::value_to_float(result);
                carry=__shfl(endpoint,Groups-1u,Groups);
            } else fallback=true;
        }
        __syncthreads();
    }
    if(!group&&live)output[cell]=!active?-INFINITY:fallback
        ?qrt_sm121_float_alignment::from_bits(qrt_deferred_qk_fallback::deferred_bits)
        :carry*qrt_blackwell_attention::kExactScale;
}
} // namespace qrt_dyadic_scan_qk
