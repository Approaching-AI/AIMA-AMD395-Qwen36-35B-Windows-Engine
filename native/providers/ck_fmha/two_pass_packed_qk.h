#pragma once
#include "deferred_qk_fallback.h"
#include "../moe_accumulator/sm121_two_pass_packed_group.h"

// Isolated exact-QK candidate. Unsupported rows/endpoints use the existing
// second-kernel original replay. No product dispatcher uses this header.
namespace qrt_two_pass_packed_qk {
namespace decoded=qrt_sm121_decoded_bf16;
template<unsigned Window,unsigned Chunk=4u>
__global__ void scores(const uint32_t* packed_query,const uint32_t* packed_key,
    const unsigned* query_flags,const unsigned* key_flags,float* output,
    unsigned query_start,unsigned query_count,unsigned stride,unsigned key_stride) {
    static_assert(Window==32u || Window==64u || Window==128u);
    constexpr unsigned Rows=16u,Keys=16u;
    __shared__ uint32_t qvalues[Rows*Window],kvalues[Window*Keys];
    const unsigned head=blockIdx.y,kv_head=head/8u;
    const unsigned query_tile=blockIdx.z*Rows,key_tile=blockIdx.x*Keys;
    const unsigned qr=threadIdx.x/Keys,kc=threadIdx.x%Keys;
    const unsigned row=query_tile+qr,key=key_tile+kc;
    const bool live=row<query_count && key<stride;
    const bool active=live && key<=query_start+row;
    const size_t cell=(size_t(row)*16u+head)*stride+key;
    const unsigned last_query=query_start+min(query_tile+Rows,query_count)-1u;
    if(key_tile>last_query){if(live)output[cell]=-INFINITY;return;}
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
            for(unsigned base=0u;base<Window;base+=16u) {
                float next;
                if(!qrt_sm121_two_pass_packed_group::accumulate<Chunk,1u,Keys>(
                    carry,qvalues+qr*Window+base,kvalues+base*Keys+kc,&next)) {fallback=true;break;}
                carry=next;
            }
        }
        __syncthreads();
    }
    if(live)output[cell]=!active?-INFINITY:fallback
        ?qrt_sm121_float_alignment::from_bits(qrt_deferred_qk_fallback::deferred_bits)
        :carry*qrt_blackwell_attention::kExactScale;
}
} // namespace qrt_two_pass_packed_qk
