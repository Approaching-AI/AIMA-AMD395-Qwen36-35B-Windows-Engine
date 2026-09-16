#ifndef QRT_PACKED_EXPONENT_QK_H
#define QRT_PACKED_EXPONENT_QK_H
#include "deferred_qk_fallback.h"
#include "../moe_accumulator/sm121_packed_exponents.h"

// Isolated exact scalar QK route. Original BF16 words plus reusable exponent
// summaries use48 bytes per K16 row, versus64 for decoded scalar operands.
// Uncertain maxima, unsupported values and carry endpoints retain complete
// original replay in the established second kernel.
namespace qrt_packed_exponent_qk {
namespace core=qrt_sm121_packed_exponents;
template<bool Key>
__global__ void prepare(const uint16_t* input,core::Row* output,uint16_t* transposed,unsigned tokens){
    constexpr unsigned heads=Key?2u:16u;
    const size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=size_t(tokens)*heads*16u)return;
    const unsigned group=row%16u,head=(row/16u)%heads,token=row/(16u*heads);
    core::Row result{};
#pragma unroll
    for(unsigned i=0u;i<16u;++i){
        result.original[i]=input[row*16u+i];
        if constexpr(Key)if(transposed)transposed[(size_t(head)*256u+group*16u+i)*tokens+token]=result.original[i];
    }
    result.metadata=core::prepare(result.original);
    output[Key?(size_t(head)*16u+group)*tokens+token:row]=result;
}
template<unsigned Window>
__global__ void scores(const core::Row* query,const core::Row* key,float* output,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride){
    static_assert(Window==32u || Window==64u || Window==128u);
    constexpr unsigned Groups=Window/16u,Pairs=Window/2u;
    __shared__ uint32_t qvalues[16][Pairs+1u],kvalues[Pairs][16];
    __shared__ uint32_t qmetadata[Groups][4][16],kmetadata[Groups][4][16];
    const unsigned tid=threadIdx.x,qr=tid/16u,kc=tid%16u,head=blockIdx.y,kv=head/8u;
    const unsigned query_tile=blockIdx.z*16u,key_tile=blockIdx.x*16u;
    const unsigned row=query_tile+qr,column=key_tile+kc;
    const bool live=row<count && column<stride,active=live && column<=start+row;
    const size_t cell=(size_t(row)*16u+head)*stride+column;
    const unsigned last=start+min(query_tile+16u,count)-1u;
    if(key_tile>last){if(live)output[cell]=-INFINITY;return;}
    bool fallback=false;float carry=0.0f;
    for(unsigned window=0u;window<256u;window+=Window){
        for(unsigned item=tid;item<32u*Groups*12u;item+=256u){
            const unsigned word=item%12u,group=(item/12u)%Groups,position=item/(12u*Groups);
            const bool is_query=position<16u;
            const unsigned local=is_query?position:position-16u;
            const unsigned token=is_query?query_tile+local:key_tile+local;
            uint32_t value=0u;
            if(token<(is_query?count:stride)){
                const core::Row* source=is_query?query+(size_t(start+token)*16u+head)*16u+window/16u+group:
                    key+(size_t(kv)*16u+window/16u+group)*key_stride+token;
                __builtin_memcpy(&value,reinterpret_cast<const unsigned char*>(source)+word*4u,4u);
            }
            if(word<8u){
                if(is_query)qvalues[local][group*8u+word]=value;
                else kvalues[group*8u+word][local]=value;
            }else{
                if(is_query)qmetadata[group][word-8u][local]=value;
                else kmetadata[group][word-8u][local]=value;
            }
        }
        __syncthreads();
        if(active && !fallback){
            for(unsigned g=0u;g<Groups;++g){
                core::Metadata a{},b{};
#pragma unroll
                for(unsigned i=0u;i<3u;++i){a.deficits[i]=qmetadata[g][i][qr];b.deficits[i]=kmetadata[g][i][kc];}
                a.maximum=int(qmetadata[g][3][qr]);b.maximum=int(kmetadata[g][3][kc]);
                qrt_sm121_float_alignment::Group products;
                if(!core::maximum(a,b,&products.maximum)){fallback=true;break;}
#pragma unroll
                for(unsigned i=0u;i<8u;++i){
                    const uint32_t left=qvalues[qr][g*8u+i],right=kvalues[g*8u+i][kc];
                    products.products[i*2u]=qrt_sm121_float_alignment::from_bits(left<<16u)*qrt_sm121_float_alignment::from_bits(right<<16u);
                    products.products[i*2u+1u]=qrt_sm121_float_alignment::from_bits(left&0xffff0000u)*qrt_sm121_float_alignment::from_bits(right&0xffff0000u);
                    if(!i)products.first_negative=((left^right)&0x8000u)!=0u;
                }
                float next;
                if(!qrt_sm121_f32_carry::accumulate<0u>(carry,products,&next)){fallback=true;break;}
                carry=next;
            }
        }
        __syncthreads();
    }
    if(live)output[cell]=!active?-INFINITY:fallback?
        qrt_sm121_float_alignment::from_bits(qrt_deferred_qk_fallback::deferred_bits):carry*qrt_blackwell_attention::kExactScale;
}
}
#endif
