#pragma once
#include "streamed_exact_attention.h"

// Isolated complete producer with sixteen resident queries. Halve the live
// native-PV accumulators and shorten the key window while preserving every
// original K16 boundary, online K32 update and complete raw-dot fallback.
namespace qrt_compact_streamed_attention {
namespace attention = qrt_blackwell_attention;
namespace decoded = qrt_sm121_decoded_bf16;
namespace bound = qrt_sm121_pv_final_bound;

template<unsigned KeyWindow>
__global__ void produce(const uint16_t* query,const uint16_t* transposed_key,
    const uint16_t* value,const uint32_t* packed_query,const uint32_t* packed_key,
    const unsigned* query_flags,const unsigned* key_flags,
    uint16_t* probabilities,float* scales,float* output,float* errors,
    float* raw_accumulator,float* raw_denominator,float* diagnostic_scores,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride,
    const unsigned char* exp2_table,const unsigned char* packed_exp,
    const unsigned char* rcp_table) {
    static_assert(KeyWindow==32u || KeyWindow==64u);
    static_assert(attention::kHeadDim==256u && attention::kQueryHeads==16u && attention::kKvHeads==2u);
    __shared__ uint32_t qvalues[16][256],kvalues[KeyWindow][32];
    __shared__ float scores[16][32],alpha[16],denominator[16];
    __shared__ uint16_t probability[16][32];
    const unsigned thread=threadIdx.x,lane=thread%32u,wave=thread/32u;
    const unsigned qr=thread/16u,kc=thread%16u;
    const unsigned head=blockIdx.x,kv_head=head/8u,row_tile=blockIdx.y*16u;
    const unsigned last_tokens=start+min(row_tile+16u,count);
    const unsigned tiles=(last_tokens+31u)/32u,tile_stride=(stride+31u)/32u;
    for(unsigned cell=thread;cell<16u*256u;cell+=256u){
        const unsigned row=cell/256u,feature=cell%256u;
        qvalues[row][feature]=row_tile+row<count
            ? packed_query[(size_t(start+row_tile+row)*16u+head)*256u+feature]
            : decoded::pack(0u);
    }
    __syncthreads();
    float running_max[2]={-INFINITY,-INFINITY},running_sum[2]={1.0f,1.0f};
    attention::MantissaF32x8 accumulator[2]{},error[2]{};
    for(unsigned tile=0u;tile<tiles;++tile){
        const unsigned key_base=tile*32u,row=row_tile+qr;
        float carry[2]{};bool active[2],fallback[2];
#pragma unroll
        for(unsigned k=0u;k<2u;++k){
            const unsigned key=key_base+kc+k*16u;
            active[k]=row<count && key<stride && key<=start+row;
            fallback[k]=active[k] && (!query_flags[(start+row)*16u+head] || !key_flags[key*2u+kv_head]);
        }
        for(unsigned window=0u;window<256u;window+=KeyWindow){
            for(unsigned cell=thread;cell<KeyWindow*32u;cell+=256u){
                const unsigned feature=cell/32u,column=cell%32u;
                kvalues[feature][column]=key_base+column<stride
                    ? packed_key[(size_t(kv_head)*256u+window+feature)*key_stride+key_base+column]
                    : decoded::pack(0u);
            }
            __syncthreads();
            for(unsigned base=0u;base<KeyWindow;base+=16u){
                if((active[0]&&!fallback[0]) || (active[1]&&!fallback[1])){
                    qrt_sm121_float_alignment::Group groups[2];
#pragma unroll
                    for(unsigned i=0u;i<16u;++i){
                        const uint32_t left=qvalues[qr][window+base+i];
#pragma unroll
                        for(unsigned k=0u;k<2u;++k)
                            decoded::set_packed(groups[k],i,left,kvalues[base+i][kc+k*16u]);
                    }
#pragma unroll
                    for(unsigned k=0u;k<2u;++k)if(active[k]&&!fallback[k]){
                        float next;
                        if(qrt_sm121_f32_carry::accumulate<0u>(carry[k],groups[k],&next))carry[k]=next;
                        else fallback[k]=true;
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for(unsigned k=0u;k<2u;++k){
            const unsigned column=kc+k*16u,key=key_base+column;
            const float score=!active[k] ? -INFINITY : fallback[k]
                ? qrt_decoded_window_qk::raw_dot(query+(size_t(start+row)*16u+head)*256u,
                    transposed_key+size_t(kv_head)*256u*key_stride+key,key_stride)
                : carry[k]*attention::kExactScale;
            scores[qr][column]=score;
            if(diagnostic_scores && row<count && key<stride)
                diagnostic_scores[(size_t(row)*16u+head)*stride+key]=score;
        }
        __syncthreads();
#pragma unroll
        for(unsigned r=0u;r<2u;++r){
            const unsigned local_row=wave+r*8u,query_row=row_tile+local_row;
            const unsigned tokens=start+query_row+1u,key=key_base+lane;
            const float score=query_row<count && key<tokens ? scores[local_row][lane] : -INFINITY;
            float p=0.0f;
            if(query_row<count && tile<(tokens+31u)/32u){
                float next_max=fmaxf(running_max[r],score);
                for(unsigned mask=16u;mask;mask>>=1u)next_max=fmaxf(next_max,__shfl_xor(next_max,mask,32u));
                const float a=qrt_sm121_exp2_native_delta::evaluate(exp2_table,packed_exp,
                    (running_max[r]-next_max)*attention::kExactLog2e);
                p=key<tokens ? qrt_sm121_exp2_native_delta::evaluate(exp2_table,packed_exp,
                    (score-next_max)*attention::kExactLog2e) : 0.0f;
                if(key<stride)probabilities[(size_t(query_row)*16u+head)*stride+key]=attention::f32_to_bf16(p);
                float sum=p;constexpr unsigned order[]={1u,4u,2u,16u,8u};
#pragma unroll
                for(unsigned step=0u;step<5u;++step)sum+=__shfl_xor(sum,order[step],32u);
                running_sum[r]=running_sum[r]*a+sum;running_max[r]=next_max;
                if(!lane){
                    alpha[local_row]=a;denominator[local_row]=running_sum[r];
                    scales[(size_t(query_row)*16u+head)*(tile_stride+1u)+tile]=a;
                }
            }
            probability[local_row][lane]=attention::f32_to_bf16(p);
        }
        __syncthreads();
        for(unsigned part=0u;part<2u;++part){
            attention::NativeOperandRow left{};
#pragma unroll
            for(unsigned i=0u;i<16u;++i)left.original[i]=probability[lane%16u][part*16u+i];
#pragma unroll
            for(unsigned c=0u;c<2u;++c){
                attention::NativeOperandRow right{};
                const unsigned column=(wave+c*8u)*16u+lane%16u;
#pragma unroll
                for(unsigned i=0u;i<16u;++i){
                    const unsigned key=key_base+part*16u+i;
                    right.original[i]=key<last_tokens ? value[(size_t(key)*2u+kv_head)*256u+column] : 0u;
                }
#pragma unroll
                for(unsigned e=0u;e<8u;++e){
                    const unsigned local_row=2u*e+lane/16u,query_row=row_tile+local_row;
                    if(!part && query_row<count && tile<(start+query_row+32u)/32u){
                        error[c][e]=bound::rescale(error[c][e],accumulator[c][e],alpha[local_row]);
                        accumulator[c][e]=bound::multiply(accumulator[c][e],alpha[local_row]);
                    }
                }
                const auto next=attention::blackwell_native_mma(left,right,accumulator[c]);
                const auto magnitudes=attention::blackwell_native_mma<true>(left,right,attention::MantissaF32x8{});
#pragma unroll
                for(unsigned e=0u;e<8u;++e){
                    const unsigned query_row=row_tile+2u*e+lane/16u;
                    if(query_row<count && tile<(start+query_row+32u)/32u){
                        error[c][e]=bound::group(error[c][e],accumulator[c][e],magnitudes[e]);
                        accumulator[c][e]=next[e];
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(unsigned c=0u;c<2u;++c){
#pragma unroll
        for(unsigned e=0u;e<8u;++e){
            const unsigned local_row=2u*e+lane/16u,row=row_tile+local_row;
            const unsigned column=(wave+c*8u)*16u+lane%16u;
            if(row<count){
                const size_t cell=(size_t(row)*16u+head)*256u+column;
                const float reciprocal=qrt_sm121_attention_rcp::evaluate(rcp_table,denominator[local_row]);
                output[cell]=accumulator[c][e]*reciprocal;
                const float final_error=bound::finalize(error[c][e],((start+row+32u)/32u)*2u);
                errors[cell]=qrt_sm121_pv_bound::finish(final_error,accumulator[c][e],reciprocal);
                if(raw_accumulator)raw_accumulator[cell]=accumulator[c][e];
                if(!column){
                    scales[(size_t(row)*16u+head)*(tile_stride+1u)+tile_stride]=denominator[local_row];
                    if(raw_denominator)raw_denominator[size_t(row)*16u+head]=denominator[local_row];
                }
            }
        }
    }
    if(diagnostic_scores)for(unsigned cell=thread;cell<16u*stride;cell+=256u){
        const unsigned row=row_tile+cell/stride,key=cell%stride;
        if(row<count && key>=tiles*32u)diagnostic_scores[(size_t(row)*16u+head)*stride+key]=-INFINITY;
    }
}
} // namespace qrt_compact_streamed_attention
