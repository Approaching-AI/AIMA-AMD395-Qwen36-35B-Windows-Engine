#pragma once
#include "joint_context_qk.h"
#include "pending_row_qk.h"

// One probability traversal repairs every possible prefix maximum and every
// uncertain BF16 probability. It propagates original-order denominator bounds
// in registers. Only unresolved context rows need a full original QK fallback.
// No reference outputs, changed bounds or approximate admission enter this path.
namespace qrt_inline_probability_repair {
using namespace qrt_blackwell_attention;
namespace attention = qrt_blackwell_attention;
namespace joint = qrt_joint_context_qk;
namespace selective = qrt_selective_qk;
namespace context = qrt_attention_context_interval;
namespace bound = qrt_sm121_pv_bound;
namespace prepared = qrt_prepared_decoded_qk;

__device__ __forceinline__ float original_score(const uint16_t* query,
    const uint16_t* transposed_key,const uint32_t* packed_query,const uint32_t* packed_key,
    const unsigned* query_flags,const unsigned* key_flags,unsigned token,unsigned head,
    unsigned key,unsigned key_stride) {
    const unsigned kv_head = head / 8u;
    const size_t query_base = (size_t(token) * 16u + head) * 256u;
    const size_t key_base = size_t(kv_head) * 256u * key_stride + key;
    bool fallback = !query_flags[token * kQueryHeads + head] ||
        !key_flags[key * kKvHeads + kv_head];
    float carry = 0.0f;
    if (!fallback) {
        for (unsigned base = 0u; base < kHeadDim; base += 16u) {
            qrt_sm121_float_alignment::Group group;
#pragma unroll
            for (unsigned i = 0u; i < 16u; ++i)
                prepared::decoded::set_packed(group, i,
                    packed_query[query_base + base + i],
                    packed_key[key_base + size_t(base + i) * key_stride]);
            float next;
            if (!qrt_sm121_f32_carry::accumulate<0u>(carry, group, &next)) {
                fallback = true; break;
            }
            carry = next;
        }
    }
    const float score = fallback ? qrt_decoded_window_qk::raw_dot(
        query + query_base, transposed_key + key_base, key_stride) : carry * kExactScale;
    return score;
}

template<bool Initial>
__global__ void probabilities(const uint16_t* query,const uint16_t* transposed_key,
    const uint32_t* packed_query,const uint32_t* packed_key,
    const unsigned* query_flags,const unsigned* key_flags,float* scores,float* errors,
    uint16_t* probability,float* scales,float* unit_scales,float* center_denominator,
    float* denominators,unsigned* pending,unsigned* statistics,
    unsigned start,unsigned stride,unsigned key_stride,
    const unsigned char* original,const unsigned char* packed) {
    const unsigned lane=threadIdx.x,head=blockIdx.x,row=blockIdx.y*16u+head;
    if constexpr(!Initial)if(!pending[row])return;
    const unsigned token=start+blockIdx.y,tokens=token+1u,tiles=(stride+31u)/32u;
    float maximum=-INFINITY,center=1.0f,low=1.0f,high=1.0f;
    unsigned repaired=0u;
    for(unsigned tile=0u;tile<(tokens+31u)/32u;++tile){
        const unsigned key=tile*32u+lane,cell=row*stride+key;
        const bool active=key<tokens;
        float score=active?scores[cell]:-INFINITY,error=active?errors[cell]:0.0f;
        if constexpr(Initial){
            const auto interval=active?selective::score_interval(score,error):selective::Interval{-INFINITY,-INFINITY};
            const float prefix_lower=selective::wave_maximum(fmaxf(maximum,interval.low));
            // A non-selected interval cannot supply this prefix maximum.
            if(active && error!=0.0f && interval.high>=prefix_lower){
                score=original_score(query,transposed_key,packed_query,packed_key,
                    query_flags,key_flags,token,head,key,key_stride);
                scores[cell]=score;errors[cell]=error=0.0f;++repaired;
            }
        }
        const float exact=active && error==0.0f?score:-INFINITY;
        const float next=selective::wave_maximum(fmaxf(maximum,exact));
        const float alpha=joint::exponential(maximum-next,original,packed);
        auto interval=active?joint::probability_interval(score,error,next,original,packed):context::Interval{0,0};
        if constexpr(Initial){
            if(active && error!=0.0f && (!isfinite(interval.low) || !isfinite(interval.high) ||
                f32_to_bf16(interval.low)!=f32_to_bf16(interval.high))){
                score=original_score(query,transposed_key,packed_query,packed_key,
                    query_flags,key_flags,token,head,key,key_stride);
                scores[cell]=score;errors[cell]=error=0.0f;++repaired;
                const float p=joint::exponential(fminf(score,next)-next,original,packed);
                interval={p,p};
            }
        }
        const float value=active?joint::exponential(fminf(score,next)-next,original,packed):0.0f;
        if(key<stride)probability[cell]=f32_to_bf16(value);
        center=center*alpha+selective::wave_sum(value);
        low=low*alpha+selective::wave_sum(interval.low);
        high=high*alpha+selective::wave_sum(interval.high);
        if(!lane){
            scales[size_t(row)*(tiles+1u)+tile]=alpha;
            unit_scales[size_t(row)*(tiles+1u)+tile]=alpha;
        }
        maximum=next;
    }
    if constexpr(Initial){
        for(unsigned mask=16u;mask;mask>>=1u)repaired+=__shfl_xor(repaired,mask,32u);
    }
    if(!lane){
        scales[size_t(row)*(tiles+1u)+tiles]=center;
        unit_scales[size_t(row)*(tiles+1u)+tiles]=1.0f;
        center_denominator[row]=center;
        denominators[size_t(row)*2u]=low;denominators[size_t(row)*2u+1u]=high;
        if constexpr(Initial){pending[row]=1u;atomicAdd(statistics,repaired);}
        else atomicAdd(statistics+1u,1u);
    }
}

__global__ void certify(const float* numerator,const float* numerator_error,const float* denominators,float* output,
    unsigned* pending,const unsigned char* reciprocal,
    bool allow_exact_point,unsigned* certified_count){
    const unsigned lane=threadIdx.x,row=blockIdx.y*16u+blockIdx.x;
    if(!pending[row])return;
    const float low=denominators[size_t(row)*2u],high=denominators[size_t(row)*2u+1u];
    const context::Interval denominator{low,high};
    const float point=qrt_sm121_attention_rcp::evaluate(reciprocal,low);
    const auto inverse=context::denominator_domain(denominator)?context::reciprocal(denominator,reciprocal):context::Interval{0,INFINITY};
    bool all_stable=true;
    for(unsigned column=lane;column<256u;column+=32u){
        const size_t cell=size_t(row)*256u+column;uint16_t fixed=0u;
        bool stable=context::stable_reciprocal(joint::numerator_interval(numerator[cell],numerator_error[cell]),inverse,&fixed);
        const float value=context::multiply(numerator[cell],point);
        if(!stable && allow_exact_point && numerator_error[cell]==0.0f && bound::bits(low)==bound::bits(high))
            stable=context::denominator_domain(denominator) && bound::finite(value) &&
                (attention::f32_to_bf16(value)&0x7f80u)!=0x7f80u;
        all_stable=all_stable&&stable;output[cell]=value;
    }
    const bool accepted=__ballot(!all_stable)==0u;
    if(!lane){
        if(accepted){pending[row]=0u;atomicAdd(certified_count,1u);}
    }
}

} // namespace qrt_inline_probability_repair
