#pragma once
#include "adaptive_denominator_qk.h"
#include "native_delta_probability.h"
#include "attention_context_interval.h"
#include "selected_microtile_qk.h"

// Isolated complete-consumer experiment. External references and original
// comparison outputs never enter these kernels. Work selection is not an
// acceptance criterion: each row must certify or finish original fallback.
namespace qrt_joint_context_qk {
namespace attention=qrt_blackwell_attention;
namespace selective=qrt_selective_qk;
namespace context=qrt_attention_context_interval;
namespace bound=qrt_sm121_pv_bound;

__device__ __forceinline__ float exponential(float x,const unsigned char* original,
    const unsigned char* packed){
    return qrt_sm121_exp2_native_delta::evaluate(original,packed,x*attention::kExactLog2e);
}
__device__ __forceinline__ context::Interval probability_interval(float score,float error,
    float maximum,const unsigned char* original,const unsigned char* packed){
    const auto interval=selective::score_interval(score,error);
    return {exponential(fminf(interval.low,maximum)-maximum,original,packed),
            exponential(fminf(interval.high,maximum)-maximum,original,packed)};
}
__device__ __forceinline__ context::Interval numerator_interval(float center,float error){
    if(!bound::finite(center) || !bound::finite(error) || error<0.0f)return {-INFINITY,INFINITY};
    if(error==0.0f)return {center,center};
    return {bound::next(center-error,false),bound::next(center+error,true)};
}

// Prefix-max repairs have already made every possible prefix maximum exact.
__global__ void collect_probabilities(const float* scores,const float* errors,float* maxima,
    unsigned start,unsigned stride,const unsigned char* original,const unsigned char* packed,
    unsigned* indices,unsigned* count){
    const unsigned lane=threadIdx.x,row=blockIdx.y*16u+blockIdx.x;
    const unsigned tokens=start+blockIdx.y+1u,tiles=(stride+31u)/32u;
    float maximum=-INFINITY;
    for(unsigned tile=0u;tile<(tokens+31u)/32u;++tile){
        const unsigned key=tile*32u+lane,cell=row*stride+key;
        const float exact=key<tokens && errors[cell]==0.0f?scores[cell]:-INFINITY;
        maximum=selective::wave_maximum(fmaxf(maximum,exact));
        if(!lane)maxima[size_t(row)*tiles+tile]=maximum;
        const auto p=key<tokens?probability_interval(scores[cell],errors[cell],maximum,original,packed):context::Interval{0,0};
        selective::collect(cell,key<tokens && errors[cell]!=0.0f &&
            (!isfinite(p.low) || !isfinite(p.high) || attention::f32_to_bf16(p.low)!=attention::f32_to_bf16(p.high)),indices,count);
    }
}

// Unit-denominator scales retain the exact alpha sequence and let the current
// native-PV envelope enclose its raw numerator via finish(error,carry,1).
__global__ void initialize(const float* scores,const float* errors,const float* maxima,
    uint16_t* probabilities,float* scales,float* unit_scales,float* lower,float* upper,
    float* center_denominator,float* maximum_cost,unsigned* pending,unsigned start,unsigned stride,
    const unsigned char* original,const unsigned char* packed){
    const unsigned lane=threadIdx.x,row=blockIdx.y*16u+blockIdx.x;
    const unsigned tokens=start+blockIdx.y+1u,tiles=(stride+31u)/32u;
    float maximum=-INFINITY,center=1.0f,cost=0.0f;
    for(unsigned tile=0u;tile<(tokens+31u)/32u;++tile){
        const unsigned key=tile*32u+lane,cell=row*stride+key;
        const float next=maxima[size_t(row)*tiles+tile];
        const float alpha=exponential(maximum-next,original,packed);
        const float value=key<tokens?exponential(fminf(scores[cell],next)-next,original,packed):0.0f;
        const auto p=key<tokens?probability_interval(scores[cell],errors[cell],next,original,packed):context::Interval{0,0};
        if(key<stride){probabilities[cell]=attention::f32_to_bf16(value);lower[cell]=p.low;upper[cell]=p.high;}
        const float width=p.high-p.low;cost=fmaxf(cost,isfinite(width)&&width>=0.0f?width:INFINITY);
        center=center*alpha+selective::wave_sum(value);
        if(!lane){
            scales[size_t(row)*(tiles+1u)+tile]=alpha;
            unit_scales[size_t(row)*(tiles+1u)+tile]=alpha;
        }
        maximum=next;
    }
    cost=selective::wave_maximum(cost);
    if(!lane){
        scales[size_t(row)*(tiles+1u)+tiles]=center;
        unit_scales[size_t(row)*(tiles+1u)+tiles]=1.0f;
        center_denominator[row]=center;maximum_cost[row]=cost;pending[row]=1u;
    }
}

// An independently scheduled original score repair updates only the chosen
// unrounded probability intervals. Their certified BF16 values stay fixed.
__global__ void update_intervals(const float* scores,const unsigned char* selected,
    const float* maxima,float* lower,float* upper,unsigned start,unsigned queries,unsigned stride,
    const unsigned char* original,const unsigned char* packed){
    const size_t cell=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(cell>=size_t(queries)*16u*stride || !selected[cell])return;
    const unsigned row=unsigned(cell/stride),key=unsigned(cell%stride),tiles=(stride+31u)/32u;
    if(key>start+row/16u)return;
    const float maximum=maxima[size_t(row)*tiles+key/32u];
    const float p=exponential(fminf(scores[cell],maximum)-maximum,original,packed);
    lower[cell]=p;upper[cell]=p;
}

// Original butterfly and ordered denominator recurrence. The optional point
// fallback is legal only after both intervals have collapsed to original
// arithmetic; it preserves subnormal/zero behavior declined by the helper.
__global__ void certify(const float* lower,const float* upper,const float* numerator,
    const float* numerator_error,const float* scales,float* denominators,float* output,
    unsigned* pending,unsigned start,unsigned stride,const unsigned char* reciprocal,
    bool allow_exact_point,unsigned* certified_count){
    const unsigned lane=threadIdx.x,row=blockIdx.y*16u+blockIdx.x;
    if(!pending[row])return;
    const unsigned tokens=start+blockIdx.y+1u,tiles=(stride+31u)/32u;
    float low=1.0f,high=1.0f;
    for(unsigned tile=0u;tile<(tokens+31u)/32u;++tile){
        const unsigned key=tile*32u+lane,cell=row*stride+key;
        const float alpha=scales[size_t(row)*(tiles+1u)+tile];
        low=low*alpha+selective::wave_sum(key<tokens?lower[cell]:0.0f);
        high=high*alpha+selective::wave_sum(key<tokens?upper[cell]:0.0f);
    }
    const context::Interval denominator{low,high};
    const float point=qrt_sm121_attention_rcp::evaluate(reciprocal,low);
    const auto inverse=context::denominator_domain(denominator)?context::reciprocal(denominator,reciprocal):context::Interval{0,INFINITY};
    bool all_stable=true;
    for(unsigned column=lane;column<256u;column+=32u){
        const size_t cell=size_t(row)*256u+column;uint16_t fixed=0u;
        bool stable=context::stable_reciprocal(numerator_interval(numerator[cell],numerator_error[cell]),inverse,&fixed);
        const float value=context::multiply(numerator[cell],point);
        if(!stable && allow_exact_point && numerator_error[cell]==0.0f && bound::bits(low)==bound::bits(high))
            stable=context::denominator_domain(denominator) && bound::finite(value) &&
                (attention::f32_to_bf16(value)&0x7f80u)!=0x7f80u;
        all_stable=all_stable&&stable;output[cell]=value;
    }
    const bool accepted=__ballot(!all_stable)==0u;
    if(!lane){
        denominators[size_t(row)*2u]=low;denominators[size_t(row)*2u+1u]=high;
        if(accepted){pending[row]=0u;atomicAdd(certified_count,1u);}
    }
}

// The first call uses the candidate's point denominator to select numerator
// work. A final call may use the independently recomputed exact denominator.
// Selection itself admits no context and never supplies comparison data.
__global__ void collect_numerators(const float* numerator,const float* errors,
    const float* center_denominator,const float* denominator_intervals,const unsigned* pending,
    unsigned cells,const unsigned char* reciprocal,unsigned* indices,unsigned* count){
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x,row=cell/256u;
    bool selected=false;
    if(cell<cells && pending[row] && errors[cell]!=0.0f){
        const float denominator=denominator_intervals?denominator_intervals[size_t(row)*2u]:center_denominator[row];
        uint16_t fixed=0u;
        selected=!context::stable(numerator_interval(numerator[cell],errors[cell]),
            {denominator,denominator},reciprocal,&fixed);
    }
    selective::collect(cell,selected,indices,count);
}
__global__ void mark_exact_numerators(float* errors,const unsigned* indices,const unsigned* count){
    for(unsigned slot=blockIdx.x*blockDim.x+threadIdx.x;slot<*count;slot+=gridDim.x*blockDim.x)
        errors[indices[slot]]=0.0f;
}
__global__ void select_remaining(const float* score_errors,const unsigned* pending,
    unsigned char* selected,unsigned start,unsigned queries,unsigned stride){
    const size_t cell=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(cell>=size_t(queries)*16u*stride)return;
    const unsigned row=unsigned(cell/stride),key=unsigned(cell%stride);
    selected[cell]=static_cast<unsigned char>(key<=start+row/16u && pending[row] && score_errors[cell]!=0.0f);
}
} // namespace qrt_joint_context_qk
