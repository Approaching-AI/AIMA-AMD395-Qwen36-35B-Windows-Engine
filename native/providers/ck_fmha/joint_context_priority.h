#pragma once
#include "joint_context_qk.h"

// Work-order heuristics only. Neither the reciprocal search nor its floating
// histogram certifies a row. The caller rechecks complete original-order
// denominator intervals and retains full QK/PV fallback after this one pass.
namespace qrt_joint_context_qk {
__device__ __forceinline__ bool row_accepts_inverse(const context::Interval* numerators,
    const uint16_t* expected,context::Interval inverse){
    bool stable=true;
#pragma unroll
    for(unsigned i=0u;i<8u;++i){
        uint16_t fixed=0u;
        const bool ok=context::stable_reciprocal(numerators[i],inverse,&fixed);
        stable=stable && ok && fixed==expected[i];
    }
    return __ballot(!stable)==0u;
}

// Each wave intersects the allowed reciprocal ranges of its 256 output
// columns. Numerators come from the candidate's native envelope or original
// selected PV replay. An unresolved point forces the complete-score cutoff.
__global__ void row_budget(const float* numerator,const float* numerator_error,
    const float* center_denominator,const float* denominator_intervals,const unsigned* pending,
    float* budgets,const unsigned char* reciprocal){
    const unsigned lane=threadIdx.x,row=blockIdx.y*16u+blockIdx.x;
    if(!pending[row]){if(!lane)budgets[row]=0.0f;return;}
    const context::Interval den{denominator_intervals[size_t(row)*2u],denominator_intervals[size_t(row)*2u+1u]};
    if(!context::denominator_domain(den)){if(!lane)budgets[row]=0.0f;return;}
    const float center=fminf(fmaxf(center_denominator[row],den.low),den.high);
    const float inverse=qrt_sm121_attention_rcp::evaluate(reciprocal,center);
    const auto enclosure=context::reciprocal(den,reciprocal);
    if(!context::valid(enclosure) || inverse<enclosure.low || inverse>enclosure.high){if(!lane)budgets[row]=0.0f;return;}
    context::Interval numbers[8];uint16_t targets[8]{};bool point_stable=true;
#pragma unroll
    for(unsigned i=0u;i<8u;++i){
        const size_t cell=size_t(row)*256u+lane+i*32u;
        numbers[i]=numerator_interval(numerator[cell],numerator_error[cell]);
        const bool ok=context::stable_reciprocal(numbers[i],{inverse,inverse},targets+i);
        point_stable=point_stable && ok;
    }
    if(__ballot(!point_stable)){if(!lane)budgets[row]=0.0f;return;}
    const uint32_t middle=bound::bits(inverse);
    uint32_t lo=bound::bits(enclosure.low),hi=middle;
    while(lo<hi){
        const uint32_t probe=lo+(hi-lo)/2u;
        if(row_accepts_inverse(numbers,targets,{bound::value(probe),inverse}))hi=probe;
        else lo=probe+1u;
    }
    const uint32_t allowed_low=lo;lo=middle;hi=bound::bits(enclosure.high);
    while(lo<hi){
        const uint32_t probe=lo+(hi-lo+1u)/2u;
        if(row_accepts_inverse(numbers,targets,{inverse,bound::value(probe)}))lo=probe;
        else hi=probe-1u;
    }
    // Approximate reciprocal inversion affects priority only. The final
    // source-table certificate remains responsible for numerical acceptance.
    const float lower_den=1.0f/bound::value(lo),upper_den=1.0f/bound::value(allowed_low);
    const float budget=fminf(center-lower_den,upper_den-center)*0.25f;
    if(!lane)budgets[row]=isfinite(budget) && budget>0.0f?budget:0.0f;
}

__device__ __forceinline__ unsigned priority_bin(float cost,float maximum){
    if(!(cost>0.0f))return 255u;
    if(!isfinite(cost) || !isfinite(maximum))return 0u;
    const int exponent=int((bound::bits(cost)>>23u)&255u);
    const int top=int((bound::bits(maximum)>>23u)&255u);
    const int bin=(top-exponent)/2;
    return unsigned(bin<0?0:bin>15?15:bin);
}

// Weight each remaining probability interval by subsequent original alpha
// products. A16-bin histogram chooses one cutoff. Exact sums/rounding are not
// inferred from these masses; an inadequate cutoff leads to original fallback.
__global__ void histogram_select(const float* errors,const float* lower,const float* upper,
    const float* scales,const float* maximum_cost,const float* budgets,const unsigned* pending,
    unsigned char* priorities,unsigned char* selected,unsigned start,unsigned stride){
    const unsigned lane=threadIdx.x,row=blockIdx.y*16u+blockIdx.x;
    const unsigned tokens=start+blockIdx.y+1u,tiles=(stride+31u)/32u,live_tiles=(tokens+31u)/32u;
    if(!pending[row]){
        for(unsigned key=lane;key<stride;key+=32u){priorities[size_t(row)*stride+key]=255u;selected[size_t(row)*stride+key]=0u;}
        return;
    }
    float masses[16]{},suffix=1.0f;
    for(unsigned next=live_tiles;next;--next){
        const unsigned tile=next-1u,key=tile*32u+lane,cell=row*stride+key;
        float cost=0.0f;unsigned bin=255u;
        if(key<tokens && errors[cell]!=0.0f){
            const float width=upper[cell]-lower[cell];
            cost=width*suffix;
            if(!isfinite(width) || width<0.0f || !isfinite(suffix) || suffix<0.0f)cost=INFINITY;
            bin=priority_bin(cost,maximum_cost[row]);
        }
        if(key<stride)priorities[cell]=static_cast<unsigned char>(bin);
#pragma unroll
        for(unsigned b=0u;b<16u;++b)masses[b]+=bin==b?cost:0.0f;
        suffix*=scales[size_t(row)*(tiles+1u)+tile];
    }
    for(unsigned key=live_tiles*32u+lane;key<stride;key+=32u)priorities[size_t(row)*stride+key]=255u;
#pragma unroll
    for(unsigned b=0u;b<16u;++b)masses[b]=selective::wave_sum(masses[b]);
    const float budget=budgets[row];const bool full=!(budget>0.0f) || !isfinite(budget);
    int cutoff=-1;float remaining=0.0f;
    for(int bin=15;bin>=0;--bin){
        const float next=remaining+masses[bin];
        if(!isfinite(next) || next>budget){cutoff=bin;break;}
        remaining=next;
    }
    for(unsigned key=lane;key<stride;key+=32u){
        const size_t cell=size_t(row)*stride+key;
        selected[cell]=static_cast<unsigned char>(key<tokens && errors[cell]!=0.0f &&
            (full || int(priorities[cell])<=cutoff));
    }
}
} // namespace qrt_joint_context_qk
