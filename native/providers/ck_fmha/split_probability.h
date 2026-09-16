#pragma once
#include "blackwell_attention.h"

// Isolated q<=8192 probability producer. Global kernel boundaries replace
// row-resident CTA phases. Only max scans are associative; each K32 butterfly
// and every row's original FP32 denominator recurrence retain their order.
namespace qrt_split_probability {
using namespace qrt_blackwell_attention;
constexpr unsigned threads=256u,waves=threads/32u;
inline size_t words(unsigned start,unsigned queries,unsigned stride){
    if(!queries||queries>128u||start>=8192u||queries>8192u-start||stride!=start+queries)return 0u;
    return size_t(3u)*queries*kQueryHeads*((stride+31u)/32u);
}
__global__ void tile_maxima(const float* scores,float* maxima,unsigned start,unsigned queries,unsigned stride){
    const unsigned lane=threadIdx.x%32u,tile_stride=(stride+31u)/32u;
    const unsigned index=blockIdx.x*waves+threadIdx.x/32u,row=index/tile_stride,tile=index%tile_stride;
    if(row>=queries*kQueryHeads)return;
    const unsigned tokens=start+row/kQueryHeads+1u;
    if(tile>=(tokens+31u)/32u)return;
    const unsigned key=tile*32u+lane;
    float maximum=key<tokens?scores[size_t(row)*stride+key]:-INFINITY;
    for(unsigned mask=16u;mask;mask>>=1u)maximum=fmaxf(maximum,__shfl_xor(maximum,mask,32u));
    if(!lane)maxima[size_t(row)*tile_stride+tile]=maximum;
}
__global__ void prefix_maxima(float* maxima,unsigned start,unsigned stride){
    const unsigned row=blockIdx.x,tile=threadIdx.x,lane=tile%32u,wave=tile/32u;
    const unsigned tile_stride=(stride+31u)/32u,tile_count=(start+row/kQueryHeads+32u)/32u;
    __shared__ float blocks[waves];
    float prefix=tile<tile_count?maxima[size_t(row)*tile_stride+tile]:-INFINITY;
    for(unsigned offset=1u;offset<32u;offset<<=1u){
        const float previous=__shfl_up(prefix,offset,32u);
        if(lane>=offset)prefix=fmaxf(previous,prefix);
    }
    if(lane==31u)blocks[wave]=prefix;
    __syncthreads();
    if(!tile){
        float preceding=-INFINITY;
        for(unsigned i=0u;i<waves;++i){const float current=blocks[i];blocks[i]=preceding;preceding=fmaxf(preceding,current);}
    }
    __syncthreads();
    if(tile<tile_count)maxima[size_t(row)*tile_stride+tile]=fmaxf(blocks[wave],prefix);
}
__global__ void tiles(const float* scores,const float* maxima,uint16_t* probability,
    float* scales,float* alphas,float* sums,unsigned start,unsigned queries,unsigned stride,
    const unsigned char* exp2,bool vllm_sum){
    const unsigned lane=threadIdx.x%32u,tile_stride=(stride+31u)/32u,rows=queries*kQueryHeads;
    const unsigned index=blockIdx.x*waves+threadIdx.x/32u,row=index/tile_stride,tile=index%tile_stride;
    if(row>=rows)return;
    const unsigned tokens=start+row/kQueryHeads+1u;
    if(tile>=(tokens+31u)/32u)return;
    const unsigned key=tile*32u+lane;
    const float next=maxima[size_t(row)*tile_stride+tile];
    const float previous=tile?maxima[size_t(row)*tile_stride+tile-1u]:-INFINITY;
    const float alpha=blackwell_attention_exp(previous-next,exp2);
    const float score=key<tokens?scores[size_t(row)*stride+key]:-INFINITY;
    const float p=key<tokens?blackwell_attention_exp(score-next,exp2):0.0f;
    if(key<stride)probability[size_t(row)*stride+key]=f32_to_bf16(p);
    float sum=p;
    if(vllm_sum){
        constexpr unsigned order[]={1u,4u,2u,16u,8u};
#pragma unroll
        for(unsigned step=0u;step<5u;++step)sum+=__shfl_xor(sum,order[step],32u);
    }else for(unsigned mask=16u;mask;mask>>=1u)sum+=__shfl_xor(sum,mask,32u);
    if(!lane){
        // Tile-major coefficients coalesce adjacent independent row folds.
        alphas[size_t(tile)*rows+row]=alpha;sums[size_t(tile)*rows+row]=sum;
        scales[size_t(row)*(tile_stride+1u)+tile]=alpha;
    }
}
__global__ void fold(const float* alphas,const float* sums,float* scales,
    unsigned start,unsigned queries,unsigned stride){
    const unsigned row=blockIdx.x*blockDim.x+threadIdx.x,rows=queries*kQueryHeads;
    if(row>=rows)return;
    const unsigned tile_stride=(stride+31u)/32u,tile_count=(start+row/kQueryHeads+32u)/32u;
    float running=1.0f;
#pragma unroll 1
    for(unsigned tile=0u;tile<tile_count;++tile){
        const size_t index=size_t(tile)*rows+row;
        running=running*alphas[index]+sums[index];
    }
    scales[size_t(row)*(tile_stride+1u)+tile_stride]=running;
}
inline int launch(const float* scores,uint16_t* probability,float* scales,float* workspace,
    size_t capacity_words,unsigned start,unsigned queries,unsigned stride,
    const unsigned char* exp2,bool vllm_sum,hipStream_t stream){
    const size_t required=words(start,queries,stride);
    if(!required||!scores||!probability||!scales||!workspace||capacity_words<required)return int(hipErrorInvalidValue);
    const unsigned rows=queries*kQueryHeads,entries=unsigned(required/3u);
    float* alphas=workspace+entries;float* sums=alphas+entries;
    hipLaunchKernelGGL(tile_maxima,dim3((entries+waves-1u)/waves),dim3(threads),0u,stream,scores,workspace,start,queries,stride);
    auto status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(prefix_maxima,dim3(rows),dim3(threads),0u,stream,workspace,start,stride);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(tiles,dim3((entries+waves-1u)/waves),dim3(threads),0u,stream,scores,workspace,probability,scales,alphas,sums,start,queries,stride,exp2,vllm_sum);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(fold,dim3((rows+threads-1u)/threads),dim3(threads),0u,stream,alphas,sums,scales,start,queries,stride);
    return int(hipGetLastError());
}
} // namespace qrt_split_probability
