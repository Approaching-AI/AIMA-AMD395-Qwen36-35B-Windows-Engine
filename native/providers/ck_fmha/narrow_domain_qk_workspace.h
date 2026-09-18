#pragma once
#include "narrow_domain_qk.h"

namespace qrt_narrow_domain_qk {
constexpr size_t query_domain_words=qrt_prepared_decoded_qk::maximum_tokens*16u;
constexpr size_t key_domain_words=qrt_prepared_decoded_qk::maximum_tokens*2u;
constexpr size_t domain_words=query_domain_words+key_domain_words+2u;
struct Workspace {
    const uint32_t* query=nullptr;
    const uint32_t* key=nullptr;
    const unsigned* query_flags=nullptr;
    const unsigned* key_flags=nullptr;
    unsigned* query_domain=nullptr;
    unsigned* key_domain=nullptr;
    unsigned* tile_counts=nullptr;
    unsigned tokens=0u;
};
inline bool valid(const Workspace& w){
    return w.query&&w.key&&w.query_flags&&w.key_flags&&w.query_domain&&
        w.key_domain&&w.tile_counts&&w.tokens&&w.tokens<=qrt_prepared_decoded_qk::maximum_tokens;
}
inline Workspace attach(const qrt_prepared_decoded_qk::Workspace& decoded,unsigned* domain){
    if(!qrt_prepared_decoded_qk::valid(decoded)||!domain)return {};
    const auto* key=decoded.words+qrt_prepared_decoded_qk::query_words;
    const auto* flags=key+qrt_prepared_decoded_qk::key_words;
    return {decoded.words,key,flags,flags+qrt_prepared_decoded_qk::query_flag_words,
        domain,domain+query_domain_words,domain+query_domain_words+key_domain_words,decoded.tokens};
}
inline int prepare_domain(const uint16_t* query,const uint16_t* key,
    const Workspace& w,hipStream_t stream){
    if(!query||!key||!valid(w))return int(hipErrorInvalidValue);
    auto status=hipMemsetAsync(w.tile_counts,0,2u*sizeof(unsigned),stream);
    if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(classify_rows,dim3(w.tokens*16u),dim3(256u),0u,stream,
        query,w.query_domain,w.tokens*16u);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(classify_rows,dim3(w.tokens*2u),dim3(256u),0u,stream,
        key,w.key_domain,w.tokens*2u);
    return int(hipGetLastError());
}
template<unsigned Queries,unsigned Keys,unsigned Window>
inline int launch_tiled_workspace(const void* state,const uint16_t* query,
    const uint16_t* transposed_key,float* output,hipStream_t stream,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride){
    if(!state||!query||!transposed_key||!output)return int(hipErrorInvalidValue);
    const auto& w=*static_cast<const Workspace*>(state);
    if(!valid(w)||!count||count>128u||start>=w.tokens||count>w.tokens-start||
        stride!=start+count||key_stride!=w.tokens)return int(hipErrorInvalidValue);
    const dim3 grid((stride+Keys*16u-1u)/(Keys*16u),16u,(count+Queries*16u-1u)/(Queries*16u));
    hipLaunchKernelGGL((scores<true,Queries,Keys,Window>),grid,dim3(256u),0u,stream,
        w.query,w.key,w.query_flags,w.key_flags,w.query_domain,w.key_domain,output,w.tile_counts,
        start,count,stride,key_stride);
    auto status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL((scores<false,Queries,Keys,Window>),grid,dim3(256u),0u,stream,
        w.query,w.key,w.query_flags,w.key_flags,w.query_domain,w.key_domain,output,w.tile_counts,
        start,count,stride,key_stride);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(qrt_deferred_qk_fallback::replay_scan,
        dim3((size_t(count)*16u*stride+255u)/256u),dim3(256u),0u,stream,
        query,transposed_key,output,start,count,stride,key_stride);
    return int(hipGetLastError());
}
inline int launch_workspace(const void* state,const uint16_t* query,
    const uint16_t* transposed_key,float* output,hipStream_t stream,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride){
    return launch_tiled_workspace<2u,4u,64u>(state,query,transposed_key,output,stream,
        start,count,stride,key_stride);
}
} // namespace qrt_narrow_domain_qk
