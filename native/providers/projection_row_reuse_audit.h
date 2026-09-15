#pragma once
#include "projection_row_identity.h"

namespace qrt_projection_row_reuse_audit {
namespace identity=qrt_projection_row_identity;
constexpr unsigned tokens=8192u,guard=128u;
__global__ void fingerprint(const uint16_t* input,unsigned width,identity::Key* output) {
    const unsigned token=blockIdx.x,lane=threadIdx.x;
    __shared__ unsigned first[256],second[256];
    unsigned a=0u,b=0u;
    for(unsigned column=lane;column<width;column+=256u) {
        const auto key=identity::word(input[size_t(token)*width+column],column);
        a^=key.first;b+=key.second;
    }
    first[lane]=a;second[lane]=b;__syncthreads();
    for(unsigned stride=128u;stride;stride>>=1u) {
        if(lane<stride){first[lane]^=first[lane+stride];second[lane]+=second[lane+stride];}
        __syncthreads();
    }
    if(!lane)output[token]={first[0],second[0]};
}
__global__ void verify(const uint16_t* input,unsigned width,const unsigned* representatives,unsigned* differences) {
    const unsigned token=blockIdx.x,lane=threadIdx.x,canonical=representatives[token];
    __shared__ unsigned count[256];unsigned mismatches=0u;
    if(canonical!=token)for(unsigned column=lane;column<width;column+=256u)
        mismatches+=input[size_t(token)*width+column]!=input[size_t(canonical)*width+column];
    count[lane]=mismatches;__syncthreads();
    for(unsigned stride=128u;stride;stride>>=1u){if(lane<stride)count[lane]+=count[lane+stride];__syncthreads();}
    if(!lane)differences[token]=count[0];
}

inline hipError_t run(const uint16_t* input,unsigned rows,unsigned token_count,unsigned width,hipStream_t stream) {
    const char* setting=std::getenv("QRT_QWEN36_Q8192_ROW_REUSE_AUDIT");
    if(!setting||!*setting||!std::strcmp(setting,"0"))return hipSuccess;
    if(std::strcmp(setting,"1"))return hipErrorInvalidValue;
    if(token_count!=tokens)return hipSuccess;
    if(!input||!rows||!width||width>4096u)return hipErrorInvalidValue;
    const auto start=std::chrono::steady_clock::now();
    auto complete=[&](){const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);for(;;){const auto result=hipStreamQuery(stream);if(result!=hipErrorNotReady)return result;if(std::chrono::steady_clock::now()>=deadline)return hipErrorLaunchTimeOut;std::this_thread::yield();}};
    hipError_t status=complete();if(status!=hipSuccess)return status;
    const std::array<size_t,3> bytes={tokens*sizeof(identity::Key),tokens*sizeof(unsigned),tokens*sizeof(unsigned)};
    std::array<void*,3> owned{};std::vector<identity::Key> keys(tokens);std::vector<unsigned> canonical,differences(tokens);
    try {status=[&]()->hipError_t {
        hipError_t result;
        for(unsigned i=0u;i<3u;++i) {
            if((result=hipMalloc(&owned[i],bytes[i]+2u*guard))!=hipSuccess)return result;
            if((result=hipMemsetAsync(owned[i],0xa5,bytes[i]+2u*guard,stream))!=hipSuccess)return result;
        }
        auto* key=reinterpret_cast<identity::Key*>(static_cast<unsigned char*>(owned[0])+guard);
        auto* map=reinterpret_cast<unsigned*>(static_cast<unsigned char*>(owned[1])+guard);
        auto* report=reinterpret_cast<unsigned*>(static_cast<unsigned char*>(owned[2])+guard);
        hipLaunchKernelGGL(fingerprint,dim3(tokens),dim3(256u),0u,stream,input,width,key);
        if((result=hipGetLastError())!=hipSuccess||(result=complete())!=hipSuccess||
            (result=hipMemcpy(keys.data(),key,bytes[0],hipMemcpyDeviceToHost))!=hipSuccess)return result;
        canonical=identity::representatives(keys);
        if((result=hipMemcpyAsync(map,canonical.data(),bytes[1],hipMemcpyHostToDevice,stream))!=hipSuccess)return result;
        hipLaunchKernelGGL(verify,dim3(tokens),dim3(256u),0u,stream,input,width,map,report);
        if((result=hipGetLastError())!=hipSuccess||(result=complete())!=hipSuccess||
            (result=hipMemcpy(differences.data(),report,bytes[2],hipMemcpyDeviceToHost))!=hipSuccess)return result;
        std::array<unsigned char,guard> redzone{};
        for(unsigned i=0u;i<3u;++i)for(size_t offset:{size_t(0u),bytes[i]+guard}) {
            if((result=hipMemcpy(redzone.data(),static_cast<unsigned char*>(owned[i])+offset,guard,hipMemcpyDeviceToHost))!=hipSuccess)return result;
            for(auto byte:redzone)if(byte!=0xa5u)return hipErrorInvalidValue;
        }
        return hipSuccess;
    }();} catch(const std::bad_alloc&) {status=hipErrorOutOfMemory;}
    const auto drained=hipStreamSynchronize(stream);if(status==hipSuccess)status=drained;
    for(auto allocation:owned)if(allocation){const auto freed=hipFree(allocation);if(status==hipSuccess)status=freed;}
    if(status!=hipSuccess)return status;
    unsigned reusable=0u,collisions=0u,largest=0u;uint64_t compared=0u;
    std::vector<unsigned> populations(tokens,1u);
    for(unsigned token=0u;token<tokens;++token)if(canonical[token]!=token) {
        compared+=width;
        if(differences[token])++collisions;
        else {++reusable;++populations[canonical[token]];}
    }
    for(unsigned population:populations)largest=(std::max)(largest,population);
    const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::fprintf(stderr,"BATCH_MARK projection_row_reuse_audit rows=%u tokens=%u k=%u verified_reusable_rows=%u unique_or_unproven_rows=%u largest_verified_class=%u hash_collision_rows=%u full_bf16_words_compared=%llu duplicate_output_cells=%llu workspace_bytes=%zu audit_wall_ms=%.6f hash_only_acceptance=0 candidate_replay_changed=0 inference_inputs_unchanged=1 redzones_pass=1 completed=1\n",
        rows,tokens,width,reusable,tokens-reusable,largest,collisions,(unsigned long long)compared,
        (unsigned long long)(uint64_t(reusable)*rows),bytes[0]+bytes[1]+bytes[2]+6u*guard,wall);
    std::fflush(stderr);return hipSuccess;
}
} // namespace qrt_projection_row_reuse_audit
