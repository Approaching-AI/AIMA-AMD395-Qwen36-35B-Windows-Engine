#include <hip/hip_runtime.h>
#include "../../native/providers/gdn/sm121_exp2_reduced_delta.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace reduced=qrt_sm121_exp2_reduced_delta;
namespace source=qrt_sm121_exp2_interpolated;
namespace raw=qrt_sm121_exp2;
constexpr size_t guard=128u;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void check(hipError_t result){if(result!=hipSuccess)throw std::runtime_error(hipGetErrorString(result));}
struct Device{
    unsigned char* allocation=nullptr;size_t size;
    explicit Device(size_t bytes):size(bytes){check(hipMalloc(reinterpret_cast<void**>(&allocation),size+2u*guard));check(hipMemset(allocation,0xa5,size+2u*guard));}
    ~Device(){if(allocation && hipFree(allocation)!=hipSuccess)std::abort();}
    template<class T=unsigned char>T* data(){return reinterpret_cast<T*>(allocation+guard);}
    std::vector<unsigned char> snapshot(){
        std::vector<unsigned char> out(size+2u*guard);check(hipMemcpy(out.data(),allocation,out.size(),hipMemcpyDeviceToHost));
        for(size_t i=0u;i<guard;++i)require(out[i]==0xa5u&&out[guard+size+i]==0xa5u,"device redzone changed");
        return out;
    }
};
void finish(){
    hipEvent_t event;check(hipEventCreate(&event));check(hipEventRecord(event));
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;){const auto status=hipEventQuery(event);if(status==hipSuccess)break;if(status!=hipErrorNotReady)check(status);require(std::chrono::steady_clock::now()<deadline,"EXP completion deadline");std::this_thread::yield();}
    check(hipEventDestroy(event));
}
template<class Function>double completed(Function function){
    const auto begin=std::chrono::steady_clock::now();function();check(hipGetLastError());finish();
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
}
struct Stats{unsigned counts[5]{};};
__global__ void verify(const unsigned char* original,const unsigned char* packed,Stats* totals){
    Stats local;
    for(uint32_t relative=blockIdx.x*blockDim.x+threadIdx.x;relative<source::end-source::begin;relative+=gridDim.x*blockDim.x){
        const uint32_t input=0x80000000u|(source::begin+relative);
        const bool admitted=reduced::admitted(input);
        ++local.counts[admitted?0u:3u];
        if(admitted)++local.counts[reduced::full::code(packed,reduced::fractional_index(input&0x7fffffffu))==3u?2u:1u];
        local.counts[4]+=raw::bits(reduced::evaluate(original,packed,raw::value(input)))!=source::decode(original,relative);
    }
#pragma unroll
    for(unsigned i=0u;i<5u;++i){
        unsigned value=local.counts[i];
#pragma unroll
        for(unsigned step=16u;step;step>>=1u)value+=__shfl_down(value,step,32u);
        if(!(threadIdx.x&31u))atomicAdd(totals->counts+i,value);
    }
}
__global__ void sample(const unsigned char* original,const unsigned char* packed,
    const uint32_t* inputs,uint32_t* outputs,unsigned count){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)outputs[i]=raw::bits(reduced::evaluate(original,packed,raw::value(inputs[i])));
}
void samples(const unsigned char* original,const unsigned char* packed,
    const std::vector<unsigned char>& table){
    std::vector<uint32_t> inputs;
    for(uint32_t sign:{0u,0x80000000u})for(uint32_t magnitude:{0u,1u,0x007fffffu,0x00800000u,
        source::begin-1u,source::begin,raw::positive_one_end-1u,raw::positive_one_end,
        reduced::magnitude_begin-1u,reduced::magnitude_begin,reduced::magnitude_begin+1u,
        reduced::magnitude_end-1u,reduced::magnitude_end,source::end-1u,source::end,
        0x7f7fffffu,0x7f800000u,0x7f800001u,0x7fc00000u,0x7fffffffu})inputs.push_back(sign|magnitude);
    Device di(inputs.size()*4u),dout(inputs.size()*4u);
    check(hipMemcpy(di.data(),inputs.data(),inputs.size()*4u,hipMemcpyHostToDevice));const auto before=di.snapshot();
    hipLaunchKernelGGL(sample,dim3(1u),dim3(256u),0u,nullptr,original,packed,di.data<uint32_t>(),dout.data<uint32_t>(),unsigned(inputs.size()));check(hipGetLastError());finish();
    const auto result=dout.snapshot();
    for(size_t i=0u;i<inputs.size();++i){uint32_t actual;std::memcpy(&actual,result.data()+guard+i*4u,4u);require(actual==raw::bits(source::evaluate(table.data(),raw::value(inputs[i]))),"exterior or boundary mismatch");}
    require(di.snapshot()==before,"sample inputs changed");
    std::printf("{\"kind\":\"reduced_exp2_boundaries\",\"cases\":%zu,\"mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true}\n",inputs.size());
}
int main(int argc,char** argv)try{
    require(argc==2,"requires verified SM121 interpolated table");
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0));require(!std::strncmp(properties.gcnArchName,"gfx1151",7u),"requires gfx1151");
    std::ifstream file(argv[1],std::ios::binary|std::ios::ate);require(file&&file.tellg()==std::streamoff(source::table_bytes),"source span");
    std::vector<unsigned char> table(source::table_bytes);file.seekg(0);file.read(reinterpret_cast<char*>(table.data()),table.size());require(file&&source::valid_layout(table.data(),table.size()),"source layout");
    Device original(table.size()),packed(reduced::packed_bytes),escapes(reduced::escape_words*4u),counts(sizeof(Stats));
    check(hipMemcpy(original.data(),table.data(),table.size(),hipMemcpyHostToDevice));const auto source_before=original.snapshot();
    const double build_ms=completed([&]{hipLaunchKernelGGL(reduced::build,dim3(4096u),dim3(256u),0u,nullptr,original.data(),packed.data());});
    check(hipMemset(escapes.data(),0,escapes.size));
    const double discover_ms=completed([&]{hipLaunchKernelGGL(reduced::find_escapes,dim3(4096u),dim3(256u),0u,nullptr,original.data(),packed.data(),escapes.data<unsigned>());});
    const auto escaped=escapes.snapshot();size_t marked=0u;
    for(size_t i=0u;i<reduced::escape_words;++i){unsigned word;std::memcpy(&word,escaped.data()+guard+i*4u,4u);marked+=unsigned(__builtin_popcount(word));}
    const double finalize_ms=completed([&]{hipLaunchKernelGGL(reduced::apply_escapes,dim3(4096u),dim3(256u),0u,nullptr,packed.data(),escapes.data<unsigned>());});
    const auto derived=packed.snapshot();size_t distribution[4]{};
    for(size_t byte=0u;byte<reduced::packed_bytes;++byte)for(unsigned part=0u;part<4u;++part)++distribution[(derived[guard+byte]>>(part*2u))&3u];
    check(hipMemset(counts.data(),0,counts.size));
    const double verify_ms=completed([&]{hipLaunchKernelGGL(verify,dim3(4096u),dim3(256u),0u,nullptr,original.data(),packed.data(),counts.data<Stats>());});
    const auto raw_counts=counts.snapshot();Stats result;std::memcpy(&result,raw_counts.data()+guard,sizeof(result));
    require(result.counts[0]==reduced::magnitude_end-reduced::magnitude_begin && result.counts[0]==result.counts[1]+result.counts[2] && result.counts[0]+result.counts[3]==source::end-source::begin && !result.counts[4],"full EXP domain comparison failed");
    samples(original.data(),packed.data(),table);
    // Deliberately damage one admitted canonical correction and require the
    // numerical check to notice it before restoring the complete saved byte.
    uint32_t changed=0u;for(;changed<reduced::fractions;++changed)if(reduced::full::code(derived.data()+guard,changed)!=3u)break;
    require(changed<reduced::fractions,"no compact native domain retained");
    const unsigned char saved=derived[guard+changed/4u];const unsigned old=(saved>>((changed%4u)*2u))&3u;
    const unsigned char corrupt=static_cast<unsigned char>((saved&~(3u<<((changed%4u)*2u)))|(((old+1u)%3u)<<((changed%4u)*2u)));
    check(hipMemcpy(packed.data()+changed/4u,&corrupt,1u,hipMemcpyHostToDevice));
    const uint32_t input=0xbf800000u|changed;Device di(4u),dout(4u);check(hipMemcpy(di.data(),&input,4u,hipMemcpyHostToDevice));const auto damaged_input_before=di.snapshot();
    hipLaunchKernelGGL(sample,dim3(1u),dim3(32u),0u,nullptr,original.data(),packed.data(),di.data<uint32_t>(),dout.data<uint32_t>(),1u);check(hipGetLastError());finish();
    const auto damaged=dout.snapshot();uint32_t damaged_bits;std::memcpy(&damaged_bits,damaged.data()+guard,4u);require(damaged_bits!=raw::bits(source::evaluate(table.data(),raw::value(input))),"corrupted correction was not detected");
    check(hipMemcpy(packed.data()+changed/4u,&saved,1u,hipMemcpyHostToDevice));
    require(di.snapshot()==damaged_input_before && packed.snapshot()==derived && escapes.snapshot()==escaped && original.snapshot()==source_before,"immutable input, table or escape bitmap changed");
    std::printf("{\"kind\":\"reduced_exp2_complete_domain\",\"verified_inputs\":%u,\"compact_inputs\":%u,\"native_corrected_inputs\":%u,\"escaped_inputs\":%u,\"original_outside_inputs\":%u,\"new_escape_fractions\":%zu,\"fraction_codes\":[%zu,%zu,%zu,%zu],\"packed_bytes\":%zu,\"temporary_escape_bytes\":%zu,\"build_host_ms\":%.6f,\"discovery_host_ms\":%.6f,\"finalize_host_ms\":%.6f,\"verification_host_ms\":%.6f,\"mismatches\":0,\"corruption_detected\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"performance_acceptance\":false,\"inference_acceptance\":false}\n",source::end-source::begin,result.counts[0],result.counts[1],result.counts[2],result.counts[3],marked,distribution[0],distribution[1],distribution[2],distribution[3],reduced::packed_bytes,escapes.size,build_ms,discover_ms,finalize_ms,verify_ms);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
