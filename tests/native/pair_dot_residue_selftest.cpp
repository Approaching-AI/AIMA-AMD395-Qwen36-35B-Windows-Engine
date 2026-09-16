#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_pair_residue.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
namespace pair = qrt_sm121_pair_residue;
constexpr unsigned guard = 67u, count = 1048576u;
struct Input { pair::Pair left,right; };
struct Result { float estimate; int32_t recovered; uint32_t residue,accepted,aligned[9]; };
void check(hipError_t value) { if(value != hipSuccess) throw std::runtime_error(hipGetErrorString(value)); }
void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }
uint32_t random_word(uint32_t& r) { r ^= r<<13u; r ^= r>>17u; r ^= r<<5u; return r; }
__global__ void execute(const Input* input, Result* output, unsigned cells) {
    const unsigned i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i >= cells) return;
    const auto in = input[i]; Result out{};
    out.estimate = pair::estimate(in.left,in.right);
    out.residue = pair::residue(in.left,in.right);
    out.accepted = unsigned(pair::recover(out.estimate,out.residue,&out.recovered));
#pragma unroll
    for(unsigned shift=0; shift<=8; ++shift) out.aligned[shift] = out.accepted ? pair::aligned(out.recovered,in.left,in.right,shift) : 0u;
    output[i] = out;
}
struct Device {
    void* p=nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&p,bytes)); check(hipMemset(p,0xa5,bytes)); }
    ~Device() { if(p) (void)hipFree(p); }
    template<class T> T* data() { return static_cast<T*>(p)+guard; }
};
void complete() {
    hipEvent_t event=nullptr; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(20);
    for(;;) { const auto status=hipEventQuery(event); if(status==hipSuccess) break;
        if(status!=hipErrorNotReady) check(status);
        require(std::chrono::steady_clock::now()<deadline,"native completion deadline"); std::this_thread::yield(); }
    check(hipEventDestroy(event));
}
template<class T> void redzones(const std::vector<T>& values) {
    const auto* bytes=reinterpret_cast<const unsigned char*>(values.data());
    for(size_t i=0;i<guard*sizeof(T);++i) require(bytes[i]==0xa5u && bytes[(values.size()-guard)*sizeof(T)+i]==0xa5u,"changed redzone");
}
void run(unsigned family) {
    std::vector<Input> input(count+2u*guard); std::memset(input.data(),0xa5,input.size()*sizeof(Input));
    std::vector<int32_t> first(count),second(count); uint32_t rng=0x91d85ac3u+family;
    for(unsigned i=0;i<count;++i) {
        unsigned m[4],s[4]; bool n[4];
        for(unsigned j=0;j<4;++j) { m[j]=128u+(random_word(rng)&127u);s[j]=random_word(rng)&7u;n[j]=(random_word(rng)&1u)!=0u; }
        if(family==0u) { // The pair-relative units used by real captures.
            s[(i&1u)?0u:1u]=0u; s[(i&2u)?2u:3u]=0u;
        } else if(family==1u) { // Exhaustive two mantissas, maximal common scaling, all signs.
            m[0]=128u+(i&127u);m[1]=128u+((i>>7u)&127u);m[2]=255u;m[3]=254u;
            for(unsigned j=0;j<4;++j) {s[j]=7u;n[j]=((i>>(14u+j))&1u)!=0u;}
        } else if(family==2u) { // Near cancellation, alternate zero and large endpoints.
            m[1]=m[0];m[3]=(i&1u)?m[2]:(m[2]==255u?254u:m[2]+1u);
            s[1]=s[0];s[3]=s[2];n[0]=n[1]=false;n[2]=false;n[3]=true;
        } // family3 retains the entire four-independent-shift domain.
        input[guard+i]={pair::prepare(m[0],s[0],n[0],m[1],s[1],n[1]),pair::prepare(m[2],s[2],n[2],m[3],s[3],n[3])};
        int32_t v[4]; for(unsigned j=0;j<4;++j) v[j]=n[j]?-int32_t(m[j]<<s[j]):int32_t(m[j]<<s[j]);
        first[i]=v[0]*v[2];second[i]=v[1]*v[3];
        const int64_t exact=int64_t(first[i])+second[i]; require(exact>=-pair::maximum_sum&&exact<=pair::maximum_sum,"CPU integer range");
    }
    Device a(input.size()*sizeof(Input)),b((count+2u*guard)*sizeof(Result));
    check(hipMemcpy(a.p,input.data(),input.size()*sizeof(Input),hipMemcpyHostToDevice));
    std::vector<Result> result(count+2u*guard),prior; double maximum_error=0;unsigned rejected=0;
    for(unsigned attempt=0;attempt<2;++attempt) {
        hipLaunchKernelGGL(execute,dim3((count+255u)/256u+1u),dim3(256u),0u,nullptr,a.data<Input>(),b.data<Result>(),count);check(hipGetLastError());complete();
        check(hipMemcpy(result.data(),b.p,result.size()*sizeof(Result),hipMemcpyDeviceToHost));redzones(result);
        if(attempt) require(std::memcmp(result.data(),prior.data(),result.size()*sizeof(Result))==0,"repeat output differs");
        else prior=result;
        for(unsigned i=0;i<count;++i) {
            const auto r=result[guard+i];const int32_t exact=first[i]+second[i];
            require(std::isfinite(r.estimate),"nonfinite DOT2 estimate");maximum_error=std::max(maximum_error,std::abs(double(r.estimate)-double(exact)));
            require(r.residue==(uint32_t(exact)&255u),"native IU8 residue differs");
            if(!r.accepted) {if(!attempt)++rejected;continue;}
            require(r.recovered==exact,"recovered integer differs");
            for(unsigned shift=0;shift<=8;++shift) require(r.aligned[shift]==uint32_t(first[i]/int32_t(1u<<shift)+second[i]/int32_t(1u<<shift)),"individual truncation differs");
        }
    }
    std::vector<Input> copy(input.size());check(hipMemcpy(copy.data(),a.p,copy.size()*sizeof(Input),hipMemcpyDeviceToHost));
    require(std::memcmp(input.data(),copy.data(),input.size()*sizeof(Input))==0,"input changed");redzones(copy);
    std::printf("{\"kind\":\"pair_dot2_integer_residue\",\"family\":%u,\"pairs\":%u,\"accepted\":%u,\"rejected_ambiguous\":%u,\"maximum_observed_absolute_error\":%.9g,\"integer_mismatches\":0,\"alignment_shifts_checked\":9,\"repeat_bit_parity\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"hardware_error_bound_proven\":false,\"inference_acceptance\":false}\n",family,count,count-rejected,rejected,maximum_error);std::fflush(stdout);
}
int main() try {
    hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));require(std::strncmp(p.gcnArchName,"gfx1151",7u)==0,"requires gfx1151");
    for(unsigned f=0;f<4;++f)run(f);return 0;
} catch(const std::exception& error) {std::fprintf(stderr,"%s\n",error.what());return 1;}
