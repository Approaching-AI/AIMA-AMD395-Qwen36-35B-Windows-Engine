#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_group16_modulo.h"
#include "../../native/providers/moe_accumulator/sm121_pv_error_bound.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
namespace canonical=qrt_q1_moe_hawkeye;
namespace bound=qrt_sm121_pv_bound;
using B16=unsigned short __attribute__((ext_vector_type(16)));
using F8=float __attribute__((ext_vector_type(8)));
constexpr unsigned guard=64u;
struct Step { float center,absolute_dot,error; };
void check(hipError_t status) { if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status)); }
void require(bool condition,const char* message) { if(!condition)throw std::runtime_error(message); }
void complete() {
    hipEvent_t event;check(hipEventCreate(&event));check(hipEventRecord(event));
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;) {
        const auto status=hipEventQuery(event);
        if(status==hipSuccess)break;
        if(status!=hipErrorNotReady)check(status);
        require(std::chrono::steady_clock::now()<deadline,"GPU completion deadline");
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
template<class T> struct Device {
    T* base=nullptr;size_t count;
    explicit Device(size_t n):count(n) {
        check(hipMalloc(reinterpret_cast<void**>(&base),(n+2u*guard)*sizeof(T)));
        check(hipMemset(base,0xa5,(n+2u*guard)*sizeof(T)));
    }
    ~Device(){if(base)(void)hipFree(base);}
    T* data(){return base+guard;}
    void upload(const std::vector<T>& input) {
        require(input.size()==count,"upload shape");
        check(hipMemcpy(data(),input.data(),count*sizeof(T),hipMemcpyHostToDevice));
    }
    std::vector<T> read(){std::vector<T> out(count);check(hipMemcpy(out.data(),data(),count*sizeof(T),hipMemcpyDeviceToHost));return out;}
    void guards() {
        std::vector<unsigned char> a(guard*sizeof(T)),b(a.size());
        check(hipMemcpy(a.data(),base,a.size(),hipMemcpyDeviceToHost));
        check(hipMemcpy(b.data(),data()+count,b.size(),hipMemcpyDeviceToHost));
        for(size_t i=0;i<a.size();++i)require(a[i]==0xa5u&&b[i]==0xa5u,"device redzone");
    }
};

// Trace actual dependent K16 WMMA calls. Absolute products use a separate
// zero-C matrix operation; no CPU reference or expected output enters a kernel.
__global__ void trace_groups(const uint16_t* input,const float* initial,
    Step* trace,unsigned width) {
    const unsigned lane=threadIdx.x,tile=blockIdx.x,groups=width/16u;
    F8 carry{},errors{};
    for(unsigned j=0u;j<8u;++j)carry[j]=initial[tile*256u+(2u*j+lane/16u)*16u+lane%16u];
    for(unsigned group=0u;group<groups;++group) {
        B16 a{},b{},aa{},bb{};
#pragma unroll
        for(unsigned i=0u;i<16u;++i) {
            a[i]=input[(size_t(tile)*32u+lane%16u)*width+group*16u+i];
            b[i]=input[(size_t(tile)*32u+16u+lane%16u)*width+group*16u+i];
            aa[i]=a[i]&0x7fffu;bb[i]=b[i]&0x7fffu;
        }
        const F8 zero{};
        const F8 absolute=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(aa,bb,zero);
        const F8 next=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,carry);
#pragma unroll
        for(unsigned j=0u;j<8u;++j) {
            errors[j]=bound::group(errors[j],carry[j],absolute[j]);
            const unsigned cell=(2u*j+lane/16u)*16u+lane%16u;
            trace[(size_t(tile)*groups+group)*256u+cell]={next[j],absolute[j],errors[j]};
        }
        carry=next;
    }
}

// These smaller coefficients are diagnostic hypotheses, not approved bounds.
// Count every undercoverage and false endpoint admission before considering
// any implementation. The established PV envelope remains a separate control.
float hypothetical(float error,float carry,float absolute_dot,unsigned shift) {
    if(!bound::finite(error)||!bound::finite(carry)||!bound::finite(absolute_dot)||error<0||absolute_dot<0)
        return bound::infinity();
    const float products=bound::upper(absolute_dot*(1.0f+0x1p-19f)+0x1p-118f);
    const float magnitude=bound::upper(bound::absolute(carry)+error+products);
    return bound::upper(error+bound::upper(magnitude*std::ldexp(1.0f,-int(shift)))+0x1p-118f);
}
float widen(uint16_t value){return bound::value(uint32_t(value)<<16u);}
struct Counts {
    uint64_t prefix_undercoverage=0,final_admitted=0,false_admissions=0;
};
void analyze(const char* kind,unsigned mode,const std::vector<uint16_t>& input,
    const std::vector<float>& initial,unsigned width,const std::vector<uint16_t>& external={}) {
    const unsigned tiles=unsigned(initial.size()/256u),groups=width/16u;
    require(input.size()==size_t(tiles)*32u*width,"input dimensions");
    require(external.empty()||external.size()==initial.size(),"external dimensions");
    Device<uint16_t> di(input.size());Device<float> dc(initial.size());Device<Step> dt(size_t(tiles)*groups*256u);
    di.upload(input);dc.upload(initial);
    hipLaunchKernelGGL(trace_groups,dim3(tiles),dim3(32u),0u,nullptr,di.data(),dc.data(),dt.data(),width);
    check(hipGetLastError());complete();const auto trace=dt.read();
    di.guards();dc.guards();dt.guards();require(di.read()==input&&dc.read()==initial,"immutable inputs");
    Counts counts[4]{};uint64_t native_bf16_differences=0,external_differences=0;
    uint64_t rne_group_differences=0,positive_native_errors=0,negative_native_errors=0,nonfinite=0;
    uint64_t gpu_cpu_envelope_bit_differences=0;
    double maximum_native_relative_error=0,maximum_canonical_error=0;
    for(unsigned tile=0;tile<tiles;++tile)for(unsigned cell=0;cell<256u;++cell) {
        const unsigned row=cell/16u,column=cell%16u;
        auto carry=canonical::value_from_float(initial[tile*256u+cell],-133);
        float center=initial[tile*256u+cell],errors[4]{};
        for(unsigned group=0u;group<groups;++group) {
            const Step step=trace[(size_t(tile)*groups+group)*256u+cell];
            canonical::Value terms[17];terms[0]=carry;
            double mathematical=double(center),magnitude=std::abs(double(center));
            for(unsigned k=0;k<16u;++k) {
                const uint16_t a=input[(size_t(tile)*32u+row)*width+group*16u+k];
                const uint16_t b=input[(size_t(tile)*32u+16u+column)*width+group*16u+k];
                terms[k+1u]=canonical::multiply_bf16(a,b,-133);
                const double product=double(widen(a))*double(widen(b));
                mathematical+=product;magnitude+=std::abs(product);
            }
            carry=canonical::group_sum<26,-133>(terms,17u);
            const float exact=canonical::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
            const double native_error=double(step.center)-mathematical;
            if(!std::isfinite(step.center)||!std::isfinite(exact))++nonfinite;
            positive_native_errors+=native_error>0;negative_native_errors+=native_error<0;
            rne_group_differences+=bound::bits(float(mathematical))!=bound::bits(step.center);
            if(magnitude>0&&std::isfinite(magnitude))maximum_native_relative_error=std::max(maximum_native_relative_error,std::abs(native_error)/magnitude);
            const double error=std::abs(double(step.center)-double(exact));
            maximum_canonical_error=std::max(maximum_canonical_error,error);
            errors[0]=bound::group(errors[0],center,step.absolute_dot);
            // Count compiler contraction differences and validate the actual
            // GPU control envelope, rather than replacing it with host values.
            gpu_cpu_envelope_bit_differences+=bound::bits(errors[0])!=bound::bits(step.error);
            errors[0]=step.error;
            for(unsigned v=1u;v<4u;++v)errors[v]=hypothetical(errors[v],center,step.absolute_dot,25u-v);
            for(unsigned v=0u;v<4u;++v) {
                counts[v].prefix_undercoverage+=error>double(errors[v]);
                if(group+1u==groups&&bound::same_bf16(step.center,errors[v])) {
                    ++counts[v].final_admitted;
                    counts[v].false_admissions+=bound::bf16(step.center)!=bound::bf16(exact);
                }
            }
            center=step.center;
        }
        const uint16_t endpoint=bound::bf16(canonical::value_to_float(qrt_sm121_group16::finish_accumulator(carry)));
        native_bf16_differences+=bound::bf16(center)!=endpoint;
        if(!external.empty())external_differences+=endpoint!=external[tile*256u+cell];
    }
    for(unsigned v=0u;v<4u;++v) {
        std::printf("{\"kind\":\"%s\",\"mode\":%u,\"envelope\":%u,\"coefficient_exponent\":%u,\"hypothesis_only\":%s,\"dots\":%zu,\"ordered_groups\":%zu,\"prefix_undercoverage\":%llu,\"final_admitted\":%llu,\"false_bf16_admissions\":%llu,\"native_bf16_differences\":%llu,\"external_reference_differences\":%llu,\"external_reference_cells\":%zu,\"native_group_rne_differences\":%llu,\"positive_native_group_errors\":%llu,\"negative_native_group_errors\":%llu,\"maximum_native_error_over_magnitude\":%.17g,\"maximum_canonical_error\":%.17g,\"nonfinite\":%llu,\"gpu_cpu_envelope_bit_differences\":%llu,\"fp64_group_reference_exact\":%s,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",kind,mode,v,v?25u-v:19u,v?"true":"false",initial.size(),initial.size()*groups,
            static_cast<unsigned long long>(counts[v].prefix_undercoverage),static_cast<unsigned long long>(counts[v].final_admitted),static_cast<unsigned long long>(counts[v].false_admissions),
            static_cast<unsigned long long>(native_bf16_differences),static_cast<unsigned long long>(external_differences),external.size(),static_cast<unsigned long long>(rne_group_differences),
            static_cast<unsigned long long>(positive_native_errors),static_cast<unsigned long long>(negative_native_errors),maximum_native_relative_error,maximum_canonical_error,static_cast<unsigned long long>(nonfinite),static_cast<unsigned long long>(gpu_cpu_envelope_bit_differences),external.empty()?"true":"false");
        std::fflush(stdout);
    }
    require(!counts[0].prefix_undercoverage&&!counts[0].false_admissions&&!external_differences&&!nonfinite,"established bound, reference or finite control failure");
}
uint32_t random_state=0x395bf16u;
uint32_t random_word(){random_state^=random_state<<13u;random_state^=random_state>>17u;random_state^=random_state<<5u;return random_state;}
void generated() {
    constexpr unsigned tiles=16u,width=1024u;
    for(unsigned mode=0u;mode<6u;++mode) {
        std::vector<uint16_t> input(size_t(tiles)*32u*width);std::vector<float> initial(tiles*256u);
        for(size_t i=0;i<input.size();++i) {
            uint16_t value=uint16_t((116u+random_word()%12u)<<7u|(random_word()&0x807fu));
            if(mode==1u)value&=0x7fffu;
            if(mode==2u)value=uint16_t((value&0x7fffu)|((i/width%32u)>=16u?0x8000u:0u));
            if(mode==3u&&(i&1u))value=input[i-1u]^uint16_t((i/width%32u)>=16u?0x8000u:0u);
            if(mode==4u&&i%7u)value=uint16_t(value&0x8000u);
            input[i]=value;
        }
        if(mode==5u)for(auto& value:initial)value=bound::value((random_word()&0x807fffffu)|(136u<<23u));
        analyze("wmma_generated_projection_envelope",mode,input,initial,width);
    }
}
std::vector<uint16_t> read(const char* path,size_t count) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    require(bool(file)&&file.tellg()==std::streamoff(count*2u),"capture size");
    std::vector<uint16_t> values(count);file.seekg(0);
    require(bool(file.read(reinterpret_cast<char*>(values.data()),std::streamsize(count*2u))),"capture read");return values;
}
void captured(const char* ipath,const char* wpath,const char* rpath) {
    constexpr unsigned rows=8192u,tokens=7169u,width=2048u,tiles=64u;
    const auto inputs=read(ipath,size_t(tokens)*width),weights=read(wpath,size_t(rows)*width),reference=read(rpath,size_t(tokens)*rows);
    std::vector<uint16_t> paired(size_t(tiles)*32u*width),external(tiles*256u);std::vector<float> initial(tiles*256u);
    for(unsigned tile=0;tile<tiles;++tile)for(unsigned i=0;i<16u;++i) {
        const unsigned token=(tile*113u+i*449u)%tokens;
        const unsigned row=tile<7u?tile*16u+i:(tile*137u+i*509u)%rows;
        std::copy_n(inputs.data()+size_t(token)*width,width,paired.data()+(size_t(tile)*32u+i)*width);
        std::copy_n(weights.data()+size_t(row)*width,width,paired.data()+(size_t(tile)*32u+16u+i)*width);
        for(unsigned q=0;q<16u;++q)external[tile*256u+q*16u+i]=reference[size_t((tile*113u+q*449u)%tokens)*rows+row];
    }
    analyze("wmma_original_qkv_sampled_envelope",0u,paired,initial,width,external);
}
}
int main(int argc,char** argv)try {
    hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));require(std::strncmp(p.gcnArchName,"gfx1151",7u)==0,"requires gfx1151");
    if(argc==2&&!std::strcmp(argv[1],"--selftest"))generated();
    else if(argc==5&&!std::strcmp(argv[1],"--real-qkv"))captured(argv[2],argv[3],argv[4]);
    else throw std::runtime_error("use --selftest or --real-qkv INPUT WEIGHT REFERENCE");
    return 0;
}catch(const std::exception& error){std::fprintf(stderr,"wmma_envelope_probe_error=%s\n",error.what());return 1;}
