// Reuse the already validated integer fixtures and intrinsic control without
// changing their standalone diagnostic or any runtime arithmetic.
#define main qrt_existing_wmma_rounding_probe_main
#include "wmma_integer_rounding_probe.cpp"
#undef main
#include <fstream>
#include <string>

namespace {
constexpr unsigned chain_variants=8u;
unsigned pair_order(unsigned variant,unsigned step) {
    return variant==1u?7u-step:variant==2u?(step%4u)*2u+step/4u:step;
}
#if defined(__HIPCC__)
__device__ __forceinline__ float hardware_pair(uint32_t a,uint32_t b,float carry) {
    using Half2=_Float16 __attribute__((ext_vector_type(2)));
    Half2 av,bv;__builtin_memcpy(&av,&a,4u);__builtin_memcpy(&bv,&b,4u);
    return __builtin_amdgcn_fdot2(av,bv,carry,false);
}
__device__ __forceinline__ float scalar_fma(float a,float b,float c) {
    float result;asm volatile("v_fma_f32 %0, %1, %2, %3" : "=v"(result) : "v"(a),"v"(b),"v"(c));return result;
}
template<unsigned Variant>
__global__ void dot_chain(const uint16_t* left,const uint16_t* right,float* output) {
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=cells)return;
    const unsigned tile=cell/256u,row=cell%256u/16u,column=cell%16u;
    const auto* a=left+tile*256u+row*16u;const auto* b=right+tile*256u+column*16u;
    float carry=0.0f,second=0.0f,terms[8];
#pragma unroll
    for(unsigned step=0u;step<8u;++step) {
        const unsigned pair=Variant==1u?7u-step:Variant==2u?(step%4u)*2u+step/4u:step;
        const unsigned i=pair*2u;
        const uint32_t av=Variant==3u?uint32_t(a[i+1u])|(uint32_t(a[i])<<16u):uint32_t(a[i])|(uint32_t(a[i+1u])<<16u);
        const uint32_t bv=Variant==3u?uint32_t(b[i+1u])|(uint32_t(b[i])<<16u):uint32_t(b[i])|(uint32_t(b[i+1u])<<16u);
        if constexpr(Variant==4u || Variant==7u)terms[step]=hardware_pair(av,bv,0.0f);
        else if constexpr(Variant==5u) {
            if(step<4u)carry=hardware_pair(av,bv,carry);else second=hardware_pair(av,bv,second);
        } else if constexpr(Variant==6u) {
            using Half2=_Float16 __attribute__((ext_vector_type(2)));
            Half2 af,bf;__builtin_memcpy(&af,&av,4u);__builtin_memcpy(&bf,&bv,4u);
            carry=scalar_fma(float(af[0]),float(bf[0]),carry);carry=scalar_fma(float(af[1]),float(bf[1]),carry);
        } else carry=hardware_pair(av,bv,carry);
    }
    if constexpr(Variant==4u) {
#pragma unroll
        for(unsigned step=0u;step<8u;++step)carry+=terms[step];
    } else if constexpr(Variant==5u)carry+=second;
    else if constexpr(Variant==7u)carry=((terms[0]+terms[1])+(terms[2]+terms[3]))+((terms[4]+terms[5])+(terms[6]+terms[7]));
    output[cell]=carry;
}
void save_words(const std::string& file,const std::vector<float>& values) {
    if(std::ifstream(file,std::ios::binary).good())throw std::runtime_error("capture already exists");
    std::ofstream out(file,std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()+guard),cells*4u);
    if(!out)throw std::runtime_error("capture write failed");
}
void check_guards(const std::vector<float>& values) {
    for(unsigned i=0;i<guard;++i)if(bits(values[i])!=0xa5a5a5a5u||bits(values[values.size()-1u-i])!=0xa5a5a5a5u)
        throw std::runtime_error("chain output guard changed");
}
void run_chains(bool bf16,const std::string& directory) {
    const Inputs input(bf16),half_input(false);
    if(input.expected!=half_input.expected)throw std::runtime_error("DOT2 operand values differ");
    Device left(input.left.size()*2u),right(input.right.size()*2u),half_left(half_input.left.size()*2u),half_right(half_input.right.size()*2u);
    Device output((cells+2u*guard)*4u);
    for(auto pair:{std::make_pair(&left,&input.left),std::make_pair(&right,&input.right),std::make_pair(&half_left,&half_input.left),std::make_pair(&half_right,&half_input.right)})
        check(hipMemcpy(pair.first->data,pair.second->data(),pair.second->size()*2u,hipMemcpyHostToDevice));
    std::vector<float> reference(cells+2u*guard),candidate(reference.size()),prior;
    for(unsigned attempt=0u;attempt<2u;++attempt) {
        check(hipMemset(output.data,0xa5,reference.size()*4u));
        if(bf16)hipLaunchKernelGGL((intrinsic_control<true>),dim3(tiles),dim3(32u),0u,nullptr,left.as<uint16_t>()+guard,right.as<uint16_t>()+guard,output.as<float>()+guard,0.0f);
        else hipLaunchKernelGGL((intrinsic_control<false>),dim3(tiles),dim3(32u),0u,nullptr,left.as<uint16_t>()+guard,right.as<uint16_t>()+guard,output.as<float>()+guard,0.0f);
        check(hipGetLastError());finish();check(hipMemcpy(reference.data(),output.data,reference.size()*4u,hipMemcpyDeviceToHost));check_guards(reference);
        if(attempt&&std::memcmp(prior.data(),reference.data(),reference.size()*4u))throw std::runtime_error("WMMA repeat changed");prior=reference;
    }
    const std::string prefix=directory+"/"+(bf16?"bf16":"fp16");save_words(prefix+"-wmma.bin",reference);
    for(unsigned variant=0u;variant<chain_variants;++variant) {
        for(unsigned attempt=0u;attempt<2u;++attempt) {
            check(hipMemset(output.data,0xa5,candidate.size()*4u));
#define QRT_CHAIN_CASE(V) case V:hipLaunchKernelGGL((dot_chain<V>),dim3((cells+255u)/256u+1u),dim3(256u),0u,nullptr,half_left.as<uint16_t>()+guard,half_right.as<uint16_t>()+guard,output.as<float>()+guard);break
            switch(variant){QRT_CHAIN_CASE(0u);QRT_CHAIN_CASE(1u);QRT_CHAIN_CASE(2u);QRT_CHAIN_CASE(3u);QRT_CHAIN_CASE(4u);QRT_CHAIN_CASE(5u);QRT_CHAIN_CASE(6u);QRT_CHAIN_CASE(7u);}
#undef QRT_CHAIN_CASE
            check(hipGetLastError());finish();check(hipMemcpy(candidate.data(),output.data,candidate.size()*4u,hipMemcpyDeviceToHost));check_guards(candidate);
            if(attempt&&std::memcmp(prior.data(),candidate.data(),candidate.size()*4u))throw std::runtime_error("DOT2 repeat changed");prior=candidate;
        }
        save_words(prefix+"-chain"+std::to_string(variant)+".bin",candidate);
        for(unsigned family=0u;family<families;++family) {
            unsigned different=0u,wrong_integer=0u,first=cells;double maximum=0;
            const unsigned begin=family*cases*permutations*256u,end=begin+cases*permutations*256u;
            for(unsigned i=begin;i<end;++i) {
                const float actual=candidate[guard+i];if(!std::isfinite(actual))throw std::runtime_error("nonfinite integer chain");
                const bool changed=bits(actual)!=bits(reference[guard+i]);different+=changed;
                if(changed&&first==cells)first=i;
                wrong_integer+=double(actual)!=double(input.expected[i]);maximum=std::max(maximum,std::abs(double(actual)-double(input.expected[i])));
            }
            if(variant==6u&&wrong_integer)throw std::runtime_error("scalar integer control not exact");
            std::printf("{\"kind\":\"wmma_dot_chain_diagnostic\",\"wmma_dtype\":\"%s\",\"chain_dtype\":\"fp16_exact_integer_encoding\",\"variant\":%u,\"family\":%u,\"cells\":%u,\"wmma_raw_mismatches\":%u,\"exact_integer_mismatches\":%u,\"maximum_integer_error\":%.12g,\"repeat_bit_parity\":true,\"redzones_pass\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",bf16?"bf16":"fp16",variant,family,end-begin,different,wrong_integer,maximum);
            if(first<cells)std::printf("{\"kind\":\"wmma_dot_chain_counterexample\",\"wmma_dtype\":\"%s\",\"variant\":%u,\"family\":%u,\"cell\":%u,\"expected_integer\":%lld,\"wmma_bits\":%u,\"chain_bits\":%u}\n",bf16?"bf16":"fp16",variant,family,first,static_cast<long long>(input.expected[first]),bits(reference[guard+first]),bits(candidate[guard+first]));
        }
        std::fflush(stdout);
    }
    for(auto pair:{std::make_pair(&left,&input.left),std::make_pair(&right,&input.right),std::make_pair(&half_left,&half_input.left),std::make_pair(&half_right,&half_input.right)}) {
        std::vector<uint16_t> copy(pair.second->size());check(hipMemcpy(copy.data(),pair.first->data,copy.size()*2u,hipMemcpyDeviceToHost));
        if(copy!=*pair.second)throw std::runtime_error("encoded input or guard changed");
    }
    std::printf("{\"kind\":\"wmma_dot_chain_inputs\",\"wmma_dtype\":\"%s\",\"immutable_inputs\":true,\"redzones_pass\":true}\n",bf16?"bf16":"fp16");
}
#endif
} // namespace

int main(int argc,char** argv)try {
#if defined(__HIPCC__)
    hipDeviceProp_t device{};check(hipGetDeviceProperties(&device,0));
    if(std::strncmp(device.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    if(argc!=2)throw std::runtime_error("requires capture directory");
    run_chains(true,argv[1]);run_chains(false,argv[1]);
#else
    (void)argc;(void)argv;const Inputs a(true),b(false);
    if(a.expected!=b.expected)throw std::runtime_error("input dtype differs");
    for(unsigned variant=0;variant<chain_variants;++variant) {
        unsigned seen=0;
        for(unsigned step=0;step<8;++step)seen|=1u<<pair_order(variant,step);
        if(seen!=255u)throw std::runtime_error("chain dropped or repeated pair");
    }
    std::printf("{\"kind\":\"wmma_dot_chain_host_inputs\",\"unique_integer_dots\":%u,\"permutations\":4,\"chain_variants\":8,\"dtypes\":2,\"native_executed\":false}\n",cells/permutations);
#endif
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
