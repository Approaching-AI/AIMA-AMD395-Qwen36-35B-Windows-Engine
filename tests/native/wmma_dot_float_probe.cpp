// Generated hardware arithmetic characterization, isolated from runtime dispatch.
#define main qrt_existing_integer_rounding_probe_main
#include "wmma_integer_rounding_probe.cpp"
#undef main
#include <fstream>
#include <string>

namespace {
constexpr unsigned float_families=8u,float_samples=64u;
constexpr unsigned float_tiles=float_families*float_samples,float_cells=float_tiles*256u;
uint32_t float_seed_salt=0u;
float from_word(uint32_t raw) {float value;std::memcpy(&value,&raw,4u);return value;}
double half_value(uint16_t raw) {
    const unsigned e=(raw>>10u)&31u,m=raw&1023u;
    const double magnitude=e?std::ldexp(double(1024u+m),int(e)-25):std::ldexp(double(m),-24);
    return raw&0x8000u?-magnitude:magnitude;
}
int half_exponent(uint16_t raw) {
    const unsigned e=(raw>>10u)&31u,m=raw&1023u;
    if(e)return int(e)-15;
    return m?-14:-126;
}
uint16_t float_operand(unsigned family,unsigned sample,unsigned row,unsigned k,bool right) {
    const unsigned index=family==2u?k/2u:k;
    const uint32_t h=hash(0x6d395a17u^float_seed_salt^sample*137u^row*65537u^index*104729u^unsigned(right)*99991u);
    unsigned sign=(h>>16u)&0x8000u;
    if(family==2u&&right&&(k&1u))sign^=0x8000u;
    if(family==3u&&k>=2u)return 0u;
    if(family==4u&&(h&3u)==0u)return uint16_t(sign);
    if(family==5u)return uint16_t(sign|((22u+((h>>8u)&7u))<<10u)|((h&127u)<<3u));
    if(family==7u&&(h&1u))return uint16_t(sign|(h&1023u));
    const unsigned exponent=family==1u?10u+((h>>10u)%11u):1u+((h>>10u)%30u);
    const unsigned mantissa=family>=6u?h&1023u:(h&127u)<<3u;
    return uint16_t(sign|(exponent<<10u)|mantissa);
}
// Empirical hypothesis from a disjoint integer fixture. This is deliberately
// checked against hardware rather than used as an accepted runtime operation.
float modeled_pair(uint16_t a0,uint16_t a1,uint16_t b0,uint16_t b1,float carry) {
    auto product_exponent=[](uint16_t a,uint16_t b) {
        return (a&0x7fffu)&&(b&0x7fffu)?half_exponent(a)+half_exponent(b):-126;
    };
    const int ce=carry?int((bits(carry)>>23u)&255u)-127:-126;
    const int e=std::max({product_exponent(a0,b0),product_exponent(a1,b1),ce-2});
    const double quantum=std::ldexp(1.0,e-24);
    double sum=std::trunc(double(carry)/quantum)+2.0*double(bits(carry)>>31u);
    for(auto pair:{std::make_pair(a0,b0),std::make_pair(a1,b1)}) {
        const double magnitude=std::trunc(std::abs(half_value(pair.first)*half_value(pair.second))/quantum);
        sum+=((pair.first^pair.second)&0x8000u)?-(magnitude+1.0):magnitude;
    }
    const float result=float(sum*quantum);return (bits(result)&0x7fffffffu)<0x00800000u?0.0f:result;
}
struct FloatInputs {
    std::vector<uint16_t> left,right;
    std::vector<float> carry,predicted;
    FloatInputs():left(float_cells+2u*guard,0x5a5au),right(left),
        carry(float_cells+2u*guard,from_word(0xa5a5a5a5u)),predicted(size_t(float_cells)*8u) {
        for(unsigned tile=0;tile<float_tiles;++tile) {
            const unsigned family=tile/float_samples,sample=tile%float_samples;
            for(unsigned row=0;row<16u;++row)for(unsigned k=0;k<16u;++k) {
                left[guard+tile*256u+row*16u+k]=float_operand(family,sample,row,k,false);
                right[guard+tile*256u+row*16u+k]=float_operand(family,sample,row,k,true);
            }
            for(unsigned row=0;row<16u;++row)for(unsigned col=0;col<16u;++col) {
                const unsigned cell=tile*256u+row*16u+col;
                const auto* a=left.data()+guard+tile*256u+row*16u;
                const auto* b=right.data()+guard+tile*256u+col*16u;
                const uint32_t h=hash(0x8a9153cdu^float_seed_salt^cell*7919u);int ep=-126;
                for(unsigned k=0;k<16u;++k)if((a[k]&0x7fffu)&&(b[k]&0x7fffu))ep=std::max(ep,half_exponent(a[k])+half_exponent(b[k]));
                const int ce=(family==1u||family==2u||family==3u||family==4u)?std::max(-100,std::min(100,ep+int((h>>16u)%25u)-12)):int((h>>16u)%121u)-60;
                const uint32_t raw=(h&0x80000000u)|(uint32_t(ce+127)<<23u)|(h&0x7fffffu);
                float c=(h&7u)<2u?from_word((h&1u)<<31u):from_word(raw);
                carry[guard+cell]=c;
                for(unsigned step=0;step<8u;++step) {
                    c=modeled_pair(a[2u*step],a[2u*step+1u],b[2u*step],b[2u*step+1u],c);
                    if(!std::isfinite(c))throw std::runtime_error("nonfinite model fixture");
                    predicted[size_t(cell)*8u+step]=c;
                }
            }
        }
    }
};

#if defined(__HIPCC__)
__device__ __forceinline__ float float_hardware_pair(uint32_t a,uint32_t b,float carry) {
    using Half2=_Float16 __attribute__((ext_vector_type(2)));
    Half2 av,bv;__builtin_memcpy(&av,&a,4u);__builtin_memcpy(&bv,&b,4u);
    return __builtin_amdgcn_fdot2(av,bv,carry,false);
}
__global__ void float_chain(const uint16_t* left,const uint16_t* right,const float* initial,float* prefix) {
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=float_cells)return;
    const unsigned tile=cell/256u,row=cell%256u/16u,col=cell%16u;
    const auto* a=left+tile*256u+row*16u;const auto* b=right+tile*256u+col*16u;
    float c=initial[cell];
#pragma unroll
    for(unsigned step=0;step<8u;++step) {
        const unsigned k=2u*step;
        c=float_hardware_pair(uint32_t(a[k])|(uint32_t(a[k+1u])<<16u),uint32_t(b[k])|(uint32_t(b[k+1u])<<16u),c);
        prefix[size_t(cell)*8u+step]=c;
    }
}
template<bool Bf16>
__global__ void float_matrix(const uint16_t* left,const uint16_t* right,const float* initial,float* output) {
    const unsigned lane=threadIdx.x,tile=blockIdx.x;U16x16 a{},b{};
#pragma unroll
    for(unsigned k=0;k<16u;++k) {
        const uint16_t av=left[tile*256u+(lane%16u)*16u+k],bv=right[tile*256u+(lane%16u)*16u+k];
        if constexpr(Bf16) {
            _Float16 ah,bh;__builtin_memcpy(&ah,&av,2u);__builtin_memcpy(&bh,&bv,2u);
            a[k]=uint16_t(__float_as_uint(float(ah))>>16u);b[k]=uint16_t(__float_as_uint(float(bh))>>16u);
        }else{a[k]=av;b[k]=bv;}
    }
    F32x8 c{},result;
#pragma unroll
    for(unsigned i=0;i<8u;++i)c[i]=initial[tile*256u+(2u*i+lane/16u)*16u+lane%16u];
    if constexpr(Bf16)result=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,c);
    else {
        using Half16=_Float16 __attribute__((ext_vector_type(16)));
        Half16 av,bv;__builtin_memcpy(&av,&a,sizeof(a));__builtin_memcpy(&bv,&b,sizeof(b));
        result=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av,bv,c);
    }
#pragma unroll
    for(unsigned i=0;i<8u;++i)output[tile*256u+(2u*i+lane/16u)*16u+lane%16u]=result[i];
}
void float_guards(const std::vector<float>& v) {
    for(unsigned i=0;i<guard;++i)if(bits(v[i])!=0xa5a5a5a5u||bits(v[v.size()-1u-i])!=0xa5a5a5a5u)throw std::runtime_error("float output guard changed");
}
template<class T>
void float_save(const std::string& path,const std::vector<T>& values,size_t begin,size_t count) {
    if(begin+count>values.size()||std::ifstream(path,std::ios::binary).good())throw std::runtime_error("invalid or existing float capture");
    std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(values.data()+begin),count*sizeof(T));
    if(!out)throw std::runtime_error("float capture write failed");
}
void run_float(const std::string& directory) {
    const FloatInputs input;constexpr unsigned bf_cells=6u*float_samples*256u;
    for(const auto* v:{&input.left,&input.right})for(unsigned i=guard;i<guard+bf_cells;++i) {
        const float value=float(half_value((*v)[i]));
        if(bits(value)&0xffffu)throw std::runtime_error("BF16 control operand changed value");
    }
    Device left(input.left.size()*2u),right(input.right.size()*2u),initial(input.carry.size()*4u);
    Device half_output((float_cells+2u*guard)*4u),bf_output((bf_cells+2u*guard)*4u),prefix((size_t(float_cells)*8u+2u*guard)*4u);
    check(hipMemcpy(left.data,input.left.data(),input.left.size()*2u,hipMemcpyHostToDevice));
    check(hipMemcpy(right.data,input.right.data(),input.right.size()*2u,hipMemcpyHostToDevice));
    check(hipMemcpy(initial.data,input.carry.data(),input.carry.size()*4u,hipMemcpyHostToDevice));
    std::vector<float> half(float_cells+2u*guard),bf(bf_cells+2u*guard),trace(size_t(float_cells)*8u+2u*guard),old_half,old_bf,old_trace;
    for(unsigned attempt=0;attempt<2u;++attempt) {
        check(hipMemset(half_output.data,0xa5,half.size()*4u));check(hipMemset(bf_output.data,0xa5,bf.size()*4u));check(hipMemset(prefix.data,0xa5,trace.size()*4u));
        hipLaunchKernelGGL((float_matrix<false>),dim3(float_tiles),dim3(32u),0u,nullptr,left.as<uint16_t>()+guard,right.as<uint16_t>()+guard,initial.as<float>()+guard,half_output.as<float>()+guard);
        check(hipGetLastError());
        hipLaunchKernelGGL((float_matrix<true>),dim3(6u*float_samples),dim3(32u),0u,nullptr,left.as<uint16_t>()+guard,right.as<uint16_t>()+guard,initial.as<float>()+guard,bf_output.as<float>()+guard);
        check(hipGetLastError());
        hipLaunchKernelGGL(float_chain,dim3((float_cells+255u)/256u+1u),dim3(256u),0u,nullptr,left.as<uint16_t>()+guard,right.as<uint16_t>()+guard,initial.as<float>()+guard,prefix.as<float>()+guard);
        check(hipGetLastError());finish();
        for(auto pair:{std::make_pair(&half_output,&half),std::make_pair(&bf_output,&bf),std::make_pair(&prefix,&trace)}) {
            check(hipMemcpy(pair.second->data(),pair.first->data,pair.second->size()*4u,hipMemcpyDeviceToHost));float_guards(*pair.second);
        }
        if(attempt&&(std::memcmp(half.data(),old_half.data(),half.size()*4u)||std::memcmp(bf.data(),old_bf.data(),bf.size()*4u)||std::memcmp(trace.data(),old_trace.data(),trace.size()*4u)))throw std::runtime_error("floating repeat changed");
        old_half=half;old_bf=bf;old_trace=trace;
    }
    for(auto pair:{std::make_pair(&left,&input.left),std::make_pair(&right,&input.right)}) {
        std::vector<uint16_t> copy(pair.second->size());check(hipMemcpy(copy.data(),pair.first->data,copy.size()*2u,hipMemcpyDeviceToHost));
        if(copy!=*pair.second)throw std::runtime_error("floating operand or guard changed");
    }
    std::vector<float> initial_copy(input.carry.size());check(hipMemcpy(initial_copy.data(),initial.data,initial_copy.size()*4u,hipMemcpyDeviceToHost));
    if(std::memcmp(initial_copy.data(),input.carry.data(),initial_copy.size()*4u))throw std::runtime_error("floating carry or guard changed");
    for(unsigned family=0;family<float_families;++family) {
        unsigned step_bad=0,recursive_bad=0,half_bad=0,bf_bad=0,first=float_cells*8u;
        const unsigned begin=family*float_samples*256u,end=begin+float_samples*256u;
        for(unsigned cell=begin;cell<end;++cell) {
            const unsigned tile=cell/256u,row=cell%256u/16u,col=cell%16u;
            const auto* a=input.left.data()+guard+tile*256u+row*16u;const auto* b=input.right.data()+guard+tile*256u+col*16u;
            for(unsigned step=0;step<8u;++step) {
                const size_t index=size_t(cell)*8u+step;const float c=step?trace[guard+index-1u]:input.carry[guard+cell];
                const float expected=modeled_pair(a[2u*step],a[2u*step+1u],b[2u*step],b[2u*step+1u],c);
                if(!std::isfinite(trace[guard+index]))throw std::runtime_error("nonfinite native floating dot");
                const bool bad=bits(expected)!=bits(trace[guard+index]);step_bad+=bad;if(bad&&first==float_cells*8u)first=unsigned(index);
                recursive_bad+=bits(input.predicted[index])!=bits(trace[guard+index]);
            }
            half_bad+=bits(half[guard+cell])!=bits(trace[guard+size_t(cell)*8u+7u]);
            if(family<6u)bf_bad+=bits(bf[guard+cell])!=bits(trace[guard+size_t(cell)*8u+7u]);
        }
        std::printf("{\"kind\":\"wmma_dot_float_diagnostic\",\"family\":%u,\"cells\":%u,\"prefix_states\":%u,\"actual_prefix_model_mismatches\":%u,\"recursive_model_mismatches\":%u,\"fp16_wmma_dot2_mismatches\":%u,\"bf16_wmma_checked\":%s,\"bf16_wmma_dot2_mismatches\":%u,\"first_model_mismatch_index\":%u,\"repeat_bit_parity\":true,\"immutable_inputs\":true,\"redzones_pass\":true,\"general_model_proven\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",family,end-begin,(end-begin)*8u,step_bad,recursive_bad,half_bad,family<6u?"true":"false",bf_bad,first);
    }
    float_save(directory+"/left-half.bin",input.left,guard,float_cells);float_save(directory+"/right-half.bin",input.right,guard,float_cells);
    float_save(directory+"/carry.bin",input.carry,guard,float_cells);float_save(directory+"/wmma-fp16.bin",half,guard,float_cells);
    float_save(directory+"/wmma-bf16.bin",bf,guard,bf_cells);float_save(directory+"/prefix.bin",trace,guard,size_t(float_cells)*8u);
    float_save(directory+"/model-prefix.bin",input.predicted,0u,input.predicted.size());
}
#endif
} // namespace

int main(int argc,char** argv)try {
    if(argc>2) {
        char* end=nullptr;const auto salt=std::strtoul(argv[2],&end,0);
        if(!end||*end||salt>UINT32_MAX)throw std::runtime_error("invalid fixture seed salt");
        float_seed_salt=uint32_t(salt);
    }
#if defined(__HIPCC__)
    hipDeviceProp_t device{};check(hipGetDeviceProperties(&device,0));
    if(std::strncmp(device.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    if(argc!=2&&argc!=3)throw std::runtime_error("requires floating capture directory and optional seed salt");
    run_float(argv[1]);
#else
    (void)argc;(void)argv;const FloatInputs input;
    std::printf("{\"kind\":\"wmma_dot_float_host_inputs\",\"fixture_seed_salt\":%u,\"cells\":%u,\"modeled_prefix_states\":%u,\"native_executed\":false,\"general_model_proven\":false}\n",float_seed_salt,float_cells,float_cells*8u);
#endif
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
