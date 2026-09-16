// Isolated test of the empirical pair model over the coarse owner's BF16 range.
#define main qrt_existing_integer_rounding_probe_main
#include "wmma_integer_rounding_probe.cpp"
#undef main
#include <fstream>
#include <string>

namespace {
constexpr unsigned wide_families=8u,wide_samples=64u,wide_tiles=wide_families*wide_samples,wide_cells=wide_tiles*256u;
float wide_float(uint32_t raw) {float result;std::memcpy(&result,&raw,4u);return result;}
uint16_t wide_operand(unsigned family,unsigned sample,unsigned row,unsigned k,bool right) {
    const unsigned index=family==3u?k/2u:k;
    const uint32_t h=hash(0xbd16395du^sample*7919u^row*104729u^index*65537u^unsigned(right)*99991u);
    unsigned sign=(h>>16u)&0x8000u,exponent=80u+((h>>7u)%95u);
    if(family==0u)sign=0u;
    if(family==1u)sign=right?0u:0x8000u;
    if(family==3u&&right&&(k&1u))sign^=0x8000u;
    if(family==4u&&k>=2u)return 0u;
    if(family==5u)exponent=k%4u==0u?174u:80u+((h>>7u)%12u);
    if(family==6u)exponent=80u+((h>>7u)%8u);
    if(family==7u)exponent=167u+((h>>7u)%8u);
    return uint16_t(sign|(exponent<<7u)|(h&127u));
}
float wide_modeled_pair(uint16_t a0,uint16_t a1,uint16_t b0,uint16_t b1,float carry) {
    auto exponent=[](uint16_t a,uint16_t b) {return (a&0x7fffu)&&(b&0x7fffu)?int((a>>7u)&255u)+int((b>>7u)&255u)-254:-126;};
    const int ce=carry?int((bits(carry)>>23u)&255u)-127:-126;
    const int e=std::max({exponent(a0,b0),exponent(a1,b1),ce-2});const double q=std::ldexp(1.0,e-24);
    double total=std::trunc(double(carry)/q)+2.0*double(bits(carry)>>31u);
    for(auto pair:{std::make_pair(a0,b0),std::make_pair(a1,b1)}) {
        const double product=double(wide_float(uint32_t(pair.first)<<16u))*double(wide_float(uint32_t(pair.second)<<16u));
        const double magnitude=std::trunc(std::abs(product)/q);
        total+=((pair.first^pair.second)&0x8000u)?-(magnitude+1.0):magnitude;
    }
    const float result=float(total*q);return (bits(result)&0x7fffffffu)<0x00800000u?0.0f:result;
}
struct WideInputs {
    std::vector<uint16_t> left,right;std::vector<float> model;
    WideInputs():left(wide_cells+2u*guard,0x5a5au),right(left),model(size_t(wide_cells)*8u) {
        for(unsigned tile=0;tile<wide_tiles;++tile) {
            const unsigned family=tile/wide_samples,sample=tile%wide_samples;
            for(unsigned row=0;row<16u;++row)for(unsigned k=0;k<16u;++k) {
                left[guard+tile*256u+row*16u+k]=wide_operand(family,sample,row,k,false);
                right[guard+tile*256u+row*16u+k]=wide_operand(family,sample,row,k,true);
            }
            for(unsigned row=0;row<16u;++row)for(unsigned col=0;col<16u;++col) {
                const unsigned cell=tile*256u+row*16u+col;float c=0.0f;
                const auto* a=left.data()+guard+tile*256u+row*16u;const auto* b=right.data()+guard+tile*256u+col*16u;
                for(unsigned step=0;step<8u;++step) {
                    c=wide_modeled_pair(a[2u*step],a[2u*step+1u],b[2u*step],b[2u*step+1u],c);
                    if(!std::isfinite(c))throw std::runtime_error("nonfinite wide model result");
                    model[size_t(cell)*8u+step]=c;
                }
            }
        }
    }
};

#if defined(__HIPCC__)
template<unsigned Prefix>
__global__ void wide_matrix(const uint16_t* left,const uint16_t* right,float* output) {
    const unsigned lane=threadIdx.x,tile=blockIdx.x;U16x16 a{},b{};
#pragma unroll
    for(unsigned k=0;k<Prefix*2u;++k) {
        a[k]=left[tile*256u+(lane%16u)*16u+k];b[k]=right[tile*256u+(lane%16u)*16u+k];
    }
    const F32x8 zero{},result=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,zero);
#pragma unroll
    for(unsigned i=0;i<8u;++i)output[(tile*256u+(2u*i+lane/16u)*16u+lane%16u)*8u+Prefix-1u]=result[i];
}
template<class T>
void wide_save(const std::string& path,const std::vector<T>& values,size_t begin,size_t count) {
    if(begin+count>values.size()||std::ifstream(path,std::ios::binary).good())throw std::runtime_error("invalid or existing wide capture");
    std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(values.data()+begin),count*sizeof(T));
    if(!out)throw std::runtime_error("wide capture write failed");
}
void run_wide(const std::string& directory) {
    const WideInputs input;Device left(input.left.size()*2u),right(input.right.size()*2u),output((size_t(wide_cells)*8u+2u*guard)*4u);
    check(hipMemcpy(left.data,input.left.data(),input.left.size()*2u,hipMemcpyHostToDevice));
    check(hipMemcpy(right.data,input.right.data(),input.right.size()*2u,hipMemcpyHostToDevice));
    std::vector<float> result(size_t(wide_cells)*8u+2u*guard),previous;
    for(unsigned attempt=0;attempt<2u;++attempt) {
        check(hipMemset(output.data,0xa5,result.size()*4u));
#define QRT_WIDE_PREFIX(N) hipLaunchKernelGGL((wide_matrix<N>),dim3(wide_tiles),dim3(32u),0u,nullptr,left.as<uint16_t>()+guard,right.as<uint16_t>()+guard,output.as<float>()+guard);check(hipGetLastError())
        QRT_WIDE_PREFIX(1u);QRT_WIDE_PREFIX(2u);QRT_WIDE_PREFIX(3u);QRT_WIDE_PREFIX(4u);
        QRT_WIDE_PREFIX(5u);QRT_WIDE_PREFIX(6u);QRT_WIDE_PREFIX(7u);QRT_WIDE_PREFIX(8u);
#undef QRT_WIDE_PREFIX
        finish();check(hipMemcpy(result.data(),output.data,result.size()*4u,hipMemcpyDeviceToHost));
        for(unsigned i=0;i<guard;++i)if(bits(result[i])!=0xa5a5a5a5u||bits(result[result.size()-1u-i])!=0xa5a5a5a5u)throw std::runtime_error("wide output guard changed");
        if(attempt&&std::memcmp(result.data(),previous.data(),result.size()*4u))throw std::runtime_error("wide native repeat changed");previous=result;
    }
    for(auto pair:{std::make_pair(&left,&input.left),std::make_pair(&right,&input.right)}) {
        std::vector<uint16_t> copy(pair.second->size());check(hipMemcpy(copy.data(),pair.first->data,copy.size()*2u,hipMemcpyDeviceToHost));
        if(copy!=*pair.second)throw std::runtime_error("wide operand or guard changed");
    }
    for(unsigned family=0;family<wide_families;++family) {
        unsigned recursive_bad=0,transition_bad=0,first=wide_cells*8u;const unsigned begin=family*wide_samples*256u,end=begin+wide_samples*256u;
        for(unsigned cell=begin;cell<end;++cell) {
            const unsigned tile=cell/256u,row=cell%256u/16u,col=cell%16u;
            const auto* a=input.left.data()+guard+tile*256u+row*16u;const auto* b=input.right.data()+guard+tile*256u+col*16u;
            for(unsigned step=0;step<8u;++step) {
                const size_t index=size_t(cell)*8u+step;const float actual=result[guard+index];
                if(!std::isfinite(actual))throw std::runtime_error("nonfinite wide native result");
                const bool bad=bits(actual)!=bits(input.model[index]);recursive_bad+=bad;
                if(bad&&first==wide_cells*8u)first=unsigned(index);
                const float carry=step?result[guard+index-1u]:0.0f;
                transition_bad+=bits(actual)!=bits(wide_modeled_pair(a[2u*step],a[2u*step+1u],b[2u*step],b[2u*step+1u],carry));
            }
        }
        std::printf("{\"kind\":\"wmma_bf16_guard_diagnostic\",\"family\":%u,\"cells\":%u,\"masked_prefix_outputs\":%u,\"recursive_model_mismatches\":%u,\"adjacent_prefix_transition_mismatches\":%u,\"first_model_mismatch_index\":%u,\"repeat_bit_parity\":true,\"immutable_inputs\":true,\"redzones_pass\":true,\"internal_states_directly_observed\":false,\"general_model_proven\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",family,end-begin,(end-begin)*8u,recursive_bad,transition_bad,first);
    }
    wide_save(directory+"/left-bf16.bin",input.left,guard,wide_cells);wide_save(directory+"/right-bf16.bin",input.right,guard,wide_cells);
    wide_save(directory+"/masked-prefix.bin",result,guard,size_t(wide_cells)*8u);wide_save(directory+"/model-prefix.bin",input.model,0u,input.model.size());
}
#endif
} // namespace
int main(int argc,char** argv)try {
#if defined(__HIPCC__)
    hipDeviceProp_t device{};check(hipGetDeviceProperties(&device,0));
    if(std::strncmp(device.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    if(argc!=2)throw std::runtime_error("requires wide BF16 capture directory");
    run_wide(argv[1]);
#else
    (void)argc;(void)argv;const WideInputs input;
    std::printf("{\"kind\":\"wmma_bf16_guard_host_inputs\",\"cells\":%u,\"prefix_states\":%u,\"native_executed\":false}\n",wide_cells,wide_cells*8u);
#endif
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
