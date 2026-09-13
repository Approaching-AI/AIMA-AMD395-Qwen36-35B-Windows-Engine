#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_row_max_projection.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace row_gpu = qrt_sm121_row_max_projection;
namespace original = qrt_q1_moe_hawkeye;
constexpr unsigned guard = 65u;
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event); if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        require(std::chrono::steady_clock::now() < end,"row maximum projection deadline"); std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
template<class T> struct Buffer {
    T* base = nullptr; size_t count;
    explicit Buffer(const std::vector<T>& values):count(values.size()) {
        check(hipMalloc(reinterpret_cast<void**>(&base),count*sizeof(T)));
        check(hipMemcpy(base,values.data(),count*sizeof(T),hipMemcpyHostToDevice));
    }
    ~Buffer() { if (base) (void)hipFree(base); }
    T* data() { return base+guard; }
    std::vector<T> read() { std::vector<T> values(count); check(hipMemcpy(values.data(),base,count*sizeof(T),hipMemcpyDeviceToHost)); return values; }
};
uint32_t mix(uint32_t value) { value ^= value<<13u; value ^= value>>17u; value ^= value<<5u; return value; }
uint16_t word(unsigned row,unsigned k,unsigned mode,unsigned side) {
    const uint32_t value = mix(0x3957169u ^ (row*977u+k*173u+side*7919u));
    if (mode == 6u) return uint16_t((120u<<7u)|(value&127u));
    if (mode == 7u) return row%3u==0u ? uint16_t(value&0x8000u) : uint16_t((value&0x807fu)|((k%257u?64u:190u)<<7u));
    if (mode == 2u) {
        const uint16_t edges[]={0u,0x8000u,1u,127u,128u,0x1f80u,0x2000u,0x5f00u,0x5f80u,0x7f80u,0xff80u,0x7fc1u};
        return k%13u ? uint16_t((value&0x807fu)|((120u+value%6u)<<7u)) : edges[(row+k+side)%12u];
    }
    if (mode == 4u) return uint16_t((value&0x807fu)|((side?4u+value%48u:108u+value%25u)<<7u));
    if (mode == 5u) return uint16_t((value&0x807fu)|((250u+value%5u)<<7u));
    if (mode == 3u) return uint16_t((120u<<7u)|127u|((side==0u && k%2u)?0x8000u:0u));
    return uint16_t((value&0x807fu)|((mode==1u?64u+value%127u:118u+value%8u)<<7u));
}
template<unsigned Variant>
__global__ void compare(const uint16_t* weights,const uint16_t* inputs,
    const uint32_t* wb,const uint32_t* ib,const unsigned* wf,const unsigned* inf,
    uint32_t* output,uint32_t* traces,uint32_t* stats,unsigned rows,unsigned tokens,unsigned width) {
    const unsigned cell=(blockIdx.x*blockDim.x+threadIdx.x)/4u;
    if(cell>=rows*tokens) return;
    const unsigned row=cell%rows,token=cell/rows,groups=width/16u;
    float value;
    if constexpr(Variant) value=row_gpu::dot<Variant == 2u>(inputs+size_t(token)*width,weights+size_t(row)*width,width,
        ib[token],wb[row],traces+size_t(cell)*groups*3u,stats+size_t(cell)*4u);
    else value=qrt_sm121_scalar_projection::validated_dot<4u>(inputs+size_t(token)*width,weights+size_t(row)*width,width,wf[row]&&inf[token]);
    if(!(threadIdx.x&3u)) output[cell]=__float_as_uint(value);
}
void run(unsigned rows,unsigned tokens,unsigned width,unsigned mode) {
    const unsigned groups=width/16u,cells=rows*tokens;
    std::vector<uint16_t> weights(size_t(rows)*width+2u*guard,0x5a5au),inputs(size_t(tokens)*width+2u*guard,0x5a5au);
    std::vector<unsigned> wf(rows+2u*guard,0xa5a5a5a5u),inf(tokens+2u*guard,0xa5a5a5a5u);
    for(unsigned side=0u;side<2u;++side) {
        auto& values=side?weights:inputs;auto& flags=side?wf:inf;const unsigned count=side?rows:tokens;
        for(unsigned row=0u;row<count;++row) {
            flags[guard+row]=1u;
            for(unsigned k=0u;k<width;++k) {
                const auto value=word(row,k,mode,side);values[guard+size_t(row)*width+k]=value;
                flags[guard+row]&=qrt_sm121_float_alignment::eligible(value);
            }
        }
    }
    std::vector<uint32_t> wb(rows+2u*guard,0xa5a5a5a5u),ib(tokens+2u*guard,0xa5a5a5a5u);
    std::vector<uint32_t> expected(cells+2u*guard,0xa5a5a5a5u),trace(size_t(cells)*groups*3u+2u*guard,0xa5a5a5a5u),stats(size_t(cells)*4u+2u*guard,0xa5a5a5a5u);
    Buffer<uint16_t> dw(weights),di(inputs);Buffer<unsigned> dfw(wf),dfi(inf);Buffer<uint32_t> dwb(wb),dib(ib),dout(expected),dt(trace),ds(stats);
    check(row_gpu::prepare(dw.data(),dwb.data(),rows,rows,width,nullptr));
    check(row_gpu::prepare(di.data(),dib.data(),tokens,tokens,width,nullptr));finish();
    for(unsigned side=0u;side<2u;++side) {
        auto& flags=side?wb:ib;const auto& values=side?weights:inputs;const unsigned count=side?rows:tokens;
        for(unsigned row=0u;row<count;++row) {
            flags[guard+row]=0u;
            for(unsigned k=0u;k<width;++k) {
                const uint16_t v=values[guard+size_t(row)*width+k];const unsigned e=(v>>7u)&255u;
                const unsigned summary=!(v&0x7fffu)?0u:e<64u||e>190u?256u:e;
                flags[guard+row]=std::max(flags[guard+row],summary);
            }
        }
    }
    require(dwb.read()==wb && dib.read()==ib,"GPU/CPU metadata or redzone mismatch");
    for(unsigned cell=0u;cell<cells;++cell) {
        original::Value carry{0u,-133,false};
        for(unsigned group=0u;group<groups;++group) {
            original::Value values[17];values[0]=carry;
            for(unsigned k=0u;k<16u;++k) values[k+1u]=original::multiply_bf16(
                inputs[guard+size_t(cell/rows)*width+group*16u+k],weights[guard+size_t(cell%rows)*width+group*16u+k],-133);
            carry=original::group_sum<26,-133>(values,17u);
            const size_t offset=guard+(size_t(cell)*groups+group)*3u;
            trace[offset]=carry.significand;trace[offset+1u]=uint32_t(int32_t(carry.exponent));trace[offset+2u]=unsigned(carry.negative);
        }
        const float value=original::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
        std::memcpy(&expected[guard+cell],&value,4u);
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(compare<0u>),dim3((cells*4u+255u)/256u),dim3(256u),0u,nullptr,
        dw.data(),di.data(),dwb.data(),dib.data(),dfw.data(),dfi.data(),dout.data(),dt.data(),ds.data(),rows,tokens,width);
    check(hipGetLastError());finish();require(dout.read()==expected,"original GPU/CPU output mismatch");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(compare<1u>),dim3((cells*4u+255u)/256u),dim3(256u),0u,nullptr,
        dw.data(),di.data(),dwb.data(),dib.data(),dfw.data(),dfi.data(),dout.data(),dt.data(),ds.data(),rows,tokens,width);
    check(hipGetLastError());finish();require(dout.read()==expected,"production GPU/CPU output mismatch");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(compare<2u>),dim3((cells*4u+255u)/256u),dim3(256u),0u,nullptr,
        dw.data(),di.data(),dwb.data(),dib.data(),dfw.data(),dfi.data(),dout.data(),dt.data(),ds.data(),rows,tokens,width);
    check(hipGetLastError());finish();require(dout.read()==expected,"row maximum audit GPU/CPU output mismatch");
    require(dt.read()==trace,"ordered canonical carry or trace redzone mismatch");
    const auto counts=ds.read();uint64_t totals[4]{};
    for(unsigned cell=0u;cell<cells;++cell) {
        unsigned total=0u;for(unsigned i=0u;i<4u;++i) {total+=counts[guard+size_t(cell)*4u+i];totals[i]+=counts[guard+size_t(cell)*4u+i];}
        require(total==groups,"incomplete group accounting");
    }
    for(unsigned i=0u;i<guard;++i) require(counts[i]==0xa5a5a5a5u && counts[guard+size_t(cells)*4u+i]==0xa5a5a5a5u,"counter redzone");
    require(dw.read()==weights && di.read()==inputs && dfw.read()==wf && dfi.read()==inf && dwb.read()==wb && dib.read()==ib,"immutable operands changed");
    std::printf("{\"kind\":\"row_max_projection_safety\",\"rows\":%u,\"tokens\":%u,\"k\":%u,\"data_mode\":%u,\"cells\":%u,\"ordered_groups\":%zu,\"raw_bit_mismatches\":0,\"canonical_carry_mismatches\":0,\"carry_dominant_groups\":%llu,\"maximum_reduction_groups\":%llu,\"ineligible_fallback_groups\":%llu,\"unused_counter3\":%llu,\"cpu_metadata_pass\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
        rows,tokens,width,mode,cells,size_t(cells)*groups,(unsigned long long)totals[0],(unsigned long long)totals[1],(unsigned long long)totals[2],(unsigned long long)totals[3]);
}
int main() try {
    hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));require(!std::strncmp(p.gcnArchName,"gfx1151",7u),"requires gfx1151");
    const unsigned shapes[][3]={{1u,1u,16u},{17u,5u,80u},{31u,7u,272u},{33u,17u,2048u},{65u,9u,4096u}};
    for(const auto& shape:shapes) for(unsigned mode=0u;mode<8u;++mode) run(shape[0],shape[1],shape[2],mode);
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"row_max_projection_error=%s\n",e.what());return 2;}
