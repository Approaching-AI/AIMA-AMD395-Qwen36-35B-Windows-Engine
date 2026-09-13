#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_tiled_projection.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace tiled=qrt_sm121_tiled_projection;
constexpr unsigned guard=65u;
void check(hipError_t status) {if(status!=hipSuccess) throw std::runtime_error(hipGetErrorString(status));}
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
void finish() {
    hipEvent_t event;check(hipEventCreate(&event));check(hipEventRecord(event));
    const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;) {
        const auto status=hipEventQuery(event);if(status==hipSuccess) break;
        if(status!=hipErrorNotReady) check(status);
        require(std::chrono::steady_clock::now()<end,"tiled projection completion deadline");std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
template<class T> struct Buffer {
    T* base=nullptr;size_t count;
    explicit Buffer(const std::vector<T>& values):count(values.size()) {
        check(hipMalloc(reinterpret_cast<void**>(&base),count*sizeof(T)));reset(values);
    }
    ~Buffer(){if(base) (void)hipFree(base);}
    T* data(){return base+guard;}
    void reset(const std::vector<T>& v){require(v.size()==count,"buffer span");check(hipMemcpy(base,v.data(),count*sizeof(T),hipMemcpyHostToDevice));}
    std::vector<T> read(){std::vector<T> v(count);check(hipMemcpy(v.data(),base,count*sizeof(T),hipMemcpyDeviceToHost));return v;}
};
float round_bf16(float value) {
    uint32_t bits;std::memcpy(&bits,&value,4u);bits=(bits+0x7fffu+((bits>>16u)&1u))&0xffff0000u;
    std::memcpy(&value,&bits,4u);return value;
}
__global__ void control_kernel(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* wf,const unsigned* inf,const unsigned* indices,unsigned count,
    float* output,unsigned rows,unsigned width) {
    const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;
    if(slot>=count) return;
    const unsigned cell=indices[slot],row=cell%rows,token=cell/rows;
    const float value=qrt_sm121_scalar_projection::validated_dot<4u>(inputs+size_t(token)*width,
        weights+size_t(row)*width,width,wf[row]&&inf[token]);
    if(!(threadIdx.x&3u)) output[cell]=tiled::rounded(value);
}
void run(unsigned rows,unsigned tokens,unsigned width,unsigned mode) {
    const size_t cells=size_t(rows)*tokens,words=(cells+31u)/32u;
    std::vector<uint16_t> weights(size_t(rows)*width+2u*guard,0x5a5au),inputs(size_t(tokens)*width+2u*guard,0x5a5au);
    for(size_t i=guard;i+guard<weights.size();++i) weights[i]=uint16_t(((i*977u)&0x807fu)|((118u+i%12u)<<7u));
    for(size_t i=guard;i+guard<inputs.size();++i) inputs[i]=uint16_t(((i*173u)&0x807fu)|((119u+i%10u)<<7u));
    for(unsigned row=0u;row<rows;++row) {
        if(row%7u==0u) weights[guard+size_t(row)*width]=0x1fffu;
        else if(row%7u==1u) weights[guard+size_t(row)*width]=0x5f80u;
        else if(row%7u==2u) weights[guard+size_t(row)*width]=0x8000u;
        else if(row%7u==3u) weights[guard+size_t(row)*width]=0x0001u;
    }
    for(unsigned token=0u;token<tokens;++token) if(token%11u==0u) inputs[guard+size_t(token)*width]=0x1f81u;
    std::vector<unsigned> selected;
    std::vector<unsigned> mask(words+2u*guard,0xa5a5a5a5u),expected_mask=mask;
    std::fill(expected_mask.begin()+guard,expected_mask.end()-guard,0u);
    std::vector<float> initial(cells+2u*guard,12345.25f),expected=initial;
    for(unsigned token=0u;token<tokens;++token) for(unsigned row=0u;row<rows;++row) {
        const unsigned cell=token*rows+row;
        const bool active=mode==3u || (mode==1u && ((cell*977u+cell/7u)%17u)==0u) ||
            (mode==2u && token<32u && row<128u && token*128u+row<=512u);
        if(!active) continue;
        selected.push_back(cell);expected_mask[guard+cell/32u] |= 1u<<(cell&31u);
        expected[guard+cell]=round_bf16(qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
            inputs.data()+guard+size_t(token)*width,weights.data()+guard+size_t(row)*width,width));
    }
    std::reverse(selected.begin(),selected.end());
    std::vector<unsigned> indices(selected.size()+2u*guard,0xa5a5a5a5u);
    std::copy(selected.begin(),selected.end(),indices.begin()+guard);
    std::vector<unsigned> wf(rows+2u*guard,0xa5a5a5a5u),inf(tokens+2u*guard,0xa5a5a5a5u);
    Buffer<uint16_t> dw(weights),di(inputs);Buffer<unsigned> dfw(wf),dfi(inf),dmask(mask),didx(indices);Buffer<float> dout(initial);
    hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,dim3(rows),dim3(256u),0u,nullptr,dw.data(),dfw.data(),rows,width);
    check(hipGetLastError());
    hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,dim3(tokens),dim3(256u),0u,nullptr,di.data(),dfi.data(),tokens,width);
    check(hipGetLastError());finish();
    wf=dfw.read();inf=dfi.read();
    const auto flags_match=[&](const auto& values,const auto& flags,unsigned count) {
        for(unsigned row=0u;row<count;++row) {
            bool valid=true;for(unsigned k=0u;k<width;++k) valid &= qrt_sm121_float_alignment::eligible(values[guard+size_t(row)*width+k]);
            require(flags[guard+row]==unsigned(valid),"CPU row flag mismatch");
        }
        for(unsigned i=0u;i<guard;++i) require(flags[i]==0xa5a5a5a5u && flags[guard+count+i]==0xa5a5a5a5u,"flag redzone");
    };
    flags_match(weights,wf,rows);flags_match(inputs,inf,tokens);
    if(!selected.empty()) {
        hipLaunchKernelGGL(control_kernel,dim3((unsigned(selected.size())*4u+255u)/256u),dim3(256u),0u,nullptr,
            dw.data(),di.data(),dfw.data(),dfi.data(),didx.data(),unsigned(selected.size()),dout.data(),rows,width);
        check(hipGetLastError());finish();
    }
    const auto control=dout.read();require(std::memcmp(control.data(),expected.data(),expected.size()*4u)==0,"original GPU replay differs from CPU");
    for(unsigned tile:{64u,128u}) {
        dout.reset(initial);
        check(tiled::mark(didx.data(),unsigned(selected.size()),unsigned(cells),dmask.data(),words,nullptr));
        check(tiled::launch(dw.data(),di.data(),dfw.data(),dfi.data(),dmask.data(),words,dout.data(),rows,tokens,width,tile,nullptr));
        finish();const auto actual=dout.read();unsigned bad=0u;
        for(size_t cell=0u;cell<actual.size();++cell) bad += std::memcmp(&actual[cell],&expected[cell],4u)!=0;
        require(!bad,"tiled output/guard differs from original CPU/GPU");
        require(dmask.read()==expected_mask,"candidate bitmap or guard mismatch");
        require(dw.read()==weights && di.read()==inputs && didx.read()==indices && dfw.read()==wf && dfi.read()==inf,"immutable operand changed");
        std::printf("{\"kind\":\"tiled_projection_safety\",\"rows\":%u,\"tokens\":%u,\"k\":%u,\"selection_mode\":%u,\"tile_rows\":%u,\"tile_tokens\":32,\"cells\":%zu,\"candidates\":%zu,\"raw_bit_mismatches\":%u,\"independent_cpu_reference\":true,\"bitmap_exact\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",rows,tokens,width,mode,tile,cells,selected.size(),bad);
    }
}
int main() try {
    hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));require(!std::strncmp(p.gcnArchName,"gfx1151",7u),"requires gfx1151");
    const unsigned shapes[][3]={{1u,1u,16u},{63u,31u,64u},{64u,32u,64u},{65u,33u,80u},{129u,65u,272u},{65u,33u,2048u},{33u,17u,4096u}};
    for(const auto& s:shapes) for(unsigned mode=0u;mode<4u;++mode) run(s[0],s[1],s[2],mode);
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"tiled_projection_error=%s\n",e.what());return 2;}
