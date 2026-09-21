#include "../../native/providers/moe_accumulator/sm121_partial_wave_projection.h"
#include "../../native/providers/moe_accumulator/sm121_tiled_projection.h"
#include "narrow_half_cases.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace candidate=qrt_sm121_partial_wave_projection;
namespace original=qrt_q1_moe_hawkeye;
constexpr unsigned guard=65u,marker=0xa5a5a5a5u;
void check(hipError_t status) {if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));}
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
struct Device {
    void* base=nullptr;size_t bytes;
    explicit Device(size_t size):bytes(size){check(hipMalloc(&base,size));reset();}
    ~Device(){if(base)(void)hipFree(base);}
    void reset(){check(hipMemset(base,0xa5,bytes));}
    template<class T>T* data(){return static_cast<T*>(base)+guard;}
};
template<class T>std::vector<T> read(Device& device,size_t count){
    std::vector<T> values(count+2u*guard);
    check(hipMemcpy(values.data(),device.base,values.size()*sizeof(T),hipMemcpyDeviceToHost));return values;
}
template<class T>void redzones(const std::vector<T>& values){
    const auto* bytes=reinterpret_cast<const unsigned char*>(values.data());
    for(size_t i=0u;i<guard*sizeof(T);++i)
        require(bytes[i]==0xa5u&&bytes[(values.size()-guard)*sizeof(T)+i]==0xa5u,"redzone");
}
void finish(){
    hipEvent_t event;check(hipEventCreate(&event));check(hipEventRecord(event));
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;){const auto status=hipEventQuery(event);if(status==hipSuccess)break;
        if(status!=hipErrorNotReady)check(status);
        if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("completion deadline");
        std::this_thread::yield();}
    check(hipEventDestroy(event));
}
uint16_t input(unsigned family,unsigned side,unsigned row,unsigned g,unsigned i,unsigned groups){
    const unsigned bits=qrt_narrow_half_cases::random_word(0x395u+side*9991u+row*7919u+g*331u+i*17u);
    if(family==0u){const auto p=qrt_narrow_half_cases::input(row+side*3u,g,i);return side?p.y:p.x;}
    if(family==1u)return uint16_t((bits&0x807fu)|((118u+(bits%17u))<<7u));
    if(family==2u)return uint16_t(bits); // Full BF16 domain, including unsupported rows.
    uint16_t value=uint16_t((bits&0x807fu)|((95u+(bits%65u))<<7u));
    if(row%5u==0u)value=uint16_t(bits&0x8000u);
    if(row%5u==1u && (i&1u))value=0u;
    if(row%5u==2u && g+1u==groups && i==15u)value=side?0x7fc1u:1u;
    return value;
}
void run(unsigned rows,unsigned tokens,unsigned width,unsigned family){
    const size_t cells=size_t(rows)*tokens,groups=width/16u,mw=(cells+31u)/32u,tw=cells*groups;
    const size_t wg=size_t(rows)*groups,ig=size_t(tokens)*groups;
    std::vector<uint16_t> w(size_t(rows)*width),x(size_t(tokens)*width);
    std::vector<unsigned> wf(rows,1u),xf(tokens,1u);
    for(unsigned side=0u;side<2u;++side){auto& values=side?w:x;auto& flags=side?wf:xf;
        for(unsigned row=0u;row<flags.size();++row)for(unsigned k=0u;k<width;++k){
            const uint16_t value=input(family,side,row,k/16u,k%16u,unsigned(groups));values[size_t(row)*width+k]=value;
            const unsigned e=(value>>7u)&255u;flags[row]&=!(value&0x7fffu)||(e>=95u&&e<=159u);
        }
    }
    Device dw((w.size()+2u*guard)*2u),dx((x.size()+2u*guard)*2u);
    Device pw((wg+2u*guard)*sizeof(candidate::Row)),px((ig+2u*guard)*sizeof(candidate::Row));
    Device fw((rows+2u*guard)*4u),fx((tokens+2u*guard)*4u);
    check(hipMemcpy(dw.data<uint16_t>(),w.data(),w.size()*2u,hipMemcpyHostToDevice));
    check(hipMemcpy(dx.data<uint16_t>(),x.data(),x.size()*2u,hipMemcpyHostToDevice));
    check(candidate::encode(dw.data<uint16_t>(),w.size(),pw.data<candidate::Row>(),wg,fw.data<unsigned>(),rows,rows,width,nullptr));
    check(candidate::encode(dx.data<uint16_t>(),x.size(),px.data<candidate::Row>(),ig,fx.data<unsigned>(),tokens,tokens,width,nullptr));finish();
    const auto packed_w=read<candidate::Row>(pw,wg),packed_x=read<candidate::Row>(px,ig);
    const auto flags_w=read<unsigned>(fw,rows),flags_x=read<unsigned>(fx,tokens);
    for(unsigned side=0u;side<2u;++side){const auto& raw=side?w:x;const auto& packed=side?packed_w:packed_x;
        const auto& flags=side?flags_w:flags_x;const auto& expected_flags=side?wf:xf;
        const unsigned count=side?rows:tokens;redzones(packed);redzones(flags);
        require(!std::memcmp(flags.data()+guard,expected_flags.data(),count*4u),"whole-row classification");
        for(unsigned row=0u;row<count;++row)for(unsigned g=0u;g<groups;++g){
            const auto expected=candidate::partial::prepare(raw.data()+size_t(row)*width+g*16u);
            const auto& actual=packed[guard+size_t(g)*count+row];
            require(!std::memcmp(&actual,&expected,sizeof(expected)),"prepared group-major row");
            for(unsigned i=0u;i<16u;++i)require(candidate::partial::original(actual,i)==raw[size_t(row)*width+g*16u+i],"lossless operand");
        }
    }
    auto immutable=[&](){
        const auto aw=read<uint16_t>(dw,w.size()),ax=read<uint16_t>(dx,x.size());redzones(aw);redzones(ax);
        require(!std::memcmp(aw.data()+guard,w.data(),w.size()*2u)&&!std::memcmp(ax.data()+guard,x.data(),x.size()*2u),"raw input changed");
        const auto apw=read<candidate::Row>(pw,wg),apx=read<candidate::Row>(px,ig);
        require(!std::memcmp(apw.data(),packed_w.data(),apw.size()*sizeof(candidate::Row))&&
            !std::memcmp(apx.data(),packed_x.data(),apx.size()*sizeof(candidate::Row)),"prepared input changed");
        require(read<unsigned>(fw,rows)==flags_w&&read<unsigned>(fx,tokens)==flags_x,"row flags changed");
    };
    immutable();
    Device mask((mw+2u*guard)*4u),output((cells+2u*guard)*4u),trace((tw+2u*guard)*4u);
    for(unsigned pattern=0u;pattern<5u;++pattern){
        std::vector<unsigned> selected,expected_mask(mw,0u),expected_output(cells,marker),expected_trace(tw,marker);
        unsigned admitted=0u;
        for(unsigned cell=0u;cell<cells;++cell){
            const unsigned row=cell%rows,token=cell/rows;
            const bool chosen=pattern==4u || (pattern==1u&&cell%37u==0u) ||
                (pattern==2u&&cell%4u==1u) || (pattern==3u&&row%16u==15u&&token%16u==7u);
            if(!chosen)continue;
            selected.push_back(cell);expected_mask[cell/32u]|=1u<<(cell&31u);admitted+=wf[row]&&xf[token];
            original::Value carry{0u,-133,false};
            for(unsigned g=0u;g<groups;++g){
                original::Value terms[17];terms[0]=carry;
                for(unsigned i=0u;i<16u;++i)terms[i+1u]=original::multiply_bf16(x[size_t(token)*width+g*16u+i],w[size_t(row)*width+g*16u+i],-133);
                carry=original::group_sum<26,-133>(terms,17u);
                expected_trace[size_t(cell)*groups+g]=candidate::group::f32::bits(original::value_to_float(carry));
            }
            expected_output[cell]=candidate::group::f32::bits(original::value_to_float(qrt_sm121_group16::finish_accumulator(carry)));
        }
        Device indices((selected.size()+2u*guard)*4u);
        if(!selected.empty())check(hipMemcpy(indices.data<unsigned>(),selected.data(),selected.size()*4u,hipMemcpyHostToDevice));
        check(qrt_sm121_tiled_projection::mark(indices.data<unsigned>(),unsigned(selected.size()),unsigned(cells),mask.data<unsigned>(),mw,nullptr));finish();
        const auto bitmap=read<unsigned>(mask,mw);redzones(bitmap);
        require(!std::memcmp(bitmap.data()+guard,expected_mask.data(),mw*4u),"original candidate bitmap");
        const auto index=read<unsigned>(indices,selected.size());redzones(index);
        auto launch=[&](bool force,bool audit){
            if(force)check(candidate::launch<true>(dw.data<uint16_t>(),dx.data<uint16_t>(),pw.data<candidate::Row>(),px.data<candidate::Row>(),fw.data<unsigned>(),fx.data<unsigned>(),mask.data<unsigned>(),mw,indices.data<unsigned>(),unsigned(selected.size()),output.data<float>(),cells,rows,tokens,width,nullptr,audit?trace.data<unsigned>():nullptr,audit?tw:0u));
            else check(candidate::launch<false>(dw.data<uint16_t>(),dx.data<uint16_t>(),pw.data<candidate::Row>(),px.data<candidate::Row>(),fw.data<unsigned>(),fx.data<unsigned>(),mask.data<unsigned>(),mw,indices.data<unsigned>(),unsigned(selected.size()),output.data<float>(),cells,rows,tokens,width,nullptr,audit?trace.data<unsigned>():nullptr,audit?tw:0u));
            finish();immutable();
            require(read<unsigned>(mask,mw)==bitmap&&read<unsigned>(indices,selected.size())==index,"candidate source changed");
        };
        for(unsigned variant=0u;variant<2u;++variant){
            output.reset();trace.reset();launch(variant!=0u,true);
            const auto actual=read<unsigned>(output,cells),states=read<unsigned>(trace,tw);redzones(actual);redzones(states);
            require(!std::memcmp(actual.data()+guard,expected_output.data(),cells*4u),"independent original output bits");
            require(!std::memcmp(states.data()+guard,expected_trace.data(),tw*4u),"original ordered carry bits or inactive writes");
            output.reset();launch(variant!=0u,false);
            require(read<unsigned>(output,cells)==actual&&read<unsigned>(trace,tw)==states,"production trace parity");
            std::printf("{\"kind\":\"partial_wave_projection\",\"rows\":%u,\"tokens\":%u,\"k\":%u,\"family\":%u,\"pattern\":%u,\"variant\":%u,\"candidates\":%zu,\"admitted_candidates\":%u,\"original_candidates\":%zu,\"ordered_carries\":%zu,\"raw_bit_mismatches\":0,\"whole_row_flags_checked\":true,\"all_encoded_words_checked\":true,\"production_trace_parity\":true,\"redzones_pass\":true,\"immutable_after_each_invocation\":true,\"inference_acceptance\":false}\n",rows,tokens,width,family,pattern,variant,selected.size(),admitted,selected.size()-admitted,selected.size()*groups);
        }
    }
}
int main()try{
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0));
    require(!std::strncmp(properties.gcnArchName,"gfx1151",7u),"requires gfx1151");
    const unsigned shapes[][3]={{1u,1u,16u},{17u,19u,272u},{33u,17u,2048u},{35u,33u,4096u},{17u,17u,8192u},{65u,3u,4112u}};
    for(const auto& shape:shapes)for(unsigned family=0u;family<4u;++family)run(shape[0],shape[1],shape[2],family);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
