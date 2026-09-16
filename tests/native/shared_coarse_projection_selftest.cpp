#include "../../native/providers/moe_accumulator/sm121_prefix_replay_projection.h"
#include "../../native/providers/moe_accumulator/sm121_shared_coarse_projection.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
namespace kernel=qrt_sm121_prefix_replay_projection;
namespace matrix=qrt_sm121_coarse_projection_matrix;
namespace shared=qrt_sm121_shared_coarse_projection;
namespace bound=qrt_sm121_coarse_projection_bound;
namespace original=qrt_q1_moe_hawkeye;
using Row=kernel::Row;
constexpr unsigned guard=64u,marker=0xa5a5a5a5u,window=262144u;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void check(hipError_t s){if(s!=hipSuccess)throw std::runtime_error(hipGetErrorString(s));}
void finish(){
    hipEvent_t event;check(hipEventCreate(&event));check(hipEventRecord(event));
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;){const auto s=hipEventQuery(event);if(s==hipSuccess)break;if(s!=hipErrorNotReady)check(s);
        require(std::chrono::steady_clock::now()<deadline,"shared C64 completion deadline");std::this_thread::yield();}
    check(hipEventDestroy(event));
}
template<class T>struct Buffer {
    T* base=nullptr;size_t count;unsigned skew;
    explicit Buffer(size_t n,unsigned shift=0u):count(n),skew(shift){check(hipMalloc(reinterpret_cast<void**>(&base),(n+2u*guard+skew)*sizeof(T)));reset();}
    ~Buffer(){if(base && hipFree(base)!=hipSuccess)std::abort();}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    T* data(){return base+guard+skew;}
    void reset(){check(hipMemset(base,0xa5,(count+2u*guard+skew)*sizeof(T)));}
    void upload(const std::vector<T>& v){require(v.size()==count,"upload span");check(hipMemcpy(data(),v.data(),count*sizeof(T),hipMemcpyHostToDevice));}
    std::vector<T> read(){std::vector<T> all(count+2u*guard+skew);check(hipMemcpy(all.data(),base,all.size()*sizeof(T),hipMemcpyDeviceToHost));
        const auto* bytes=reinterpret_cast<const unsigned char*>(all.data());
        for(size_t i=0u;i<(guard+skew)*sizeof(T);++i)require(bytes[i]==0xa5u,"buffer leading guard changed");
        for(size_t i=0u;i<guard*sizeof(T);++i)require(bytes[(count+guard+skew)*sizeof(T)+i]==0xa5u,"buffer trailing guard changed");
        return std::vector<T>(all.begin()+guard+skew,all.end()-guard);}

    void unchanged(const std::vector<T>& expected){const auto actual=read();require(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),count*sizeof(T)),"immutable buffer changed");}
};
template<class T>void marker_tail(const std::vector<T>& v,size_t first){
    const auto* bytes=reinterpret_cast<const unsigned char*>(v.data());
    for(size_t i=first*sizeof(T);i<v.size()*sizeof(T);++i)require(bytes[i]==0xa5u,"unused workspace tail changed");
}
original::Value cpu_prefix(const uint16_t* w,const uint16_t* x,unsigned width){
    original::Value carry{0u,-133,false};
    for(unsigned base=0u;base<width;base+=16u){original::Value terms[17];terms[0]=carry;
        for(unsigned i=0u;i<16u;++i)terms[i+1u]=original::multiply_bf16(w[base+i],x[base+i],-133);
        carry=original::group_sum<26,-133>(terms,17u);}
    return carry;
}
float as_float(original::Value value){return original::value_to_float(qrt_sm121_group16::finish_accumulator(value));}
std::vector<uint16_t> read_words(const char* name,size_t words){
    std::ifstream file(name,std::ios::binary|std::ios::ate);require(bool(file)&&file.tellg()==std::streamoff(words*2u),"capture span");
    std::vector<uint16_t> v(words);file.seekg(0);file.read(reinterpret_cast<char*>(v.data()),words*2u);require(bool(file),"capture read");return v;
}
struct Measurement {double ms=0.0,prefix_ms=0.0;unsigned selected=0u;};
constexpr unsigned variants=5u;
const char* names[variants]={"c64_scalar","c64_vector","shared64x64","shared128x32","shared64x64_prefetch"};
struct Test {
    unsigned rows,tokens,width,cells;bool captured;
    std::vector<uint16_t> weights,inputs,reference;
    std::vector<float> canonical,baseline_centers,baseline_errors;
    std::vector<Row> expected_pw,expected_px;
    std::vector<unsigned> expected_wf,expected_xf;
    Buffer<uint16_t> w,x;
    Buffer<Row> pw,px;
    Buffer<unsigned> wf,xf,ids,counter;
    Buffer<float> centers,errors,output,dc;
    unsigned cpu_dots=0u;
    Test(unsigned r,unsigned n,unsigned k,std::vector<uint16_t> hw,std::vector<uint16_t> hx,
        std::vector<uint16_t> ref={},unsigned weight_offset=0u,unsigned input_offset=0u)
        :rows(r),tokens(n),width(k),cells(r*n),captured(!ref.empty()),weights(std::move(hw)),inputs(std::move(hx)),reference(std::move(ref)),
         w(weights.size(),weight_offset),x(inputs.size(),input_offset),pw(size_t(r)*(k/16u)),px(size_t(n)*(k/16u)),
         wf(r),xf(n),ids(cells),counter(1u),centers(cells),errors(cells),output(cells),dc(cells) {
        require(rows&&tokens&&width&&width%16u==0u&&width<=8192u,"test shape");w.upload(weights);x.upload(inputs);
        prepare();full(false,dc.data(),cells);finish();canonical=dc.read();
        if(!captured){reference.resize(cells);for(unsigned i=0u;i<cells;++i)reference[i]=bound::scalar::bf16(canonical[i]);}
        require(reference.size()==cells,"reference span");
        for(unsigned i=0u;i<cells;++i)require(bound::scalar::finite(canonical[i])&&bound::scalar::bf16(canonical[i])==reference[i],"original canonical differs from BF16 reference");
        const unsigned checks=captured?256u:cells;
        for(unsigned sample=0u;sample<checks;++sample){const unsigned cell=captured?unsigned((uint64_t(sample)*2654435761ull+1013904223ull)%cells):sample;
            const float expected=as_float(cpu_prefix(weights.data()+size_t(cell%rows)*width,inputs.data()+size_t(cell/rows)*width,width));
            require(bound::scalar::bits(expected)==bound::scalar::bits(canonical[cell]),"original GPU differs from independent CPU dot");++cpu_dots;}
        for(unsigned side=0u;side<2u;++side){
            const auto& raw=side?inputs:weights;auto& prepared=side?expected_px:expected_pw;auto& flags=side?expected_xf:expected_wf;
            prepared.resize(raw.size()/16u);flags.assign(side?tokens:rows,1u);
            for(size_t i=0u;i<prepared.size();++i)prepared[i]=qrt_sm121_scaled_half_products::prepare(raw.data()+i*16u);
            for(size_t i=0u;i<raw.size();++i)flags[i/width]&=unsigned(bound::eligible(raw[i]));
        }
    }
    void prepare(){
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((pw.count+255u)/256u),dim3(256u),0u,nullptr,w.data(),pw.data(),rows,width);check(hipGetLastError());
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((px.count+255u)/256u),dim3(256u),0u,nullptr,x.data(),px.data(),tokens,width);check(hipGetLastError());
    }
    void full(bool indexed,float* destination,unsigned count){
        for(unsigned offset=0u;offset<count;offset+=window){const unsigned n=std::min(window,count-offset);
            if(indexed)hipLaunchKernelGGL((kernel::full_replay<true>),dim3((n+63u)/64u),dim3(256u),0u,nullptr,pw.data(),px.data(),ids.data(),destination,rows,width,offset,n);
            else hipLaunchKernelGGL((kernel::full_replay<false>),dim3((n+63u)/64u),dim3(256u),0u,nullptr,pw.data(),px.data(),nullptr,destination,rows,width,offset,n);
            check(hipGetLastError());}
    }
    template<unsigned Rows,bool Prefetch>void shared_producer(){
        constexpr unsigned Tokens=shared::Layout<Rows>::tokens;
        hipLaunchKernelGGL((shared::produce<Rows,Prefetch>),dim3((rows+Rows-1u)/Rows,(tokens+Tokens-1u)/Tokens),dim3(256u),0u,nullptr,
            w.data(),x.data(),wf.data(),xf.data(),centers.data(),errors.data(),rows,tokens,width);check(hipGetLastError());
    }
    void producer(unsigned variant){
        hipLaunchKernelGGL(matrix::eligibility,dim3(rows),dim3(256u),0u,nullptr,w.data(),wf.data(),rows,width);check(hipGetLastError());
        hipLaunchKernelGGL(matrix::eligibility,dim3(tokens),dim3(256u),0u,nullptr,x.data(),xf.data(),tokens,width);check(hipGetLastError());
        if(variant==0u)hipLaunchKernelGGL((matrix::produce<64u,1u,false>),dim3((rows+127u)/128u,(tokens+15u)/16u),dim3(256u),0u,nullptr,w.data(),x.data(),wf.data(),xf.data(),centers.data(),errors.data(),rows,tokens,width);
        else if(variant==1u)hipLaunchKernelGGL((matrix::produce<64u,1u,true>),dim3((rows+127u)/128u,(tokens+15u)/16u),dim3(256u),0u,nullptr,w.data(),x.data(),wf.data(),xf.data(),centers.data(),errors.data(),rows,tokens,width);
        else if(variant==2u)shared_producer<64u,false>();
        else if(variant==3u)shared_producer<128u,false>();
        else shared_producer<64u,true>();
        check(hipGetLastError());
    }
    Measurement run(unsigned variant){
        ids.reset();centers.reset();errors.reset();output.reset();check(hipMemset(counter.data(),0,4u));finish();
        const auto begin=std::chrono::steady_clock::now();prepare();producer(variant);
        hipLaunchKernelGGL(matrix::compact,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,centers.data(),errors.data(),output.data(),ids.data(),counter.data(),cells);
        check(hipGetLastError());finish();Measurement result;
        check(hipMemcpy(&result.selected,counter.data(),4u,hipMemcpyDeviceToHost));require(result.selected<=cells,"candidate capacity");
        result.prefix_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        full(true,output.data(),result.selected);finish();
        result.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        verify(variant,result);return result;
    }
    void verify(unsigned variant,const Measurement& measurement){
        const auto out=output.read(),c=centers.read(),e=errors.read();const auto selected=ids.read();
        marker_tail(selected,measurement.selected);require(counter.read()[0]==measurement.selected,"counter changed after replay");
        if(baseline_centers.empty()){require(variant==0u,"baseline must run first");baseline_centers=c;baseline_errors=e;}
        else require(!std::memcmp(c.data(),baseline_centers.data(),size_t(cells)*4u)&&!std::memcmp(e.data(),baseline_errors.data(),size_t(cells)*4u),"C64 raw center or envelope changed");
        std::vector<unsigned char> seen(cells,0u);
        for(unsigned i=0u;i<measurement.selected;++i){require(selected[i]<cells&&!seen[selected[i]],"coarse candidate permutation");seen[selected[i]]=1u;}
        for(unsigned cell=0u;cell<cells;++cell){
            require(bool(seen[cell])==!bound::certified({c[cell],e[cell]}),"coarse candidate mask");
            require(std::abs(double(canonical[cell])-double(c[cell]))<=double(e[cell]),"coarse original interval undercoverage");
            require(bound::scalar::finite(out[cell])&&bound::scalar::bf16(out[cell])==reference[cell],"final BF16 differs from reference");
            require(bound::scalar::bits(out[cell])==bound::scalar::bits(seen[cell]?canonical[cell]:c[cell]),"raw replay or unselected output changed");
        }
        w.unchanged(weights);x.unchanged(inputs);pw.unchanged(expected_pw);px.unchanged(expected_px);wf.unchanged(expected_wf);xf.unchanged(expected_xf);
    }
    void execute(unsigned mode){
        Measurement warm[variants];double samples[variants][3]{},prefix[variants][3]{};
        const unsigned attempts=captured?4u:1u;
        for(unsigned attempt=0u;attempt<attempts;++attempt)for(unsigned position=0u;position<variants;++position){
            const unsigned variant=(position+attempt)%variants;const auto result=run(variant);
            if(!attempt)warm[variant]=result;
            else {require(result.selected==warm[variant].selected,"candidate work changed across samples");samples[variant][attempt-1u]=result.ms;prefix[variant][attempt-1u]=result.prefix_ms;}
        }
        dc.unchanged(canonical);
        for(unsigned variant=0u;variant<variants;++variant){
            require(warm[variant].selected==warm[0].selected,"candidate count changed across producers");
            auto sorted=std::array<double,3>{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(sorted.begin(),sorted.end());
            std::printf("{\"kernel\":\"shared_c64_projection\",\"variant\":\"%s\",\"variant_id\":%u,\"rows\":%u,\"tokens\":%u,\"width\":%u,\"mode\":%u,\"weight_skew_words\":%u,\"input_skew_words\":%u,\"captured\":%s,\"cells\":%u,\"candidates\":%u,\"replay_groups\":%llu,",
                names[variant],variant,rows,tokens,width,mode,w.skew,x.skew,captured?"true":"false",cells,warm[variant].selected,static_cast<unsigned long long>(warm[variant].selected)*(width/16u));
            std::printf("\"owner_ms\":%.7f,\"owner_samples_ms\":[%.7f,%.7f,%.7f],\"prefix_ms\":%.7f,\"prefix_samples_ms\":[%.7f,%.7f,%.7f],\"warm_samples\":%u,\"measured_samples\":%u,\"independent_cpu_dots\":%u,",
                captured?sorted[1]:warm[variant].ms,samples[variant][0],samples[variant][1],samples[variant][2],warm[variant].prefix_ms,prefix[variant][0],prefix[variant][1],prefix[variant][2],unsigned(captured),captured?3u:1u,cpu_dots);
            std::printf("\"bf16_mismatches\":0,\"full_replay_raw_mismatches\":0,\"interval_undercoverage\":0,\"raw_centers_and_intervals_unchanged\":true,\"complete_candidate_permutation_checked\":true,\"all_attempts_verified\":true,\"all_prepared_words_checked\":true,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"hardware_error_bound_proven\":false,\"real_model_prompt\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n");
            std::fflush(stdout);
        }
    }
};
void generated(unsigned rows,unsigned tokens,unsigned width,unsigned mode,unsigned weight_offset=0u,unsigned input_offset=0u){
    std::vector<uint16_t> weights(size_t(rows)*width),inputs(size_t(tokens)*width);
    for(size_t i=0u;i<weights.size();++i)weights[i]=uint16_t(((i*37u+i/19u)&0x807fu)|((119u+i%9u)<<7u));
    for(size_t i=0u;i<inputs.size();++i)inputs[i]=uint16_t(((i*53u+i/23u)&0x807fu)|((121u+i%7u)<<7u));
    if(mode==1u){for(size_t i=0u;i<weights.size();i+=7u)weights[i]=i%2u?0u:0x8000u;for(size_t i=0u;i<inputs.size();i+=11u)inputs[i]=i%2u?0u:0x8000u;}
    if(mode==2u){weights[width-1u]=1u;inputs[width-17u]=uint16_t((175u<<7u)|19u);}
    if(mode==3u){for(size_t i=0u;i<weights.size();++i)weights[i]=uint16_t(0x3f81u|(i%2u?0x8000u:0u));std::fill(inputs.begin(),inputs.end(),0x3f85u);}
    if(mode==4u){std::fill(weights.begin(),weights.end(),0u);}
    Test test(rows,tokens,width,std::move(weights),std::move(inputs),{},weight_offset,input_offset);test.execute(mode);
}
void captured(const char* mode,const char* input_file,const char* weight_file,const char* reference_file){
    const bool out=!std::strcmp(mode,"--out");const unsigned rows=out?2048u:8192u,width=out?4096u:2048u;
    auto inputs=read_words(input_file,size_t(7169u)*width);const auto weights=read_words(weight_file,size_t(rows)*width);
    auto reference=read_words(reference_file,size_t(out?8192u:7169u)*rows);
    auto extend=[](std::vector<uint16_t>& data,unsigned stride){data.resize(size_t(8192u)*stride);std::copy_n(data.begin(),size_t(1023u)*stride,data.begin()+size_t(7169u)*stride);};
    extend(inputs,width);if(!out)extend(reference,rows);
    Test test(rows,8192u,width,weights,std::move(inputs),std::move(reference));test.execute(0u);
}
} // namespace
int main(int argc,char** argv)try{
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0));require(!std::strncmp(properties.gcnArchName,"gfx1151",7u),"requires gfx1151");
    if(argc==5&&(!std::strcmp(argv[1],"--out")||!std::strcmp(argv[1],"--qkv"))){captured(argv[1],argv[2],argv[3],argv[4]);return 0;}
    require(argc==2&&!std::strcmp(argv[1],"--selftest"),"use --selftest or --out/--qkv INPUT WEIGHT REFERENCE");
    generated(1u,1u,16u,0u);generated(19u,17u,80u,0u);generated(33u,17u,272u,2u);generated(17u,19u,512u,1u);generated(33u,17u,2048u,2u);
    generated(19u,17u,4096u,3u);generated(17u,3u,8192u,4u);generated(67u,19u,4096u,0u);
    generated(19u,17u,80u,0u,1u,0u);generated(33u,17u,272u,2u,0u,3u);generated(17u,19u,512u,1u,1u,3u);
    generated(65u,67u,64u,0u);generated(129u,65u,80u,0u);generated(129u,67u,272u,2u);generated(63u,65u,512u,3u);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"shared_c64_error=%s\n",e.what());return 1;}
