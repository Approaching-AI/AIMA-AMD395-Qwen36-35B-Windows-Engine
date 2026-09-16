#include "../../native/providers/moe_accumulator/sm121_prefix_replay_projection.h"
#include "../../native/providers/moe_accumulator/sm121_macro_norm_projection.h"
#include "../../native/providers/moe_accumulator/sm121_cooperative_norm_metadata.h"
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
namespace bound=qrt_sm121_coarse_projection_bound;
namespace original=qrt_q1_moe_hawkeye;
using Row=kernel::Row;
namespace macro=qrt_sm121_macro_norm_projection;
namespace cooperative=qrt_sm121_cooperative_norm_metadata;
using Summary=macro::Summary;
constexpr unsigned guard=64u,marker=0xa5a5a5a5u,window=262144u;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void check(hipError_t s){if(s!=hipSuccess)throw std::runtime_error(hipGetErrorString(s));}
void finish(){
    hipEvent_t event;check(hipEventCreate(&event));check(hipEventRecord(event));
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;){const auto s=hipEventQuery(event);if(s==hipSuccess)break;if(s!=hipErrorNotReady)check(s);
        require(std::chrono::steady_clock::now()<deadline,"prefix replay completion deadline");std::this_thread::yield();}
    check(hipEventDestroy(event));
}
template<class T>struct Buffer {
    T* base=nullptr;size_t count;
    explicit Buffer(size_t n):count(n){check(hipMalloc(reinterpret_cast<void**>(&base),(n+2u*guard)*sizeof(T)));reset();}
    ~Buffer(){if(base && hipFree(base)!=hipSuccess)std::abort();}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    T* data(){return base+guard;}
    void reset(){check(hipMemset(base,0xa5,(count+2u*guard)*sizeof(T)));}
    void upload(const std::vector<T>& v){require(v.size()==count,"upload span");check(hipMemcpy(data(),v.data(),count*sizeof(T),hipMemcpyHostToDevice));}
    std::vector<T> read(){std::vector<T> all(count+2u*guard);check(hipMemcpy(all.data(),base,all.size()*sizeof(T),hipMemcpyDeviceToHost));
        const auto* bytes=reinterpret_cast<const unsigned char*>(all.data());
        for(size_t i=0u;i<guard*sizeof(T);++i)require(bytes[i]==0xa5u && bytes[(count+guard)*sizeof(T)+i]==0xa5u,"buffer guard changed");
        return std::vector<T>(all.begin()+guard,all.end()-guard);}
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
struct Measurement {double ms=0.0,prefix_ms=0.0;unsigned selected=0u;double audit_ms[5]{};};
struct Configuration {unsigned groups,metadata;};
// Metadata0 is the original per-thread F64 scan,1 cooperative F64,2 F32.
constexpr Configuration variants[]={{0u,0u},{8u,0u},{16u,0u},{32u,0u},{64u,0u},
    {8u,1u},{16u,1u},{32u,1u},{64u,1u},{8u,2u},{16u,2u},{32u,2u},{64u,2u}};
constexpr unsigned variant_count=unsigned(sizeof(variants)/sizeof(variants[0]));
struct Test {
    unsigned rows,tokens,width,cells;bool captured;
    std::vector<uint16_t> weights,inputs,reference;
    std::vector<float> canonical;
    Buffer<uint16_t> w,x;
    Buffer<Row> pw,px;
    Buffer<unsigned> wf,xf,ids,counter;
    Buffer<float> centers,errors,output,dc;
    Buffer<Summary> wn,xn;
    unsigned cpu_dots=0u;
    Test(unsigned r,unsigned n,unsigned k,std::vector<uint16_t> hw,std::vector<uint16_t> hx,
        std::vector<uint16_t> ref={})
        : rows(r),tokens(n),width(k),cells(r*n),captured(!ref.empty()),weights(std::move(hw)),inputs(std::move(hx)),reference(std::move(ref)),
          w(weights.size()),x(inputs.size()),pw(size_t(r)*(k/16u)),px(size_t(n)*(k/16u)),wf(r),xf(n),ids(cells),counter(1u),
          centers(cells),errors(cells),output(cells),dc(cells),wn(size_t(r)*((k+127u)/128u)),xn(size_t(n)*((k+127u)/128u)) {
        require(rows && tokens && width%16u==0u && width<=8192u,"test shape");w.upload(weights);x.upload(inputs);
        prepare();full(false,dc.data(),cells);finish();canonical=dc.read();
        if(!captured){reference.resize(cells);for(unsigned i=0u;i<cells;++i)reference[i]=bound::scalar::bf16(canonical[i]);}
        require(reference.size()==cells,"reference span");
        for(unsigned i=0u;i<cells;++i)require(bound::scalar::finite(canonical[i])&&bound::scalar::bf16(canonical[i])==reference[i],"original canonical differs from BF16 reference");
        const unsigned checks=captured?256u:cells;
        for(unsigned sample=0u;sample<checks;++sample){const unsigned cell=captured?unsigned((uint64_t(sample)*2654435761ull+1013904223ull)%cells):sample;
            const float expected=as_float(cpu_prefix(weights.data()+size_t(cell%rows)*width,inputs.data()+size_t(cell/rows)*width,width));
            require(bound::scalar::bits(expected)==bound::scalar::bits(canonical[cell]),"original GPU differs from independent CPU dot");++cpu_dots;}
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
    template<unsigned Groups>void encode(unsigned method){
        const unsigned chunks=(width+Groups*16u-1u)/(Groups*16u);
        for(unsigned side=0u;side<2u;++side){
            const unsigned n=side?tokens:rows;const auto* input=side?x.data():w.data();auto* output=side?xn.data():wn.data();
            if(method==0u)hipLaunchKernelGGL((macro::prepare<Groups>),dim3((size_t(n)*chunks+255u)/256u),dim3(256u),0u,nullptr,input,output,n,width);
            else if(method==1u)hipLaunchKernelGGL((cooperative::prepare<Groups,double>),dim3((size_t(n)*chunks+7u)/8u),dim3(256u),0u,nullptr,input,output,n,width);
            else if(method==2u)hipLaunchKernelGGL((cooperative::prepare<Groups,float>),dim3((size_t(n)*chunks+7u)/8u),dim3(256u),0u,nullptr,input,output,n,width);
            else throw std::runtime_error("metadata method");
            check(hipGetLastError());
        }
    }
    void metadata(unsigned variant,unsigned method){
        switch(variant){case 0u:require(!method,"control metadata method");break;case 8u:encode<8u>(method);break;case 16u:encode<16u>(method);break;case 32u:encode<32u>(method);break;case 64u:encode<64u>(method);break;default:throw std::runtime_error("metadata variant");}
    }
    template<unsigned Groups>void produce(){
        hipLaunchKernelGGL((macro::produce<Groups>),dim3((rows+127u)/128u,(tokens+15u)/16u),dim3(256u),0u,nullptr,
            w.data(),x.data(),wf.data(),xf.data(),wn.data(),xn.data(),centers.data(),errors.data(),rows,tokens,width);check(hipGetLastError());
    }
    void producer(unsigned variant){
        switch(variant){
            case 0u:hipLaunchKernelGGL((matrix::produce<64u>),dim3((rows+127u)/128u,(tokens+15u)/16u),dim3(256u),0u,nullptr,
                w.data(),x.data(),wf.data(),xf.data(),centers.data(),errors.data(),rows,tokens,width);check(hipGetLastError());break;
            case 8u:produce<8u>();break;case 16u:produce<16u>();break;case 32u:produce<32u>();break;case 64u:produce<64u>();break;
            default:throw std::runtime_error("producer variant");
        }
    }
    void verify_metadata(unsigned variant){
        for(unsigned side=0u;side<2u;++side){
            const auto actual=side?xn.read():wn.read();const auto& input=side?inputs:weights;const unsigned n=side?tokens:rows;
            if(!variant){marker_tail(actual,0u);continue;}
            const unsigned chunk=variant*16u,chunks=(width+chunk-1u)/chunk;marker_tail(actual,size_t(n)*chunks);
            for(unsigned row=0u;row<n;++row)for(unsigned block=0u;block<chunks;++block){
                double squares=0.0,maximum=0.0;bool valid=true;
                for(unsigned k=block*chunk;k<std::min(width,(block+1u)*chunk);++k){const uint16_t word=input[size_t(row)*width+k];
                    valid=valid&&bound::eligible(word);const double value=bound::scalar::value(uint32_t(word&0x7fffu)<<16u);squares+=value*value;maximum=std::max(maximum,value);}
                const auto value=actual[size_t(row)*chunks+block];
                if(!valid){require(std::isinf(value.norm)&&std::isinf(value.maximum),"unsupported metadata not conservative");continue;}
                require(double(value.maximum)==maximum&&std::isfinite(value.norm)&&value.norm>=0.0f,"metadata maximum or domain");
                require(double(value.norm)*double(value.norm)>=squares,"norm undercoverage");
                require(squares?double(value.norm)<=std::sqrt(squares)*1.00003:value.norm==0.0f,"norm excessive inflation");
            }
        }
    }
    Measurement run(unsigned variant,unsigned method,bool audit){
        ids.reset();wn.reset();xn.reset();output.reset();centers.reset();errors.reset();check(hipMemset(counter.data(),0,4u));
        finish();const auto begin=std::chrono::steady_clock::now();auto stage=begin;Measurement result;prepare();
        auto completed_stage=[&](unsigned index){finish();const auto now=std::chrono::steady_clock::now();result.audit_ms[index]=std::chrono::duration<double,std::milli>(now-stage).count();stage=now;};
        hipLaunchKernelGGL(matrix::eligibility,dim3(rows),dim3(256u),0u,nullptr,w.data(),wf.data(),rows,width);check(hipGetLastError());
        hipLaunchKernelGGL(matrix::eligibility,dim3(tokens),dim3(256u),0u,nullptr,x.data(),xf.data(),tokens,width);check(hipGetLastError());
        if(audit)completed_stage(0u);metadata(variant,method);if(audit)completed_stage(1u);
        std::vector<Summary> before_w,before_x;
        if(audit){before_w=wn.read();before_x=xn.read();verify_metadata(variant);stage=std::chrono::steady_clock::now();}
        producer(variant);
        if(audit)completed_stage(2u);
        hipLaunchKernelGGL(matrix::compact,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,centers.data(),errors.data(),output.data(),ids.data(),counter.data(),cells);
        check(hipGetLastError());finish();check(hipMemcpy(&result.selected,counter.data(),4u,hipMemcpyDeviceToHost));require(result.selected<=cells,"candidate capacity");
        if(audit){const auto now=std::chrono::steady_clock::now();result.audit_ms[3]=std::chrono::duration<double,std::milli>(now-stage).count();stage=now;}
        result.prefix_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        full(true,output.data(),result.selected);finish();result.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        if(audit)result.audit_ms[4]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-stage).count();
        verify_metadata(variant);if(audit){wn.unchanged(before_w);xn.unchanged(before_x);}verify(result);return result;
    }
    void verify(const Measurement& measurement){
        const auto out=output.read(),c=centers.read(),e=errors.read();const auto selected=ids.read();marker_tail(selected,measurement.selected);counter.read();
        std::vector<unsigned char> seen(cells,0u);for(unsigned i=0u;i<measurement.selected;++i){require(selected[i]<cells&&!seen[selected[i]],"candidate permutation");seen[selected[i]]=1u;}
        for(unsigned cell=0u;cell<cells;++cell){
            require(bool(seen[cell])==!bound::certified({c[cell],e[cell]}),"candidate mask");
            require(std::abs(double(canonical[cell])-double(c[cell]))<=double(e[cell]),"canonical interval undercoverage");
            require(bound::scalar::finite(out[cell])&&bound::scalar::bf16(out[cell])==reference[cell],"final BF16 differs from reference");
            require(bound::scalar::bits(out[cell])==bound::scalar::bits(seen[cell]?canonical[cell]:c[cell]),"replay raw value or unselected output changed");
        }
    }
    void immutable(){
        w.unchanged(weights);x.unchanged(inputs);const auto wp=pw.read(),xp=px.read();const auto wflags=wf.read(),xflags=xf.read();
        for(unsigned side=0u;side<2u;++side){const auto& raw=side?inputs:weights;const auto& encoded=side?xp:wp;const auto& flags=side?xflags:wflags;const unsigned n=side?tokens:rows;
            for(size_t group=0u;group<encoded.size();++group){const auto expected=qrt_sm121_scaled_half_products::prepare(raw.data()+group*16u);require(!std::memcmp(&expected,&encoded[group],sizeof(Row)),"prepared encoding changed");}
            for(unsigned row=0u;row<n;++row){bool valid=true;for(unsigned k=0u;k<width;++k)valid=valid&&bound::eligible(raw[size_t(row)*width+k]);require(flags[row]==unsigned(valid),"eligibility flags changed");}}
    }
    void execute(unsigned mode){
        double samples[variant_count][3]{},prefix_samples[variant_count][3]{};Measurement first[variant_count],audited[variant_count];const unsigned attempts=captured?4u:1u;
        for(unsigned attempt=0u;attempt<attempts;++attempt)for(unsigned position=0u;position<variant_count;++position){
            const unsigned index=(attempt*5u+position)%variant_count;const auto config=variants[index];const auto result=run(config.groups,config.metadata,false);
            if(!attempt)first[index]=result;else{require(result.selected==first[index].selected,"selection changed between attempts");samples[index][attempt-1u]=result.ms;prefix_samples[index][attempt-1u]=result.prefix_ms;}
        }
        for(unsigned index=0u;index<variant_count;++index){const auto config=variants[index];audited[index]=run(config.groups,config.metadata,true);require(audited[index].selected==first[index].selected,"audit selection parity");}
        immutable();
        for(unsigned index=0u;index<variant_count;++index){
            const unsigned variant=variants[index].groups;std::array<double,3> total{samples[index][0],samples[index][1],samples[index][2]},prefix{prefix_samples[index][0],prefix_samples[index][1],prefix_samples[index][2]};std::sort(total.begin(),total.end());std::sort(prefix.begin(),prefix.end());
            const size_t entries=variant?size_t(rows+tokens)*((width+variant*16u-1u)/(variant*16u)):0u;
            std::printf("{\"metadata_method\":%u,\"audit_phase_ms\":[%.6f,%.6f,%.6f,%.6f,%.6f],",variants[index].metadata,audited[index].audit_ms[0],audited[index].audit_ms[1],audited[index].audit_ms[2],audited[index].audit_ms[3],audited[index].audit_ms[4]);
            std::printf("\"kind\":\"cooperative_norm_projection\",\"captured\":%s,\"rows\":%u,\"tokens\":%u,\"width\":%u,\"mode\":%u,\"variant\":%u,\"cells\":%u,\"candidates\":%u,\"metadata_entries\":%zu,\"metadata_bytes\":%zu,\"selected_groups\":%zu,\"completed_total_ms\":%.6f,\"preparation_producer_selection_ms\":%.6f,\"completed_samples_ms\":[%.6f,%.6f,%.6f],\"preparation_producer_selection_samples_ms\":[%.6f,%.6f,%.6f],\"warmups\":%u,\"measured_attempts\":%u,\"cpu_full_dots\":%u,\"bf16_mismatches\":0,\"selected_raw_mismatches\":0,\"canonical_interval_undercoverage\":0,\"all_attempts_verified\":true,\"complete_candidate_permutation_checked\":true,\"all_metadata_bounds_checked\":true,\"metadata_immutable_in_audit\":true,\"production_audit_selection_parity\":true,\"all_prepared_words_checked\":true,\"redzones_pass\":true,\"unused_workspace_tail_pass\":true,\"immutable_inputs\":true,\"hardware_error_bound_proven\":false,\"real_model_prompt\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",captured?"true":"false",rows,tokens,width,mode,variant,cells,first[index].selected,entries,entries*sizeof(Summary),size_t(first[index].selected)*(width/16u),captured?total[1]:first[index].ms,captured?prefix[1]:first[index].prefix_ms,samples[index][0],samples[index][1],samples[index][2],prefix_samples[index][0],prefix_samples[index][1],prefix_samples[index][2],unsigned(captured),captured?3u:1u,cpu_dots);
            std::fflush(stdout);
        }
    }
};
void generated(unsigned rows,unsigned tokens,unsigned width,unsigned mode){
    std::vector<uint16_t> weights(size_t(rows)*width),inputs(size_t(tokens)*width);
    for(size_t i=0u;i<weights.size();++i)weights[i]=uint16_t(((i*37u+i/19u)&0x807fu)|((119u+i%9u)<<7u));
    for(size_t i=0u;i<inputs.size();++i)inputs[i]=uint16_t(((i*53u+i/23u)&0x807fu)|((121u+i%7u)<<7u));
    if(mode==1u){for(size_t i=0u;i<weights.size();i+=7u)weights[i]=i%2u?0u:0x8000u;for(size_t i=0u;i<inputs.size();i+=11u)inputs[i]=i%2u?0u:0x8000u;}
    if(mode==2u){weights[width-1u]=1u;inputs[width-17u]=uint16_t((175u<<7u)|19u);}
    if(mode==3u){for(size_t i=0u;i<weights.size();++i)weights[i]=uint16_t(0x3f81u|(i%2u?0x8000u:0u));std::fill(inputs.begin(),inputs.end(),0x3f85u);}
    if(mode==4u){std::fill(weights.begin(),weights.end(),0u);}
    Test test(rows,tokens,width,std::move(weights),std::move(inputs));test.execute(mode);
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
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"macro_norm_error=%s\n",e.what());return 1;}
