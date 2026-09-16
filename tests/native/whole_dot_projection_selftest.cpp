#include "../../native/providers/moe_accumulator/sm121_prefix_replay_projection.h"
#include "../../native/providers/moe_accumulator/sm121_whole_dot_projection.h"
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
namespace whole=qrt_sm121_whole_dot_projection;
using Summary=whole::Summary;
namespace bound=qrt_sm121_coarse_projection_bound;
namespace original=qrt_q1_moe_hawkeye;
using Row=kernel::Row;
using Work=kernel::Work;
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
struct Measurement {double ms=0.0,prefix_ms=0.0;unsigned selected=0u;std::vector<unsigned> phases;size_t groups=0u;};
struct Test {
    unsigned rows,tokens,width,cells;bool captured;
    std::vector<uint16_t> weights,inputs,reference;
    std::vector<float> canonical,baseline_centers,baseline_errors,whole_centers,whole_errors;
    std::vector<Summary> audit_w,audit_x;
    Buffer<uint16_t> w,x;
    Buffer<Row> pw,px;
    Buffer<unsigned> wf,xf,ids,counter,finished;
    Buffer<float> centers,errors,output,dc,steps;
    Buffer<Summary> wn,xn;
    Buffer<bound::State> snapshots;
    Buffer<Work> work_a,work_b;
    unsigned cpu_dots=0u,cpu_prefixes=0u;
    Test(unsigned r,unsigned n,unsigned k,std::vector<uint16_t> hw,std::vector<uint16_t> hx,
        std::vector<uint16_t> ref={})
        : rows(r),tokens(n),width(k),cells(r*n),captured(!ref.empty()),weights(std::move(hw)),inputs(std::move(hx)),reference(std::move(ref)),
          w(weights.size()),x(inputs.size()),pw(size_t(r)*(k/16u)),px(size_t(n)*(k/16u)),wf(r),xf(n),ids(cells),counter(1u),finished(cells),
          centers(cells),errors(cells),output(cells),dc(cells),steps(cells),wn(size_t(r)*4u),xn(size_t(n)*4u),snapshots(size_t(cells)*3u),work_a(cells),work_b(cells) {
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
    void verify_metadata(unsigned variant){
        for(unsigned side=0u;side<2u;++side){
            const auto actual=side?xn.read():wn.read();const auto& input=side?inputs:weights;const unsigned n=side?tokens:rows;
            if(!variant){marker_tail(actual,0u);continue;}
            const unsigned groups=width/16u;
            for(unsigned row=0u;row<n;++row)for(unsigned part=0u;part<4u;++part){
                double squares=0.0,maximum=0.0;bool valid=true;
                for(unsigned k=(part*groups/4u)*16u;k<((part+1u)*groups/4u)*16u;++k){const uint16_t word=input[size_t(row)*width+k];
                    valid=valid&&bound::eligible(word);const double value=bound::scalar::value(uint32_t(word&0x7fffu)<<16u);squares+=value*value;maximum=std::max(maximum,value);}
                const auto value=actual[size_t(row)*4u+part];
                if(!valid){require(std::isinf(value.norm)&&std::isinf(value.maximum),"unsupported metadata not conservative");continue;}
                require(double(value.maximum)==maximum&&std::isfinite(value.norm)&&value.norm>=0.0f,"metadata maximum or domain");
                require(double(value.norm)*double(value.norm)>=squares,"norm undercoverage");
                require(squares?double(value.norm)<=std::sqrt(squares)*1.00003:value.norm==0.0f,"norm excessive inflation");
            }
        }
    }
    template<unsigned Parts>void produce(){
        hipLaunchKernelGGL((whole::produce<Parts>),dim3((rows+127u)/128u,(tokens+15u)/16u),dim3(256u),0u,nullptr,
            w.data(),x.data(),wf.data(),xf.data(),wn.data(),xn.data(),centers.data(),errors.data(),rows,tokens,width,snapshots.data(),steps.data());check(hipGetLastError());
    }
    void producer(unsigned variant,bool audit){
        hipLaunchKernelGGL(matrix::eligibility,dim3(rows),dim3(256u),0u,nullptr,w.data(),wf.data(),rows,width);check(hipGetLastError());
        hipLaunchKernelGGL(matrix::eligibility,dim3(tokens),dim3(256u),0u,nullptr,x.data(),xf.data(),tokens,width);check(hipGetLastError());
        if(variant){
            hipLaunchKernelGGL(whole::prepare,dim3((size_t(rows)*4u+7u)/8u),dim3(256u),0u,nullptr,w.data(),wn.data(),rows,width);check(hipGetLastError());
            hipLaunchKernelGGL(whole::prepare,dim3((size_t(tokens)*4u+7u)/8u),dim3(256u),0u,nullptr,x.data(),xn.data(),tokens,width);check(hipGetLastError());
        }
        if(audit){finish();audit_w=wn.read();audit_x=xn.read();verify_metadata(variant);}
        if(!variant){hipLaunchKernelGGL((matrix::produce<64u>),dim3((rows+127u)/128u,(tokens+15u)/16u),dim3(256u),0u,nullptr,w.data(),x.data(),wf.data(),xf.data(),centers.data(),errors.data(),rows,tokens,width);check(hipGetLastError());}
        else if(variant==1u)produce<1u>();
        else if(variant==3u)produce<2u>();
        else produce<4u>();
    }
    unsigned count(){unsigned value;check(hipMemcpy(&value,counter.data(),4u,hipMemcpyDeviceToHost));require(value<=cells,"candidate capacity");return value;}
    template<unsigned Parts,bool Audit>void launch(unsigned stage,unsigned n,Work* previous,Work* next){
        for(unsigned offset=0u;offset<n;offset+=window){const unsigned size=std::min(window,n-offset);
            hipLaunchKernelGGL((whole::phase<Parts,Audit>),dim3((size+63u)/64u),dim3(256u),0u,nullptr,pw.data(),px.data(),ids.data(),previous,
                centers.data(),steps.data(),snapshots.data(),output.data(),next,counter.data(),rows,tokens,width,stage,offset,size,Audit?finished.data():nullptr);
            check(hipGetLastError());}
    }
    Measurement run(unsigned variant,bool audit){
        ids.reset();snapshots.reset();steps.reset();wn.reset();xn.reset();output.reset();centers.reset();errors.reset();check(hipMemset(counter.data(),0,4u));
        if(audit)check(hipMemset(finished.data(),0,size_t(cells)*4u));
        finish();const auto begin=std::chrono::steady_clock::now();prepare();producer(variant,audit);
        hipLaunchKernelGGL(matrix::compact,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,centers.data(),errors.data(),output.data(),ids.data(),counter.data(),cells);
        check(hipGetLastError());finish();Measurement result;result.selected=count();
        result.prefix_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        const unsigned parts=variant==3u?2u:4u;unsigned current=result.selected;
        std::vector<unsigned> alive,old_finished;
        if(audit && variant>=3u){const auto selected=ids.read();alive.assign(selected.begin(),selected.begin()+current);old_finished.assign(cells,0u);}
        if(variant<3u){full(true,output.data(),current);finish();result.groups=size_t(current)*(width/16u);}
        else for(unsigned stage=0u;stage<parts;++stage){
            result.phases.push_back(current);result.groups+=size_t(current)*((stage+1u)*(width/16u)/parts-stage*(width/16u)/parts);
            auto& next=stage%2u?work_b:work_a;auto& previous=stage%2u?work_a:work_b;
            if(audit)next.reset();
            check(hipMemset(counter.data(),0,4u));
            if(parts==2u){if(audit)launch<2u,true>(stage,current,previous.data(),next.data());else launch<2u,false>(stage,current,previous.data(),next.data());}
            else{if(audit)launch<4u,true>(stage,current,previous.data(),next.data());else launch<4u,false>(stage,current,previous.data(),next.data());}
            finish();const unsigned remaining=count();require(remaining<=current,"continuation count grew");
            if(audit){
                const auto records=next.read();marker_tail(records,remaining);
                const auto stages=finished.read();std::vector<unsigned char> before(cells,0u),seen(cells,0u);
                for(unsigned cell:alive){require(cell<cells&&!before[cell],"input continuation membership");before[cell]=1u;}
                std::vector<unsigned> new_alive;new_alive.reserve(remaining);
                const unsigned step=captured?std::max(1u,remaining/128u):1u;
                for(unsigned i=0u;i<remaining;++i){const auto item=records[i];require(item.cell<cells&&before[item.cell]&&!seen[item.cell]&&!(item.state&0xfffe0000u),"compacted continuation identity or state");
                    seen[item.cell]=1u;new_alive.push_back(item.cell);
                    if(i%step==0u){const auto expected=cpu_prefix(weights.data()+size_t(item.cell%rows)*width,inputs.data()+size_t(item.cell/rows)*width,((stage+1u)*(width/16u)/parts)*16u);
                        require(item.significand==expected.significand && uint16_t(item.state)==uint16_t(expected.exponent) && bool(item.state&65536u)==expected.negative,"continuation differs from CPU canonical carry");++cpu_prefixes;}}
                for(unsigned cell=0u;cell<cells;++cell){
                    if(before[cell])require(stages[cell]==(seen[cell]?0u:stage+1u),"continuation/completion partition differs");
                    else require(stages[cell]==old_finished[cell],"unselected completion state changed");}
                alive=std::move(new_alive);old_finished=stages;
            }
            current=remaining;
        }
        if(variant>=3u)require(!current,"unfinished final continuation");
        result.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        verify_metadata(variant);if(audit){wn.unchanged(audit_w);xn.unchanged(audit_x);}
        verify(variant,result,audit);return result;
    }
    void verify(unsigned variant,const Measurement& measurement,bool audit){
        const auto out=output.read(),c=centers.read(),e=errors.read();const auto selected=ids.read();
        marker_tail(selected,measurement.selected);const auto counter_words=counter.read();(void)counter_words;
        if(!variant){
            if(baseline_centers.empty()){baseline_centers=c;baseline_errors=e;}
            else require(!std::memcmp(c.data(),baseline_centers.data(),size_t(cells)*4u)&&!std::memcmp(e.data(),baseline_errors.data(),size_t(cells)*4u),"original envelope changed");
        }else{
            if(whole_centers.empty()){whole_centers=c;whole_errors=e;}
            else require(!std::memcmp(c.data(),whole_centers.data(),size_t(cells)*4u)&&!std::memcmp(e.data(),whole_errors.data(),size_t(cells)*4u),"snapshots changed whole-dot center or envelope");
        }
        std::vector<unsigned char> seen(cells,0u);
        for(unsigned i=0u;i<measurement.selected;++i){require(selected[i]<cells&&!seen[selected[i]],"coarse candidate permutation");seen[selected[i]]=1u;}
        const auto stages=audit&&variant>=3u?finished.read():std::vector<unsigned>{};
        for(unsigned cell=0u;cell<cells;++cell){
            require(bool(seen[cell])==!bound::certified({c[cell],e[cell]}),"coarse candidate mask");
            require(std::abs(double(canonical[cell])-double(c[cell]))<=double(e[cell]),"coarse original interval undercoverage");
            require(bound::scalar::finite(out[cell])&&bound::scalar::bf16(out[cell])==reference[cell],"final BF16 differs from reference");
            if(!seen[cell])require(bound::scalar::bits(out[cell])==bound::scalar::bits(c[cell]),"unselected output changed");
            if(seen[cell]&&(variant<3u || (audit&&stages[cell]==(variant==3u?2u:4u))))
                require(bound::scalar::bits(out[cell])==bound::scalar::bits(canonical[cell]),"full replay changed raw endpoint");
            if(audit&&variant>=3u)require(seen[cell]?(stages[cell]>=1u&&stages[cell]<=(variant==3u?2u:4u)):stages[cell]==0u,"completion coverage");
        }
        if(variant>=2u){
            const unsigned parts=variant==3u?2u:4u;const auto snap=snapshots.read();marker_tail(snap,size_t(parts-1u)*cells);const auto step=steps.read();
            const unsigned checks=captured?256u:cells;
            for(unsigned sample=0u;sample<checks;++sample){const unsigned cell=captured?unsigned((uint64_t(sample)*2654435761ull+1013904223ull)%cells):sample;
                for(unsigned stage=0u;stage<parts-1u;++stage){const auto state=snap[size_t(stage)*cells+cell];const unsigned end=(stage+1u)*(width/16u)/parts;
                    const float exact=as_float(cpu_prefix(weights.data()+size_t(cell%rows)*width,inputs.data()+size_t(cell/rows)*width,end*16u));
                    const auto suffix=whole::bound::suffix(c[cell],state.center,exact,state.error,step[cell],width/16u-end);
                    require(std::abs(double(canonical[cell])-double(suffix.center))<=double(suffix.error),"native suffix interval undercoverage");
                    if(bound::certified(suffix))require(bound::scalar::bf16(suffix.center)==reference[cell],"native suffix false certificate");
                    ++cpu_prefixes;
                }
            }
        }else{marker_tail(snapshots.read(),0u);marker_tail(steps.read(),0u);}
    }

    void immutable(){
        w.unchanged(weights);x.unchanged(inputs);
        const auto wp=pw.read(),xp=px.read();const auto wflags=wf.read(),xflags=xf.read();
        for(unsigned side=0u;side<2u;++side){const auto& raw=side?inputs:weights;const auto& encoded=side?xp:wp;const auto& flags=side?xflags:wflags;const unsigned n=side?tokens:rows;
            for(size_t group=0u;group<encoded.size();++group){const auto expected=qrt_sm121_scaled_half_products::prepare(raw.data()+group*16u);require(!std::memcmp(&expected,&encoded[group],sizeof(Row)),"complete prepared encoding changed");}
            for(unsigned row=0u;row<n;++row){bool valid=true;for(unsigned k=0u;k<width;++k)valid=valid&&bound::eligible(raw[size_t(row)*width+k]);require(flags[row]==unsigned(valid),"eligibility flags changed");}}
    }
    void execute(unsigned mode){
        double samples[5][3]{},prefix_samples[5][3]{};Measurement first[5];
        const unsigned attempts=captured?4u:1u;
        for(unsigned attempt=0u;attempt<attempts;++attempt)for(unsigned position=0u;position<5u;++position){
            const unsigned variant=(attempt+position)%5u;const auto result=run(variant,false);
            if(!attempt)first[variant]=result;
            else{require(result.selected==first[variant].selected&&result.phases==first[variant].phases&&result.groups==first[variant].groups,"phase work changed between attempts");samples[variant][attempt-1u]=result.ms;prefix_samples[variant][attempt-1u]=result.prefix_ms;}
        }
        for(unsigned variant=0u;variant<5u;++variant){const auto audited=run(variant,true);
            require(audited.selected==first[variant].selected&&audited.phases==first[variant].phases&&audited.groups==first[variant].groups,"production/audit work parity");}
        immutable();
        for(unsigned variant=0u;variant<5u;++variant){
            std::array<double,3> sorted{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(sorted.begin(),sorted.end());
            std::array<double,3> prefix{prefix_samples[variant][0],prefix_samples[variant][1],prefix_samples[variant][2]};std::sort(prefix.begin(),prefix.end());
            const auto& result=first[variant];const size_t total=size_t(result.selected)*(width/16u);
            std::printf("{\"preparation_producer_selection_ms\":%.6f,\"preparation_producer_selection_samples_ms\":[%.6f,%.6f,%.6f],",captured?prefix[1]:result.prefix_ms,prefix_samples[variant][0],prefix_samples[variant][1],prefix_samples[variant][2]);
            std::printf("\"kind\":\"whole_dot_projection\",\"captured\":%s,\"rows\":%u,\"tokens\":%u,\"width\":%u,\"mode\":%u,\"variant\":%u,\"cells\":%u,\"candidates\":%u,\"original_groups\":%zu,\"executed_groups\":%zu,\"skipped_groups\":%zu,\"phase_input_counts\":[",captured?"true":"false",rows,tokens,width,mode,variant,cells,result.selected,total,result.groups,total-result.groups);
            for(unsigned i=0u;i<result.phases.size();++i)std::printf("%s%u",i?",":"",result.phases[i]);
            std::printf("],\"completed_total_ms\":%.6f,\"completed_samples_ms\":[%.6f,%.6f,%.6f],\"warmups\":%u,\"measured_attempts\":%u,\"cpu_full_dots\":%u,\"cpu_prefix_checks\":%u,\"bf16_mismatches\":0,\"full_replay_raw_mismatches\":0,\"interval_undercoverage\":0,\"whole_centers_and_bounds_unchanged_across_replay_variants\":true,\"all_metadata_bounds_checked\":true,\"metadata_immutable_in_audit\":true,\"all_attempts_verified\":true,\"complete_candidate_permutation_checked\":true,\"compacted_carry_membership_checked\":true,\"production_audit_work_parity\":true,\"all_prepared_words_checked\":true,\"redzones_pass\":true,\"unused_work_tail_pass\":true,\"immutable_inputs\":true,\"hardware_error_bound_proven\":false,\"real_model_prompt\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
                captured?sorted[1]:result.ms,samples[variant][0],samples[variant][1],samples[variant][2],unsigned(captured),captured?3u:1u,cpu_dots,cpu_prefixes);
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
}catch(const std::exception& e){std::fprintf(stderr,"whole_dot_error=%s\n",e.what());return 1;}
