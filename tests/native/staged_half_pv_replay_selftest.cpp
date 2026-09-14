#include "../../native/providers/ck_fmha/prepared_decoded_qk.h"
#include "../../native/providers/ck_fmha/staged_half_pv_replay.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace qrt_blackwell_attention;
constexpr unsigned guard = 64u;
constexpr unsigned variants[] = {0u, 1u};
constexpr unsigned variant_count = sizeof(variants) / sizeof(variants[0]);
void check(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
struct Device {
    void* pointer = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&pointer, bytes)); }
    ~Device() { if (pointer) (void)hipFree(pointer); }
    template<class T> T* as() { return static_cast<T*>(pointer); }
};
template<class T> void upload(Device& d, const std::vector<T>& values) {
    check(hipMemcpy(d.pointer, values.data(), values.size()*sizeof(T), hipMemcpyHostToDevice));
}
template<class T> std::vector<T> download(Device& d, size_t count) {
    std::vector<T> values(count);
    check(hipMemcpy(values.data(), d.pointer, count*sizeof(T), hipMemcpyDeviceToHost));
    return values;
}
template<class T> void unchanged(Device& d, const std::vector<T>& expected) {
    const auto actual = download<T>(d, expected.size());
    if (std::memcmp(actual.data(), expected.data(), expected.size()*sizeof(T)))
        throw std::runtime_error("QK input, encoding or redzone changed");
}
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("QK completion deadline");
        // Keep the explicit deadline without quantizing each short GPU slab
        // through a fixed host sleep. The benchmark reports completed host time.
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, sizeof(u)); return u; }
double elapsed(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-begin).count();
}
struct Prepared {
    unsigned tokens;
    std::vector<uint32_t> qpacked, kpacked, qflags, kflags;
    Device qp, kp, qf, kf;
    double ms = 0.0;
    Prepared(const uint16_t* q, const uint16_t* k, uint16_t* transposed,
        const uint16_t* hq, const uint16_t* hk, unsigned n)
        : tokens(n), qpacked(size_t(n)*kQueryHeads*kHeadDim+2u*guard,0xa5a5a5a5u),
          kpacked(size_t(n)*kKvHeads*kHeadDim+2u*guard,0xa5a5a5a5u),
          qflags(n*kQueryHeads+2u*guard,0xa5a5a5a5u), kflags(n*kKvHeads+2u*guard,0xa5a5a5a5u),
          qp(qpacked.size()*4u), kp(kpacked.size()*4u), qf(qflags.size()*4u), kf(kflags.size()*4u) {
        for(Device* d:{&qp,&kp,&qf,&kf}) {
            const size_t words=d==&qp?qpacked.size():d==&kp?kpacked.size():d==&qf?qflags.size():kflags.size();
            check(hipMemset(d->pointer,0xa5,words*4u));
        }
        const auto begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::prepare<false>),dim3(n*kQueryHeads),dim3(kHeadDim),0u,nullptr,
            q,qp.as<uint32_t>()+guard,qf.as<unsigned>()+guard,nullptr,n);
        check(hipGetLastError());
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::prepare<true>),dim3(n*kKvHeads),dim3(kHeadDim),0u,nullptr,
            k,kp.as<uint32_t>()+guard,kf.as<unsigned>()+guard,transposed,n);
        check(hipGetLastError());finish();ms=elapsed(begin);
        for(unsigned key=0u;key<2u;++key) {
            const unsigned heads=key?kKvHeads:kQueryHeads;
            const auto* source=key?hk:hq;auto& packed=key?kpacked:qpacked;auto& flags=key?kflags:qflags;
            for(unsigned row=0u;row<n*heads;++row) {
                bool eligible=true;
                for(unsigned c=0u;c<kHeadDim;++c) {
                    const uint16_t x=source[size_t(row)*kHeadDim+c];
                    const unsigned e=(x>>7u)&255u;
                    eligible &= !(x&0x7fffu) || (e>=64u && e<=190u);
                    const int exponent=(x&0x7fffu)?int(e)-127:-512;
                    const size_t index=key?(size_t(row%heads)*kHeadDim+c)*n+row/heads:size_t(row)*kHeadDim+c;
                    packed[guard+index]=(uint32_t(x)<<16u)|uint16_t(exponent);
                }
                flags[guard+row]=unsigned(eligible);
            }
        }
        verify();
    }
    void verify() { unchanged(qp,qpacked);unchanged(kp,kpacked);unchanged(qf,qflags);unchanged(kf,kflags); }
};

using PackedRow = qrt_staged_half_pv::Row;
struct Plan {
    unsigned tokens,capacity;
    size_t pp_words,vp_words;
    Device pp,vp;
    std::vector<PackedRow> expected_value;
    double value_ms=0.0;
    Plan(unsigned n,unsigned count,const uint16_t* tv,const uint16_t* host_tv)
        :tokens(n),capacity(count),pp_words(size_t(count)*kQueryHeads*((n+15u)/16u)),
         vp_words(size_t(kKvHeads)*kHeadDim*((n+15u)/16u)),
         pp((pp_words+2u*guard)*sizeof(PackedRow)),vp((vp_words+2u*guard)*sizeof(PackedRow)) {
        check(hipMemset(vp.pointer,0xa5,(vp_words+2u*guard)*sizeof(PackedRow)));finish();
        const auto begin=std::chrono::steady_clock::now();
        check(hipError_t(qrt_staged_half_pv::prepare<false>(tv,n,0u,0u,vp.as<PackedRow>()+guard,vp_words,nullptr)));
        finish();value_ms=elapsed(begin);
        expected_value=cpu(host_tv,n,0u,0u,false,vp_words);unchanged(vp,expected_value);
    }
    static std::vector<PackedRow> cpu(const uint16_t* source,unsigned stride,unsigned start,
        unsigned count,bool probability,size_t allocation) {
        PackedRow sentinel;std::memset(&sentinel,0xa5,sizeof(sentinel));
        std::vector<PackedRow> expected(allocation+2u*guard,sentinel);
        const unsigned rows=probability?count*kQueryHeads:kKvHeads*kHeadDim,groups=(stride+15u)/16u;
        for(unsigned row=0u;row<rows;++row)for(unsigned group=0u;group<groups;++group) {
            const unsigned extent=probability?start+row/kQueryHeads+1u:stride;
            uint16_t values[16];unsigned maximum=0u,minimum=255u,nonzero=0u;bool valid=true;
            for(unsigned i=0u;i<16u;++i) {
                const unsigned key=group*16u+i;const uint16_t x=key<extent?source[size_t(row)*stride+key]:0u;
                values[i]=x;
                if(x&0x7fffu){const unsigned e=(x>>7u)&255u;maximum=std::max(maximum,e);minimum=std::min(minimum,e);valid&=e&&e<255u;nonzero|=1u<<i;}
            }
            const auto packed=qrt_sm121_scaled_half_products::prepare(values);
            const bool supported=valid&&(!nonzero||maximum-minimum<=29u);
            const int expected_unit=supported?(nonzero?int(maximum)-142:-15):-32768;
            if(packed.control!=((nonzero<<16u)|uint16_t(expected_unit)))throw std::runtime_error("PV independent encoded control");
            for(unsigned i=0u;i<16u;++i){
                const uint16_t encoded=uint16_t(packed.pairs[i/2u]>>(i%2u*16u));
                uint16_t restored=encoded;
                if(supported&&(encoded&0x7fffu))restored=uint16_t((encoded&0x8000u)|(unsigned(int((encoded>>10u)&31u)+112+expected_unit)<<7u)|((encoded&1023u)>>3u));
                if(restored!=values[i])throw std::runtime_error("PV original operand roundtrip");
            }
            expected[guard+size_t(row)*groups+group]=packed;
        }
        return expected;
    }
    void reset(){check(hipMemset(pp.pointer,0xa5,(pp_words+2u*guard)*sizeof(PackedRow)));}
    void prepare(const uint16_t* probability,unsigned start,unsigned count,unsigned stride) {
        const size_t words=size_t(count)*kQueryHeads*((stride+15u)/16u);
        if(words>pp_words||count>capacity||stride>tokens)throw std::runtime_error("PV plan capacity");
        check(hipError_t(qrt_staged_half_pv::prepare<true>(probability,stride,start,count,pp.as<PackedRow>()+guard,pp_words,nullptr)));
    }
    void verify(const uint16_t* probability,unsigned start,unsigned count,unsigned stride) {
        unchanged(pp,cpu(probability,stride,start,count,true,pp_words));unchanged(vp,expected_value);
    }
};
template<bool Audit=false>void replay(unsigned variant,const uint16_t* value,const uint16_t* probability,
    const float* scales,float* out,unsigned start,unsigned queries,unsigned output_start,unsigned stride,
    const unsigned char* rcp,float* acc,float* den,const unsigned* indices,const unsigned* count,
    const uint16_t* tv,unsigned value_stride,Plan& plan,unsigned long long* stats=nullptr) {
    const unsigned blocks=std::min(1024u,(queries*kQueryHeads*kHeadDim+63u)/64u);
    if(!variant) {
        hipLaunchKernelGGL((blackwell_compacted_pv_replay_kernel<true>),dim3(blocks),dim3(256u),0u,nullptr,
            value,probability,scales,out,start,output_start,stride,rcp,acc,den,indices,count,tv,value_stride,0u);
    }else {
        if(variant!=1u)throw std::runtime_error("PV variant");
        check(hipError_t(qrt_staged_half_pv::launch<Audit>(plan.pp.as<PackedRow>()+guard,plan.pp_words,
            plan.vp.as<PackedRow>()+guard,plan.vp_words,scales,out,start,queries,output_start,stride,value_stride,
            rcp,acc,den,indices,count,nullptr,stats)));
    }
    check(hipGetLastError());
}
void collect(const float* output,const float* errors,unsigned offset,unsigned cells,Device& indices) {
    auto* count=indices.as<unsigned>()+guard+cells;check(hipMemset(count,0,4u));
    hipLaunchKernelGGL(blackwell_collect_pv_replay_kernel,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,
        output,errors,offset,cells,indices.as<unsigned>()+guard,count);check(hipGetLastError());finish();
}
void equal(Device& d,const std::vector<uint32_t>& expected,const char* kind) {
    const auto actual=download<uint32_t>(d,expected.size());
    for(size_t i=0u;i<actual.size();++i)if(actual[i]!=expected[i]) {
        std::fprintf(stderr,"%s index=%zu expected=%08x actual=%08x\n",kind,i,expected[i],actual[i]);
        throw std::runtime_error("PV raw mismatch");
    }
}
void generated(unsigned start,unsigned queries,unsigned input_mode,unsigned selection_mode) {
    constexpr unsigned output_start=3u;
    const unsigned tokens=start+queries,rows=queries*kQueryHeads,cells=rows*kHeadDim,tiles=(tokens+31u)/32u;
    std::vector<uint16_t> v(size_t(tokens)*kKvHeads*kHeadDim+2u*guard,0x5a5au);
    std::vector<uint16_t> p(size_t(rows)*tokens+2u*guard,0x5a5au),tv(v.size(),0x5a5au);
    std::vector<float> scales(size_t(rows)*(tiles+1u)+2u*guard,12345.0f),errors(cells+2u*guard,12345.0f);
    for(size_t i=guard;i+guard<v.size();++i)v[i]=qrt_sm121_pv_bound::bf16(float(int((i*173u+i/17u)%63u)-31)/64.0f);
    for(unsigned row=0u;row<rows;++row)for(unsigned key=0u;key<tokens;++key) {
        uint16_t word=qrt_sm121_pv_bound::bf16(float((key*47u+row*13u)%31u)/32.0f);
        if(input_mode==1u&&key%97u==0u)word=0x0001u;
        if(input_mode==2u&&(key/16u)%3u)word=(key&1u)?0x8000u:0u;
        if(input_mode==3u&&key>=16u&&(key/16u)%5u)word=uint16_t((70u+key%3u)<<7u|127u);
        if(input_mode==4u)word=0u;
        p[guard+size_t(row)*tokens+key]=word;
    }
    if(input_mode==1u)for(unsigned feature=0u;feature<512u;++feature)
        v[guard+feature]=feature%3u?0x0001u:0x5f80u;
    if(input_mode==4u)for(size_t i=guard;i+guard<v.size();++i)v[i]=uint16_t(i%3u?0x7fc1u:0x8000u);
    for(unsigned row=0u;row<rows;++row) {
        for(unsigned tile=0u;tile<tiles;++tile)
            scales[guard+size_t(row)*(tiles+1u)+tile]=tile%11u?1.0f:tile%3u?0.875f:0.0f;
        scales[guard+size_t(row)*(tiles+1u)+tiles]=1.0f+float(row%31u)/16.0f;
    }
    unsigned selected=0u;
    for(unsigned cell=0u;cell<cells;++cell) {
        const bool active=selection_mode==1u||(selection_mode==2u&&(cell%251u==3u||cell%257u==0u));
        errors[guard+cell]=active?qrt_sm121_pv_bound::infinity():0.0f;selected+=active;
    }
    for(unsigned key=0u;key<tokens;++key)for(unsigned c=0u;c<512u;++c)tv[guard+size_t(c)*tokens+key]=v[guard+size_t(key)*512u+c];
    std::vector<float> initial(size_t(output_start+queries)*4096u+2u*guard,12345.0f);
    std::fill(initial.begin()+guard+output_start*4096u,initial.begin()+guard+(output_start+queries)*4096u,0.75f);
    std::vector<uint32_t> untouched(initial.size(),0xa5a5a5a5u),den_initial(size_t(output_start+queries)*16u+2u*guard,0xa5a5a5a5u);
    std::vector<unsigned> index_initial(cells+1u+2u*guard,0xa5a5a5a5u);
    Device dv(v.size()*2u),dp(p.size()*2u),dtv(tv.size()*2u),ds(scales.size()*4u),de(errors.size()*4u);
    Device out(initial.size()*4u),acc(untouched.size()*4u),den(den_initial.size()*4u),indices(index_initial.size()*4u);
    upload(dv,v);upload(dp,p);upload(dtv,tv);upload(ds,scales);upload(de,errors);
    upload(out,initial);upload(acc,untouched);upload(den,den_initial);
    hipLaunchKernelGGL(blackwell_probability_value_kernel,dim3(16u,queries),dim3(256u),0u,nullptr,
        dv.as<uint16_t>()+guard,dp.as<uint16_t>()+guard,ds.as<float>()+guard,out.as<float>()+guard,
        start,output_start,tokens,nullptr,acc.as<float>()+guard,den.as<float>()+guard,de.as<float>()+guard);
    check(hipGetLastError());finish();
    const auto expected=download<uint32_t>(out,initial.size()),expected_acc=download<uint32_t>(acc,untouched.size()),expected_den=download<uint32_t>(den,den_initial.size());
    Plan plan(tokens,queries,dtv.as<uint16_t>()+guard,tv.data()+guard);
    plan.reset();plan.prepare(dp.as<uint16_t>()+guard,start,queries,tokens);finish();plan.verify(p.data()+guard,start,queries,tokens);
    for(unsigned variant:variants) {
        upload(out,initial);upload(acc,untouched);upload(den,den_initial);upload(indices,index_initial);
        collect(out.as<float>()+guard,de.as<float>()+guard,output_start,cells,indices);
        const auto captured_indices=download<unsigned>(indices,index_initial.size());
        if(captured_indices[guard+cells]!=selected)throw std::runtime_error("PV candidate count");
        std::vector<bool> seen(cells,false);
        for(unsigned i=0u;i<selected;++i) {
            const unsigned cell=captured_indices[guard+i];
            if(cell>=cells||seen[cell]||errors[guard+cell]==0.0f)throw std::runtime_error("PV candidate identity");seen[cell]=true;
        }
        for(size_t i=0u;i<captured_indices.size();++i)
            if((i<guard||(i>=guard+selected&&i<guard+cells)||i>guard+cells)&&captured_indices[i]!=0xa5a5a5a5u)throw std::runtime_error("PV candidate guards");
        replay(variant,dv.as<uint16_t>()+guard,dp.as<uint16_t>()+guard,ds.as<float>()+guard,out.as<float>()+guard,
            start,queries,output_start,tokens,nullptr,acc.as<float>()+guard,den.as<float>()+guard,
            indices.as<unsigned>()+guard,indices.as<unsigned>()+guard+cells,dtv.as<uint16_t>()+guard,tokens,plan);
        finish();equal(out,expected,"output");equal(acc,expected_acc,"accumulator");equal(den,expected_den,"denominator");
        unchanged(indices,captured_indices);plan.verify(p.data()+guard,start,queries,tokens);
        std::printf("{\"kind\":\"staged_half_pv_safety\",\"query_start\":%u,\"query_count\":%u,\"input_mode\":%u,\"selection_mode\":%u,\"variant\":%u,\"cells\":%u,\"selected_cells\":%u,\"raw_bit_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"unique_candidates\":true,\"complete_cpu_metadata_check\":true,\"inference_acceptance\":false}\n",
            start,queries,input_mode,selection_mode,variant,cells,selected);std::fflush(stdout);
    }
    unchanged(dv,v);unchanged(dp,p);unchanged(dtv,tv);unchanged(ds,scales);unchanged(de,errors);
}

template<class T>std::vector<T> read_values(const char* path,size_t count) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file||file.tellg()!=std::streamoff(count*sizeof(T)))throw std::runtime_error("PV capture size");
    std::vector<T> result(count);file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(result.data()),std::streamsize(count*sizeof(T))))throw std::runtime_error("PV capture read");
    return result;
}
__global__ void compare(const uint32_t* expected,const uint32_t* actual,size_t count,unsigned* bad) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count&&expected[i]!=actual[i])atomicAdd(bad,1u);
}
void compare_device(Device& a,Device& b,size_t words,Device& bad) {
    hipLaunchKernelGGL(compare,dim3((words+255u)/256u),dim3(256u),0u,nullptr,
        a.as<uint32_t>(),b.as<uint32_t>(),words,bad.as<unsigned>());check(hipGetLastError());
}
float cpu_pv(const uint16_t* p,const uint16_t* v,const float* scales,unsigned tokens,unsigned column) {
    float accumulator=0.0f;
    for(unsigned tile=0u;tile<(tokens+31u)/32u;++tile) {
        volatile float scaled=accumulator*scales[tile];
        auto partial=qrt_q1_moe_hawkeye::value_from_float(scaled,-133);
        for(unsigned group=0u;group<2u;++group) {
            qrt_q1_moe_hawkeye::Value values[17];values[0]=partial;
            for(unsigned i=0u;i<16u;++i) {
                const unsigned key=tile*32u+group*16u+i;
                values[i+1u]=qrt_q1_moe_hawkeye::multiply_bf16(key<tokens?p[key]:0u,key<tokens?v[size_t(key)*512u+column]:0u,-133);
            }
            partial=qrt_q1_moe_hawkeye::group_sum<26,-133>(values,17u);
            partial=qrt_q1_moe_hawkeye::value_from_float(qrt_q1_moe_hawkeye::value_to_float(
                qrt_sm121_group16::finish_accumulator(partial)),-133);
        }
        accumulator=qrt_q1_moe_hawkeye::value_to_float(partial);
    }
    return accumulator;
}
void captured(const char* qfile,const char* kfile,const char* vfile,const char* reference_file,
    const char* exp_file,const char* rcp_file) {
    constexpr unsigned tokens=7169u,batch=128u,attempts=4u;
    auto q=read_values<uint16_t>(qfile,size_t(tokens)*4096u),k=read_values<uint16_t>(kfile,size_t(tokens)*512u),v=read_values<uint16_t>(vfile,size_t(tokens)*512u);
    const auto golden=read_values<uint16_t>(reference_file,size_t(tokens)*4096u);
    const auto exp=read_values<unsigned char>(exp_file,exp2_backend::table_bytes);
    const auto rcp=read_values<unsigned char>(rcp_file,qrt_sm121_attention_rcp::table_bytes);
    if(!exp2_backend::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("PV table layout");
    for(auto* a:{&q,&k,&v}){a->insert(a->begin(),guard,0x5a5au);a->insert(a->end(),guard,0x5a5au);}
    std::vector<uint16_t> tv(v.size(),0x5a5au);
    for(unsigned key=0u;key<tokens;++key)for(unsigned c=0u;c<512u;++c)tv[guard+size_t(c)*tokens+key]=v[guard+size_t(key)*512u+c];
    const size_t score_capacity=size_t(batch)*16u*tokens,output_capacity=size_t(batch)*4096u,
        output_words=output_capacity+2u*guard,den_words=batch*16u+2u*guard,scale_capacity=size_t(batch)*16u*((tokens+31u)/32u+1u);
    Device dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),dkt(k.size()*2u),dvt(v.size()*2u),dex(exp.size()),drcp(rcp.size());
    upload(dq,q);upload(dk,k);upload(dv,v);upload(dex,exp);upload(drcp,rcp);
    check(hipMemset(dkt.pointer,0x5a,k.size()*2u));check(hipMemset(dvt.pointer,0x5a,v.size()*2u));finish();
    auto begin=std::chrono::steady_clock::now();
    check(hipError_t(transpose_keys(dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,size_t(tokens)*512u,tokens,nullptr)));
    finish();const double transpose_ms=elapsed(begin);unchanged(dvt,tv);
    Prepared prepared(dq.as<uint16_t>()+guard,dk.as<uint16_t>()+guard,dkt.as<uint16_t>()+guard,q.data()+guard,k.data()+guard,tokens);
    const auto kt_before=download<uint16_t>(dkt,k.size());
    Plan plan(tokens,batch,dvt.as<uint16_t>()+guard,tv.data()+guard);
    Device scores((score_capacity+2u*guard)*4u),prob((score_capacity+2u*guard)*2u),scales((scale_capacity+2u*guard)*4u),errors(output_words*4u);
    Device approximate(output_words*4u),control(output_words*4u),candidate(output_words*4u),control_acc(output_words*4u),candidate_acc(output_words*4u),control_den(den_words*4u),candidate_den(den_words*4u);
    Device indices((output_capacity+1u+2u*guard)*4u),bad(4u),audit((5u+2u*guard)*8u);
    double samples[4][3]{},maximum[4]{},common_ms=0.0;
    unsigned long long stats[4][5]{},selected_total=0u,expected_groups=0u;
    size_t endpoints=0u,raw_comparisons[4]{};unsigned cpu_dots=0u;
    for(unsigned start=0u;start<tokens;start+=batch) {
        const unsigned queries=std::min(batch,tokens-start),stride=start+queries,rows=queries*16u,cells=rows*256u,tiles=(stride+31u)/32u;
        const size_t score_cells=size_t(rows)*stride;
        for(Device* d:{&scores,&scales,&errors,&approximate})check(hipMemset(d->pointer,0xa5,d==&scores?(score_capacity+2u*guard)*4u:d==&scales?(scale_capacity+2u*guard)*4u:output_words*4u));
        check(hipMemset(prob.pointer,0x5a,(score_capacity+2u*guard)*2u));check(hipMemset(indices.pointer,0xa5,(output_capacity+1u+2u*guard)*4u));finish();
        begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::scores<128u,true,16u,16u>),dim3((stride+15u)/16u,16u,(queries+15u)/16u),dim3(256u),0u,nullptr,
            dq.as<uint16_t>()+guard,dkt.as<uint16_t>()+guard,prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,prepared.qf.as<unsigned>()+guard,prepared.kf.as<unsigned>()+guard,scores.as<float>()+guard,start,queries,stride,tokens);
        check(hipGetLastError());
        hipLaunchKernelGGL(blackwell_online_probability_kernel,dim3(16u,queries),dim3(32u),0u,nullptr,
            scores.as<float>()+guard,prob.as<uint16_t>()+guard,scales.as<float>()+guard,start,stride,dex.as<unsigned char>(),true);check(hipGetLastError());
        hipLaunchKernelGGL((blackwell_mantissa_value_kernel<true,false,true,true,true>),dim3(kHeadDim/kIntegerMatrixColumns,16u,(queries+15u)/16u),dim3(256u),0u,nullptr,
            dv.as<uint16_t>()+guard,prob.as<uint16_t>()+guard,scales.as<float>()+guard,approximate.as<float>()+guard,start,queries,0u,stride,drcp.as<unsigned char>(),nullptr,nullptr,nullptr,nullptr,errors.as<float>()+guard);check(hipGetLastError());
        collect(approximate.as<float>()+guard,errors.as<float>()+guard,0u,cells,indices);common_ms+=elapsed(begin);
        const auto index_before=download<unsigned>(indices,output_capacity+1u+2u*guard);
        const unsigned selected=index_before[guard+cells];if(selected>cells)throw std::runtime_error("captured PV count");selected_total+=selected;
        std::vector<bool> seen(cells,false);
        for(unsigned i=0u;i<selected;++i) {
            const unsigned cell=index_before[guard+i];if(cell>=cells||seen[cell])throw std::runtime_error("captured PV identity");seen[cell]=true;
            expected_groups+=2u*((start+cell/4096u+32u)/32u);
        }
        for(size_t i=0u;i<index_before.size();++i)
            if((i<guard||(i>=guard+selected&&i<guard+cells)||i>guard+cells)&&index_before[i]!=0xa5a5a5a5u)throw std::runtime_error("captured PV index guard");
        const auto probability=download<uint16_t>(prob,score_capacity+2u*guard);
        const auto scale_words=download<float>(scales,scale_capacity+2u*guard);
        // Establish the original replay result, with independent resets for
        // each of its warmup and three completed timing samples.
        std::vector<uint32_t> first_output,first_acc,first_den;
        for(unsigned attempt=0u;attempt<attempts;++attempt) {
            check(hipMemcpy(control.pointer,approximate.pointer,output_words*4u,hipMemcpyDeviceToDevice));
            check(hipMemset(control_acc.pointer,0xa5,output_words*4u));check(hipMemset(control_den.pointer,0xa5,den_words*4u));finish();
            begin=std::chrono::steady_clock::now();
            replay(0u,dv.as<uint16_t>()+guard,prob.as<uint16_t>()+guard,scales.as<float>()+guard,control.as<float>()+guard,start,queries,0u,stride,drcp.as<unsigned char>(),control_acc.as<float>()+guard,control_den.as<float>()+guard,
                indices.as<unsigned>()+guard,indices.as<unsigned>()+guard+cells,dvt.as<uint16_t>()+guard,tokens,plan);
            finish();const double wall=elapsed(begin);if(attempt){samples[0][attempt-1u]+=wall;maximum[0]=std::max(maximum[0],wall);}
            if(!attempt) {
                first_output=download<uint32_t>(control,output_words);first_acc=download<uint32_t>(control_acc,output_words);first_den=download<uint32_t>(control_den,den_words);
            }else {
                equal(control,first_output,"control output");equal(control_acc,first_acc,"control accumulator");equal(control_den,first_den,"control denominator");raw_comparisons[0]+=cells;
            }
        }
        const auto host_control=download<float>(control,output_words),host_acc=download<float>(control_acc,output_words);
        for(unsigned i=0u;i<cells;++i)if(qrt_sm121_pv_bound::bf16(host_control[guard+i])!=golden[size_t(start)*4096u+i])throw std::runtime_error("original PV differs from complete external GB10 context");
        for(size_t i=0u;i<output_words;++i)if((i<guard||i>=guard+cells)&&bits(host_control[i])!=0xa5a5a5a5u)throw std::runtime_error("captured output guard");
        endpoints+=cells;
        for(unsigned sample=0u;sample<std::min(4u,selected);++sample) {
            const unsigned cell=index_before[guard+size_t(sample)*(selected-1u)/std::max(1u,std::min(4u,selected)-1u)],row=cell/256u,column=cell%256u,head=row%16u;
            const float expected=cpu_pv(probability.data()+guard+size_t(row)*stride,v.data()+guard,scale_words.data()+guard+size_t(row)*(tiles+1u),start+row/16u+1u,(head/8u)*256u+column);
            if(bits(expected)!=bits(host_acc[guard+cell]))throw std::runtime_error("captured PV differs from wide CPU recurrence");++cpu_dots;
        }
        for(unsigned position=0u;position<1u;++position) {
            const unsigned variant=1u+position;
            for(unsigned attempt=0u;attempt<attempts;++attempt) {
                check(hipMemcpy(candidate.pointer,approximate.pointer,output_words*4u,hipMemcpyDeviceToDevice));
                check(hipMemset(candidate_acc.pointer,0xa5,output_words*4u));check(hipMemset(candidate_den.pointer,0xa5,den_words*4u));check(hipMemset(bad.pointer,0,4u));plan.reset();finish();
                begin=std::chrono::steady_clock::now();plan.prepare(prob.as<uint16_t>()+guard,start,queries,stride);
                replay(variant,dv.as<uint16_t>()+guard,prob.as<uint16_t>()+guard,scales.as<float>()+guard,candidate.as<float>()+guard,start,queries,0u,stride,drcp.as<unsigned char>(),candidate_acc.as<float>()+guard,candidate_den.as<float>()+guard,
                    indices.as<unsigned>()+guard,indices.as<unsigned>()+guard+cells,dvt.as<uint16_t>()+guard,tokens,plan);
                finish();const double wall=elapsed(begin);if(attempt){samples[variant][attempt-1u]+=wall;maximum[variant]=std::max(maximum[variant],wall);}
                compare_device(control,candidate,output_words,bad);compare_device(control_acc,candidate_acc,output_words,bad);compare_device(control_den,candidate_den,den_words,bad);finish();
                if(download<unsigned>(bad,1u)[0])throw std::runtime_error("captured PV candidate raw mismatch");raw_comparisons[variant]+=cells;
            }
            plan.verify(probability.data()+guard,start,queries,stride);
            check(hipMemset(audit.pointer,0xa5,(5u+2u*guard)*8u));check(hipMemset(audit.as<unsigned long long>()+guard,0,5u*8u));
            replay<true>(variant,dv.as<uint16_t>()+guard,prob.as<uint16_t>()+guard,scales.as<float>()+guard,candidate.as<float>()+guard,start,queries,0u,stride,drcp.as<unsigned char>(),candidate_acc.as<float>()+guard,candidate_den.as<float>()+guard,
                indices.as<unsigned>()+guard,indices.as<unsigned>()+guard+cells,dvt.as<uint16_t>()+guard,tokens,plan,audit.as<unsigned long long>()+guard);finish();
            const auto observed=download<unsigned long long>(audit,5u+2u*guard);
            for(size_t i=0u;i<observed.size();++i)if((i<guard||i>=guard+5u)&&observed[i]!=0xa5a5a5a5a5a5a5a5ull)throw std::runtime_error("PV audit guard");
            if(observed[guard]!=observed[guard+1]+observed[guard+2]+observed[guard+3]+observed[guard+4])throw std::runtime_error("PV audit partition");
            for(unsigned i=0u;i<5u;++i)stats[variant][i]+=observed[guard+i];
            compare_device(control,candidate,output_words,bad);compare_device(control_acc,candidate_acc,output_words,bad);compare_device(control_den,candidate_den,den_words,bad);finish();
            if(download<unsigned>(bad,1u)[0])throw std::runtime_error("PV audit production parity");
        }
        unchanged(indices,index_before);unchanged(prob,probability);unchanged(scales,scale_words);
        (void)score_cells;
    }
    unchanged(dq,q);unchanged(dk,k);unchanged(dv,v);unchanged(dkt,kt_before);unchanged(dvt,tv);unchanged(dex,exp);unchanged(drcp,rcp);prepared.verify();unchanged(plan.vp,plan.expected_value);
    if(endpoints!=size_t(tokens)*4096u)throw std::runtime_error("incomplete PV external context");
    for(unsigned variant:variants) {
        double sorted[3]={samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(sorted,sorted+3);
        if(variant&&stats[variant][0]!=expected_groups)throw std::runtime_error("PV audited group count");
        std::printf("{\"kind\":\"staged_half_pv_original_q7169\",\"variant\":%u,\"tokens\":7169,\"query_batch\":128,\"external_bf16_cells\":%zu,\"external_bf16_mismatches\":0,\"selected_cells\":%llu,\"cpu_selected_dots\":%u,\"raw_compared_output_cells\":%zu,\"raw_bit_mismatches\":0,\"common_attention_ms\":%.6f,\"query_key_preparation_ms\":%.6f,\"value_transpose_ms\":%.6f,\"value_encoding_ms\":%.6f,\"replay_including_probability_encoding_ms\":%.6f,\"replay_with_all_encoding_ms\":%.6f,\"replay_samples_ms\":[%.6f,%.6f,%.6f],\"maximum_completed_slab_ms\":%.6f,\"warmup_per_slab\":1,\"samples_per_slab\":3,\"audit_groups_total_reserved_reserved_transformed_original\":[%llu,%llu,%llu,%llu,%llu],\"audit_excluded_from_timing\":true,\"all_attempts_verified\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"unique_candidates\":true,\"complete_cpu_metadata_check\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            variant,endpoints,selected_total,cpu_dots,raw_comparisons[variant],common_ms,prepared.ms,transpose_ms,variant?plan.value_ms:0.0,sorted[1],sorted[1]+(variant?plan.value_ms:0.0),samples[variant][0],samples[variant][1],samples[variant][2],maximum[variant],stats[variant][0],stats[variant][1],stats[variant][2],stats[variant][3],stats[variant][4]);std::fflush(stdout);
    }
}
}
int main(int argc,char** argv)try {
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));
    if(std::strncmp(prop.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    if(argc==8&&!std::strcmp(argv[1],"--q7169")){captured(argv[2],argv[3],argv[4],argv[5],argv[6],argv[7]);return 0;}
    if(argc==2&&!std::strcmp(argv[1],"--long-selftest")){
        for(unsigned input:{0u,1u})for(auto shape:{std::pair<unsigned,unsigned>{8191u,2u},{65536u,32u},{131072u,32u},{263168u,32u},{264735u,1u}})
            generated(shape.first,shape.second,input,2u);
        return 0;
    }
    if(argc!=2||std::strcmp(argv[1],"--selftest"))throw std::runtime_error("use --selftest or --q7169 Q K V reference exp2 rcp");
    for(unsigned input=0u;input<5u;++input) {
        for(auto shape:{std::pair<unsigned,unsigned>{0u,1u},{31u,2u},{17u,32u},{64u,3u},{8191u,1u},{17u,65u},{31u,128u}})
            for(unsigned selection:{0u,1u,2u})generated(shape.first,shape.second,input,selection);
        generated(8064u,128u,input,2u);
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"staged_half_pv_error=%s\n",e.what());return 1;}
