#include "../../native/providers/ck_fmha/prepared_decoded_qk.h"
#include "../../native/providers/ck_fmha/wmma_pv_operands.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace qrt_blackwell_attention;
constexpr unsigned guard = 64u;
constexpr unsigned variants[] = {0u, 1u, 2u, 3u};
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
        throw std::runtime_error("PV source, encoding or redzone changed");
}
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("PV completion deadline");
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

using PackedRow=qrt_wmma_pv_operands::Row;
struct ValueView {
    unsigned tokens; size_t words; Device device; std::vector<uint16_t> expected; double ms;
    ValueView(const uint16_t* value,const uint16_t* host,unsigned count)
        :tokens(count),words(size_t((count+31u)/32u*2u)*512u*16u),device((words+2u*guard)*2u),expected(words+2u*guard,0x5a5au) {
        check(hipMemset(device.pointer,0x5a,(words+2u*guard)*2u));finish();
        auto begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL(qrt_wmma_pv_operands::prepare_value,dim3((words/16u+255u)/256u),dim3(256u),0u,nullptr,value,data(),count);
        check(hipGetLastError());finish();ms=elapsed(begin);
        for(unsigned group=0u;group<(count+31u)/32u*2u;++group)for(unsigned feature=0u;feature<512u;++feature)for(unsigned i=0u;i<16u;++i) {
            const unsigned key=group*16u+i;
            expected[guard+(size_t(group)*512u+feature)*16u+i]=key<count?host[size_t(key)*512u+feature]:0u;
        }
        verify();
    }
    PackedRow* data(){return reinterpret_cast<PackedRow*>(device.as<uint16_t>()+guard);}
    void verify(){unchanged(device,expected);}
};
struct Shape {
    unsigned start,queries,stride,capacity,maximum_stride,output_start;
    size_t pwords()const{return size_t(capacity)*16u*maximum_stride+2u*guard;}
    size_t swords()const{return size_t(capacity)*16u*((maximum_stride+31u)/32u+1u)+2u*guard;}
    size_t packed_words()const{return size_t((maximum_stride+31u)/32u*2u)*16u*capacity*16u+2u*guard;}
    size_t owords()const{return size_t(output_start+capacity)*4096u+2u*guard;}
    size_t dwords()const{return size_t(output_start+capacity)*16u+2u*guard;}
    size_t ewords()const{return size_t(capacity)*4096u+2u*guard;}
    unsigned cells()const{return queries*4096u;}
};
struct Stage {
    Shape shape;Device p,s,packed,o,a,d,e;
    explicit Stage(Shape q):shape(q),p(q.pwords()*2u),s(q.swords()*4u),packed(q.packed_words()*2u),o(q.owords()*4u),a(q.owords()*4u),d(q.dwords()*4u),e(q.ewords()*4u){reset();}
    PackedRow* rows(){return reinterpret_cast<PackedRow*>(packed.as<uint16_t>()+guard);}
    void reset() {
        check(hipMemset(p.pointer,0x5a,shape.pwords()*2u));check(hipMemset(packed.pointer,0x5a,shape.packed_words()*2u));
        for(Device* v:{&s,&o,&a,&d,&e}) {
            const size_t words=v==&s?shape.swords():v==&d?shape.dwords():v==&e?shape.ewords():shape.owords();
            check(hipMemset(v->pointer,0xa5,words*4u));
        }
    }
    void launch(unsigned variant,const float* scores,const uint16_t* value,ValueView& view,const unsigned char* exp,const unsigned char* rcp,bool vllm) {
        const auto& q=shape;
        if(variant<2u) {
            hipLaunchKernelGGL(blackwell_online_probability_kernel,dim3(16u,q.queries),dim3(32u),0u,nullptr,
                scores,p.as<uint16_t>()+guard,s.as<float>()+guard,q.start,q.stride,exp,vllm);
        }else {
            hipLaunchKernelGGL(qrt_wmma_pv_operands::probability,dim3(16u,q.queries),dim3(32u),0u,nullptr,
                scores,p.as<uint16_t>()+guard,s.as<float>()+guard,rows(),q.start,q.stride,q.capacity,exp,vllm);
        }
        check(hipGetLastError());
        const dim3 grid(2u,16u,(q.queries+15u)/16u);
        if(!variant) {
            hipLaunchKernelGGL((blackwell_mantissa_value_kernel<true,false,true,true,true>),grid,dim3(256u),0u,nullptr,
                value,p.as<uint16_t>()+guard,s.as<float>()+guard,o.as<float>()+guard,q.start,q.queries,q.output_start,q.stride,rcp,
                a.as<float>()+guard,d.as<float>()+guard,nullptr,nullptr,e.as<float>()+guard);
        }else {
#define PACKED_PV_CASE(v,packed,paired) if(variant==v) hipLaunchKernelGGL((qrt_wmma_pv_operands::value<packed,paired>),grid,dim3(256u),0u,nullptr,p.as<uint16_t>()+guard,s.as<float>()+guard,rows(),view.data(),o.as<float>()+guard,q.start,q.queries,q.output_start,q.stride,q.capacity,rcp,a.as<float>()+guard,d.as<float>()+guard,e.as<float>()+guard)
            PACKED_PV_CASE(1u,false,false);
            else PACKED_PV_CASE(2u,true,false);
            else PACKED_PV_CASE(3u,true,true);
            else throw std::runtime_error("packed PV variant");
#undef PACKED_PV_CASE
        }
        check(hipGetLastError());
    }
    void compare(Stage& expected,Device& expected_packed,bool uses_packed,Device& bad) {
        compare_device(expected.p,p,shape.pwords()/2u,bad);compare_device(expected.s,s,shape.swords(),bad);
        compare_device(expected.o,o,shape.owords(),bad);compare_device(expected.a,a,shape.owords(),bad);
        compare_device(expected.d,d,shape.dwords(),bad);compare_device(expected.e,e,shape.ewords(),bad);
        compare_device(uses_packed?expected_packed:expected.packed,packed,shape.packed_words()/2u,bad);
    }
};
void guards(Device& device,size_t words,size_t begin,size_t count) {
    const auto data=download<uint32_t>(device,words);
    for(size_t i=0u;i<words;++i)if((i<begin||i>=begin+count)&&data[i]!=0xa5a5a5a5u)
        throw std::runtime_error("PV output guard, prefix or tail changed");
}
std::vector<uint16_t> packed_expected(Stage& original) {
    const auto& q=original.shape;const auto p=download<uint16_t>(original.p,q.pwords());
    const auto scales=download<float>(original.s,q.swords());
    std::vector<uint16_t> result(q.packed_words(),0x5a5au);
    for(unsigned query=0u;query<q.queries;++query)for(unsigned head=0u;head<16u;++head) {
        const unsigned row=query*16u+head,tokens=q.start+query+1u,groups=(tokens+31u)/32u*2u;
        for(unsigned group=0u;group<groups;++group)for(unsigned i=0u;i<16u;++i) {
            const unsigned key=group*16u+i;
            result[guard+((size_t(group)*16u+head)*q.capacity+query)*16u+i]=key<tokens?p[guard+size_t(row)*q.stride+key]:0u;
        }
    }
    for(size_t i=0u;i<p.size();++i) {
        bool live=i>=guard&&i<guard+size_t(q.queries)*16u*q.stride;
        if(live) {
            const unsigned row=unsigned((i-guard)/q.stride),key=unsigned((i-guard)%q.stride);
            const unsigned tokens=q.start+row/16u+1u;
            live=key<(tokens+31u)/32u*32u;
            if(live&&key>=tokens&&p[i]!=0u)throw std::runtime_error("PV probability causal zero");
        }
        if(!live&&p[i]!=0x5a5au)throw std::runtime_error("PV probability guard or tail");
    }
    const unsigned tiles=(q.stride+31u)/32u;
    for(size_t i=0u;i<scales.size();++i) {
        bool live=i>=guard&&i<guard+size_t(q.queries)*16u*(tiles+1u);
        if(live) {
            const unsigned row=unsigned((i-guard)/(tiles+1u)),tile=unsigned((i-guard)%(tiles+1u));
            live=tile<(q.start+row/16u+32u)/32u||tile==tiles;
        }
        if(!live&&bits(scales[i])!=0xa5a5a5a5u)throw std::runtime_error("PV scale guard or tail");
    }
    guards(original.o,q.owords(),guard+q.output_start*4096u,q.cells());
    guards(original.a,q.owords(),guard+q.output_start*4096u,q.cells());
    guards(original.d,q.dwords(),guard+q.output_start*16u,q.queries*16u);
    guards(original.e,q.ewords(),guard,q.cells());
    return result;
}
float score_value(unsigned row,unsigned key,unsigned mode) {
    uint32_t random=(row+3u)*747796405u+key*2891336453u;random=(random^(random>>16u))*2246822519u;
    if(mode==0u)return float(int(random%10241u)-5120)/256.0f;
    if(mode==1u)return key%7u==0u?8.0f:-float(random%4096u)/128.0f;
    if(mode==2u)return float(key/32u)*256.0f-float(key%32u);
    if(mode==3u)return key%2u?-0.0f:0.0f;
    if(mode==4u)return std::ldexp(float(int(random%33u)-16),-129);
    return (key/32u)%3u==0u?-1000.0f:float(int(random%65u)-32)/8.0f;
}
void generated(unsigned start,unsigned queries,unsigned mode,bool vllm) {
    Shape q{start,queries,start+queries,queries+3u,start+queries+7u,3u};
    std::vector<float> scores(q.pwords(),12345.0f);
    std::vector<uint16_t> values(size_t(q.stride)*512u+2u*guard,0x5a5au);
    for(unsigned row=0u;row<queries*16u;++row)for(unsigned key=0u;key<start+row/16u+1u;++key)
        scores[guard+size_t(row)*q.stride+key]=score_value(row,key,mode);
    for(size_t i=guard;i+guard<values.size();++i) {
        uint16_t word=qrt_sm121_pv_bound::bf16(float(int((i*173u+i/17u)%63u)-31)/64.0f);
        if(mode==1u&&i%7u==0u)word=uint16_t(i%127u+1u);
        if(mode==2u&&i%3u==0u)word=uint16_t((i&1u?0x8000u:0u)|0x5f80u);
        if(mode==3u&&i%3u==0u)word=(i&1u)?0x8000u:0u;
        if(mode==4u)word=uint16_t((i&1u?0x8000u:0u)|(i%3u?0x0101u:0x0001u));
        if(mode==5u&&i%251u==0u)word=(i&1u)?0xffc1u:0x7fc1u;
        values[i]=word;
    }
    Device ds(scores.size()*4u),dv(values.size()*2u),bad(4u);upload(ds,scores);upload(dv,values);
    ValueView view(dv.as<uint16_t>()+guard,values.data()+guard,q.stride);
    Stage original(q),candidate(q);original.launch(0u,ds.as<float>()+guard,dv.as<uint16_t>()+guard,view,nullptr,nullptr,vllm);finish();
    const auto expected=packed_expected(original);Device dp(expected.size()*2u);upload(dp,expected);
    if(mode==3u) {
        const auto p=download<uint16_t>(original.p,q.pwords());const auto s=download<float>(original.s,q.swords());
        const unsigned tiles=(q.stride+31u)/32u;
        for(unsigned row=0u;row<queries*16u;++row) {
            const unsigned tokens=start+row/16u+1u;
            for(unsigned key=0u;key<tokens;++key)if(p[guard+size_t(row)*q.stride+key]!=0x3f80u)throw std::runtime_error("PV CPU unit probability");
            if(s[guard+size_t(row)*(tiles+1u)+tiles]!=float(tokens))throw std::runtime_error("PV CPU constant denominator");
        }
    }
    for(unsigned variant:variants) {
        candidate.reset();check(hipMemset(bad.pointer,0,4u));finish();
        candidate.launch(variant,ds.as<float>()+guard,dv.as<uint16_t>()+guard,view,nullptr,nullptr,vllm);finish();
        candidate.compare(original,dp,variant>=2u,bad);finish();
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("WMMA operand generated raw parity");
        std::printf("{\"kind\":\"wmma_pv_operands_safety\",\"query_start\":%u,\"query_count\":%u,\"input_mode\":%u,\"vllm_sum\":%s,\"variant\":%u,\"cells\":%u,\"probability_cells\":%zu,\"raw_bit_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"complete_cpu_layout_check\":true,\"inference_acceptance\":false}\n",start,queries,mode,vllm?"true":"false",variant,q.cells(),size_t(queries)*16u*q.stride);std::fflush(stdout);
    }
    unchanged(ds,scores);unchanged(dv,values);unchanged(dp,expected);view.verify();
}

std::vector<unsigned> collect(Stage& stage,Device& indices) {
    const auto& q=stage.shape;const size_t words=size_t(q.capacity)*4096u+1u+2u*guard;
    check(hipMemset(indices.pointer,0xa5,words*4u));auto* count=indices.as<unsigned>()+guard+q.cells();check(hipMemset(count,0,4u));
    hipLaunchKernelGGL(blackwell_collect_pv_replay_kernel,dim3((q.cells()+255u)/256u),dim3(256u),0u,nullptr,
        stage.o.as<float>()+guard,stage.e.as<float>()+guard,q.output_start,q.cells(),indices.as<unsigned>()+guard,count);check(hipGetLastError());finish();
    auto result=download<unsigned>(indices,words);const unsigned n=result[guard+q.cells()];
    if(n>q.cells())throw std::runtime_error("WMMA PV candidate capacity");
    std::vector<bool> seen(q.cells(),false);
    for(unsigned i=0u;i<n;++i){const unsigned c=result[guard+i];if(c>=q.cells()||seen[c])throw std::runtime_error("WMMA PV duplicate candidate");seen[c]=true;}
    for(size_t i=0u;i<words;++i)if((i<guard||(i>=guard+n&&i<guard+q.cells())||i>guard+q.cells())&&result[i]!=0xa5a5a5a5u)throw std::runtime_error("WMMA PV candidate guard");
    std::sort(result.begin()+guard,result.begin()+guard+n);
    return result;
}
void exact(Stage& stage,Device& indices,const uint16_t* v,const uint16_t* tv,unsigned value_stride,const unsigned char* rcp) {
    const auto& q=stage.shape;
    hipLaunchKernelGGL((blackwell_compacted_pv_replay_kernel<true>),dim3(std::min(1024u,(q.cells()+63u)/64u)),dim3(256u),0u,nullptr,
        v,stage.p.as<uint16_t>()+guard,stage.s.as<float>()+guard,stage.o.as<float>()+guard,q.start,q.output_start,q.stride,rcp,
        stage.a.as<float>()+guard,stage.d.as<float>()+guard,indices.as<unsigned>()+guard,indices.as<unsigned>()+guard+q.cells(),tv,value_stride,0u);check(hipGetLastError());finish();
}
void captured(const char* qfile,const char* kfile,const char* vfile,const char* reference_file,const char* exp_file,const char* rcp_file) {
    constexpr unsigned tokens=7169u,batch=128u;
    auto hq=read_values<uint16_t>(qfile,size_t(tokens)*4096u),hk=read_values<uint16_t>(kfile,size_t(tokens)*512u),hv=read_values<uint16_t>(vfile,size_t(tokens)*512u);
    const auto golden=read_values<uint16_t>(reference_file,size_t(tokens)*4096u);
    const auto exp=read_values<unsigned char>(exp_file,exp2_backend::table_bytes),rcp=read_values<unsigned char>(rcp_file,qrt_sm121_attention_rcp::table_bytes);
    if(!exp2_backend::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("WMMA PV table layout");
    for(auto* p:{&hq,&hk,&hv}){p->insert(p->begin(),guard,0x5a5au);p->insert(p->end(),guard,0x5a5au);}
    Device dq(hq.size()*2u),dk(hk.size()*2u),dv(hv.size()*2u),dkt(hk.size()*2u),dvt(hv.size()*2u),dex(exp.size()),drcp(rcp.size());
    upload(dq,hq);upload(dk,hk);upload(dv,hv);upload(dex,exp);upload(drcp,rcp);
    check(hipMemset(dkt.pointer,0x5a,hk.size()*2u));check(hipMemset(dvt.pointer,0x5a,hv.size()*2u));finish();
    std::vector<uint16_t> tv(hv.size(),0x5a5au);for(unsigned key=0u;key<tokens;++key)for(unsigned f=0u;f<512u;++f)tv[guard+size_t(f)*tokens+key]=hv[guard+size_t(key)*512u+f];
    auto begin=std::chrono::steady_clock::now();check(hipError_t(transpose_keys(dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,size_t(tokens)*512u,tokens,nullptr)));finish();const double transpose_ms=elapsed(begin);unchanged(dvt,tv);
    Prepared prepared(dq.as<uint16_t>()+guard,dk.as<uint16_t>()+guard,dkt.as<uint16_t>()+guard,hq.data()+guard,hk.data()+guard,tokens);
    const auto kt_before=download<uint16_t>(dkt,hk.size());ValueView view(dv.as<uint16_t>()+guard,hv.data()+guard,tokens);
    Shape q{0u,batch,batch,batch,tokens,0u};Stage original(q),candidate(q);
    Device scores(q.pwords()*4u),score_copy(q.pwords()*4u),expected_packed(q.packed_words()*2u),bad(4u);
    Device indices((size_t(batch)*4096u+1u+2u*guard)*4u),other_indices((size_t(batch)*4096u+1u+2u*guard)*4u),corrected(q.owords()*4u),corrected_acc(q.owords()*4u),corrected_den(q.dwords()*4u);
    double samples[4][3]{},maximum[4]{},qk_ms=0.0,exact_ms=0.0;size_t endpoints=0u,score_cells=0u,raw_cells[4]{};unsigned long long selected_total=0u;unsigned cpu_dots=0u;
    for(unsigned start=0u;start<tokens;start+=batch) {
        q.start=start;q.queries=std::min(batch,tokens-start);q.stride=start+q.queries;original.shape=q;candidate.shape=q;
        check(hipMemset(scores.pointer,0xa5,q.pwords()*4u));finish();begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::scores<128u,true,16u,16u>),dim3((q.stride+15u)/16u,16u,(q.queries+15u)/16u),dim3(256u),0u,nullptr,
            dq.as<uint16_t>()+guard,dkt.as<uint16_t>()+guard,prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,prepared.qf.as<unsigned>()+guard,prepared.kf.as<unsigned>()+guard,scores.as<float>()+guard,start,q.queries,q.stride,tokens);
        check(hipGetLastError());finish();qk_ms+=elapsed(begin);check(hipMemcpy(score_copy.pointer,scores.pointer,q.pwords()*4u,hipMemcpyDeviceToDevice));
        original.reset();original.launch(0u,scores.as<float>()+guard,dv.as<uint16_t>()+guard,view,dex.as<unsigned char>(),drcp.as<unsigned char>(),true);finish();
        const auto packed=packed_expected(original);upload(expected_packed,packed);
        for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<4u;++position) {
            const unsigned variant=(position+start/batch+attempt)%4u;
            candidate.reset();check(hipMemset(bad.pointer,0,4u));finish();begin=std::chrono::steady_clock::now();
            candidate.launch(variant,scores.as<float>()+guard,dv.as<uint16_t>()+guard,view,dex.as<unsigned char>(),drcp.as<unsigned char>(),true);finish();
            const double wall=elapsed(begin);if(attempt){samples[variant][attempt-1u]+=wall;maximum[variant]=std::max(maximum[variant],wall);}
            candidate.compare(original,expected_packed,variant>=2u,bad);finish();
            if(download<unsigned>(bad,1u)[0])throw std::runtime_error("captured WMMA PV probability,scale,output,accumulator,denominator,bound or layout mismatch");
            raw_cells[variant]+=q.cells();
        }
        // Independently collect each producer's actual candidates and run the
        // unchanged exact correction, outside the producer timing.
        const auto expected_indices=collect(original,indices);const unsigned selected=expected_indices[guard+q.cells()];selected_total+=selected;
        begin=std::chrono::steady_clock::now();exact(original,indices,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>());exact_ms+=elapsed(begin);
        const auto output=download<float>(original.o,q.owords()),acc=download<float>(original.a,q.owords());
        const auto hp=download<uint16_t>(original.p,q.pwords());const auto hs=download<float>(original.s,q.swords());
        for(unsigned cell=0u;cell<q.cells();++cell)if(qrt_sm121_pv_bound::bf16(output[guard+cell])!=golden[size_t(start)*4096u+cell])throw std::runtime_error("complete original GB10 attention mismatch");
        for(unsigned sample=0u;sample<std::min(4u,selected);++sample) {
            const unsigned cell=expected_indices[guard+size_t(sample)*(selected-1u)/std::max(1u,std::min(4u,selected)-1u)],row=cell/256u,head=row%16u;
            const float expected=cpu_pv(hp.data()+guard+size_t(row)*q.stride,hv.data()+guard,hs.data()+guard+size_t(row)*((q.stride+31u)/32u+1u),start+row/16u+1u,(head/8u)*256u+cell%256u);
            if(bits(expected)!=bits(acc[guard+cell]))throw std::runtime_error("WMMA PV independent CPU selected dot");++cpu_dots;
        }
        check(hipMemcpy(corrected.pointer,original.o.pointer,q.owords()*4u,hipMemcpyDeviceToDevice));
        check(hipMemcpy(corrected_acc.pointer,original.a.pointer,q.owords()*4u,hipMemcpyDeviceToDevice));check(hipMemcpy(corrected_den.pointer,original.d.pointer,q.dwords()*4u,hipMemcpyDeviceToDevice));
        for(unsigned variant:variants) {
            candidate.reset();candidate.launch(variant,scores.as<float>()+guard,dv.as<uint16_t>()+guard,view,dex.as<unsigned char>(),drcp.as<unsigned char>(),true);finish();
            const auto actual_indices=collect(candidate,other_indices);if(actual_indices!=expected_indices)throw std::runtime_error("WMMA PV changed candidate identities");
            exact(candidate,other_indices,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>());
            check(hipMemset(bad.pointer,0,4u));compare_device(corrected,candidate.o,q.owords(),bad);compare_device(corrected_acc,candidate.a,q.owords(),bad);compare_device(corrected_den,candidate.d,q.dwords(),bad);finish();
            if(download<unsigned>(bad,1u)[0])throw std::runtime_error("WMMA PV complete corrected parity");
        }
        check(hipMemset(bad.pointer,0,4u));compare_device(score_copy,scores,q.pwords(),bad);finish();if(download<unsigned>(bad,1u)[0])throw std::runtime_error("WMMA PV score source changed");
        unchanged(expected_packed,packed);endpoints+=q.cells();score_cells+=size_t(q.queries)*16u*q.stride;
    }
    unchanged(dq,hq);unchanged(dk,hk);unchanged(dv,hv);unchanged(dkt,kt_before);unchanged(dvt,tv);unchanged(dex,exp);unchanged(drcp,rcp);prepared.verify();view.verify();
    if(endpoints!=size_t(tokens)*4096u)throw std::runtime_error("WMMA PV incomplete external context");
    for(unsigned variant:variants) {
        double sorted[3]={samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(sorted,sorted+3);
        std::printf("{\"kind\":\"wmma_pv_operands_original_q7169\",\"variant\":%u,\"tokens\":7169,\"query_batch\":128,\"external_bf16_cells\":%zu,\"external_bf16_mismatches\":0,\"selected_cells\":%llu,\"cpu_selected_dots\":%u,\"probability_cells\":%zu,\"raw_compared_output_cells\":%zu,\"raw_bit_mismatches\":0,\"query_key_ms\":%.6f,\"query_key_preparation_ms\":%.6f,\"exact_replay_ms\":%.6f,\"value_transpose_ms\":%.6f,\"packed_value_preparation_ms\":%.6f,\"probability_plus_approximate_pv_ms\":%.6f,\"producer_with_all_preparation_ms\":%.6f,\"producer_samples_ms\":[%.6f,%.6f,%.6f],\"maximum_completed_slab_ms\":%.6f,\"warmup_per_slab\":1,\"samples_per_slab\":3,\"all_attempts_verified\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"unique_candidates\":true,\"complete_cpu_layout_check\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            variant,endpoints,selected_total,cpu_dots,score_cells,raw_cells[variant],qk_ms,prepared.ms,exact_ms,transpose_ms,variant?view.ms:0.0,sorted[1],sorted[1]+(variant?view.ms:0.0),samples[variant][0],samples[variant][1],samples[variant][2],maximum[variant]);std::fflush(stdout);
    }
}
}
#ifndef QRT_WMMA_PV_OPERANDS_NO_MAIN
int main(int argc,char** argv)try {
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));if(std::strncmp(prop.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    if(argc==8&&!std::strcmp(argv[1],"--q7169")){captured(argv[2],argv[3],argv[4],argv[5],argv[6],argv[7]);return 0;}
    if(argc!=2||std::strcmp(argv[1],"--selftest"))throw std::runtime_error("use --selftest or --q7169 Q K V reference exp2 rcp");
    for(auto shape:{std::pair<unsigned,unsigned>{0u,1u},{31u,2u},{17u,32u},{64u,3u},{8191u,1u},{8064u,128u},{17u,65u},{31u,128u}})
        for(unsigned mode=0u;mode<6u;++mode)for(bool vllm:{false,true})generated(shape.first,shape.second,mode,vllm);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"wmma_pv_operands_error=%s\n",e.what());return 1;}
#endif
