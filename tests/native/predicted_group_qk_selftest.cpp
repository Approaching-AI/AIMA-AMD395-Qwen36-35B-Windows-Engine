#include "../../native/providers/ck_fmha/prepared_decoded_qk.h"
#include "../../native/providers/ck_fmha/predicted_group_qk.h"
#include "../../native/providers/ck_fmha/fused_predicted_group_qk.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace qrt_blackwell_attention;
constexpr unsigned guard = 64u;
constexpr unsigned variants[] = {0u, 1u, 2u, 3u, 4u};
constexpr unsigned variant_count = sizeof(variants) / sizeof(variants[0]);
void check(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
struct Device {
    void* pointer = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&pointer, bytes)); }
    ~Device() { if (pointer && hipFree(pointer) != hipSuccess) std::abort(); }
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
__global__ void verify_plan_storage(const uint32_t* storage,size_t words,size_t used,unsigned* bad) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=words)return;
    if(i<2u*guard||i>=2u*guard+used*2u){if(storage[i]!=0xa5a5a5a5u)atomicAdd(bad,1u);}
    else if(i%2u){const unsigned control=storage[i];if(control!=0x80000000u&&(control&0xffe00000u))atomicAdd(bad,1u);}
}
__global__ void audit_parity(const uint32_t* actual,const uint32_t* expected,size_t cells,unsigned* bad) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=cells+2u*guard)return;
    if(i<guard||i>=cells+guard){if(actual[i]!=0xa5a5a5a5u)atomicAdd(bad,1u);}
    else if(actual[i]!=expected[i-guard])atomicAdd(bad,1u);
}
struct Scratch {
    size_t capacity;Device storage,bad,counters;
    const uint16_t* q=nullptr;const uint16_t* k=nullptr;float* out=nullptr;
    const unsigned* qflags=nullptr;const unsigned* kflags=nullptr;
    unsigned start=0u,count=0u,stride=0u,key_stride=0u;
    size_t last_hits=0u,last_fallbacks=0u;
    Scratch(unsigned tokens,unsigned queries):capacity(size_t(tokens)*queries*kQueryHeads*16u),
        storage((capacity+2u*guard)*sizeof(qrt_sm121_predicted_group::Plan)),bad(4u),counters(16u){reset();}
    qrt_sm121_predicted_group::Plan* data(){return storage.as<qrt_sm121_predicted_group::Plan>()+guard;}
    void reset(){check(hipMemset(storage.pointer,0xa5,(capacity+2u*guard)*8u));q=nullptr;last_hits=last_fallbacks=0u;}
    size_t verify(unsigned variant,unsigned actual_stride,unsigned queries){
        const size_t used=variant==1u||variant==2u?size_t(actual_stride)*queries*kQueryHeads*16u:0u;
        if(used>capacity)throw std::runtime_error("predicted QK plan capacity");
        const size_t words=(capacity+2u*guard)*2u;check(hipMemset(bad.pointer,0,4u));
        hipLaunchKernelGGL(verify_plan_storage,dim3((words+255u)/256u),dim3(256u),0u,nullptr,
            storage.as<uint32_t>(),words,used,bad.as<unsigned>());check(hipGetLastError());finish();
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("plan control/guard/unused tail changed");
        if(variant){
            if(!q||stride!=actual_stride||count!=queries)throw std::runtime_error("audit context");
            const size_t cells=size_t(count)*kQueryHeads*stride;
            Device audit((cells+2u*guard)*4u);check(hipMemset(audit.pointer,0xa5,(cells+2u*guard)*4u));
            check(hipMemset(counters.pointer,0,16u));
            if(variant<=2u){
                const auto audited=variant==2u?qrt_predicted_group_qk::finish<true,true>:qrt_predicted_group_qk::finish<true,false>;
                hipLaunchKernelGGL(audited,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,
                    q,k,data(),audit.as<float>()+guard,start,count,stride,key_stride,counters.as<unsigned long long>());
            }else{
                if(!qflags||!kflags)throw std::runtime_error("fused audit flags missing");
                const auto audited=variant==3u?qrt_fused_predicted_group_qk::scores<4u,true>:qrt_fused_predicted_group_qk::scores<16u,true>;
                hipLaunchKernelGGL(audited,dim3((stride+15u)/16u,kQueryHeads,(count+15u)/16u),dim3(256u),0u,nullptr,
                    q,k,qflags,kflags,audit.as<float>()+guard,start,count,stride,key_stride,counters.as<unsigned long long>());
            }
            check(hipGetLastError());finish();
            hipLaunchKernelGGL(audit_parity,dim3((cells+2u*guard+255u)/256u),dim3(256u),0u,nullptr,
                audit.as<uint32_t>(),reinterpret_cast<const uint32_t*>(out),cells,bad.as<unsigned>());
            check(hipGetLastError());finish();
            if(download<unsigned>(bad,1u)[0])throw std::runtime_error("predicted QK audit parity/guard mismatch");
            const auto stats=download<unsigned long long>(counters,2u);last_hits=size_t(stats[0]);last_fallbacks=size_t(stats[1]);
        }
        return last_fallbacks;
    }
};
void staged_variant(unsigned variant,const uint16_t* q,const uint16_t* k,float* out,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride,Prepared& prepared,Scratch& scratch) {
    const dim3 grid((stride+15u)/16u,kQueryHeads,(count+15u)/16u);
    const auto* qp=prepared.qp.as<uint32_t>()+guard;const auto* kp=prepared.kp.as<uint32_t>()+guard;
    const auto* qf=prepared.qf.as<unsigned>()+guard;const auto* kf=prepared.kf.as<unsigned>()+guard;
    if(variant){
        scratch.q=q;scratch.k=k;scratch.out=out;scratch.start=start;scratch.count=count;scratch.stride=stride;scratch.key_stride=key_stride;
        scratch.qflags=qf;scratch.kflags=kf;
    }
    if(!variant){
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::scores<128u,true,16u,16u>),grid,dim3(kThreads),0u,nullptr,
            q,k,qp,kp,qf,kf,out,start,count,stride,key_stride);
    }else if(variant==1u||variant==2u){
        const auto builder=variant==2u?qrt_predicted_group_qk::prepare<true>:qrt_predicted_group_qk::prepare<false>;
        hipLaunchKernelGGL(builder,grid,dim3(kThreads),0u,nullptr,
            q,k,qf,kf,scratch.data(),start,count,stride,key_stride);check(hipGetLastError());
        const auto consumer=variant==2u?qrt_predicted_group_qk::finish<false,true>:qrt_predicted_group_qk::finish<false,false>;
        hipLaunchKernelGGL(consumer,
            dim3((size_t(count)*kQueryHeads*stride+255u)/256u),dim3(256u),0u,nullptr,
            q,k,scratch.data(),out,start,count,stride,key_stride,nullptr);
    }else if(variant==3u||variant==4u){
        const auto fused=variant==3u?qrt_fused_predicted_group_qk::scores<4u,false>:qrt_fused_predicted_group_qk::scores<16u,false>;
        hipLaunchKernelGGL(fused,grid,dim3(kThreads),0u,nullptr,
            q,k,qf,kf,out,start,count,stride,key_stride,nullptr);
    }else throw std::runtime_error("invalid predicted QK variant");
    check(hipGetLastError());
}

void original(const uint16_t* q, const uint16_t* k, float* out,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    hipLaunchKernelGGL(blackwell_tiled_exact_scores_kernel,
        dim3((stride+31u)/32u,kQueryHeads,(count+7u)/8u),dim3(kThreads),0u,nullptr,
        q,k,out,start,count,stride,key_stride);
    check(hipGetLastError());
}
struct Case { unsigned tokens, start, count, mode; };
void run(Case c) {
    const unsigned stride = c.start+c.count;
    const size_t cells = size_t(c.count)*kQueryHeads*stride;
    std::vector<uint16_t> q(size_t(c.tokens)*kQueryHeads*kHeadDim+2u*guard,0x5a5au);
    std::vector<uint16_t> k(size_t(c.tokens)*kKvHeads*kHeadDim+2u*guard,0x5a5au);
    for(size_t i=guard;i+guard<q.size();++i)
        q[i]=uint16_t(((i*37u+i/19u)&0x807fu)|((123u+i%8u)<<7u));
    for(size_t i=guard;i+guard<k.size();++i)
        k[i]=uint16_t(((i*53u+i/23u)&0x807fu)|((121u+i%10u)<<7u));
    if(c.mode==1u) {
        for(size_t i=guard;i+guard<q.size();i+=7u)q[i]=i%3u?0u:0x8000u;
        for(size_t i=guard;i+guard<k.size();i+=11u)k[i]=i%3u?0u:0x8000u;
    }
    if(c.mode==2u || c.mode==3u) {
        const uint16_t edges[]={1u,0x8001u,0x007fu,0x807fu,uint16_t(63u<<7u|19u),uint16_t(192u<<7u|11u),0u,0x8000u};
        for(unsigned j=0u;j<8u;++j) {
            q[guard+size_t(c.start)*kQueryHeads*kHeadDim+j]=edges[j];
            k[guard+j]=edges[7u-j];
        }
    }
    if(c.mode==4u) {
        for(size_t i=guard;i+guard<q.size();++i)q[i]=uint16_t(127u<<7u|127u);
        for(size_t i=guard;i+guard<k.size();++i)
            k[i]=uint16_t(127u<<7u|127u|((i/16u)&1u?0x8000u:0u));
    }
    if(c.mode==5u) {
        q[guard+size_t(c.start)*kQueryHeads*kHeadDim+233u]=0x0001u;
        k[guard+195u]=uint16_t(192u<<7u|37u);
    }
    if(c.mode==6u) {
        for(size_t i=guard;i+guard<q.size();++i)q[i]=uint16_t((190u<<7u)|127u);
        for(size_t i=guard;i+guard<k.size();++i)k[i]=uint16_t((190u<<7u)|127u);
    }
    if(c.mode==7u) {
        for(size_t i=guard;i+guard<q.size();++i)q[i]=i%2u ? 0u : 0x8000u;
    }
    std::vector<uint16_t> transposed(k.size(),0x5a5au);
    for(unsigned token=0u;token<c.tokens;++token)
        for(unsigned feature=0u;feature<kKvHeads*kHeadDim;++feature)
            transposed[guard+size_t(feature)*c.tokens+token]=k[guard+size_t(token)*kKvHeads*kHeadDim+feature];
    Device dq(q.size()*2u),dk(k.size()*2u),dt(transposed.size()*2u);
    Device reference((cells+2u*guard)*4u),candidate((cells+2u*guard)*4u);
    upload(dq,q);upload(dk,k);upload(dt,transposed);
    Prepared prepared(dq.as<uint16_t>()+guard,dk.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,
        q.data()+guard,k.data()+guard,c.tokens);
    Scratch scratch(c.tokens,c.count);
    check(hipMemset(reference.pointer,0xa5,(cells+2u*guard)*4u));
    original(dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,reference.as<float>()+guard,c.start,c.count,stride,c.tokens);
    finish();const auto a=download<uint32_t>(reference,cells+2u*guard);
    size_t original_hits=0u,original_fallbacks=0u;
    for(unsigned variant:variants) {
        const unsigned groups=variant%100u;
        check(hipMemset(candidate.pointer,0xa5,(cells+2u*guard)*4u));
        scratch.reset();
        staged_variant(variant,dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,candidate.as<float>()+guard,c.start,c.count,stride,c.tokens,prepared,scratch);
        finish();const auto b=download<uint32_t>(candidate,cells+2u*guard);
        for(size_t i=0u;i<a.size();++i) {
            if(i<guard || i>=cells+guard) {
                if(a[i]!=0xa5a5a5a5u || b[i]!=0xa5a5a5a5u)throw std::runtime_error("QK score redzone changed");
            }else if(a[i]!=b[i]) {
                std::fprintf(stderr,"groups=%u tokens=%u start=%u count=%u mode=%u index=%zu expected=%08x actual=%08x\n",groups,c.tokens,c.start,c.count,c.mode,i-guard,a[i],b[i]);
                throw std::runtime_error("predicted QK differs from original scores");
            }
        }
        for(unsigned sample=0u;sample<64u;++sample) {
            const unsigned row=sample%c.count,head=(sample/4u)%kQueryHeads;
            const unsigned key=sample%2u?(sample*797u)%(c.start+row+1u):0u;
            const auto* left=q.data()+guard+(size_t(c.start+row)*kQueryHeads+head)*kHeadDim;
            const auto* right=k.data()+guard+(size_t(key)*kKvHeads+head/(kQueryHeads/kKvHeads))*kHeadDim;
            const float expected=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,left,right,kHeadDim)*kExactScale;
            if(bits(expected)!=b[guard+(size_t(row)*kQueryHeads+head)*stride+key])
                throw std::runtime_error("QK differs from independent wide CPU accumulator");
        }
        scratch.verify(variant,stride,c.count);
        if(variant==1u){original_hits=scratch.last_hits;original_fallbacks=scratch.last_fallbacks;}
        if(variant>=2u&&(original_hits!=scratch.last_hits||original_fallbacks!=scratch.last_fallbacks))
            throw std::runtime_error("layout changed exact plan decisions");
        std::printf("{\"kind\":\"predicted_group_qk_safety\",\"tokens\":%u,\"query_start\":%u,\"query_count\":%u,\"mode\":%u,\"variant\":%u,\"fused_plan_groups\":%u,\"global_plan_live_bytes\":%zu,\"cells\":%zu,\"cpu_dots\":64,\"raw_bit_mismatches\":0,\"plan_hits\":%zu,\"fallback_scores\":%zu,\"audit_parity_pass\":true,\"layout_plan_decisions_equal\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",c.tokens,c.start,c.count,c.mode,variant,variant==3u?4u:variant==4u?16u:0u,(variant==1u||variant==2u)?cells*16u*8u:0u,cells,scratch.last_hits,scratch.last_fallbacks);
        std::fflush(stdout);
    }
    unchanged(dq,q);unchanged(dk,k);unchanged(dt,transposed);prepared.verify();
}

std::vector<uint16_t> read_words(const char* name, size_t words) {
    std::ifstream file(name,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()!=std::streamoff(words*2u))throw std::runtime_error("QK capture size differs");
    std::vector<uint16_t> result(words);file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(result.data()),std::streamsize(words*2u)))throw std::runtime_error("QK capture read failed");
    return result;
}
__global__ void compare_scores(const uint32_t* expected, const uint32_t* actual,
    size_t count, unsigned* mismatches) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count && expected[i]!=actual[i])atomicAdd(mismatches,1u);
}
void captured(const char* qfile,const char* kfile,unsigned tokens) {
    constexpr unsigned original_tokens=7169u,batch=128u,attempts=4u;
    if(tokens!=original_tokens && tokens!=8192u)throw std::runtime_error("invalid extension");
    auto q=read_words(qfile,size_t(original_tokens)*kQueryHeads*kHeadDim);
    auto k=read_words(kfile,size_t(original_tokens)*kKvHeads*kHeadDim);
    if(tokens>original_tokens) {
        // Explicit component extension: repeat original rows, not a new model
        // prompt or a GB10 token capture. All extended scores use full canonical
        // replay and independent CPU dot checks, exactly like original rows.
        q.resize(size_t(tokens)*kQueryHeads*kHeadDim); k.resize(size_t(tokens)*kKvHeads*kHeadDim);
        std::copy_n(q.begin(),size_t(tokens-original_tokens)*kQueryHeads*kHeadDim,q.begin()+size_t(original_tokens)*kQueryHeads*kHeadDim);
        std::copy_n(k.begin(),size_t(tokens-original_tokens)*kKvHeads*kHeadDim,k.begin()+size_t(original_tokens)*kKvHeads*kHeadDim);
    }
    q.insert(q.begin(),guard,0x5a5au);q.insert(q.end(),guard,0x5a5au);
    k.insert(k.begin(),guard,0x5a5au);k.insert(k.end(),guard,0x5a5au);
    const size_t capacity=size_t(batch)*kQueryHeads*tokens,total_words=capacity+2u*guard;
    Device dq(q.size()*2u),dk(k.size()*2u),dt(k.size()*2u);
    Device reference(total_words*4u),candidate(total_words*4u),bad(4u);
    upload(dq,q);upload(dk,k);check(hipMemset(dt.pointer,0x5a,k.size()*2u));
    check(hipMemset(bad.pointer,0,4u));
    Prepared prepared(dq.as<uint16_t>()+guard,dk.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,
        q.data()+guard,k.data()+guard,tokens);
    Scratch scratch(tokens,batch);
    std::vector<uint16_t> transposed(k.size(),0x5a5au);
    for(unsigned token=0u;token<tokens;++token)
        for(unsigned feature=0u;feature<kKvHeads*kHeadDim;++feature)
            transposed[guard+size_t(feature)*tokens+token]=k[guard+size_t(token)*kKvHeads*kHeadDim+feature];
    unchanged(dt,transposed);
    double samples[variant_count][3]{},maximum_stage_ms[variant_count]{};
    size_t flagged_tiles[variant_count]{},accepted_groups[variant_count]{};
    size_t compared=0u;unsigned cpu_dots=0u;
    for(unsigned start=0u;start<tokens;start+=batch) {
        const unsigned count=std::min(batch,tokens-start),stride=start+count;
        const size_t cells=size_t(count)*kQueryHeads*stride;
        check(hipMemset(reference.pointer,0xa5,total_words*4u));
        original(dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,reference.as<float>()+guard,start,count,stride,tokens);
        finish();
        // Rotate variant order across slabs; each receives the same warmup and
        // three completed samples. All runs are checked after their timer.
        for(unsigned position=0u;position<variant_count;++position) {
            const unsigned mode=(position+start/batch)%variant_count,variant=variants[mode];
            for(unsigned attempt=0u;attempt<attempts;++attempt) {
                check(hipMemset(candidate.pointer,0xa5,total_words*4u));
                scratch.reset();finish();
                const auto begin=std::chrono::steady_clock::now();
                staged_variant(variant,dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,candidate.as<float>()+guard,
                    start,count,stride,tokens,prepared,scratch);
                finish();const double wall=elapsed(begin);
                if(attempt){samples[mode][attempt-1u]+=wall;maximum_stage_ms[mode]=std::max(maximum_stage_ms[mode],wall);}
                // Includes both redzones and the complete unused score tail.
                hipLaunchKernelGGL(compare_scores,dim3((total_words+255u)/256u),dim3(256u),0u,nullptr,
                    reference.as<uint32_t>(),candidate.as<uint32_t>(),total_words,bad.as<unsigned>());
                check(hipGetLastError());finish();compared+=cells;
                if(download<unsigned>(bad,1u)[0])throw std::runtime_error("captured QK score or tail differs from original");
                scratch.verify(variant,stride,count);
            }
            flagged_tiles[mode]+=scratch.last_fallbacks;accepted_groups[mode]+=scratch.last_hits;
            for(unsigned sample=0u;sample<4u;++sample) {
                const unsigned row=sample*(count-1u)/3u,head=(start/batch+sample*5u)%kQueryHeads;
                const unsigned key=(start+row)*sample/3u;
                const float expected=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
                    q.data()+guard+(size_t(start+row)*kQueryHeads+head)*kHeadDim,
                    k.data()+guard+(size_t(key)*kKvHeads+head/(kQueryHeads/kKvHeads))*kHeadDim,kHeadDim)*kExactScale;
                uint32_t actual;
                check(hipMemcpy(&actual,candidate.as<uint32_t>()+guard+(size_t(row)*kQueryHeads+head)*stride+key,4u,hipMemcpyDeviceToHost));
                if(bits(expected)!=actual)throw std::runtime_error("captured QK differs from independent wide CPU sum");
                ++cpu_dots;
            }
        }
    }
    unchanged(dq,q);unchanged(dk,k);unchanged(dt,transposed);prepared.verify();
    for(unsigned mode=2u;mode<variant_count;++mode)
        if(accepted_groups[1]!=accepted_groups[mode]||flagged_tiles[1]!=flagged_tiles[mode])
            throw std::runtime_error("capture layout changed exact plan decisions");
    for(unsigned mode=0u;mode<variant_count;++mode) {
        const unsigned variant=variants[mode];
        double sorted[3]={samples[mode][0],samples[mode][1],samples[mode][2]};std::sort(sorted,sorted+3);
        const double preparation_ms=prepared.ms;
        std::printf("{\"kind\":\"predicted_group_qk_capture\",\"variant\":%u,\"fused_plan_groups\":%u,\"global_plan_live_bytes\":%zu,\"query_rows\":16,\"key_columns\":16,\"verified_fallback_scores_after_last_attempt\":%zu,\"verified_plan_hits_after_last_attempt\":%zu,\"plan_workspace_bytes\":%zu,\"audit_parity_pass\":true,\"layout_plan_decisions_equal\":true,\"tokens\":%u,\"original_capture_tokens\":7169,\"real_model_prompt\":false,\"query_batch\":128,\"unique_score_cells\":%zu,\"compared_score_cells\":%zu,\"cpu_dots\":%u,\"raw_bit_mismatches\":0,\"one_time_preparation_ms\":%.6f,\"completed_query_with_replay_ms\":%.6f,\"completed_total_ms\":%.6f,\"completed_query_samples_ms\":[%.6f,%.6f,%.6f],\"warmup_per_slab\":1,\"samples_per_slab\":3,\"maximum_completed_slab_ms\":%.6f,\"all_attempts_verified\":true,\"redzones_pass\":true,\"unused_score_tail_pass\":true,\"immutable_inputs\":true,\"complete_cpu_encoding_check\":true,\"plan_guards_and_tail_pass\":true,\"all_attempts_plans_and_audit_verified\":true,\"plan_preparation_and_replay_timing_included\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            variant,variant==3u?4u:variant==4u?16u:0u,(variant==1u||variant==2u)?scratch.capacity*8u:0u,flagged_tiles[mode],accepted_groups[mode],scratch.capacity*8u,tokens,compared/variant_count/attempts,compared/variant_count,cpu_dots/variant_count,
            preparation_ms,sorted[1],sorted[1]+preparation_ms,samples[mode][0],samples[mode][1],samples[mode][2],maximum_stage_ms[mode]);
        std::fflush(stdout);
    }
}

}
int main(int argc,char** argv) {
    try {
        hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));
        if(std::strncmp(p.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
        if(argc==4 && (!std::strcmp(argv[1],"--q7169") || !std::strcmp(argv[1],"--q8192"))){captured(argv[2],argv[3],!std::strcmp(argv[1],"--q8192")?8192u:7169u);return 0;}
        if(argc!=2 || std::strcmp(argv[1],"--selftest"))throw std::runtime_error("use --selftest, --q7169 Q K or --q8192 Q K");
        const Case cases[]={{17,0,8,0},{17,0,9,2},{33,0,16,0},{33,1,31,1},{1,0,1,0},{9,0,9,1},{35,3,17,2},{67,33,32,3},{67,64,3,4},
            {129,1,128,0},{7169,7041,128,1},{8192,8064,128,0},{8192,8191,1,2},{8193,8191,2,3},
            {17,0,9,5},{67,33,32,5},{33,1,31,6},{33,0,16,7},{129,1,128,5},{129,1,128,7}};
        for(auto c:cases)run(c);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
