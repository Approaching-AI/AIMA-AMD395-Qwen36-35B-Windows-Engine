// Reuses the established deferred-QK fixture geometry and independent CPU reference.
#include "../../native/providers/ck_fmha/prepared_decoded_qk.h"
#include "../../native/providers/ck_fmha/packed_exponent_qk.h"
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
constexpr unsigned variants[] = {0u, 1u, 3u, 4u, 5u};
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
struct Compressed {
    using Row=qrt_sm121_packed_exponents::Row;
    std::vector<Row> query,key;
    Device qrows,krows;
    double ms=0.0;
    Compressed(const uint16_t* q,const uint16_t* k,uint16_t* transposed,const uint16_t* hq,const uint16_t* hk,unsigned n)
        :query(size_t(n)*16u*16u+2u*guard),key(size_t(n)*2u*16u+2u*guard),qrows(query.size()*sizeof(Row)),krows(key.size()*sizeof(Row)){
        std::memset(query.data(),0xa5,query.size()*sizeof(Row));std::memset(key.data(),0xa5,key.size()*sizeof(Row));
        check(hipMemset(qrows.pointer,0xa5,query.size()*sizeof(Row)));check(hipMemset(krows.pointer,0xa5,key.size()*sizeof(Row)));finish();
        const auto begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL((qrt_packed_exponent_qk::prepare<false>),dim3((size_t(n)*16u*16u+255u)/256u),dim3(256u),0u,nullptr,q,qrows.as<Row>()+guard,nullptr,n);
        check(hipGetLastError());
        hipLaunchKernelGGL((qrt_packed_exponent_qk::prepare<true>),dim3((size_t(n)*2u*16u+255u)/256u),dim3(256u),0u,nullptr,k,krows.as<Row>()+guard,transposed,n);
        check(hipGetLastError());finish();ms=elapsed(begin);
        for(unsigned is_key=0u;is_key<2u;++is_key){
            const unsigned heads=is_key?2u:16u;const auto* source=is_key?hk:hq;
            auto& target=is_key?key:query;
            for(size_t row=0u;row<size_t(n)*heads*16u;++row){
                Row value{};int maximum=0;bool good=true;
                for(unsigned i=0u;i<16u;++i){
                    const uint16_t word=source[row*16u+i];value.original[i]=word;
                    const unsigned exponent=(word>>7u)&255u;
                    good &= !(word&32767u)||(exponent>=64u&&exponent<=190u);
                    if(word&32767u)maximum=std::max(maximum,int(exponent));
                }
                value.metadata.maximum=good?maximum:-1;
                for(unsigned i=0u;i<16u;++i){
                    const uint16_t word=value.original[i];
                    const unsigned delta=(word&32767u)?unsigned(std::min(15,maximum-int((word>>7u)&255u))):15u;
                    value.metadata.deficits[i/6u]|=delta<<(i%6u*5u);
                }
                const unsigned group=row%16u,head=(row/16u)%heads,token=row/(heads*16u);
                target[guard+(is_key?(size_t(head)*16u+group)*n+token:row)]=value;
            }
        }
        verify();
    }
    void verify(){unchanged(qrows,query);unchanged(krows,key);}
};
struct Prepared {
    unsigned tokens;
    std::vector<uint32_t> qpacked, kpacked, qflags, kflags;
    Device qp, kp, qf, kf;
    double ms = 0.0;
    Compressed compact;
    Prepared(const uint16_t* q, const uint16_t* k, uint16_t* transposed,
        const uint16_t* hq, const uint16_t* hk, unsigned n)
        : tokens(n), qpacked(size_t(n)*kQueryHeads*kHeadDim+2u*guard,0xa5a5a5a5u),
          kpacked(size_t(n)*kKvHeads*kHeadDim+2u*guard,0xa5a5a5a5u),
          qflags(n*kQueryHeads+2u*guard,0xa5a5a5a5u), kflags(n*kKvHeads+2u*guard,0xa5a5a5a5u),
          qp(qpacked.size()*4u), kp(kpacked.size()*4u), qf(qflags.size()*4u), kf(kflags.size()*4u),compact(q,k,transposed,hq,hk,n) {
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
    void verify() { unchanged(qp,qpacked);unchanged(kp,kpacked);unchanged(qf,qflags);unchanged(kf,kflags);compact.verify(); }
};
struct Scratch {
    size_t capacity; Device flags;
    Scratch(unsigned tokens,unsigned queries)
        : capacity(size_t((tokens+15u)/16u)*kQueryHeads*((queries+15u)/16u)),flags((capacity+2u*guard)*4u) { reset(); }
    unsigned* data(){return flags.as<unsigned>()+guard;}
    void reset(){check(hipMemset(flags.pointer,0xa5,(capacity+2u*guard)*4u));}
    size_t live(unsigned stride,unsigned queries) const {
        const size_t n=size_t((stride+15u)/16u)*kQueryHeads*((queries+15u)/16u);
        if(n>capacity)throw std::runtime_error("deferred tile capacity");return n;
    }
    size_t verify(unsigned variant,unsigned stride,unsigned queries) {
        const auto actual=download<unsigned>(flags,capacity+2u*guard);
        const size_t used=variant==2u?live(stride,queries):0u;size_t selected=0u;
        for(size_t i=0u;i<actual.size();++i) {
            if(i>=guard&&i<guard+used) {
                if(actual[i]>1u)throw std::runtime_error("invalid deferred tile flag");selected+=actual[i];
            }else if(actual[i]!=0xa5a5a5a5u)throw std::runtime_error("deferred tile guard or unused tail changed");
        }
        return selected;
    }
};
void staged_variant(unsigned variant,const uint16_t* q,const uint16_t* k,float* out,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride,Prepared& prepared,Scratch&) {
    const dim3 grid((stride+15u)/16u,kQueryHeads,(count+15u)/16u);
    const auto* qp=prepared.qp.as<uint32_t>()+guard;const auto* kp=prepared.kp.as<uint32_t>()+guard;
    const auto* qf=prepared.qf.as<unsigned>()+guard;const auto* kf=prepared.kf.as<unsigned>()+guard;
    if(!variant) {
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::scores<128u,true,16u,16u>),grid,dim3(kThreads),0u,nullptr,
            q,k,qp,kp,qf,kf,out,start,count,stride,key_stride);
    }else {
        if(variant==1u)hipLaunchKernelGGL((qrt_deferred_qk_fallback::scores<false>),grid,dim3(kThreads),0u,nullptr,
            qp,kp,qf,kf,out,start,count,stride,key_stride,nullptr);
#define QRT_PACKED_EXPONENT_CASE(V,W) else if(variant==V)hipLaunchKernelGGL((qrt_packed_exponent_qk::scores<W>),grid,dim3(kThreads),0u,nullptr,prepared.compact.qrows.as<Compressed::Row>()+guard,prepared.compact.krows.as<Compressed::Row>()+guard,out,start,count,stride,key_stride)
        QRT_PACKED_EXPONENT_CASE(3u,32u);
        QRT_PACKED_EXPONENT_CASE(4u,64u);
        QRT_PACKED_EXPONENT_CASE(5u,128u);
        else throw std::runtime_error("invalid packed-exponent QK variant");
#undef QRT_PACKED_EXPONENT_CASE
        check(hipGetLastError());
        hipLaunchKernelGGL(qrt_deferred_qk_fallback::replay_scan,
            dim3((size_t(count)*kQueryHeads*stride+255u)/256u),dim3(256u),0u,nullptr,q,k,out,start,count,stride,key_stride);
    }
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
    if(c.mode==8u){
        for(size_t i=guard;i+guard<q.size();++i)q[i]=uint16_t(((i-guard)%16u==0u?190u:64u)<<7u|((i&1u)?0x8000u:0u));
        for(size_t i=guard;i+guard<k.size();++i)k[i]=uint16_t(((i-guard)%16u==1u?190u:64u)<<7u|((i&2u)?0x8000u:0u));
    }
    if(c.mode==9u){
        for(size_t i=guard;i+guard<q.size();++i)q[i]=uint16_t(64u<<7u|((i&1u)?0x8000u:0u));
        for(size_t i=guard;i+guard<k.size();++i)k[i]=uint16_t(64u<<7u|((i&2u)?0x8000u:0u));
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
                throw std::runtime_error("packed-exponent QK differs from original scores");
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
        std::printf("{\"kind\":\"packed_exponent_qk_safety\",\"tokens\":%u,\"query_start\":%u,\"query_count\":%u,\"mode\":%u,\"variant\":%u,\"cells\":%zu,\"cpu_dots\":64,\"raw_bit_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",c.tokens,c.start,c.count,c.mode,variant,cells);
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
    size_t flagged_tiles[variant_count]{};
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
            }
            flagged_tiles[mode]+=scratch.verify(variant,stride,count);
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
    for(unsigned mode=0u;mode<variant_count;++mode) {
        const unsigned variant=variants[mode];
        double sorted[3]={samples[mode][0],samples[mode][1],samples[mode][2]};std::sort(sorted,sorted+3);
        const double preparation_ms=variant>=3u?prepared.compact.ms:prepared.ms;
        std::printf("{\"kind\":\"packed_exponent_qk_capture\",\"variant\":%u,\"query_rows\":16,\"key_columns\":16,\"unused_flag_tiles\":%zu,\"tokens\":%u,\"original_capture_tokens\":7169,\"real_model_prompt\":false,\"query_batch\":128,\"unique_score_cells\":%zu,\"compared_score_cells\":%zu,\"cpu_dots\":%u,\"raw_bit_mismatches\":0,\"one_time_preparation_ms\":%.6f,\"completed_query_with_replay_ms\":%.6f,\"completed_total_ms\":%.6f,\"completed_query_samples_ms\":[%.6f,%.6f,%.6f],\"warmup_per_slab\":1,\"samples_per_slab\":3,\"maximum_completed_slab_ms\":%.6f,\"all_attempts_verified\":true,\"redzones_pass\":true,\"unused_score_tail_pass\":true,\"immutable_inputs\":true,\"complete_cpu_encoding_check\":true,\"unused_auxiliary_guards_pass\":true,\"complete_fallback_timing_included\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            variant,flagged_tiles[mode],tokens,compared/variant_count/attempts,compared/variant_count,cpu_dots/variant_count,
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
            {17,0,9,5},{67,33,32,5},{33,1,31,6},{33,0,16,7},{129,1,128,5},{129,1,128,7},{33,1,31,8},{35,3,17,9}};
        for(auto c:cases)run(c);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
