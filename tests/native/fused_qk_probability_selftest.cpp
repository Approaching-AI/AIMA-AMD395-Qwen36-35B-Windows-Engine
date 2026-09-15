#include "../../native/providers/ck_fmha/prepared_decoded_qk.h"
#include "../../native/providers/ck_fmha/fused_qk_probability.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace qrt_blackwell_attention;
constexpr unsigned guard = 64u;
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

void separate(const uint16_t* q, const uint16_t* k, Prepared& prepared,
    float* scores, uint16_t* p, float* s, unsigned start, unsigned count,
    unsigned stride, unsigned tokens, const unsigned char* table, bool vllm, bool original) {
    if (original) {
        hipLaunchKernelGGL(blackwell_tiled_exact_scores_kernel,
            dim3((stride+31u)/32u,kQueryHeads,(count+7u)/8u),dim3(kThreads),0u,nullptr,
            q,k,scores,start,count,stride,tokens);
    } else {
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::scores<128u,true>),
            dim3((stride+15u)/16u,kQueryHeads,(count+15u)/16u),dim3(kThreads),0u,nullptr,
            q,k,prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,
            prepared.qf.as<unsigned>()+guard,prepared.kf.as<unsigned>()+guard,
            scores,start,count,stride,tokens);
    }
    check(hipGetLastError());
    hipLaunchKernelGGL(blackwell_online_probability_kernel,dim3(kQueryHeads,count),dim3(32u),0u,nullptr,
        scores,p,s,start,stride,table,vllm);
    check(hipGetLastError());
}
void fused(const uint16_t* q, const uint16_t* k, Prepared& prepared,
    float* scores, uint16_t* p, float* s, unsigned start, unsigned count,
    unsigned stride, unsigned tokens, const unsigned char* table, bool vllm) {
    hipLaunchKernelGGL((qrt_fused_qk_probability::run<8u>),
        dim3(kQueryHeads,(count+7u)/8u),dim3(kThreads),0u,nullptr,
        q,k,prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,
        prepared.qf.as<unsigned>()+guard,prepared.kf.as<unsigned>()+guard,
        p,s,scores,start,count,stride,tokens,table,vllm,0u);
    check(hipGetLastError());
}
template<class T> size_t differences(const std::vector<T>& a, const std::vector<T>& b) {
    if(a.size()!=b.size())throw std::runtime_error("comparison size differs");
    size_t bad=0u;
    for(size_t i=0u;i<a.size();++i)bad+=std::memcmp(&a[i],&b[i],sizeof(T))!=0;
    return bad;
}
struct Outputs {
    size_t cells, scale_cells;
    Device scores, probability, scales;
    Outputs(unsigned count,unsigned stride):cells(size_t(count)*kQueryHeads*stride),
        scale_cells(size_t(count)*kQueryHeads*((stride+31u)/32u+1u)),
        scores((cells+2u*guard)*4u),probability((cells+2u*guard)*2u),scales((scale_cells+2u*guard)*4u) { reset(); }
    void reset() {
        check(hipMemset(scores.pointer,0xa5,(cells+2u*guard)*4u));
        check(hipMemset(probability.pointer,0xa5,(cells+2u*guard)*2u));
        check(hipMemset(scales.pointer,0xa5,(scale_cells+2u*guard)*4u));
    }
};

void safety(unsigned tokens,unsigned start,unsigned count,unsigned mode,bool vllm,const unsigned char* table) {
    const unsigned stride=start+count;
    std::vector<uint16_t> q(size_t(tokens)*kQueryHeads*kHeadDim+2u*guard,0x5a5au);
    std::vector<uint16_t> k(size_t(tokens)*kKvHeads*kHeadDim+2u*guard,0x5a5au);
    for(size_t i=guard;i+guard<q.size();++i)
        q[i]=uint16_t(((i*37u+i/19u)&0x807fu)|((123u+i%8u)<<7u));
    for(size_t i=guard;i+guard<k.size();++i)
        k[i]=uint16_t(((i*53u+i/23u)&0x807fu)|((121u+i%10u)<<7u));
    if(mode==1u) {
        for(size_t i=guard;i+guard<q.size();i+=7u)q[i]=i%3u?0u:0x8000u;
        for(size_t i=guard;i+guard<k.size();i+=11u)k[i]=i%3u?0u:0x8000u;
    }
    if(mode==2u) {
        const uint16_t edges[]={1u,0x8001u,0x007fu,0x807fu,uint16_t(63u<<7u|19u),uint16_t(192u<<7u|11u),0u,0x8000u};
        for(unsigned i=0u;i<8u;++i) {
            q[guard+size_t(start)*kQueryHeads*kHeadDim+i]=edges[i];
            k[guard+i]=edges[7u-i];
        }
    }
    if(mode==3u) {
        for(size_t i=guard;i+guard<q.size();++i)q[i]=uint16_t(127u<<7u|127u);
        for(size_t i=guard;i+guard<k.size();++i)k[i]=uint16_t(127u<<7u|127u|((i/16u)&1u?0x8000u:0u));
    }
    if(mode==4u)for(size_t i=guard;i+guard<q.size();++i)q[i]=i%2u?0u:0x8000u;
    std::vector<uint16_t> transposed(k.size(),0x5a5au);
    for(unsigned token=0u;token<tokens;++token)
        for(unsigned feature=0u;feature<kKvHeads*kHeadDim;++feature)
            transposed[guard+size_t(feature)*tokens+token]=k[guard+size_t(token)*kKvHeads*kHeadDim+feature];
    Device dq(q.size()*2u),dk(k.size()*2u),dt(k.size()*2u);
    upload(dq,q);upload(dk,k);upload(dt,transposed);
    Prepared prepared(dq.as<uint16_t>()+guard,dk.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,
        q.data()+guard,k.data()+guard,tokens);
    Outputs expected(count,stride),actual(count,stride);
    separate(dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,prepared,expected.scores.as<float>()+guard,
        expected.probability.as<uint16_t>()+guard,expected.scales.as<float>()+guard,start,count,stride,tokens,table,vllm,true);
    finish();
    const auto scores=download<uint32_t>(expected.scores,expected.cells+2u*guard);
    const auto probabilities=download<uint16_t>(expected.probability,expected.cells+2u*guard);
    const auto scales=download<uint32_t>(expected.scales,expected.scale_cells+2u*guard);
    for(unsigned variant=0u;variant<3u;++variant) {
        actual.reset();
        if(!variant)separate(dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,prepared,actual.scores.as<float>()+guard,
            actual.probability.as<uint16_t>()+guard,actual.scales.as<float>()+guard,start,count,stride,tokens,table,vllm,false);
        else fused(dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,prepared,variant==1u?actual.scores.as<float>()+guard:nullptr,
            actual.probability.as<uint16_t>()+guard,actual.scales.as<float>()+guard,start,count,stride,tokens,table,vllm);
        finish();
        const size_t bad_scores=variant<2u?differences(scores,download<uint32_t>(actual.scores,actual.cells+2u*guard)):0u;
        const size_t bad_p=differences(probabilities,download<uint16_t>(actual.probability,actual.cells+2u*guard));
        const size_t bad_s=differences(scales,download<uint32_t>(actual.scales,actual.scale_cells+2u*guard));
        if(variant==2u)unchanged(actual.scores,std::vector<uint32_t>(actual.cells+2u*guard,0xa5a5a5a5u));
        if(bad_scores || bad_p || bad_s) {
            std::fprintf(stderr,"start=%u count=%u mode=%u variant=%u scores=%zu probability=%zu scales=%zu\n",
                start,count,mode,variant,bad_scores,bad_p,bad_s);
            throw std::runtime_error("fused QK/probability differs from original");
        }
    }
    for(size_t i=0u;i<scores.size();++i) {
        const bool guarded=i<guard || i>=guard+expected.cells;
        if(guarded && (scores[i]!=0xa5a5a5a5u || probabilities[i]!=0xa5a5u))
            throw std::runtime_error("score/probability redzone changed");
        if(!guarded) {
            const unsigned row=unsigned((i-guard)/stride),key=unsigned((i-guard)%stride);
            const unsigned causal=start+row/kQueryHeads+1u;
            if(key>=causal && scores[i]!=bits(-INFINITY))throw std::runtime_error("score mask changed");
            if(key>=((causal+31u)/32u)*32u && probabilities[i]!=0xa5a5u)
                throw std::runtime_error("probability unused tail changed");
            if(mode==4u && key<((causal+31u)/32u)*32u && probabilities[i]!=(key<causal?0x3f80u:0u))
                throw std::runtime_error("constant probability CPU check");
        }
    }
    const unsigned tile_stride=(stride+31u)/32u;
    for(size_t i=0u;i<scales.size();++i) {
        const bool guarded=i<guard || i>=guard+expected.scale_cells;
        if(guarded && scales[i]!=0xa5a5a5a5u)throw std::runtime_error("scale redzone changed");
        if(!guarded) {
            const unsigned row=unsigned((i-guard)/(tile_stride+1u)),tile=unsigned((i-guard)%(tile_stride+1u));
            const unsigned causal=start+row/kQueryHeads+1u;
            if(tile<tile_stride && tile>=(causal+31u)/32u && scales[i]!=0xa5a5a5a5u)
                throw std::runtime_error("scale unused tail changed");
            if(mode==4u && (tile<(causal+31u)/32u || tile==tile_stride) &&
                scales[i]!=bits(tile==tile_stride?float(causal):tile==0u?0.0f:1.0f))
                throw std::runtime_error("constant scale/denominator CPU check");
        }
    }
    for(unsigned sample=0u;sample<64u;++sample) {
        const unsigned row=sample%count,head=(sample/4u)%kQueryHeads;
        const unsigned key=sample%2u?(sample*797u)%(start+row+1u):0u;
        const auto* left=q.data()+guard+(size_t(start+row)*kQueryHeads+head)*kHeadDim;
        const auto* right=k.data()+guard+(size_t(key)*kKvHeads+head/(kQueryHeads/kKvHeads))*kHeadDim;
        const float value=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,left,right,kHeadDim)*kExactScale;
        if(bits(value)!=scores[guard+(size_t(row)*kQueryHeads+head)*stride+key])
            throw std::runtime_error("score differs from independent wide CPU accumulator");
    }
    unchanged(dq,q);unchanged(dk,k);unchanged(dt,transposed);prepared.verify();
    std::printf("{\"kind\":\"fused_qk_probability_safety\",\"tokens\":%u,\"query_start\":%u,\"query_count\":%u,\"mode\":%u,\"vllm_sum\":%s,\"sm121_exp2_table\":%s,\"score_cells\":%zu,\"scale_cells\":%zu,\"cpu_dots\":64,\"variants\":3,\"raw_bit_mismatches\":0,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"production_and_diagnostic_agree\":true,\"inference_acceptance\":false}\n",
        tokens,start,count,mode,vllm?"true":"false",table?"true":"false",expected.cells,expected.scale_cells);
    std::fflush(stdout);
}

std::vector<uint16_t> read_words(const char* name,size_t words) {
    std::ifstream file(name,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()!=std::streamoff(words*2u))throw std::runtime_error("capture size differs");
    std::vector<uint16_t> result(words);file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(result.data()),std::streamsize(words*2u)))throw std::runtime_error("capture read failed");
    return result;
}
__global__ void compare_words(const unsigned char* expected,const unsigned char* actual,size_t bytes,unsigned* bad) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<bytes && expected[index]!=actual[index])atomicAdd(bad,1u);
}
__global__ void compare_sentinel(const unsigned char* actual,size_t bytes,unsigned* bad) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<bytes && actual[index]!=0xa5u)atomicAdd(bad,1u);
}

void captured(const char* qfile,const char* kfile,unsigned tokens,const unsigned char* table) {
    constexpr unsigned source_tokens=7169u,batch=128u;
    auto q=read_words(qfile,size_t(source_tokens)*kQueryHeads*kHeadDim);
    auto k=read_words(kfile,size_t(source_tokens)*kKvHeads*kHeadDim);
    if(tokens!=source_tokens) {
        if(tokens!=8192u)throw std::runtime_error("unsupported captured geometry");
        const auto original_q=q,original_k=k;
        q.insert(q.end(),original_q.begin(),original_q.begin()+size_t(tokens-source_tokens)*kQueryHeads*kHeadDim);
        k.insert(k.end(),original_k.begin(),original_k.begin()+size_t(tokens-source_tokens)*kKvHeads*kHeadDim);
    }
    q.insert(q.begin(),guard,0x5a5au);q.insert(q.end(),guard,0x5a5au);
    k.insert(k.begin(),guard,0x5a5au);k.insert(k.end(),guard,0x5a5au);
    std::vector<uint16_t> transposed(k.size(),0x5a5au);
    for(unsigned token=0u;token<tokens;++token)
        for(unsigned feature=0u;feature<kKvHeads*kHeadDim;++feature)
            transposed[guard+size_t(feature)*tokens+token]=k[guard+size_t(token)*kKvHeads*kHeadDim+feature];
    Device dq(q.size()*2u),dk(k.size()*2u),dt(k.size()*2u),bad(4u);
    upload(dq,q);upload(dk,k);upload(dt,transposed);check(hipMemset(bad.pointer,0,4u));
    Prepared prepared(dq.as<uint16_t>()+guard,dk.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,
        q.data()+guard,k.data()+guard,tokens);
    Outputs reference(batch,tokens),actual(batch,tokens);
    double samples[2][3]{},maximum_slab[2]{};
    size_t unique_scores=0u,unique_scales=0u;unsigned cpu_dots=0u;
    for(unsigned start=0u;start<tokens;start+=batch) {
        const unsigned count=std::min(batch,tokens-start),stride=start+count;
        reference.reset();
        separate(dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,prepared,reference.scores.as<float>()+guard,
            reference.probability.as<uint16_t>()+guard,reference.scales.as<float>()+guard,
            start,count,stride,tokens,table,true,true);
        finish();
        unique_scores+=size_t(count)*kQueryHeads*stride;
        unique_scales+=size_t(count)*kQueryHeads*((stride+31u)/32u+1u);
        for(unsigned position=0u;position<2u;++position) {
            const unsigned variant=(position+start/batch)%2u;
            for(unsigned attempt=0u;attempt<4u;++attempt) {
                actual.reset();finish();
                const auto begin=std::chrono::steady_clock::now();
                if(!variant)separate(dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,prepared,actual.scores.as<float>()+guard,
                    actual.probability.as<uint16_t>()+guard,actual.scales.as<float>()+guard,
                    start,count,stride,tokens,table,true,false);
                else fused(dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,prepared,nullptr,
                    actual.probability.as<uint16_t>()+guard,actual.scales.as<float>()+guard,
                    start,count,stride,tokens,table,true);
                finish();const double wall=elapsed(begin);
                if(attempt) { samples[variant][attempt-1u]+=wall;maximum_slab[variant]=std::max(maximum_slab[variant],wall); }
                for(unsigned surface=0u;surface<3u;++surface) {
                    if(variant && surface==0u) {
                        const size_t bytes=(actual.cells+2u*guard)*4u;
                        hipLaunchKernelGGL(compare_sentinel,dim3((bytes+255u)/256u),dim3(256u),0u,nullptr,
                            actual.scores.as<unsigned char>(),bytes,bad.as<unsigned>());
                        check(hipGetLastError());continue;
                    }
                    auto& left=surface==0u?reference.scores:surface==1u?reference.probability:reference.scales;
                    auto& right=surface==0u?actual.scores:surface==1u?actual.probability:actual.scales;
                    const size_t bytes=surface==2u?(reference.scale_cells+2u*guard)*4u:
                        (reference.cells+2u*guard)*(surface==1u?2u:4u);
                    hipLaunchKernelGGL(compare_words,dim3((bytes+255u)/256u),dim3(256u),0u,nullptr,
                        left.as<unsigned char>(),right.as<unsigned char>(),bytes,bad.as<unsigned>());
                    check(hipGetLastError());
                }
                finish();
                if(download<unsigned>(bad,1u)[0])throw std::runtime_error("captured probability/scales or unused tail differs");
            }
        }
        for(unsigned sample=0u;sample<4u;++sample) {
            const unsigned row=sample*(count-1u)/3u,head=(start/batch+sample*5u)%kQueryHeads;
            const unsigned key=(start+row)*sample/3u;
            const auto* left=q.data()+guard+(size_t(start+row)*kQueryHeads+head)*kHeadDim;
            const auto* right=k.data()+guard+(size_t(key)*kKvHeads+head/(kQueryHeads/kKvHeads))*kHeadDim;
            const float expected=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,left,right,kHeadDim)*kExactScale;
            uint32_t actual_score;
            check(hipMemcpy(&actual_score,reference.scores.as<uint32_t>()+guard+
                (size_t(row)*kQueryHeads+head)*stride+key,4u,hipMemcpyDeviceToHost));
            if(actual_score!=bits(expected))throw std::runtime_error("captured reference score differs from CPU");
            ++cpu_dots;
        }
    }
    unchanged(dq,q);unchanged(dk,k);unchanged(dt,transposed);prepared.verify();
    for(unsigned variant=0u;variant<2u;++variant) {
        auto sorted=std::vector<double>(samples[variant],samples[variant]+3u);std::sort(sorted.begin(),sorted.end());
        std::printf("{\"kind\":\"fused_qk_probability_capture\",\"source_capture_tokens\":7169,\"tokens\":%u,\"repeated_prefix_rows\":%u,\"fused\":%s,\"unique_score_probability_cells\":%zu,\"unique_scale_cells\":%zu,\"cpu_dots\":%u,\"samples_ms\":[%.9f,%.9f,%.9f],\"median_completed_ms\":%.9f,\"shared_preparation_ms\":%.9f,\"preparation_plus_median_ms\":%.9f,\"maximum_completed_slab_ms\":%.9f,\"warmups_per_slab\":1,\"timed_attempts_per_slab\":3,\"all_attempts_compared\":true,\"score_matrix_materialized\":%s,\"raw_bit_mismatches\":0,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
            tokens,tokens-source_tokens,variant?"true":"false",unique_scores,unique_scales,cpu_dots,
            samples[variant][0],samples[variant][1],samples[variant][2],sorted[1],prepared.ms,prepared.ms+sorted[1],
            maximum_slab[variant],variant?"false":"true");
        std::fflush(stdout);
    }
}
} // namespace

int main(int argc,char** argv) try {
    if(argc!=3 && argc!=5)throw std::runtime_error("requires --selftest TABLE or --q7169/--q8192 TABLE Q K");
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0));
    if(std::strncmp(properties.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    std::ifstream file(argv[2],std::ios::binary|std::ios::ate);
    if(!file || file.tellg()!=std::streamoff(exp2_backend::table_bytes))throw std::runtime_error("exp2 table size mismatch");
    std::vector<unsigned char> table(exp2_backend::table_bytes);file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(table.data()),table.size()) || !exp2_backend::valid_layout(table.data(),table.size()))
        throw std::runtime_error("exp2 table read/layout failure");
    Device dt(table.size());upload(dt,table);
    if(argc==3 && !std::strcmp(argv[1],"--selftest")) {
        for(auto shape:{std::pair<unsigned,unsigned>{0u,1u},{0u,7u},{7u,2u},{31u,2u},{17u,32u},
                       {511u,65u},{7167u,2u},{8191u,1u},{8064u,128u},{0u,128u}})
            for(unsigned mode=0u;mode<5u;++mode)
                for(bool vllm:{false,true})for(bool use_table:{false,true})
                    safety(shape.first+shape.second,shape.first,shape.second,mode,vllm,use_table?dt.as<unsigned char>():nullptr);
    } else if(argc==5 && (!std::strcmp(argv[1],"--q7169") || !std::strcmp(argv[1],"--q8192"))) {
        captured(argv[3],argv[4],!std::strcmp(argv[1],"--q7169")?7169u:8192u,dt.as<unsigned char>());
    } else throw std::runtime_error("invalid component arguments");
    unchanged(dt,table);
    return 0;
} catch(const std::exception& error) {
    std::fprintf(stderr,"fused_qk_probability_error=%s\n",error.what());return 2;
}
