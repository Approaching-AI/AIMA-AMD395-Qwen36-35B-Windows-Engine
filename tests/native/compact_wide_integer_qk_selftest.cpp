#include "../../native/providers/ck_fmha/wide_integer_qk.h"
#include "../../native/providers/ck_fmha/prepared_decoded_qk.h"
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
using Row = qrt_wide_integer_qk::Row;
using CompactRow = qrt_wide_integer_qk::CompactRow;
constexpr unsigned guard = 64u;
constexpr unsigned variants[] = {132u,68u};
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
template<IntegerRowKind Kind> void prepare(const uint16_t* input, Row* output,
    unsigned tokens, unsigned start, unsigned count) {
    const size_t rows = integer_row_count(Kind,tokens,count);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_wide_integer_qk::prepare_rows<Kind>),
        dim3((rows+kThreads-1u)/kThreads),dim3(kThreads),0u,nullptr,
        input,output,tokens,start,count);
    check(hipGetLastError());
}
template<IntegerRowKind Kind> void prepare_compact(const uint16_t* input, CompactRow* output,
    unsigned tokens, unsigned start, unsigned count) {
    const size_t rows=integer_row_count(Kind,tokens,count);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_wide_integer_qk::prepare_compact_rows<Kind>),
        dim3((rows+kThreads-1u)/kThreads),dim3(kThreads),0u,nullptr,input,output,tokens,start,count);
    check(hipGetLastError());
}
void staged_variant(unsigned variant, const Row* q, const Row* k,
    const CompactRow* cq, const CompactRow* ck, float* out,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    const dim3 grid((stride+kIntegerMatrixColumns-1u)/kIntegerMatrixColumns,kQueryHeads,(count+15u)/16u);
    if(variant==132u) {
        hipLaunchKernelGGL(qrt_wide_integer_qk::scores,grid,dim3(kThreads),0u,nullptr,
            q,k,out,start,count,stride,key_stride);
    } else if(variant==68u) {
        hipLaunchKernelGGL(qrt_wide_integer_qk::compact_scores,grid,dim3(kThreads),0u,nullptr,
            cq,ck,out,start,count,stride,key_stride);
    } else throw std::runtime_error("invalid compact wide variant");
    check(hipGetLastError());
}
void original(const uint16_t* q, const uint16_t* k, float* out,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    hipLaunchKernelGGL(blackwell_tiled_exact_scores_kernel,
        dim3((stride+31u)/32u,kQueryHeads,(count+7u)/8u),dim3(kThreads),0u,nullptr,
        q,k,out,start,count,stride,key_stride);
    check(hipGetLastError());
}
void byte_guards(const std::vector<unsigned char>& data, size_t body_bytes) {
    for (size_t i=0u;i<guard;++i)
        if(data[i]!=0xa5u || data[guard+body_bytes+i]!=0xa5u)
            throw std::runtime_error("encoded operand redzone changed");
}
struct Case { unsigned tokens, start, count, mode; };
void run(Case c) {
    const unsigned stride = c.start+c.count;
    const size_t cells = size_t(c.count)*kQueryHeads*stride;
    const size_t qrows = integer_row_count(IntegerRowKind::Query,c.tokens,c.count);
    const size_t krows = integer_row_count(IntegerRowKind::Key,c.tokens,c.count);
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
    std::vector<uint16_t> transposed(k.size(),0x5a5au);
    for(unsigned token=0u;token<c.tokens;++token)
        for(unsigned feature=0u;feature<kKvHeads*kHeadDim;++feature)
            transposed[guard+size_t(feature)*c.tokens+token]=k[guard+size_t(token)*kKvHeads*kHeadDim+feature];
    Device dq(q.size()*2u),dk(k.size()*2u),dt(transposed.size()*2u);
    Device qp(qrows*sizeof(Row)+2u*guard),kp(krows*sizeof(Row)+2u*guard);
    Device cqp(qrows*sizeof(CompactRow)+2u*guard),ckp(krows*sizeof(CompactRow)+2u*guard);
    Device reference((cells+2u*guard)*4u),candidate((cells+2u*guard)*4u);
    auto* prepared_q=reinterpret_cast<Row*>(qp.as<unsigned char>()+guard);
    auto* prepared_k=reinterpret_cast<Row*>(kp.as<unsigned char>()+guard);
    auto* compact_q=reinterpret_cast<CompactRow*>(cqp.as<unsigned char>()+guard);
    auto* compact_k=reinterpret_cast<CompactRow*>(ckp.as<unsigned char>()+guard);
    upload(dq,q);upload(dk,k);upload(dt,transposed);
    check(hipMemset(qp.pointer,0xa5,qrows*sizeof(Row)+2u*guard));
    check(hipMemset(kp.pointer,0xa5,krows*sizeof(Row)+2u*guard));
    check(hipMemset(cqp.pointer,0xa5,qrows*sizeof(CompactRow)+2u*guard));
    check(hipMemset(ckp.pointer,0xa5,krows*sizeof(CompactRow)+2u*guard));
    prepare_compact<IntegerRowKind::Key>(dk.as<uint16_t>()+guard,compact_k,c.tokens,0u,0u);
    prepare_compact<IntegerRowKind::Query>(dq.as<uint16_t>()+guard,compact_q,c.tokens,c.start,c.count);
    prepare<IntegerRowKind::Key>(dk.as<uint16_t>()+guard,prepared_k,c.tokens,0u,0u);
    prepare<IntegerRowKind::Query>(dq.as<uint16_t>()+guard,prepared_q,c.tokens,c.start,c.count);
    finish();
    const auto qbefore=download<unsigned char>(qp,qrows*sizeof(Row)+2u*guard);
    const auto kbefore=download<unsigned char>(kp,krows*sizeof(Row)+2u*guard);
    byte_guards(qbefore,qrows*sizeof(Row));byte_guards(kbefore,krows*sizeof(Row));
    const auto cqbefore=download<unsigned char>(cqp,qrows*sizeof(CompactRow)+2u*guard);
    const auto ckbefore=download<unsigned char>(ckp,krows*sizeof(CompactRow)+2u*guard);
    byte_guards(cqbefore,qrows*sizeof(CompactRow));byte_guards(ckbefore,krows*sizeof(CompactRow));
    for(unsigned side=0u;side<2u;++side) {
        const auto& full=side?kbefore:qbefore;const auto& compact=side?ckbefore:cqbefore;
        for(size_t row=0;row<(side?krows:qrows);++row) {
            CompactRow packed;std::memcpy(&packed,compact.data()+guard+row*sizeof(CompactRow),sizeof(packed));
            const auto expanded=qrt_sm121_compact_wide_core::expand(packed);
            if(std::memcmp(&expanded,full.data()+guard+row*sizeof(Row),sizeof(Row)))
                throw std::runtime_error("GPU compact rows do not expand to original wide rows");
        }
    }
    // Check independently prepared CPU rows, including first and final rows.
    for(unsigned sample=0u;sample<16u;++sample)for(unsigned side=0u;side<2u;++side) {
        const auto kind=side?IntegerRowKind::Key:IntegerRowKind::Query;
        const size_t count=side?krows:qrows, row=size_t(sample)*(count-1u)/15u;
        Row expected{};
        for(unsigned col=0u;col<16u;++col) {
            const size_t index=integer_row_input_index(kind,row,col,c.tokens,c.start,c.count);
            expected.original[col]=(side?k:q)[guard+index];
        }
        qrt_sm121_wide_core::prepare(expected);
        const auto& before=side?kbefore:qbefore;
        if(std::memcmp(&expected,before.data()+guard+row*sizeof(Row),sizeof(Row)))
            throw std::runtime_error("prepared row differs from CPU encoding");
    }
    check(hipMemset(reference.pointer,0xa5,(cells+2u*guard)*4u));
    original(dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,reference.as<float>()+guard,c.start,c.count,stride,c.tokens);
    finish();const auto a=download<uint32_t>(reference,cells+2u*guard);
    for(unsigned variant:variants) {
        const unsigned groups=variant%100u;
        check(hipMemset(candidate.pointer,0xa5,(cells+2u*guard)*4u));
        staged_variant(variant,prepared_q,prepared_k,compact_q,compact_k,candidate.as<float>()+guard,c.start,c.count,stride,c.tokens);
        finish();const auto b=download<uint32_t>(candidate,cells+2u*guard);
        for(size_t i=0u;i<a.size();++i) {
            if(i<guard || i>=cells+guard) {
                if(a[i]!=0xa5a5a5a5u || b[i]!=0xa5a5a5a5u)throw std::runtime_error("QK score redzone changed");
            }else if(a[i]!=b[i]) {
                std::fprintf(stderr,"groups=%u tokens=%u start=%u count=%u mode=%u index=%zu expected=%08x actual=%08x\n",groups,c.tokens,c.start,c.count,c.mode,i-guard,a[i],b[i]);
                throw std::runtime_error("staged integer QK differs from original scores");
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
        unchanged(qp,qbefore);unchanged(kp,kbefore);unchanged(cqp,cqbefore);unchanged(ckp,ckbefore);
        std::printf("{\"kind\":\"compact_wide_integer_qk_safety\",\"core_bits\":18,\"row_bytes\":%u,\"tokens\":%u,\"query_start\":%u,\"query_count\":%u,\"mode\":%u,\"cells\":%zu,\"cpu_dots\":64,\"raw_bit_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",variant,c.tokens,c.start,c.count,c.mode,cells);
        std::fflush(stdout);
    }
    unchanged(dq,q);unchanged(dk,k);unchanged(dt,transposed);
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
void captured(const char* qfile,const char* kfile) {
    constexpr unsigned tokens=7169u,batch=128u;
    const auto q=read_words(qfile,size_t(tokens)*kQueryHeads*kHeadDim);
    const auto k=read_words(kfile,size_t(tokens)*kKvHeads*kHeadDim);
    const size_t capacity=size_t(batch)*kQueryHeads*tokens;
    const size_t qrows=integer_row_count(IntegerRowKind::Query,tokens,batch);
    const size_t krows=integer_row_count(IntegerRowKind::Key,tokens,batch);
    Device dq(q.size()*2u),dk(k.size()*2u),dt(k.size()*2u);
    Device qp(qrows*sizeof(Row)),kp(krows*sizeof(Row));
    Device cqp(qrows*sizeof(CompactRow)),ckp(krows*sizeof(CompactRow));
    Device decoded(qrt_prepared_decoded_qk::workspace_words*sizeof(uint32_t));
    qrt_prepared_decoded_qk::Workspace selected{decoded.as<uint32_t>(),tokens};
    Device reference((capacity+2u*guard)*4u),candidate((capacity+2u*guard)*4u),bad(4u);
    upload(dq,q);upload(dk,k);
    check(hipMemset(reference.pointer,0xa5,(capacity+2u*guard)*4u));
    check(hipMemset(candidate.pointer,0xa5,(capacity+2u*guard)*4u));
    check(hipMemset(bad.pointer,0,4u));
    auto begin=std::chrono::steady_clock::now();
    check(hipError_t(transpose_keys(dk.as<uint16_t>(),dt.as<uint16_t>(),k.size(),tokens,nullptr)));
    finish();const double transpose_ms=elapsed(begin);
    begin=std::chrono::steady_clock::now();
    prepare<IntegerRowKind::Key>(dk.as<uint16_t>(),kp.as<Row>(),tokens,0u,0u);
    finish();const double key_encoding_ms=elapsed(begin);
    begin=std::chrono::steady_clock::now();
    prepare_compact<IntegerRowKind::Key>(dk.as<uint16_t>(),ckp.as<CompactRow>(),tokens,0u,0u);
    finish();const double compact_key_encoding_ms=elapsed(begin);
    begin=std::chrono::steady_clock::now();
    check(hipError_t(qrt_prepared_decoded_qk::prepare_workspace(dq.as<uint16_t>(),dk.as<uint16_t>(),dt.as<uint16_t>(),selected,nullptr)));
    finish();const double selected_preparation_ms=elapsed(begin);
    const auto key_before=download<unsigned char>(kp,krows*sizeof(Row));
    const auto compact_key_before=download<unsigned char>(ckp,krows*sizeof(CompactRow));
    const auto transposed_before=download<uint16_t>(dt,k.size());
    double selected_query_ms=0.0;
    double original_ms=0.0,staged_ms[variant_count]{},maximum_stage_ms[variant_count]{};
    size_t compared=0u;unsigned cpu_dots=0u;
    for(unsigned start=0u;start<tokens;start+=batch) {
        const unsigned count=std::min(batch,tokens-start),stride=start+count;
        const size_t cells=size_t(count)*kQueryHeads*stride;
        begin=std::chrono::steady_clock::now();
        original(dq.as<uint16_t>(),dt.as<uint16_t>(),reference.as<float>()+guard,start,count,stride,tokens);
        finish();original_ms+=elapsed(begin);
        check(hipMemset(candidate.pointer,0xa5,(capacity+2u*guard)*4u));
        begin=std::chrono::steady_clock::now();
        check(hipError_t(qrt_prepared_decoded_qk::launch_workspace(&selected,dq.as<uint16_t>(),dt.as<uint16_t>(),candidate.as<float>()+guard,nullptr,start,count,stride,tokens)));
        finish();selected_query_ms+=elapsed(begin);
        hipLaunchKernelGGL(compare_scores,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,
            reference.as<uint32_t>()+guard,candidate.as<uint32_t>()+guard,cells,bad.as<unsigned>());
        check(hipGetLastError());finish();
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("selected prepared QK differs from original scores");
        for(unsigned mode=0u;mode<variant_count;++mode) {
            check(hipMemset(candidate.pointer,0xa5,(capacity+2u*guard)*4u));
            begin=std::chrono::steady_clock::now();
            if(variants[mode]==68u)prepare_compact<IntegerRowKind::Query>(dq.as<uint16_t>(),cqp.as<CompactRow>(),tokens,start,count);
            else prepare<IntegerRowKind::Query>(dq.as<uint16_t>(),qp.as<Row>(),tokens,start,count);
            staged_variant(variants[mode],qp.as<Row>(),kp.as<Row>(),cqp.as<CompactRow>(),ckp.as<CompactRow>(),candidate.as<float>()+guard,start,count,stride,tokens);
            finish();const double wall=elapsed(begin);staged_ms[mode]+=wall;
            maximum_stage_ms[mode]=std::max(maximum_stage_ms[mode],wall);
            hipLaunchKernelGGL(compare_scores,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,
                reference.as<uint32_t>()+guard,candidate.as<uint32_t>()+guard,cells,bad.as<unsigned>());
            check(hipGetLastError());finish();compared+=cells;
            if(download<unsigned>(bad,1u)[0])throw std::runtime_error("captured QK differs from original scores");
            for(unsigned sample=0u;sample<4u;++sample) {
                const unsigned row=sample*(count-1u)/3u,head=(start/128u+sample*5u)%kQueryHeads;
                const unsigned key=(start+row)*sample/3u;
                const float expected=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
                    q.data()+(size_t(start+row)*kQueryHeads+head)*kHeadDim,
                    k.data()+(size_t(key)*kKvHeads+head/(kQueryHeads/kKvHeads))*kHeadDim,kHeadDim)*kExactScale;
                uint32_t actual;
                check(hipMemcpy(&actual,candidate.as<uint32_t>()+guard+(size_t(row)*kQueryHeads+head)*stride+key,4u,hipMemcpyDeviceToHost));
                if(bits(expected)!=actual)throw std::runtime_error("captured QK differs from wide CPU sum");
                ++cpu_dots;
            }
        }
    }
    for(Device* d:{&reference,&candidate})for(size_t offset:{size_t(0u),capacity+guard}) {
        std::vector<uint32_t> words(guard);
        check(hipMemcpy(words.data(),d->as<uint32_t>()+offset,guard*4u,hipMemcpyDeviceToHost));
        for(auto word:words)if(word!=0xa5a5a5a5u)throw std::runtime_error("captured QK redzone changed");
    }
    unchanged(dq,q);unchanged(dk,k);unchanged(dt,transposed_before);
    unchanged(kp,key_before);unchanged(ckp,compact_key_before);
    for(unsigned mode=0u;mode<variant_count;++mode) {
        const double encoding=variants[mode]==68u?compact_key_encoding_ms:key_encoding_ms;
        std::printf("{\"kind\":\"compact_wide_integer_original_q7169\",\"tokens\":7169,\"query_batch\":128,\"row_bytes\":%u,\"compared_score_cells\":%zu,\"cpu_dots\":%u,\"raw_bit_mismatches\":0,\"original_query_ms\":%.6f,\"key_transpose_ms\":%.6f,\"wide_query_and_encoding_ms\":%.6f,\"key_encoding_ms\":%.6f,\"selected_prepared_query_ms\":%.6f,\"selected_preparation_ms\":%.6f,\"maximum_completed_slab_ms\":%.6f,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            variants[mode],compared/variant_count,cpu_dots/variant_count,original_ms,transpose_ms,staged_ms[mode],encoding,selected_query_ms,selected_preparation_ms,maximum_stage_ms[mode]);
    }
}
}
int main(int argc,char** argv) {
    try {
        hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));
        if(std::strncmp(p.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
        if(argc==4 && !std::strcmp(argv[1],"--q7169")){captured(argv[2],argv[3]);return 0;}
        if(argc!=2 || std::strcmp(argv[1],"--selftest"))throw std::runtime_error("use --selftest or --q7169 Q K");
        const Case cases[]={{17,0,8,0},{17,0,9,2},{33,0,16,0},{33,1,31,1},{1,0,1,0},{9,0,9,1},{35,3,17,2},{67,33,32,3},{67,64,3,4},
            {129,1,128,0},{7169,7041,128,1},{8192,8064,128,0},{8192,8191,1,2},{8193,8191,2,3}};
        for(auto c:cases)run(c);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
