#include "../../native/providers/ck_fmha/blackwell_attention.h"
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
using Row = qrt_sm121_integer_core::Row;
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
template<IntegerRowKind Kind> void prepare(const uint16_t* input, Row* output,
    unsigned tokens, unsigned start, unsigned count) {
    const size_t rows = integer_row_count(Kind,tokens,count);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_prepare_integer_core_rows_kernel<Kind>),
        dim3((rows+kThreads-1u)/kThreads),dim3(kThreads),0u,nullptr,
        input,output,tokens,start,count);
    check(hipGetLastError());
}
template<unsigned Groups, bool CacheRows = false> void staged(const Row* q, const Row* k, float* out,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_staged_integer_scores_kernel<Groups, CacheRows>),
        dim3((stride+15u)/16u,kQueryHeads,(count+15u)/16u),dim3(kThreads),0u,nullptr,
        q,k,out,start,count,stride,key_stride);
    check(hipGetLastError());
}
void staged_variant(unsigned variant, const Row* q, const Row* k, float* out,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    if(variant==8u)staged<8u>(q,k,out,start,count,stride,key_stride);
    else if(variant==16u)staged<16u>(q,k,out,start,count,stride,key_stride);
    else if(variant==104u)staged<4u,true>(q,k,out,start,count,stride,key_stride);
    else if(variant==108u)staged<8u,true>(q,k,out,start,count,stride,key_stride);
    else throw std::runtime_error("invalid staging variant");
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
    Device reference((cells+2u*guard)*4u),candidate((cells+2u*guard)*4u);
    auto* prepared_q=reinterpret_cast<Row*>(qp.as<unsigned char>()+guard);
    auto* prepared_k=reinterpret_cast<Row*>(kp.as<unsigned char>()+guard);
    upload(dq,q);upload(dk,k);upload(dt,transposed);
    check(hipMemset(qp.pointer,0xa5,qrows*sizeof(Row)+2u*guard));
    check(hipMemset(kp.pointer,0xa5,krows*sizeof(Row)+2u*guard));
    prepare<IntegerRowKind::Key>(dk.as<uint16_t>()+guard,prepared_k,c.tokens,0u,0u);
    prepare<IntegerRowKind::Query>(dq.as<uint16_t>()+guard,prepared_q,c.tokens,c.start,c.count);
    finish();
    const auto qbefore=download<unsigned char>(qp,qrows*sizeof(Row)+2u*guard);
    const auto kbefore=download<unsigned char>(kp,krows*sizeof(Row)+2u*guard);
    byte_guards(qbefore,qrows*sizeof(Row));byte_guards(kbefore,krows*sizeof(Row));
    // Check independently prepared CPU rows, including first and final rows.
    for(unsigned sample=0u;sample<16u;++sample)for(unsigned side=0u;side<2u;++side) {
        const auto kind=side?IntegerRowKind::Key:IntegerRowKind::Query;
        const size_t count=side?krows:qrows, row=size_t(sample)*(count-1u)/15u;
        Row expected{};
        for(unsigned col=0u;col<16u;++col) {
            const size_t index=integer_row_input_index(kind,row,col,c.tokens,c.start,c.count);
            expected.original[col]=(side?k:q)[guard+index];
        }
        qrt_sm121_integer_core::prepare(expected);
        const auto& before=side?kbefore:qbefore;
        if(std::memcmp(&expected,before.data()+guard+row*sizeof(Row),sizeof(Row)))
            throw std::runtime_error("prepared row differs from CPU encoding");
    }
    check(hipMemset(reference.pointer,0xa5,(cells+2u*guard)*4u));
    original(dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,reference.as<float>()+guard,c.start,c.count,stride,c.tokens);
    finish();const auto a=download<uint32_t>(reference,cells+2u*guard);
    for(unsigned variant:{8u,16u,104u,108u}) {
        const unsigned groups=variant%100u;
        check(hipMemset(candidate.pointer,0xa5,(cells+2u*guard)*4u));
        staged_variant(variant,prepared_q,prepared_k,candidate.as<float>()+guard,c.start,c.count,stride,c.tokens);
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
        unchanged(qp,qbefore);unchanged(kp,kbefore);
        std::printf("{\"kind\":\"staged_integer_qk_safety\",\"groups_per_stage\":%u,\"shared_operand_rows\":%s,\"tokens\":%u,\"query_start\":%u,\"query_count\":%u,\"mode\":%u,\"cells\":%zu,\"cpu_dots\":64,\"raw_bit_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",groups,variant>100u?"true":"false",c.tokens,c.start,c.count,c.mode,cells);
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
    double sleep_minimum=1.0e30,sleep_maximum=0.0,sleep_total=0.0;
    for(unsigned i=0u;i<16u;++i) {
        const auto before=std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const double wall=elapsed(before);
        sleep_total+=wall;sleep_minimum=std::min(sleep_minimum,wall);sleep_maximum=std::max(sleep_maximum,wall);
    }
    std::fprintf(stderr,"HOST_WAIT_DIAGNOSTIC requested_sleep_ms=1 samples=16 minimum_ms=%.6f maximum_ms=%.6f mean_ms=%.6f measurement_polling=yield deadline_seconds=30\n",sleep_minimum,sleep_maximum,sleep_total/16.0);
    const auto q=read_words(qfile,size_t(tokens)*kQueryHeads*kHeadDim);
    const auto k=read_words(kfile,size_t(tokens)*kKvHeads*kHeadDim);
    const size_t capacity=size_t(batch)*kQueryHeads*tokens;
    const size_t qrows=integer_row_count(IntegerRowKind::Query,tokens,batch);
    const size_t krows=integer_row_count(IntegerRowKind::Key,tokens,batch);
    Device dq(q.size()*2u),dk(k.size()*2u),dt(k.size()*2u);
    Device qp(qrows*sizeof(Row)),kp(krows*sizeof(Row));
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
    double original_ms=0.0,staged_ms[4]{},maximum_stage_ms[4]{};
    const unsigned variants[]={8u,16u,104u,108u};
    size_t compared=0u;unsigned cpu_dots=0u;
    for(unsigned start=0u;start<tokens;start+=batch) {
        const unsigned count=std::min(batch,tokens-start),stride=start+count;
        const size_t cells=size_t(count)*kQueryHeads*stride;
        begin=std::chrono::steady_clock::now();
        original(dq.as<uint16_t>(),dt.as<uint16_t>(),reference.as<float>()+guard,start,count,stride,tokens);
        finish();original_ms+=elapsed(begin);
        for(unsigned mode=0u;mode<4u;++mode) {
            begin=std::chrono::steady_clock::now();
            prepare<IntegerRowKind::Query>(dq.as<uint16_t>(),qp.as<Row>(),tokens,start,count);
            staged_variant(variants[mode],qp.as<Row>(),kp.as<Row>(),candidate.as<float>()+guard,start,count,stride,tokens);
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
    unchanged(dq,q);unchanged(dk,k);
    std::printf("{\"kind\":\"staged_integer_original_q7169\",\"tokens\":7169,\"query_batch\":128,\"compared_score_cells\":%zu,\"cpu_dots\":%u,\"raw_bit_mismatches\":0,\"original_query_ms\":%.6f,\"key_transpose_ms\":%.6f,\"staged_8_query_and_encoding_ms\":%.6f,\"staged_16_query_and_encoding_ms\":%.6f,\"staged_shared_4_query_and_encoding_ms\":%.6f,\"staged_shared_8_query_and_encoding_ms\":%.6f,\"key_encoding_ms\":%.6f,\"maximum_completed_slab_ms\":[%.6f,%.6f,%.6f,%.6f],\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",compared,cpu_dots,original_ms,transpose_ms,staged_ms[0],staged_ms[1],staged_ms[2],staged_ms[3],key_encoding_ms,maximum_stage_ms[0],maximum_stage_ms[1],maximum_stage_ms[2],maximum_stage_ms[3]);
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
