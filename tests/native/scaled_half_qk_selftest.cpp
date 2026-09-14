#include "../../native/providers/ck_fmha/prepared_decoded_qk.h"
#include "../../native/providers/ck_fmha/scaled_half_qk.h"
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
constexpr unsigned variants[] = {0u, 64u, 128u, 256u};
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
using Row = qrt_scaled_half_qk::Row;
template<IntegerRowKind Kind> void encode_rows(const uint16_t* input, Row* output,
    unsigned tokens, unsigned start, unsigned count) {
    const size_t rows=integer_row_count(Kind,tokens,count);
    hipLaunchKernelGGL((qrt_scaled_half_qk::prepare_rows<Kind>),
        dim3((rows+kThreads-1u)/kThreads),dim3(kThreads),0u,nullptr,input,output,tokens,start,count);
    check(hipGetLastError());
}
struct IntegerRows {
    unsigned tokens;
    size_t qbytes,kbytes;
    Device qp,kp;
    std::vector<unsigned char> expected_keys;
    double key_ms=0.0;
    IntegerRows(unsigned n,unsigned count,const uint16_t* device_key,const uint16_t* host_key)
        : tokens(n),qbytes(integer_row_count(IntegerRowKind::Query,n,count)*sizeof(Row)),
          kbytes(integer_row_count(IntegerRowKind::Key,n,0u)*sizeof(Row)),
          qp(qbytes+2u*guard),kp(kbytes+2u*guard) {
        check(hipMemset(kp.pointer,0xa5,kbytes+2u*guard));
        const auto begin=std::chrono::steady_clock::now();
        encode_rows<IntegerRowKind::Key>(device_key,key(),tokens,0u,0u);finish();key_ms=elapsed(begin);
        expected_keys=cpu_encoding(IntegerRowKind::Key,host_key,0u,0u,kbytes);
        unchanged(kp,expected_keys);
    }
    Row* query(){return reinterpret_cast<Row*>(qp.as<unsigned char>()+guard);}
    Row* key(){return reinterpret_cast<Row*>(kp.as<unsigned char>()+guard);}
    void reset_query(){check(hipMemset(qp.pointer,0xa5,qbytes+2u*guard));}
    std::vector<unsigned char> cpu_encoding(IntegerRowKind kind,const uint16_t* input,
        unsigned start,unsigned count,size_t capacity) const {
        std::vector<unsigned char> expected(capacity+2u*guard,0xa5u);
        const size_t rows=integer_row_count(kind,tokens,count);
        if(rows*sizeof(Row)>capacity)throw std::runtime_error("matrix row capacity");
        for(size_t row=0u;row<rows;++row) {
            uint16_t original[16];
            for(unsigned c=0u;c<16u;++c)
                original[c]=input[integer_row_input_index(kind,row,c,tokens,start,count)];
            const Row value=qrt_scaled_half_qk::prepare(original);
            std::memcpy(expected.data()+guard+row*sizeof(Row),&value,sizeof(Row));
        }
        return expected;
    }
    void verify_query(const uint16_t* input,unsigned start,unsigned count) {
        unchanged(qp,cpu_encoding(IntegerRowKind::Query,input,start,count,qbytes));
    }
    void verify_keys(){unchanged(kp,expected_keys);}
};
void staged_variant(unsigned variant,const uint16_t* q,const uint16_t* k,float* out,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride,Prepared& prepared,IntegerRows& matrix) {
    if(!variant) {
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::scores<128u,true,16u,16u>),
            dim3((stride+15u)/16u,kQueryHeads,(count+15u)/16u),dim3(kThreads),0u,nullptr,
            q,k,prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,
            prepared.qf.as<unsigned>()+guard,prepared.kf.as<unsigned>()+guard,out,start,count,stride,key_stride);
    } else {
        // Query encoding is repeated for every slab and included in its timer.
        encode_rows<IntegerRowKind::Query>(q,matrix.query(),key_stride,start,count);
        const dim3 grid((stride+15u)/16u,kQueryHeads,(count+15u)/16u);
#define PREPARED_PAIRS_CASE(w) if(variant==w) hipLaunchKernelGGL((qrt_scaled_half_qk::scores<w>),grid,dim3(kThreads),0u,nullptr,matrix.query(),matrix.key(),out,start,count,stride,key_stride)
        PREPARED_PAIRS_CASE(64u);
        else PREPARED_PAIRS_CASE(128u);
        else PREPARED_PAIRS_CASE(256u);
        else throw std::runtime_error("invalid scaled half QK variant");
#undef PREPARED_PAIRS_CASE
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

__global__ void paired_group_values(const Row* left,const Row* right,const uint32_t* carry,
    uint32_t* output,unsigned* paths,unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const qrt_q1_moe_hawkeye::Value input{carry[3u*i],int16_t(int32_t(carry[3u*i+1u])),carry[3u*i+2u]!=0u};
    unsigned path=0u;const auto value=qrt_sm121_scaled_half_products::accumulate(input,left[i],right[i],&path);
    atomicAdd(paths+path,1u);
    output[3u*i]=value.significand;output[3u*i+1u]=uint32_t(int32_t(value.exponent));output[3u*i+2u]=unsigned(value.negative);
}
void paired_preflight() {
    constexpr unsigned count=131072u;
    Row sentinel;std::memset(&sentinel,0xa5,sizeof(sentinel));
    std::vector<Row> left(count+2u*guard,sentinel),right=left;
    std::vector<uint32_t> carry(size_t(count)*3u+2u*guard,0xa5a5a5a5u),expected=carry;
    const uint16_t controls[]={0,0x8000,1,0x7f,0x80,0x807f,0x3f80,0xbf80,0x3fff,0xbfff,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc1,0xffff};
    qrt_q1_moe_hawkeye::Value previous{0u,-133,false};
    unsigned state=0x3958192u;auto random=[&](){state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;};
    for(unsigned group=0u;group<count;++group) {
        uint16_t a[16],b[16];qrt_q1_moe_hawkeye::Value values[17];
        int maximum=-133;
        for(unsigned i=0u;i<16u;++i) {
            if(group<65536u){a[i]=uint16_t(group+i*4093u);b[i]=controls[i];}
            else {
                const unsigned n=group-65536u,spread=1u+n%30u;
                a[i]=uint16_t((random()&0x807fu)|((100u+n%31u+random()%spread)<<7u));
                b[i]=uint16_t((random()&0x807fu)|((102u+(n*7u)%29u+random()%spread)<<7u));
                if(n%11u==0u && i%3u==0u)a[i]=uint16_t(a[i]&0x8000u);
                if(n%13u==0u && i%3u==1u)b[i]=uint16_t(b[i]&0x8000u);
                if(n%17u==0u)a[i]=uint16_t(a[i]&0x8000u);
            }
            values[i+1u]=qrt_q1_moe_hawkeye::multiply_bf16(a[i],b[i],-133);
            maximum=std::max(maximum,int(values[i+1u].exponent));
        }
        values[0]=group%3u==0u?previous:group%3u==1u?qrt_q1_moe_hawkeye::Value{0u,-133,false}:
            qrt_q1_moe_hawkeye::Value{0x800000u|(group*797u&0x7fffffu),int16_t(maximum+int(group%63u)-21),bool(group&1u)};
        previous=qrt_q1_moe_hawkeye::group_sum<26,-133>(values,17u);
        left[guard+group]=qrt_sm121_scaled_half_products::prepare(a);right[guard+group]=qrt_sm121_scaled_half_products::prepare(b);
        carry[guard+3u*group]=values[0].significand;carry[guard+3u*group+1u]=uint32_t(int32_t(values[0].exponent));carry[guard+3u*group+2u]=unsigned(values[0].negative);
        expected[guard+3u*group]=previous.significand;expected[guard+3u*group+1u]=uint32_t(int32_t(previous.exponent));expected[guard+3u*group+2u]=unsigned(previous.negative);
    }
    Device a(left.size()*sizeof(Row)),b(right.size()*sizeof(Row)),c(carry.size()*4u),out(expected.size()*4u),paths(16u);
    upload(a,left);upload(b,right);upload(c,carry);check(hipMemset(out.pointer,0xa5,expected.size()*4u));check(hipMemset(paths.pointer,0,16u));
    hipLaunchKernelGGL(paired_group_values,dim3((count+255u)/256u),dim3(256u),0u,nullptr,
        a.as<Row>()+guard,b.as<Row>()+guard,c.as<uint32_t>()+guard,out.as<uint32_t>()+guard,paths.as<unsigned>(),count);
    check(hipGetLastError());finish();unchanged(out,expected);unchanged(a,left);unchanged(b,right);unchanged(c,carry);
    const auto counts=download<unsigned>(paths,4u);
    if(counts[0]+counts[1]+counts[2]+counts[3]!=count || counts[0]<65536u || !counts[1] || counts[2]<30000u || counts[3]<1000u)
        throw std::runtime_error("scaled half preflight lacks arithmetic path coverage");
    std::printf("{\"kind\":\"scaled_half_preflight\",\"all_bf16_encodings\":65536,\"controls\":16,\"original_products\":2097152,\"raw_carry_states\":131072,\"general_encoding_groups\":65536,\"scaled_normal_groups\":65536,\"native_path_fallback_zero_full_partial\":[%u,%u,%u,%u],\"raw_bit_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",counts[0],counts[1],counts[2],counts[3]);
    std::fflush(stdout);
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
    IntegerRows matrix(c.tokens,c.count,dk.as<uint16_t>()+guard,k.data()+guard);
    check(hipMemset(reference.pointer,0xa5,(cells+2u*guard)*4u));
    original(dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,reference.as<float>()+guard,c.start,c.count,stride,c.tokens);
    finish();const auto a=download<uint32_t>(reference,cells+2u*guard);
    for(unsigned variant:variants) {
        const unsigned groups=variant%100u;
        check(hipMemset(candidate.pointer,0xa5,(cells+2u*guard)*4u));
        matrix.reset_query();
        staged_variant(variant,dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,candidate.as<float>()+guard,c.start,c.count,stride,c.tokens,prepared,matrix);
        finish();const auto b=download<uint32_t>(candidate,cells+2u*guard);
        for(size_t i=0u;i<a.size();++i) {
            if(i<guard || i>=cells+guard) {
                if(a[i]!=0xa5a5a5a5u || b[i]!=0xa5a5a5a5u)throw std::runtime_error("QK score redzone changed");
            }else if(a[i]!=b[i]) {
                std::fprintf(stderr,"groups=%u tokens=%u start=%u count=%u mode=%u index=%zu expected=%08x actual=%08x\n",groups,c.tokens,c.start,c.count,c.mode,i-guard,a[i],b[i]);
                throw std::runtime_error("scaled half QK differs from original scores");
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
        if(variant)matrix.verify_query(q.data()+guard,c.start,c.count);
        matrix.verify_keys();
        std::printf("{\"kind\":\"scaled_half_qk_safety\",\"tokens\":%u,\"query_start\":%u,\"query_count\":%u,\"mode\":%u,\"variant\":%u,\"cells\":%zu,\"cpu_dots\":64,\"raw_bit_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",c.tokens,c.start,c.count,c.mode,variant,cells);
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
void captured(const char* qfile,const char* kfile) {
    constexpr unsigned tokens=7169u,batch=128u,attempts=4u;
    auto q=read_words(qfile,size_t(tokens)*kQueryHeads*kHeadDim);
    auto k=read_words(kfile,size_t(tokens)*kKvHeads*kHeadDim);
    q.insert(q.begin(),guard,0x5a5au);q.insert(q.end(),guard,0x5a5au);
    k.insert(k.begin(),guard,0x5a5au);k.insert(k.end(),guard,0x5a5au);
    const size_t capacity=size_t(batch)*kQueryHeads*tokens,total_words=capacity+2u*guard;
    Device dq(q.size()*2u),dk(k.size()*2u),dt(k.size()*2u);
    Device reference(total_words*4u),candidate(total_words*4u),bad(4u);
    upload(dq,q);upload(dk,k);check(hipMemset(dt.pointer,0x5a,k.size()*2u));
    check(hipMemset(bad.pointer,0,4u));
    Prepared prepared(dq.as<uint16_t>()+guard,dk.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,
        q.data()+guard,k.data()+guard,tokens);
    IntegerRows matrix(tokens,batch,dk.as<uint16_t>()+guard,k.data()+guard);
    std::vector<uint16_t> transposed(k.size(),0x5a5au);
    for(unsigned token=0u;token<tokens;++token)
        for(unsigned feature=0u;feature<kKvHeads*kHeadDim;++feature)
            transposed[guard+size_t(feature)*tokens+token]=k[guard+size_t(token)*kKvHeads*kHeadDim+feature];
    unchanged(dt,transposed);
    double samples[variant_count][3]{},maximum_stage_ms[variant_count]{};
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
                matrix.reset_query();finish();
                const auto begin=std::chrono::steady_clock::now();
                staged_variant(variant,dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,candidate.as<float>()+guard,
                    start,count,stride,tokens,prepared,matrix);
                finish();const double wall=elapsed(begin);
                if(attempt){samples[mode][attempt-1u]+=wall;maximum_stage_ms[mode]=std::max(maximum_stage_ms[mode],wall);}
                // Includes both redzones and the complete unused score tail.
                hipLaunchKernelGGL(compare_scores,dim3((total_words+255u)/256u),dim3(256u),0u,nullptr,
                    reference.as<uint32_t>(),candidate.as<uint32_t>(),total_words,bad.as<unsigned>());
                check(hipGetLastError());finish();compared+=cells;
                if(download<unsigned>(bad,1u)[0])throw std::runtime_error("captured QK score or tail differs from original");
            }
            if(variant)matrix.verify_query(q.data()+guard,start,count);
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
    unchanged(dq,q);unchanged(dk,k);unchanged(dt,transposed);prepared.verify();matrix.verify_keys();
    for(unsigned mode=0u;mode<variant_count;++mode) {
        const unsigned variant=variants[mode];
        double sorted[3]={samples[mode][0],samples[mode][1],samples[mode][2]};std::sort(sorted,sorted+3);
        const double preparation_ms=variant?matrix.key_ms:prepared.ms;
        std::printf("{\"kind\":\"scaled_half_original_q7169\",\"variant\":%u,\"query_rows\":16,\"key_columns\":%u,\"tokens\":7169,\"query_batch\":128,\"unique_score_cells\":%zu,\"compared_score_cells\":%zu,\"cpu_dots\":%u,\"raw_bit_mismatches\":0,\"one_time_preparation_ms\":%.6f,\"completed_query_with_slab_encoding_ms\":%.6f,\"completed_total_ms\":%.6f,\"completed_query_samples_ms\":[%.6f,%.6f,%.6f],\"warmup_per_slab\":1,\"samples_per_slab\":3,\"maximum_completed_slab_ms\":%.6f,\"all_attempts_verified\":true,\"redzones_pass\":true,\"unused_score_tail_pass\":true,\"immutable_inputs\":true,\"complete_cpu_encoding_check\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            variant,16u,compared/variant_count/attempts,compared/variant_count,cpu_dots/variant_count,
            preparation_ms,sorted[1],sorted[1]+preparation_ms,samples[mode][0],samples[mode][1],samples[mode][2],maximum_stage_ms[mode]);
        std::fflush(stdout);
    }
}

}
int main(int argc,char** argv) {
    try {
        hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));
        if(std::strncmp(p.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
        if(argc==4 && !std::strcmp(argv[1],"--q7169")){captured(argv[2],argv[3]);return 0;}
        if(argc!=2 || std::strcmp(argv[1],"--selftest"))throw std::runtime_error("use --selftest or --q7169 Q K");
        paired_preflight();
        const Case cases[]={{17,0,8,0},{17,0,9,2},{33,0,16,0},{33,1,31,1},{1,0,1,0},{9,0,9,1},{35,3,17,2},{67,33,32,3},{67,64,3,4},
            {129,1,128,0},{7169,7041,128,1},{8192,8064,128,0},{8192,8191,1,2},{8193,8191,2,3},
            {17,0,9,5},{67,33,32,5},{33,1,31,6},{33,0,16,7},{129,1,128,5},{129,1,128,7}};
        for(auto c:cases)run(c);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
