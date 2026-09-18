#include "../../native/providers/ck_fmha/rz_tree_qk.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace qrt_blackwell_attention;
constexpr unsigned guard = 64u;
constexpr unsigned variants[] = {1u};
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
        hipLaunchKernelGGL((qrt_rz_tree_qk::prepare<false>),dim3(n*kQueryHeads),dim3(kHeadDim),0u,nullptr,
            q,qp.as<uint32_t>()+guard,qf.as<unsigned>()+guard,nullptr,n);
        check(hipGetLastError());
        hipLaunchKernelGGL((qrt_rz_tree_qk::prepare<true>),dim3(n*kKvHeads),dim3(kHeadDim),0u,nullptr,
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
                    eligible &= !(x&0x7fffu) || (e>=96u && e<=158u);
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
void original(const uint16_t* q, const uint16_t* k, float* out,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    hipLaunchKernelGGL(blackwell_tiled_exact_scores_kernel,
        dim3((stride+31u)/32u,kQueryHeads,(count+7u)/8u),dim3(kThreads),0u,nullptr,
        q,k,out,start,count,stride,key_stride);
    check(hipGetLastError());
}


float reference_dot(const uint16_t* left, const uint16_t* right) {
    bool supported=true;
    for(unsigned i=0u;i<256u;++i)supported &= qrt_rz_tree_qk::eligible(left[i]) && qrt_rz_tree_qk::eligible(right[i]);
    if(!supported)return qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,left,right,256u)*kExactScale;
    float carry=0.0f;
    for(unsigned group=0u;group<256u;group+=16u) {
        float products[16];
        for(unsigned i=0u;i<16u;++i)products[i]=qrt_sm121_float_alignment::from_bits(uint32_t(left[group+i])<<16u)*
            qrt_sm121_float_alignment::from_bits(uint32_t(right[group+i])<<16u);
        carry=qrt_sm121_rz_tree::group_spec(carry,products);
    }
    return (carry==0.0f?0.0f:carry)*kExactScale;
}
__global__ void add_pairs(const float* left,const float* right,float* output,
    unsigned long long* audit,unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned before=qrt_rz_tree_qk::native::enter();
    unsigned during;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 2)" : "=s"(during) : : "memory");
    if(i<count)output[i]=qrt_rz_tree_qk::add(left[i],right[i]);
    qrt_rz_tree_qk::native::leave(before);
    unsigned after;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 2)" : "=s"(after) : : "memory");
    if(!(threadIdx.x%32u)) {
        atomicAdd(audit,1ull);
        if(during!=3u || after!=before)atomicAdd(audit+1u,1ull);
    }
}
void add_preflight() {
    constexpr unsigned count=1048576u;
    const float sentinel=qrt_sm121_float_alignment::from_bits(0xa5a5a5a5u);
    std::vector<float> left(count+2u*guard,sentinel),right=left,expected=left;
    unsigned state=0x3958192u;auto random=[&](){state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;};
    for(unsigned i=0u;i<count;++i) {
        uint32_t a=(random()&0x807fffffu)|((51u+random()%149u)<<23u);
        uint32_t b=(random()&0x807fffffu)|((51u+random()%149u)<<23u);
        if(i%11u==0u)b=a^0x80000000u;
        if(i%13u==0u)a&=0x80000000u;
        if(i%17u==0u)b&=0x80000000u;
        left[guard+i]=qrt_sm121_float_alignment::from_bits(a);
        right[guard+i]=qrt_sm121_float_alignment::from_bits(b);
        expected[guard+i]=qrt_rz_tree_qk::add_spec(left[guard+i],right[guard+i]);
    }
    Device a(left.size()*4u),b(right.size()*4u),out(expected.size()*4u),audit(16u);
    upload(a,left);upload(b,right);check(hipMemset(out.pointer,0xa5,expected.size()*4u));check(hipMemset(audit.pointer,0,16u));
    hipLaunchKernelGGL(add_pairs,dim3((count+255u)/256u),dim3(256u),0u,nullptr,
        a.as<float>()+guard,b.as<float>()+guard,out.as<float>()+guard,audit.as<unsigned long long>(),count);
    check(hipGetLastError());finish();unchanged(out,expected);unchanged(a,left);unchanged(b,right);
    const auto counts=download<unsigned long long>(audit,2u);
    if(counts[0]!=count/32u || counts[1])throw std::runtime_error("native RZ mode not restored");
    std::printf("{\"kind\":\"rz_tree_add_preflight\",\"native_additions\":1048576,\"waves\":%llu,\"mode_errors\":%llu,\"spec_bit_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",counts[0],counts[1]);std::fflush(stdout);
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
    if(c.mode==8u) {
        // Only the second query/key cell owned by a thread is unsupported.
        q[guard+(size_t(c.start+16u)*kQueryHeads+3u)*kHeadDim+233u]=0x7fc1u;
        k[guard+(size_t(16u)*kKvHeads+1u)*kHeadDim+195u]=0x8001u;
    }
    if(c.mode==9u || c.mode==10u) {
        // Exercise both admitted exponent endpoints, cancellation and wide
        // exponent gaps inside the actual tree, not only the scalar add probe.
        for(size_t i=guard;i+guard<q.size();++i)
            q[i]=uint16_t(((i*37u)&0x807fu)|((i%3u?96u:158u)<<7u));
        for(size_t i=guard;i+guard<k.size();++i)
            k[i]=uint16_t(((i*53u)&0x807fu)|((i%5u?158u:96u)<<7u));
        if(c.mode==10u) {
            for(size_t i=guard;i+guard<q.size();++i)q[i]=uint16_t((158u<<7u)|127u);
            for(size_t i=guard;i+guard<k.size();++i)
                k[i]=uint16_t((158u<<7u)|127u|((i&1u)?0x8000u:0u));
        }
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
    Device audit(24u);
    check(hipMemset(reference.pointer,0xa5,(cells+2u*guard)*4u));
    original(dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,reference.as<float>()+guard,c.start,c.count,stride,c.tokens);
    finish();const auto a=download<uint32_t>(reference,cells+2u*guard);
    for(unsigned variant:variants) {
        (void)variant;
        check(hipMemset(candidate.pointer,0xa5,(cells+2u*guard)*4u));
        check(hipMemset(audit.pointer,0,24u));
        hipLaunchKernelGGL((qrt_rz_tree_qk::scores<true>),dim3((stride+31u)/32u,16u,(c.count+31u)/32u),dim3(256u),0u,nullptr,
            prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,prepared.qf.as<unsigned>()+guard,
            prepared.kf.as<unsigned>()+guard,candidate.as<float>()+guard,c.start,c.count,stride,c.tokens,audit.as<unsigned long long>());
        check(hipGetLastError());
        hipLaunchKernelGGL(qrt_deferred_qk_fallback::replay_scan,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,
            dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,candidate.as<float>()+guard,c.start,c.count,stride,c.tokens);
        check(hipGetLastError());
        finish();const auto b=download<uint32_t>(candidate,cells+2u*guard);
        size_t different=0u,unsupported=0u;double maximum_error=0.0;
        for(size_t i=0u;i<a.size();++i) {
            if(i<guard || i>=cells+guard) {
                if(a[i]!=0xa5a5a5a5u || b[i]!=0xa5a5a5a5u)throw std::runtime_error("QK score redzone changed");
            }else {
                const size_t cell=i-guard;const unsigned key=unsigned(cell%stride),head=unsigned(cell/stride%16u),row=unsigned(cell/stride/16u);
                if(key>c.start+row) {
                    if(a[i]!=bits(-INFINITY) || b[i]!=bits(-INFINITY))throw std::runtime_error("causal mask differs");
                }else {
                    const bool fallback=!prepared.qflags[guard+(c.start+row)*16u+head] || !prepared.kflags[guard+key*2u+head/8u];
                    if(fallback) { ++unsupported;if(a[i]!=b[i])throw std::runtime_error("unsupported dot differs from original replay"); }
                    else {
                        const float actual=qrt_sm121_float_alignment::from_bits(b[i]),control=qrt_sm121_float_alignment::from_bits(a[i]);
                        if(!std::isfinite(actual))throw std::runtime_error("eligible tree score is nonfinite");
                        maximum_error=std::max(maximum_error,std::fabs(double(actual)-control));
                    }
                    different+=a[i]!=b[i];
                }
            }
        }

        for(unsigned sample=0u;sample<64u;++sample) {
            const unsigned row=sample%c.count,head=(sample/4u)%kQueryHeads;
            const unsigned key=sample%2u?(sample*797u)%(c.start+row+1u):0u;
            const auto* left=q.data()+guard+(size_t(c.start+row)*kQueryHeads+head)*kHeadDim;
            const auto* right=k.data()+guard+(size_t(key)*kKvHeads+head/(kQueryHeads/kKvHeads))*kHeadDim;
            const float expected=reference_dot(left,right);
            if(bits(expected)!=b[guard+(size_t(row)*kQueryHeads+head)*stride+key])
                throw std::runtime_error("QK differs from independent RZ-tree/original-fallback CPU specification");
        }
        const auto mode=download<unsigned long long>(audit,3u);
        unsigned long long waves=0u;
        for(unsigned row=0u;row<c.count;row+=32u)waves+=(uint64_t(c.start+std::min(row+32u,c.count)-1u)/32u+1u)*16u*8u;
        if(mode[0]!=waves || mode[1] || mode[2]!=unsupported)throw std::runtime_error("rounding-scope or fallback accounting mismatch");
        std::printf("{\"kind\":\"rz_tree_qk_safety\",\"tokens\":%u,\"query_start\":%u,\"query_count\":%u,\"mode\":%u,\"cells\":%zu,\"cpu_spec_dots\":64,\"spec_bit_mismatches\":0,\"original_score_bit_differences\":%zu,\"maximum_original_absolute_error\":%.12g,\"original_fallback_cells\":%zu,\"mode_waves\":%llu,\"mode_errors\":%llu,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",c.tokens,c.start,c.count,c.mode,cells,different,maximum_error,unsupported,mode[0],mode[1]);
        std::fflush(stdout);
    }
    unchanged(dq,q);unchanged(dk,k);unchanged(dt,transposed);prepared.verify();
}


} // namespace
int main(int argc,char** argv) {
    try {
        hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));
        if(std::strncmp(p.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
        if(argc!=2 || std::strcmp(argv[1],"--selftest"))throw std::runtime_error("use --selftest");
        add_preflight();
        const Case cases[]={{17,0,8,9},{17,0,9,2},{33,0,16,10},{33,1,31,1},{1,0,1,0},{9,0,9,1},{35,3,17,2},{67,33,32,3},{67,64,3,4},
            {129,1,128,0},{7169,7041,128,1},{8192,8064,128,0},{8192,8191,1,2},{8193,8191,2,3},
            {17,0,9,5},{67,33,32,5},{33,1,31,6},{33,0,16,7},{129,1,128,5},{129,1,128,7},{65,0,31,0},{65,0,32,1},{65,0,33,5},{65,31,33,8}};
        for(auto c:cases)run(c);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
