#include "../../native/providers/ck_fmha/blackwell_attention.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace qrt_blackwell_attention;
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
        throw std::runtime_error("QK input or input redzone changed");
}
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("QK completion deadline");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(hipEventDestroy(event));
}
uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, sizeof(u)); return u; }
struct Case { unsigned tokens, start, count, mode; };
void run(Case c) {
    constexpr unsigned guard = 64u;
    const unsigned stride = c.start+c.count;
    const size_t cells = size_t(c.count)*kQueryHeads*stride;
    std::vector<uint16_t> q(size_t(c.tokens)*kQueryHeads*kHeadDim+2u*guard, 0x5a5au);
    std::vector<uint16_t> k(size_t(c.tokens)*kKvHeads*kHeadDim+2u*guard, 0x5a5au);
    for (size_t i=guard; i+guard<q.size(); ++i)
        q[i] = uint16_t(((i*37u+i/19u)&0x807fu) | ((123u+i%8u)<<7u));
    for (size_t i=guard; i+guard<k.size(); ++i)
        k[i] = uint16_t(((i*53u+i/23u)&0x807fu) | ((121u+i%10u)<<7u));
    if (c.mode == 1u) {
        for (size_t i=guard; i+guard<q.size(); i+=7u) q[i] = i%3u ? 0u : 0x8000u;
        for (size_t i=guard; i+guard<k.size(); i+=11u) k[i] = i%3u ? 0u : 0x8000u;
    }
    if (c.mode == 2u || c.mode == 3u) {
        // Ineligible operands force a uniform raw reload, including partial
        // query and key tiles. Significands still follow the original model.
        const uint16_t edges[] = {1u,0x8001u,0x007fu,0x807fu,uint16_t(63u<<7u|19u),uint16_t(192u<<7u|11u),0u,0x8000u};
        for (unsigned j=0u;j<8u;++j) {
            q[guard+size_t(c.start)*kQueryHeads*kHeadDim+j] = edges[j];
            k[guard+j] = edges[7u-j];
        }
    }
    if (c.mode == 4u) {
        for (size_t i=guard; i+guard<q.size(); ++i) q[i] = uint16_t(127u<<7u|127u);
        for (size_t i=guard; i+guard<k.size(); ++i)
            k[i] = uint16_t(127u<<7u|127u|((i/16u)&1u ? 0x8000u : 0u));
    }
    std::vector<uint16_t> transposed(k.size(),0x5a5au);
    for (unsigned token=0u;token<c.tokens;++token)
        for (unsigned feature=0u;feature<kKvHeads*kHeadDim;++feature)
            transposed[guard+size_t(feature)*c.tokens+token] = k[guard+size_t(token)*kKvHeads*kHeadDim+feature];
    Device dq(q.size()*2u), dk(k.size()*2u), dt(transposed.size()*2u);
    Device original((cells+2u*guard)*4u), subgroup((cells+2u*guard)*4u);
    upload(dq,q); upload(dk,k);
    std::vector<uint16_t> initial(transposed.size(),0x5a5au); upload(dt,initial);
    check(hipError_t(transpose_keys(dk.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,
        size_t(c.tokens)*kKvHeads*kHeadDim,c.tokens,nullptr)));
    finish(); unchanged(dt,transposed);
    check(hipMemset(original.pointer,0xa5,(cells+2u*guard)*4u));
    check(hipMemset(subgroup.pointer,0xa5,(cells+2u*guard)*4u));
    hipLaunchKernelGGL(blackwell_tiled_exact_scores_kernel,
        dim3((stride+31u)/32u,kQueryHeads,(c.count+7u)/8u),dim3(kThreads),0u,nullptr,
        dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,original.as<float>()+guard,c.start,c.count,stride,c.tokens);
    check(hipGetLastError()); finish();
    hipLaunchKernelGGL(blackwell_subgroup_tiled_scores_kernel,
        dim3((stride+7u)/8u,kQueryHeads,(c.count+7u)/8u),dim3(kThreads),0u,nullptr,
        dq.as<uint16_t>()+guard,dt.as<uint16_t>()+guard,subgroup.as<float>()+guard,c.start,c.count,stride,c.tokens);
    check(hipGetLastError()); finish();
    const auto a = download<uint32_t>(original,cells+2u*guard);
    const auto b = download<uint32_t>(subgroup,cells+2u*guard);
    for (size_t i=0u;i<a.size();++i) {
        if (i<guard || i>=cells+guard) {
            if (a[i]!=0xa5a5a5a5u || b[i]!=0xa5a5a5a5u) throw std::runtime_error("QK score redzone changed");
        } else if (a[i]!=b[i]) throw std::runtime_error("Subgroup QK differs from original scores");
    }
    unsigned host_dots=0u;
    for (unsigned sample=0u;sample<64u;++sample) {
        const unsigned row=sample%c.count, head=(sample/4u)%kQueryHeads;
        const unsigned key=sample%2u ? (sample*797u)%(c.start+row+1u) : 0u;
        const auto* left=q.data()+guard+(size_t(c.start+row)*kQueryHeads+head)*kHeadDim;
        const auto* right=k.data()+guard+(size_t(key)*kKvHeads+head/8u)*kHeadDim;
        const float reference=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,left,right,kHeadDim)*kExactScale;
        if (bits(reference)!=b[guard+(size_t(row)*kQueryHeads+head)*stride+key])
            throw std::runtime_error("QK differs from independent wide CPU accumulator");
        ++host_dots;
    }
    unchanged(dq,q); unchanged(dk,k); unchanged(dt,transposed);
    std::printf("{\"kind\":\"subgroup_tiled_qk_safety\",\"tokens\":%u,\"query_start\":%u,\"query_count\":%u,\"mode\":%u,\"cells\":%zu,\"cpu_dots\":%u,\"raw_bit_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
        c.tokens,c.start,c.count,c.mode,cells,host_dots);
    std::fflush(stdout);
}
}
int main() {
    try {
        hipDeviceProp_t p{}; check(hipGetDeviceProperties(&p,0));
        if (std::strncmp(p.gcnArchName,"gfx1151",7u)) throw std::runtime_error("requires gfx1151");
        const Case cases[]={{1,0,1,0},{9,0,9,1},{35,3,17,2},{67,33,32,3},{67,64,3,4},
            {129,1,128,0},{7169,7041,128,1},{8192,8064,128,0},{8192,8191,1,2},{8193,8191,2,3}};
        for (auto c:cases) run(c);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
