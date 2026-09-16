#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
#include "../../native/providers/moe_accumulator/sm121_native_rz_carry.h"
#include "../../native/providers/moe_accumulator/sm121_canonical_normalize.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
namespace native = qrt_sm121_native_rz_carry;
constexpr unsigned guard = 64u, samples = 4096u, count = 311u * samples * 2u + 1u;
constexpr unsigned waves = (count + 255u) / 256u * 8u, fields = 9u;
#if defined(__HIPCC__)
#define QRT_CASE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_CASE_INLINE inline
#endif
struct Case { uint32_t magnitude; int maximum; bool negative; };
QRT_CASE_INLINE Case input(unsigned index) {
    const unsigned sample = index / 2u % samples;
    uint32_t x = index / 2u * 7919u + 0x3958192u;
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    if (sample < 96u) {
        const unsigned bit = sample / 3u;
        x = (uint32_t(1u) << bit) + sample % 3u - 1u;
    } else if (sample < 102u) {
        x = sample < 99u ? qrt_sm121_group16::kMaxMagnitude + sample - 97u :
            qrt_sm121_group16::kMinNegativeModulo + sample - 100u;
    } else if (sample == 102u) x = 0xffffffffu;
    return {x, int(index / (2u * samples)) - 160, bool(index & 1u)};
}
#undef QRT_CASE_INLINE
uint32_t expected(Case c) {
    const auto value = qrt_sm121_canonical::normalize(c.magnitude, c.negative, c.maximum);
    return qrt_sm121_f32_carry::bits(qrt_q1_moe_hawkeye::value_to_float(
        qrt_sm121_group16::finish_accumulator(value)));
}
struct Result { uint32_t bits, accepted; };
void verify(const Result* results, unsigned& accepted, unsigned& rejected) {
    for (unsigned i = 0u; i < count; ++i) {
        const auto c = input(i);
        const bool valid = c.maximum >= -101 && c.maximum <= 121;
        const uint32_t want = valid ? expected(c) : 0x42f60000u;
        if (results[i].accepted != unsigned(valid) || results[i].bits != want) {
            std::fprintf(stderr,"index=%u magnitude=%u maximum=%d sign=%u expected=%08x/%u actual=%08x/%u\n",
                i,c.magnitude,c.maximum,unsigned(c.negative),want,unsigned(valid),results[i].bits,results[i].accepted);
            throw std::runtime_error("native RZ normalization differs from independent canonical value");
        }
        accepted += unsigned(valid); rejected += unsigned(!valid);
    }
}
#if defined(__HIPCC__)
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
struct Device {
    void* pointer = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&pointer, bytes)); }
    ~Device() { if (pointer && hipFree(pointer) != hipSuccess) std::abort(); }
    template<class T> T* as() { return static_cast<T*>(pointer); }
};
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("native RZ completion deadline");
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
template<unsigned Initial>
__global__ void probe(Result* output, uint32_t* metadata) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned before, initial, during, after, final;
    float during_positive, during_negative, after_positive, after_negative;
    const float one = 1.0f, half = 0x1p-24f;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 8)\n\t"
                 "s_round_mode %2\n\t"
                 "s_getreg_b32 %1, hwreg(HW_REG_MODE, 0, 8)\n\t"
                 : "=&s"(before), "=&s"(initial) : "n"(Initial * 5u) : "memory");
    const unsigned saved = native::enter();
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 8)\n\t"
                 "v_add_f32_e64 %1, %3, %4\n\t"
                 "v_add_f32_e64 %2, -%3, -%4\n\t"
                 : "=&s"(during), "=&v"(during_positive), "=&v"(during_negative)
                 : "v"(one), "v"(half) : "memory");
    if (index < count) {
        const auto c = input(index); float value = 123.0f;
        const bool accepted = native::normalize(c.magnitude, c.negative, c.maximum, &value);
        output[index] = {qrt_sm121_f32_carry::bits(value), unsigned(accepted)};
    }
    native::leave(saved);
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 8)\n\t"
                 "v_add_f32_e64 %1, %4, %5\n\t"
                 "v_add_f32_e64 %2, -%4, -%5\n\t"
                 "s_setreg_b32 hwreg(HW_REG_MODE, 0, 8), %6\n\t"
                 "s_getreg_b32 %3, hwreg(HW_REG_MODE, 0, 8)\n\t"
                 : "=&s"(after), "=&v"(after_positive), "=&v"(after_negative), "=&s"(final)
                 : "v"(one), "v"(half), "s"(before) : "memory");
    if (!(threadIdx.x % 32u)) {
        auto* m = metadata + (blockIdx.x * 8u + threadIdx.x / 32u) * fields;
        m[0]=before;m[1]=initial;m[2]=during;m[3]=after;m[4]=final;
        m[5]=qrt_sm121_f32_carry::bits(during_positive);m[6]=qrt_sm121_f32_carry::bits(during_negative);
        m[7]=qrt_sm121_f32_carry::bits(after_positive);m[8]=qrt_sm121_f32_carry::bits(after_negative);
    }
}
void run(unsigned mode) {
    std::vector<Result> result(count + 2u * guard);
    std::vector<uint32_t> metadata(size_t(waves) * fields + 2u * guard);
    Device d(result.size() * sizeof(Result)), m(metadata.size() * 4u);
    check(hipMemset(d.pointer,0xa5,result.size()*sizeof(Result)));
    check(hipMemset(m.pointer,0xa5,metadata.size()*4u));
#define QRT_PROBE(M) case M: hipLaunchKernelGGL((probe<M>),dim3((count+255u)/256u),dim3(256u),0u,nullptr,d.as<Result>()+guard,m.as<uint32_t>()+guard);break
    switch(mode){QRT_PROBE(0u);QRT_PROBE(1u);QRT_PROBE(2u);QRT_PROBE(3u);}
#undef QRT_PROBE
    check(hipGetLastError());finish();
    check(hipMemcpy(result.data(),d.pointer,result.size()*sizeof(Result),hipMemcpyDeviceToHost));
    check(hipMemcpy(metadata.data(),m.pointer,metadata.size()*4u,hipMemcpyDeviceToHost));
    for(unsigned i=0u;i<guard;++i) {
        const auto a=result[i],b=result[guard+count+i];
        if(a.bits!=0xa5a5a5a5u || a.accepted!=0xa5a5a5a5u || b.bits!=0xa5a5a5a5u || b.accepted!=0xa5a5a5a5u ||
            metadata[i]!=0xa5a5a5a5u || metadata[metadata.size()-1u-i]!=0xa5a5a5a5u)throw std::runtime_error("native RZ redzone changed");
    }
    for(unsigned wave=0u;wave<waves;++wave) {
        const auto* x=metadata.data()+guard+wave*fields;
        if(x[1]!=((x[0]&~15u)|mode*5u) || x[2]!=((x[1]&~3u)|3u) || x[3]!=x[1] || x[4]!=x[0] ||
            x[5]!=0x3f800000u || x[6]!=0xbf800000u || x[7]!=(mode==1u?0x3f800001u:0x3f800000u) ||
            x[8]!=(mode==2u?0xbf800001u:0xbf800000u))throw std::runtime_error("native RZ state restoration or scalar control failed");
    }
    unsigned accepted=0u,rejected=0u;verify(result.data()+guard,accepted,rejected);
    std::printf("{\"kind\":\"native_rz_carry_primitive\",\"initial_round_mode\":%u,\"cases\":%u,\"accepted\":%u,\"rejected\":%u,\"waves\":%u,\"raw_bit_mismatches\":0,\"round_state_restored\":true,\"unrelated_mode_bits_preserved\":true,\"scalar_round_control_pass\":true,\"redzones_pass\":true,\"inference_acceptance\":false}\n",mode,count,accepted,rejected,waves);
    std::fflush(stdout);
}
#endif
} // namespace
int main() try {
#if defined(__HIPCC__)
    hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));
    if(std::strncmp(p.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    for(unsigned mode=0u;mode<4u;++mode)run(mode);
#else
    std::vector<Result> results(count);
    for(unsigned i=0u;i<count;++i){const auto c=input(i);float value=123.0f;const bool ok=native::normalize(c.magnitude,c.negative,c.maximum,&value);results[i]={qrt_sm121_f32_carry::bits(value),unsigned(ok)};}
    unsigned accepted=0u,rejected=0u;verify(results.data(),accepted,rejected);
    std::printf("{\"kind\":\"native_rz_carry_host_specification\",\"cases\":%u,\"accepted\":%u,\"rejected\":%u,\"raw_bit_mismatches\":0,\"hardware_instructions_executed\":false}\n",count,accepted,rejected);
#endif
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
