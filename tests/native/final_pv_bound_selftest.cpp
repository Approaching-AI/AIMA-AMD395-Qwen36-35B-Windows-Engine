#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define BOUND_TEST_INLINE __host__ __device__ inline
#else
#define BOUND_TEST_INLINE inline
#endif
#include "../../native/providers/moe_accumulator/sm121_pv_final_bound.h"

namespace b = qrt_sm121_pv_bound;
namespace f = qrt_sm121_pv_final_bound;
struct Result { unsigned checks, underestimated, false_admissions; float worst_ratio; };
BOUND_TEST_INLINE unsigned random(unsigned& x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}
BOUND_TEST_INLINE float weakened(float x, bool down) {
    if (!down || !b::finite(x)) return x;
    unsigned bits = b::bits(x);
    bits = bits > 4u ? bits - 4u : 0u;
    return bits < 0x00800000u ? 0.0f : b::value(bits);
}
BOUND_TEST_INLINE Result sequence(unsigned trial) {
    unsigned seed = 0x39513171u ^ (trial * 977u + 1u);
    float original = 0.0f, state = 0.0f;
    Result result{};
    const unsigned groups = 2u * (1u + trial % 256u);
    const bool down = (trial & 256u) != 0u;
    for (unsigned g = 0; g < groups; ++g) {
        // Broad independent carry/dot exponents, cancellation, zeros, gradual
        // underflow and inputs that force the new conservative overflow cap.
        const unsigned exponent = trial % 4u == 0u ? random(seed) % 254u
            : trial % 4u == 1u ? 1u + random(seed) % 32u : 110u + random(seed) % 32u;
        const float carry = b::value((random(seed) & 0x807fffffu) | (exponent << 23u));
        float dot = b::value((random(seed) & 0x007fffffu) | (exponent << 23u));
        if (g % 7u == 0u) dot = 0.0f;
        if (g % 2u == 0u) {
            const unsigned mode = (trial + g / 2u) % 8u;
            float alpha = 1.0f;
            if (mode == 0u) alpha = 0.0f;
            if (mode == 1u) alpha = 0.99609375f;
            if (mode == 2u) alpha = 0.625f;
            if (mode == 3u) alpha = b::value(0x3f7fffffu);
            if (mode == 4u) alpha = b::value((random(seed) % 127u << 23u) | (random(seed) & 0x007fffffu));
            if ((trial / 512u) % 4u == 1u) alpha = 1.0f;
            if ((trial / 512u) % 4u == 2u) alpha = b::value(0x3f7fffffu);
            if ((trial / 512u) % 4u == 3u) alpha = 0.99609375f;
            original = b::rescale(original, carry, alpha);
            state = weakened(f::rescale(state, carry, alpha), down);
        }
        original = b::group(original, carry, dot);
        state = weakened(f::group(state, carry, dot), down);
        if (g % 2u == 1u) {
            const float enlarged = f::finalize(state, g + 1u);
            ++result.checks;
            if (!(enlarged >= original)) ++result.underestimated;
            if (b::finite(enlarged) && original > 0.0f) {
                const float ratio = original / enlarged;
                if (ratio > result.worst_ratio) result.worst_ratio = ratio;
            }
            const float rcp = b::value(0x39800000u + (random(seed) & 0x01ffffffu));
            const float old_final = b::finish(original, carry, rcp);
            const float new_final = b::finish(enlarged, carry, rcp);
            if (!(new_final >= old_final)) ++result.underestimated;
            if (b::same_bf16(carry * rcp, new_final) &&
                !b::same_bf16(carry * rcp, old_final)) ++result.false_admissions;
        }
    }
    return result;
}

#if defined(__HIPCC__)
__global__ void evaluate(Result* out, unsigned count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = sequence(i);
}
void check(hipError_t status) { if (status != hipSuccess) { std::fprintf(stderr,"HIP %s\n",hipGetErrorString(status)); std::abort(); } }
#endif

int main() {
    constexpr unsigned trials = 16384u;
    std::vector<Result> results(trials);
    const char* backend = "cpu";
    const char* guarded = "false";
#if defined(__HIPCC__)
    backend = "gfx1151";
    guarded = "true";
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    assert(std::strstr(properties.gcnArchName, "gfx1151") != nullptr);
    Result* device = nullptr; check(hipMalloc(&device, (trials+2u)*sizeof(Result)));
    check(hipMemset(device, 0x5a, (trials+2u)*sizeof(Result)));
    hipLaunchKernelGGL(evaluate, dim3((trials+255u)/256u), dim3(256u), 0, 0, device+1, trials);
    check(hipGetLastError()); check(hipDeviceSynchronize());
    check(hipMemcpy(results.data(), device+1, trials*sizeof(Result), hipMemcpyDeviceToHost));
    Result guards[2];
    check(hipMemcpy(guards, device, sizeof(Result), hipMemcpyDeviceToHost));
    check(hipMemcpy(guards+1, device+trials+1, sizeof(Result), hipMemcpyDeviceToHost));
    for (unsigned char byte : *reinterpret_cast<unsigned char(*)[sizeof(guards)]>(guards)) assert(byte == 0x5au);
    check(hipFree(device));
#else
    for (unsigned i = 0; i < trials; ++i) results[i] = sequence(i);
#endif
    unsigned checks = 0, underestimated = 0, false_admissions = 0;
    float worst_ratio = 0.0f;
    for (const auto& r : results) {
        checks += r.checks; underestimated += r.underestimated; false_admissions += r.false_admissions;
        if (r.worst_ratio > worst_ratio) worst_ratio = r.worst_ratio;
    }
    assert(checks == 2105344u && underestimated == 0u && false_admissions == 0u);
    for (float invalid : {b::infinity(), b::value(0x7fc00000u), -1.0f}) {
        assert(!b::finite(f::group(invalid, 0.0f, 1.0f)));
        assert(!b::finite(f::group(0.0f, 1.0f, invalid)));
        assert(!b::finite(f::rescale(invalid, 1.0f, 0.0f)));
        assert(!b::finite(f::rescale(0.0f, 1.0f, invalid)));
    }
    assert(!b::finite(f::rescale(0.0f, b::infinity(), 0.0f)));
    assert(!b::finite(f::rescale(0.0f, 1.0f, 1.00001f)));
    assert(!b::finite(f::finalize(1.0f, 514u)));
    assert(!b::finite(f::finalize(1.0f, 1u)));
    assert(!b::finite(f::finalize(1.0f, 0u)));
    assert(f::rescale(1.0f, 1.0f, 0.0f) == 0.0f);
    assert(f::rescale(1.0f, 1.0f, 1.0f) == 1.0f);
    std::printf("{\"backend\":\"%s\",\"sequences\":%u,\"envelope_comparisons\":%u,\"underestimates\":%u,\"false_admissions\":%u,\"max_old_over_new\":%.9g,\"redzones_checked\":%s,\"redzones_pass\":%s,\"inference_acceptance\":false}\n",
        backend, trials, checks*2u, underestimated, false_admissions, worst_ratio, guarded, guarded);
}
