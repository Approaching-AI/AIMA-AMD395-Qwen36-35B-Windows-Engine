#include "../../native/providers/moe_accumulator/sm121_subgroup.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

constexpr unsigned cases = 4099u;
template<unsigned Groups, bool Certified>
__global__ void probe(const uint16_t* a, const uint16_t* b, float* output,
                      unsigned count, unsigned k, qrt_sm121_dot_certificate::Stats* counts) {
    constexpr unsigned Lanes = 16u;
    qrt_sm121_dot_certificate::Stats stats;
    const unsigned index = (blockIdx.x * blockDim.x + threadIdx.x) / Lanes;
    if (index >= count) return;
    const float value = qrt_sm121_subgroup::dot<Lanes, Groups, Certified>(a + index * (k + 3u) + 1u,
                                                     b + index * (k + 3u) + 1u, k, &stats);
    if ((threadIdx.x & (Lanes - 1u)) == 0u) { output[index] = value; counts[index] = stats; }
}
uint32_t seed = 0x3958192u;
uint32_t random_word() { seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u; return seed; }
void check(hipError_t error) { if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error)); }
uint32_t bits(float value) { uint32_t result; std::memcpy(&result, &value, 4u); return result; }
void wait_for(hipEvent_t event) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    for (;;) {
        const hipError_t status = hipEventQuery(event);
        if (status == hipSuccess) return;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("GPU completion deadline exceeded");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
int main() try {
    unsigned long long all_accepted = 0u;
    if (!QRT_SM121_COMPACT_NORMALIZE) throw std::runtime_error("requires compact normalization");
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    for (unsigned k : {16u, 48u, 80u, 128u, 512u, 2048u, 2064u}) {
        std::vector<uint16_t> a(size_t(cases) * (k + 3u), 0x5a5au), b = a;
        for (unsigned row = 0u; row < cases; ++row) for (unsigned i = 0u; i < k; ++i) {
            const unsigned spread = row % 3u == 0u ? 30u : 10u, base = row % 3u == 0u ? 1u + row % 200u : 119u;
            uint16_t left = uint16_t((random_word() & 0x807fu) | ((base + random_word() % spread) << 7u));
            uint16_t right = uint16_t((random_word() & 0x807fu) | ((base + random_word() % spread) << 7u));
            if (row % 41u == 0u) left = uint16_t((row & 1u) << 15u);
            if (row % 43u == 0u) right &= 0x807fu; // Signed subnormals.
            if (row % 47u == 0u) { left = uint16_t(0x3fffu | ((row & 1u) << 15u)); right = 0x3fffu; }
            if (row % 53u == 0u) { left = uint16_t(0x3f80u | ((i & 1u) << 15u)); right = 0x3f80u; }
            a[size_t(row) * (k + 3u) + 1u + i] = left; b[size_t(row) * (k + 3u) + 1u + i] = right;
        }
        std::vector<float> expected(cases), result(cases + 128u, 12345.0f);
        for (unsigned row = 0u; row < cases; ++row)
            expected[row] = qrt_q1_moe_hawkeye::dot_bf16_hopper(a.data() + row * (k + 3u) + 1u,
                                                                            b.data() + row * (k + 3u) + 1u, k);
        uint16_t *da = nullptr, *db = nullptr; float* dout = nullptr;
        using Stats = qrt_sm121_dot_certificate::Stats;
        Stats* dc = nullptr;
        std::vector<Stats> counts(cases + 128u, {0xa5a5a5a5u, 0xa5a5a5a5u});
        check(hipMalloc(reinterpret_cast<void**>(&dc), counts.size() * sizeof(Stats)));
        check(hipMalloc(reinterpret_cast<void**>(&da), a.size() * 2u));
        check(hipMalloc(reinterpret_cast<void**>(&db), b.size() * 2u));
        check(hipMalloc(reinterpret_cast<void**>(&dout), result.size() * 4u));
        check(hipMemcpy(da, a.data(), a.size() * 2u, hipMemcpyHostToDevice));
        check(hipMemcpy(db, b.data(), b.size() * 2u, hipMemcpyHostToDevice));
        for (unsigned choice : {0u, 1u, 2u}) {
            const unsigned groups = choice == 2u ? 8u : 4u;
            const bool certified = choice != 0u;
            std::fill(counts.begin(), counts.end(), Stats{0xa5a5a5a5u, 0xa5a5a5a5u});
            check(hipMemcpy(dc, counts.data(), counts.size()*sizeof(Stats), hipMemcpyHostToDevice));
            constexpr unsigned lanes = 16u;
            std::fill(result.begin(), result.end(), 12345.0f);
            check(hipMemcpy(dout, result.data(), result.size() * 4u, hipMemcpyHostToDevice));
            const dim3 grid((cases * lanes + 255u) / 256u);
            hipEvent_t begin, end; check(hipEventCreate(&begin)); check(hipEventCreate(&end));
            check(hipEventRecord(begin));
            if (choice == 0u) hipLaunchKernelGGL(HIP_KERNEL_NAME(probe<4u, false>), grid, dim3(256u), 0u, nullptr, da, db, dout + 64u, cases, k, dc + 64u);
            else if (choice == 1u) hipLaunchKernelGGL(HIP_KERNEL_NAME(probe<4u, true>), grid, dim3(256u), 0u, nullptr, da, db, dout + 64u, cases, k, dc + 64u);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(probe<8u, true>), grid, dim3(256u), 0u, nullptr, da, db, dout + 64u, cases, k, dc + 64u);
            check(hipGetLastError()); check(hipEventRecord(end)); wait_for(end);
            float ms = 0; check(hipEventElapsedTime(&ms, begin, end));
            check(hipMemcpy(result.data(), dout, result.size() * 4u, hipMemcpyDeviceToHost));
            check(hipMemcpy(counts.data(), dc, counts.size()*sizeof(Stats), hipMemcpyDeviceToHost));
            unsigned long long attempted = 0u, accepted = 0u;
            for (unsigned i = 0u; i < counts.size(); ++i) {
                const auto entry = counts[i];
                if (i < 64u || i >= cases + 64u) {
                    if (entry.attempted != 0xa5a5a5a5u || entry.accepted != 0xa5a5a5a5u) throw std::runtime_error("stats redzone");
                } else {
                    if (entry.attempted != (k + groups*16u - 1u)/(groups*16u) || entry.accepted > entry.attempted || (!certified && entry.accepted)) throw std::runtime_error("invalid tile counts");
                    attempted += entry.attempted; accepted += entry.accepted;
                }
            }
            all_accepted += accepted;
            unsigned bad = 0u;
            for (unsigned i = 0u; i < cases; ++i) bad += bits(result[i + 64u]) != bits(expected[i]);
            for (unsigned i = 0u; i < result.size(); ++i) if ((i < 64u || i >= cases + 64u) && result[i] != 12345.0f) throw std::runtime_error("output redzone");
            std::printf("{\"kind\":\"sm121_certified_dot\",\"lanes\":%u,\"staging_groups\":%u,\"k\":%u,\"dots\":%u,\"bit_mismatches\":%u,\"ms\":%.6f,\"certified\":%s,\"attempted_tiles\":%llu,\"accepted_tiles\":%llu,\"compact_normalize\":true,\"dpp_reduction\":%s,\"inference_acceptance\":false}\n", lanes, groups, k, cases, bad, ms, certified ? "true" : "false", attempted, accepted, QRT_SM121_DPP_REDUCTION ? "true" : "false");
            check(hipEventDestroy(end)); check(hipEventDestroy(begin));
            if (bad) throw std::runtime_error("staged dot versus independent CPU reference mismatch");
        }
        std::vector<uint16_t> after(a.size());
        check(hipMemcpy(after.data(), da, a.size() * 2u, hipMemcpyDeviceToHost));
        if (after != a) throw std::runtime_error("left input changed");
        check(hipMemcpy(after.data(), db, b.size() * 2u, hipMemcpyDeviceToHost));
        if (after != b) throw std::runtime_error("right input changed");
        check(hipFree(dc)); check(hipFree(dout)); check(hipFree(db)); check(hipFree(da));
    }
    if (!all_accepted) throw std::runtime_error("certificate path never used");
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "certified_dot_selftest_error=%s\n", error.what()); return 2; }
