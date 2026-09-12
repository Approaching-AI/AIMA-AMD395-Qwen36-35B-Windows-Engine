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
template<unsigned Lanes>
__global__ void probe(const uint16_t* a, const uint16_t* b, float* output,
                      unsigned count, unsigned k) {
    const unsigned index = (blockIdx.x * blockDim.x + threadIdx.x) / Lanes;
    if (index >= count) return;
    const float value = qrt_sm121_subgroup::dot<Lanes>(a + index * (k + 3u) + 1u,
                                                     b + index * (k + 3u) + 1u, k);
    if ((threadIdx.x & (Lanes - 1u)) == 0u) output[index] = value;
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
struct Reduction {
    int maximum_dpp, maximum_shuffle;
    uint32_t sum_dpp, sum_shuffle;
};
static_assert(sizeof(Reduction) == 16);
__host__ __device__ bool active_group(unsigned group, unsigned mode) {
    return mode == 0u || (mode == 1u ? (group & 1u) != 0u : ((0x96u >> (group & 7u)) & 1u) != 0u);
}
template<unsigned Lanes>
__global__ void reduce_probe(const int* input, Reduction* output, unsigned groups, unsigned mode) {
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned group = lane / Lanes;
    if (group >= groups || !active_group(group, mode)) return;
    const int value = input[lane];
    output[lane] = {qrt_sm121_lane_reduce::maximum<Lanes, true>(value),
                    qrt_sm121_lane_reduce::maximum<Lanes, false>(value),
                    qrt_sm121_lane_reduce::sum<Lanes, true>(uint32_t(value)),
                    qrt_sm121_lane_reduce::sum<Lanes, false>(uint32_t(value))};
}
template<unsigned Lanes>
void check_reductions() {
    constexpr unsigned groups = 65539u, count = groups * Lanes, guard = 64u;
    std::vector<int> input(count);
    for (unsigned i = 0; i < count; ++i) {
        uint32_t word = random_word();
        switch ((i / Lanes) % 16u) {
            case 0: word = 0u; break;
            case 1: word = 0x80000000u; break;
            case 2: word = 0x7fffffffu; break;
            case 3: word = 0xffffffffu; break;
            case 4: word = (i & 1u) ? 0x80000000u : 0x7fffffffu; break;
            case 5: word = uint32_t(-133 - int(i % Lanes)); break;
        }
        std::memcpy(&input[i], &word, sizeof(word));
    }
    int* device_input = nullptr; Reduction* device_output = nullptr;
    std::vector<Reduction> expected(count + 2u * guard), result(expected.size());
    check(hipMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(int)));
    check(hipMalloc(reinterpret_cast<void**>(&device_output), result.size() * sizeof(Reduction)));
    check(hipMemcpy(device_input, input.data(), input.size() * sizeof(int), hipMemcpyHostToDevice));
    for (unsigned mode : {0u, 1u, 2u}) {
        std::memset(expected.data(), 0xa5, expected.size() * sizeof(Reduction));
        check(hipMemcpy(device_output, expected.data(), expected.size() * sizeof(Reduction), hipMemcpyHostToDevice));
        unsigned active = 0u;
        for (unsigned group = 0u; group < groups; ++group) {
            if (!active_group(group, mode)) continue;
            ++active;
            int maximum = INT_MIN; uint32_t sum = 0u;
            for (unsigned lane = 0u; lane < Lanes; ++lane) {
                const int value = input[group * Lanes + lane];
                maximum = std::max(maximum, value); sum += uint32_t(value);
            }
            for (unsigned lane = 0u; lane < Lanes; ++lane)
                expected[guard + group * Lanes + lane] = {maximum, maximum, sum, sum};
        }
        hipEvent_t done; check(hipEventCreate(&done));
        hipLaunchKernelGGL(reduce_probe<Lanes>, dim3((count + 255u) / 256u), dim3(256u), 0u, nullptr,
                           device_input, device_output + guard, groups, mode);
        check(hipGetLastError()); check(hipEventRecord(done)); wait_for(done);
        check(hipMemcpy(result.data(), device_output, result.size() * sizeof(Reduction), hipMemcpyDeviceToHost));
        check(hipEventDestroy(done));
        unsigned bad = 0u;
        for (unsigned i = 0u; i < result.size(); ++i)
            bad += std::memcmp(&result[i], &expected[i], sizeof(Reduction)) != 0;
        std::printf("{\"kind\":\"sm121_lane_reductions\",\"lanes\":%u,\"mask_mode\":%u,\"groups\":%u,\"active_groups\":%u,\"checked_lanes\":%u,\"bit_mismatches\":%u,\"includes_inactive_and_redzones\":true,\"inference_acceptance\":false}\n",
                    Lanes, mode, groups, active, count, bad);
        if (bad) throw std::runtime_error("DPP/shuffle versus scalar integer reduction mismatch");
    }
    std::vector<int> after(input.size());
    check(hipMemcpy(after.data(), device_input, input.size() * sizeof(int), hipMemcpyDeviceToHost));
    if (input != after) throw std::runtime_error("reduction input changed");
    check(hipFree(device_output)); check(hipFree(device_input));
}
int main() try {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    check_reductions<4u>(); check_reductions<8u>(); check_reductions<16u>();
    for (unsigned k : {16u, 512u, 2048u}) {
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
        check(hipMalloc(reinterpret_cast<void**>(&da), a.size() * 2u));
        check(hipMalloc(reinterpret_cast<void**>(&db), b.size() * 2u));
        check(hipMalloc(reinterpret_cast<void**>(&dout), result.size() * 4u));
        check(hipMemcpy(da, a.data(), a.size() * 2u, hipMemcpyHostToDevice));
        check(hipMemcpy(db, b.data(), b.size() * 2u, hipMemcpyHostToDevice));
        for (unsigned lanes : {16u, 8u, 4u}) {
            std::fill(result.begin(), result.end(), 12345.0f);
            check(hipMemcpy(dout, result.data(), result.size() * 4u, hipMemcpyHostToDevice));
            const dim3 grid((cases * lanes + 255u) / 256u);
            hipEvent_t begin, end; check(hipEventCreate(&begin)); check(hipEventCreate(&end));
            check(hipEventRecord(begin));
            if (lanes == 16u) hipLaunchKernelGGL(probe<16u>, grid, dim3(256u), 0u, nullptr, da, db, dout, cases, k);
            else if (lanes == 8u) hipLaunchKernelGGL(probe<8u>, grid, dim3(256u), 0u, nullptr, da, db, dout, cases, k);
            else hipLaunchKernelGGL(probe<4u>, grid, dim3(256u), 0u, nullptr, da, db, dout, cases, k);
            check(hipGetLastError()); check(hipEventRecord(end)); wait_for(end);
            float ms = 0; check(hipEventElapsedTime(&ms, begin, end));
            check(hipMemcpy(result.data(), dout, result.size() * 4u, hipMemcpyDeviceToHost));
            unsigned bad = 0u;
            for (unsigned i = 0u; i < cases; ++i) bad += bits(result[i]) != bits(expected[i]);
            for (unsigned i = cases; i < result.size(); ++i) if (result[i] != 12345.0f) throw std::runtime_error("output redzone");
            std::printf("{\"kind\":\"sm121_subgroup_dot\",\"lanes\":%u,\"k\":%u,\"dots\":%u,\"bit_mismatches\":%u,\"ms\":%.6f,\"dpp_reduction\":%s,\"inference_acceptance\":false}\n", lanes, k, cases, bad, ms, QRT_SM121_DPP_REDUCTION ? "true" : "false");
            check(hipEventDestroy(end)); check(hipEventDestroy(begin));
            if (bad) throw std::runtime_error("subgroup dot mismatch");
        }
        std::vector<uint16_t> after(a.size());
        check(hipMemcpy(after.data(), da, a.size() * 2u, hipMemcpyDeviceToHost));
        if (after != a) throw std::runtime_error("left input changed");
        check(hipMemcpy(after.data(), db, b.size() * 2u, hipMemcpyDeviceToHost));
        if (after != b) throw std::runtime_error("right input changed");
        check(hipFree(dout)); check(hipFree(db)); check(hipFree(da));
    }
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "subgroup_selftest_error=%s\n", error.what()); return 2; }
