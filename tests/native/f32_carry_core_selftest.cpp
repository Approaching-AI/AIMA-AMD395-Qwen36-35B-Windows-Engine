#include <hip/hip_runtime.h>
#include "f32_carry_cases.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cases = qrt_float_alignment_cases;
namespace fast_cases = qrt_f32_carry_cases;
constexpr unsigned rows = 131072u, guard = 64u;
__global__ void compare(cases::Result* output, uint32_t* traces) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) output[row] = row < 65536u ? fast_cases::candidate<0u>(row, traces + row * 16u) : fast_cases::candidate<1u>(row - 65536u, traces + row * 16u);
}
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
int main() try {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    cases::Result* output = nullptr;
    uint32_t* traces = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&output), (rows + 2u * guard) * sizeof(*output)));
    check(hipMalloc(reinterpret_cast<void**>(&traces), (rows * 16u + 2u * guard) * 4u));
    check(hipMemset(output, 0xa5, (rows + 2u * guard) * sizeof(*output)));
    check(hipMemset(traces, 0xa5, (rows * 16u + 2u * guard) * 4u));
    hipLaunchKernelGGL(compare, dim3(rows / 256u), dim3(256u), 0u, nullptr, output + guard, traces + guard);
    check(hipGetLastError()); check(hipDeviceSynchronize());
    std::vector<cases::Result> results(rows + 2u * guard);
    std::vector<uint32_t> groups(rows * 16u + 2u * guard);
    check(hipMemcpy(results.data(), output, results.size() * sizeof(*output), hipMemcpyDeviceToHost));
    check(hipMemcpy(groups.data(), traces, groups.size() * 4u, hipMemcpyDeviceToHost));
    check(hipFree(output)); check(hipFree(traces));
    unsigned mismatches = 0u, group_mismatches = 0u, accepted = 0u, fallback = 0u;
    for (unsigned row = 0u; row < rows; ++row) {
        uint32_t expected_groups[16];
        const auto result = results[row + guard]; const auto expected = cases::reference(row % 65536u, expected_groups);
        for (unsigned group = 0u; group < 16u; ++group) {
            if (expected_groups[group] != groups[guard + row * 16u + group]) {
                if (group_mismatches < 4u) std::printf("GROUP_DIFF row=%u group=%u expected=%08x actual=%08x\n", row, group, expected_groups[group], groups[guard + row * 16u + group]);
                ++group_mismatches;
            }
        }
        if (result.bits != expected) {
            if (mismatches < 4u) std::printf("DIFF row=%u expected=%08x actual=%08x\n", row, expected, result.bits);
            ++mismatches;
        }
        if (result.accepted + result.fallback != 16u) throw std::runtime_error("incomplete ordered group sequence");
        accepted += result.accepted; fallback += result.fallback;
    }
    const auto* bytes = reinterpret_cast<const unsigned char*>(results.data()); bool guards = true;
    for (size_t i = 0u; i < guard * sizeof(*output); ++i)
        guards = guards && bytes[i] == 0xa5 && bytes[(rows + guard) * sizeof(*output) + i] == 0xa5;
    for (unsigned i = 0u; i < guard; ++i)
        guards = guards && groups[i] == 0xa5a5a5a5u && groups[rows * 16u + guard + i] == 0xa5a5a5a5u;
    std::printf("{\"kind\":\"f32_carry_alignment\",\"dots\":131072,\"variants\":2,\"ordered_groups\":2097152,\"raw_bit_mismatches\":%u,\"group_mismatches\":%u,\"float_groups\":%u,\"fallback_groups\":%u,\"redzones_pass\":%s,\"immutable_inputs\":true,\"inference_acceptance\":false}\n", mismatches, group_mismatches, accepted, fallback, guards ? "true" : "false");
    return !mismatches && !group_mismatches && accepted && fallback && guards ? 0 : 2;
} catch (const std::exception& error) { std::fprintf(stderr, "float_alignment_error=%s\n", error.what()); return 1; }
