#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_float_subgroup.h"
#include "float_alignment_cases.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cases = qrt_float_alignment_cases;
constexpr unsigned guard = 64u;
struct Result { float value; unsigned float_groups; };
template<unsigned Lanes, unsigned Staging>
__global__ void execute(const uint16_t* left, const uint16_t* right,
    Result* output, uint32_t* traces, unsigned rows, unsigned width) {
    const unsigned row = (blockIdx.x * blockDim.x + threadIdx.x) / Lanes;
    if (row >= rows) return;
    unsigned accepted = 0u;
    const float value = qrt_sm121_float_subgroup::dot<Lanes, Staging>(left + size_t(row) * width,
        right + size_t(row) * width, width, traces + size_t(row) * (width / 16u), &accepted);
    if (!(threadIdx.x & (Lanes - 1u))) output[row] = {value, accepted};
}
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
struct Device {
    void* value = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&value, bytes)); check(hipMemset(value, 0xa5, bytes)); }
    ~Device() { if (value && hipFree(value) != hipSuccess) std::abort(); }
    template<class T> T* data() { return static_cast<T*>(value) + guard; }
};
template<class T> std::vector<T> read(Device& device, size_t count) {
    std::vector<T> result(count + 2u * guard);
    check(hipMemcpy(result.data(), device.value, result.size() * sizeof(T), hipMemcpyDeviceToHost));
    return result;
}
template<class T> void guards(const std::vector<T>& values) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
    for (size_t i = 0u; i < guard * sizeof(T); ++i)
        if (bytes[i] != 0xa5 || bytes[(values.size() - guard) * sizeof(T) + i] != 0xa5)
            throw std::runtime_error("subgroup redzone changed");
}
template<unsigned Lanes, unsigned Staging>
void variant(unsigned rows, unsigned width, Device& left, Device& right,
    Device& output, Device& traces, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
    const std::vector<uint32_t>& expected) {
    const size_t groups = size_t(rows) * (width / 16u);
    check(hipMemset(output.value, 0xa5, (rows + 2u * guard) * sizeof(Result)));
    check(hipMemset(traces.value, 0xa5, (groups + 2u * guard) * 4u));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(execute<Lanes, Staging>), dim3((rows * Lanes + 255u) / 256u),
        dim3(256u), 0u, nullptr, left.data<uint16_t>(), right.data<uint16_t>(), output.data<Result>(),
        traces.data<uint32_t>(), rows, width);
    check(hipGetLastError()); check(hipDeviceSynchronize());
    const auto values = read<Result>(output, rows); const auto actual = read<uint32_t>(traces, groups);
    guards(values); guards(actual);
    unsigned mismatches = 0u, accepted = 0u;
    for (size_t i = 0u; i < groups; ++i) if (actual[i + guard] != expected[i]) {
        if (mismatches < 4u) std::printf("DIFF lanes=%u staging=%u width=%u row=%zu group=%zu expected=%08x actual=%08x\n",
            Lanes, Staging, width, i / (width / 16u), i % (width / 16u), expected[i], actual[i + guard]);
        ++mismatches;
    }
    for (unsigned row = 0u; row < rows; ++row) {
        uint32_t bits; __builtin_memcpy(&bits, &values[row + guard].value, 4u);
        if (bits != expected[(size_t(row) + 1u) * (width / 16u) - 1u]) ++mismatches;
        if (values[row + guard].float_groups > width / 16u) throw std::runtime_error("invalid subgroup group count");
        accepted += values[row + guard].float_groups;
    }
    const auto after_a = read<uint16_t>(left, size_t(rows) * width), after_b = read<uint16_t>(right, size_t(rows) * width);
    guards(after_a); guards(after_b);
    if (std::memcmp(after_a.data() + guard, a.data(), a.size() * 2u) ||
        std::memcmp(after_b.data() + guard, b.data(), b.size() * 2u)) throw std::runtime_error("subgroup input changed");
    std::printf("{\"kind\":\"float_subgroup_safety\",\"lanes\":%u,\"staging_groups\":%u,\"rows\":%u,\"width\":%u,\"ordered_groups\":%zu,\"raw_bit_mismatches\":%u,\"float_groups\":%u,\"fallback_groups\":%zu,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
        Lanes, Staging, rows, width, groups, mismatches, accepted, groups - accepted);
    std::fflush(stdout);
    if (mismatches || !accepted || accepted == groups) throw std::runtime_error("subgroup arithmetic or coverage failure");
}
void run(unsigned rows, unsigned width) {
    const size_t words = size_t(rows) * width, groups = words / 16u;
    std::vector<uint16_t> a(words), b(words); std::vector<uint32_t> expected(groups);
    for (unsigned row = 0u; row < rows; ++row) {
        cases::original::Value carry{0u, -133, false};
        for (unsigned group = 0u; group < width / 16u; ++group) {
            cases::original::Value values[17]; values[0] = carry;
            for (unsigned i = 0u; i < 16u; ++i) {
                const auto pair = cases::input(row, group, i); const size_t index = size_t(row) * width + group * 16u + i;
                a[index] = pair.left; b[index] = pair.right;
                values[i + 1u] = cases::original::multiply_bf16(pair.left, pair.right, -133);
            }
            carry = cases::original::group_sum<26, -133>(values, 17u);
            expected[size_t(row) * (width / 16u) + group] = cases::output_bits(carry);
        }
    }
    Device left((words + 2u * guard) * 2u), right((words + 2u * guard) * 2u);
    Device output((rows + 2u * guard) * sizeof(Result)), traces((groups + 2u * guard) * 4u);
    check(hipMemcpy(left.data<uint16_t>(), a.data(), a.size() * 2u, hipMemcpyHostToDevice));
    check(hipMemcpy(right.data<uint16_t>(), b.data(), b.size() * 2u, hipMemcpyHostToDevice));
    variant<4u, 1u>(rows, width, left, right, output, traces, a, b, expected);
    variant<4u, 4u>(rows, width, left, right, output, traces, a, b, expected);
    variant<8u, 1u>(rows, width, left, right, output, traces, a, b, expected);
    variant<16u, 1u>(rows, width, left, right, output, traces, a, b, expected);
    variant<16u, 4u>(rows, width, left, right, output, traces, a, b, expected);
    variant<16u, 8u>(rows, width, left, right, output, traces, a, b, expected);
}
int main() try {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    run(4096u, 272u); run(2048u, 2048u); run(1024u, 4096u);
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "float_subgroup_error=%s\n", error.what()); return 1; }
