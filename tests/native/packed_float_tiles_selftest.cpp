#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_strong_float_subgroup.h"
#include "strong_float_replay_cases.h"
#include "../../native/providers/moe_accumulator/sm121_packed_float_tiles.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cases = qrt_strong_replay_cases;
constexpr unsigned guard = 64u;
struct Result { float value; unsigned certified_groups, attempted, accepted; };
template<unsigned Lanes, unsigned Staging, bool Trace>
__global__ void execute(const uint16_t* left, const uint16_t* right, const unsigned* flags,
    Result* output, uint32_t* traces, unsigned rows, unsigned width) {
    const unsigned row = (blockIdx.x * blockDim.x + threadIdx.x) / Lanes;
    if (row >= rows) return;
    const unsigned accepted = flags[row] && width <= 4096u ? width / 16u : 0u;
    qrt_sm121_packed_tiles::Stats stats;
    const float value = qrt_sm121_packed_tiles::dot<Lanes, Staging, (Staging > 1u), Trace>(left + size_t(row) * width,
        right + size_t(row) * width, width, flags[row] != 0u, Trace ? traces + size_t(row) * (width / 16u) : nullptr, Trace ? &stats : nullptr);
    if (!(threadIdx.x & (Lanes - 1u))) output[row] = {value, accepted, stats.attempted, stats.accepted};
}
__global__ void certify_rows(const uint16_t* left, const uint16_t* right, unsigned* flags,
    unsigned rows, unsigned width) {
    const unsigned row=blockIdx.x;if(row>=rows)return;
    __shared__ unsigned invalid;
    if(!threadIdx.x)invalid=0u;
    __syncthreads();
    bool bad=false;
    for(unsigned k=threadIdx.x;k<width;k+=blockDim.x)
        bad |= !qrt_sm121_strong_float::eligible(left[size_t(row)*width+k]) ||
               !qrt_sm121_strong_float::eligible(right[size_t(row)*width+k]);
    if(bad)atomicOr(&invalid,1u);
    __syncthreads();
    if(!threadIdx.x)flags[row]=invalid==0u;
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
    Device& output, Device& traces, Device& flags, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
    const std::vector<uint32_t>& expected) {
    const size_t groups = size_t(rows) * (width / 16u);
    check(hipMemset(output.value, 0xa5, (rows + 2u * guard) * sizeof(Result)));
    check(hipMemset(traces.value, 0xa5, (groups + 2u * guard) * 4u));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(execute<Lanes, Staging, true>), dim3((rows * Lanes + 255u) / 256u),
        dim3(256u), 0u, nullptr, left.data<uint16_t>(), right.data<uint16_t>(), flags.data<unsigned>(), output.data<Result>(),
        traces.data<uint32_t>(), rows, width);
    check(hipGetLastError()); check(hipDeviceSynchronize());
    const auto values = read<Result>(output, rows); const auto actual = read<uint32_t>(traces, groups);
    guards(values); guards(actual);
    unsigned mismatches = 0u, accepted = 0u, attempted_tiles = 0u, accepted_tiles = 0u;
    for (size_t i = 0u; i < groups; ++i) if (actual[i + guard] != expected[i]) {
        if (mismatches < 4u) std::printf("DIFF lanes=%u staging=%u width=%u row=%zu group=%zu expected=%08x actual=%08x\n",
            Lanes, Staging, width, i / (width / 16u), i % (width / 16u), expected[i], actual[i + guard]);
        ++mismatches;
    }
    for (unsigned row = 0u; row < rows; ++row) {
        uint32_t bits; __builtin_memcpy(&bits, &values[row + guard].value, 4u);
        if (bits != expected[(size_t(row) + 1u) * (width / 16u) - 1u]) ++mismatches;
        const unsigned expected_count = cases::row_certificate(row,width) && width<=4096u ? width/16u : 0u;
        if (values[row + guard].certified_groups != expected_count) throw std::runtime_error("invalid certified group count");
        accepted += values[row + guard].certified_groups;
        attempted_tiles += values[row + guard].attempted; accepted_tiles += values[row + guard].accepted;
        const unsigned attempts = expected_count && Staging > 1u ? (width + 16u * Staging - 1u)/(16u * Staging) : 0u;
        if(values[row + guard].attempted != attempts || values[row + guard].accepted > attempts) throw std::runtime_error("tile statistics differ");
    }
    check(hipMemset(output.value, 0xa5, (rows + 2u * guard) * sizeof(Result)));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(execute<Lanes, Staging, false>), dim3((rows * Lanes + 255u) / 256u),
        dim3(256u), 0u, nullptr, left.data<uint16_t>(), right.data<uint16_t>(), flags.data<unsigned>(), output.data<Result>(),
        traces.data<uint32_t>(), rows, width);
    check(hipGetLastError()); check(hipDeviceSynchronize());
    const auto production = read<Result>(output,rows); guards(production);
    for(unsigned row=0;row<rows;row++) if(std::memcmp(&production[row+guard].value,&values[row+guard].value,4u))
        throw std::runtime_error("production/diagnostic result differs");
    if(read<uint32_t>(traces,groups)!=actual) throw std::runtime_error("production wrote diagnostic trace");
    const auto after_flags=read<unsigned>(flags,rows);guards(after_flags);
    for(unsigned row=0;row<rows;row++)if(after_flags[guard+row]!=unsigned(cases::row_certificate(row,width)))
        throw std::runtime_error("row certificate differs from CPU predicate");
    const auto after_a = read<uint16_t>(left, size_t(rows) * width), after_b = read<uint16_t>(right, size_t(rows) * width);
    guards(after_a); guards(after_b);
    if (std::memcmp(after_a.data() + guard, a.data(), a.size() * 2u) ||
        std::memcmp(after_b.data() + guard, b.data(), b.size() * 2u)) throw std::runtime_error("subgroup input changed");
    std::printf("{\"kind\":\"packed_float_tiles_safety\",\"lanes\":%u,\"staging_groups\":%u,\"rows\":%u,\"width\":%u,\"ordered_groups\":%zu,\"raw_bit_mismatches\":%u,\"certified_groups\":%u,\"fallback_groups\":%zu,\"attempted_tiles\":%u,\"accepted_tiles\":%u,\"production_diagnostic_parity\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
        Lanes, Staging, rows, width, groups, mismatches, accepted, groups - accepted, attempted_tiles, accepted_tiles);
    std::fflush(stdout);
    if (mismatches || (width<=4096u ? (!accepted || accepted==groups) : accepted!=0u)) throw std::runtime_error("strong replay arithmetic or coverage failure");
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
                a[index] = pair.x; b[index] = pair.y;
                values[i + 1u] = cases::original::multiply_bf16(pair.x, pair.y, -133);
            }
            carry = cases::original::group_sum<26, -133>(values, 17u);
            expected[size_t(row) * (width / 16u) + group] = cases::output_bits(carry);
        }
    }
    Device left((words + 2u * guard) * 2u), right((words + 2u * guard) * 2u);
    Device output((rows + 2u * guard) * sizeof(Result)), traces((groups + 2u * guard) * 4u), flags((rows+2u*guard)*4u);
    check(hipMemcpy(left.data<uint16_t>(), a.data(), a.size() * 2u, hipMemcpyHostToDevice));
    check(hipMemcpy(right.data<uint16_t>(), b.data(), b.size() * 2u, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(certify_rows,dim3(rows),dim3(256),0u,nullptr,
        left.data<uint16_t>(),right.data<uint16_t>(),flags.data<unsigned>(),rows,width);
    check(hipGetLastError());check(hipDeviceSynchronize());
    variant<4u, 1u>(rows, width, left, right, output, traces, flags, a, b, expected);
    variant<4u, 4u>(rows, width, left, right, output, traces, flags, a, b, expected);
    variant<4u, 8u>(rows, width, left, right, output, traces, flags, a, b, expected);
}
int main() try {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    run(257u,16u); run(4096u, 272u); run(2048u, 2048u); run(1024u, 4096u); run(128u,4112u);
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "strong_float_replay_error=%s\n", error.what()); return 1; }
