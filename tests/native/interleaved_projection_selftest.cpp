#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_strong_float_subgroup.h"
#include "strong_float_replay_cases.h"
#include "../../native/providers/moe_accumulator/sm121_interleaved_projection.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cases = qrt_strong_replay_cases;
constexpr unsigned guard = 65u; // Exercise unaligned packed operand loads.
struct Result { float value; unsigned floating, integer; };
template<unsigned Outputs, bool Trace>
__global__ void execute(const uint16_t* left, const uint16_t* right, const unsigned* flags,
    Result* output, uint32_t* traces, unsigned rows, unsigned width, bool shared) {
    const unsigned first = ((blockIdx.x * blockDim.x + threadIdx.x) / 4u) * Outputs;
    if (first >= rows) return;
    const unsigned active = (rows - first < Outputs) ? rows - first : Outputs;
    const uint16_t* a[Outputs]; const uint16_t* b[Outputs]; uint32_t* history[Outputs];
    bool eligible[Outputs]; float result[Outputs]; qrt_sm121_interleaved_projection::Stats stats[Outputs];
#pragma unroll
    for (unsigned slot = 0u; slot < Outputs; ++slot) if (slot < active) {
        const unsigned row = first + slot;
        a[slot] = left + size_t(shared ? first : row) * width;
        b[slot] = right + size_t(row) * width;
        eligible[slot] = flags[row] != 0u;
        history[slot] = traces + size_t(row) * (width / 16u);
    }
    qrt_sm121_interleaved_projection::dot<Outputs, Trace>(a, b, eligible, active,
        width, result, Trace ? history : nullptr, Trace ? stats : nullptr);
    if (!(threadIdx.x & 3u)) {
#pragma unroll
        for (unsigned slot = 0u; slot < Outputs; ++slot) if (slot < active)
            output[first + slot] = {result[slot], stats[slot].floating, stats[slot].integer};
    }
}
__global__ void certify_rows(const uint16_t* left, const uint16_t* right, unsigned* flags,
    unsigned rows, unsigned width) {
    const unsigned row=blockIdx.x;if(row>=rows)return;
    __shared__ unsigned invalid;
    if(!threadIdx.x)invalid=0u;
    __syncthreads();
    bool bad=false;
    for(unsigned k=threadIdx.x;k<width;k+=blockDim.x)
        bad |= !qrt_sm121_float_alignment::eligible(left[size_t(row)*width+k]) ||
               !qrt_sm121_float_alignment::eligible(right[size_t(row)*width+k]);
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
template<unsigned Outputs>
void variant(bool shared, unsigned rows, unsigned width, Device& left, Device& right,
    Device& output, Device& traces, Device& flags, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
    const std::vector<uint32_t>& expected, const std::vector<unsigned>& expected_flags,
    const std::vector<unsigned>& expected_floating) {
    const size_t groups = size_t(rows) * (width / 16u);
    check(hipMemset(output.value, 0xa5, (rows + 2u * guard) * sizeof(Result)));
    check(hipMemset(traces.value, 0xa5, (groups + 2u * guard) * 4u));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(execute<Outputs, true>), dim3((rows + Outputs * 64u - 1u) / (Outputs * 64u)),
        dim3(256u), 0u, nullptr, left.data<uint16_t>(), right.data<uint16_t>(), flags.data<unsigned>(), output.data<Result>(),
        traces.data<uint32_t>(), rows, width, shared);
    check(hipGetLastError()); check(hipDeviceSynchronize());
    const auto values = read<Result>(output, rows); const auto actual = read<uint32_t>(traces, groups);
    guards(values); guards(actual);
    unsigned mismatches = 0u, floating = 0u, integer = 0u;
    for (size_t i = 0u; i < groups; ++i) if (actual[i + guard] != expected[i]) {
        if (mismatches < 4u) std::printf("DIFF lanes=4 width=%u row=%zu group=%zu expected=%08x actual=%08x\n",
            width, i / (width / 16u), i % (width / 16u), expected[i], actual[i + guard]);
        ++mismatches;
    }
    for (unsigned row = 0u; row < rows; ++row) {
        uint32_t bits; __builtin_memcpy(&bits, &values[row + guard].value, 4u);
        if (bits != expected[(size_t(row) + 1u) * (width / 16u) - 1u]) ++mismatches;
        const auto& result = values[row + guard];
        if (result.floating != expected_floating[row] || result.integer != width / 16u - expected_floating[row])
            throw std::runtime_error("float/fallback count differs from independent original exponent check");
        floating += result.floating; integer += result.integer;
    }
    check(hipMemset(output.value, 0xa5, (rows + 2u * guard) * sizeof(Result)));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(execute<Outputs, false>), dim3((rows + Outputs * 64u - 1u) / (Outputs * 64u)),
        dim3(256u), 0u, nullptr, left.data<uint16_t>(), right.data<uint16_t>(), flags.data<unsigned>(), output.data<Result>(),
        traces.data<uint32_t>(), rows, width, shared);
    check(hipGetLastError()); check(hipDeviceSynchronize());
    const auto production = read<Result>(output,rows); guards(production);
    for(unsigned row=0;row<rows;row++) if(std::memcmp(&production[row+guard].value,&values[row+guard].value,4u))
        throw std::runtime_error("production/diagnostic result differs");
    if(read<uint32_t>(traces,groups)!=actual) throw std::runtime_error("production wrote diagnostic trace");
    const auto after_flags=read<unsigned>(flags,rows);guards(after_flags);
    for(unsigned row=0;row<rows;row++)if(after_flags[guard+row]!=expected_flags[row])
        throw std::runtime_error("row certificate differs from CPU predicate");
    const auto after_a = read<uint16_t>(left, size_t(rows) * width), after_b = read<uint16_t>(right, size_t(rows) * width);
    guards(after_a); guards(after_b);
    if (std::memcmp(after_a.data() + guard, a.data(), a.size() * 2u) ||
        std::memcmp(after_b.data() + guard, b.data(), b.size() * 2u)) throw std::runtime_error("subgroup input changed");
    std::printf("{\"kind\":\"interleaved_projection_safety\",\"lanes\":4,\"outputs_per_subgroup\":%u,\"shared_inputs\":%s,\"rows\":%u,\"width\":%u,\"ordered_groups\":%zu,\"raw_bit_mismatches\":%u,\"floating_groups\":%u,\"integer_groups\":%u,\"unaligned_operands\":true,\"production_diagnostic_parity\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
        Outputs, shared ? "true" : "false", rows, width, groups, mismatches, floating, integer);
    std::fflush(stdout);
    if (mismatches || !floating || !integer || floating + integer != groups)
        throw std::runtime_error("interleaved arithmetic or coverage failure");
}
void run(unsigned rows, unsigned width, bool shared) {
    const size_t words = size_t(rows) * width, groups = words / 16u;
    std::vector<uint16_t> a(words), b(words); std::vector<uint32_t> expected(groups);
    std::vector<unsigned> expected_flags(rows, 1u), expected_floating(rows, 0u);
    for (unsigned row = 0u; row < rows; ++row) {
        for (unsigned group = 0u; group < width / 16u; ++group) for (unsigned i = 0u; i < 16u; ++i) {
            auto pair = cases::input(row, group, i);
            if (shared) pair.x = cases::input((row / 4u) * 4u, group, i).x;
            expected_flags[row] &= unsigned(cases::alignment::eligible(pair.x) && cases::alignment::eligible(pair.y));
        }
        cases::original::Value carry{0u, -133, false};
        for (unsigned group = 0u; group < width / 16u; ++group) {
            cases::original::Value values[17]; values[0] = carry;
            for (unsigned i = 0u; i < 16u; ++i) {
                auto pair = cases::input(row, group, i);
                if (shared) pair.x = cases::input((row / 4u) * 4u, group, i).x;
                const size_t index = size_t(row) * width + group * 16u + i;
                a[index] = pair.x; b[index] = pair.y;
                values[i + 1u] = cases::original::multiply_bf16(pair.x, pair.y, -133);
            }
            int maximum = -133;
            for (const auto& value : values) maximum = std::max(maximum, int(value.exponent));
            expected_floating[row] += expected_flags[row] &&
                ((maximum == -133 && !carry.significand) || (maximum >= -101 && maximum <= 151));
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
    variant<2u>(shared, rows, width, left, right, output, traces, flags, a, b, expected, expected_flags, expected_floating);
    variant<4u>(shared, rows, width, left, right, output, traces, flags, a, b, expected, expected_flags, expected_floating);
}
int main() try {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    for (bool shared : {false, true}) {
        run(257u,16u,shared); run(4096u,272u,shared); run(2048u,2048u,shared);
        run(1024u,4096u,shared); run(129u,4112u,shared); run(129u,8192u,shared);
    }
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "interleaved_projection_error=%s\n", error.what()); return 1; }
