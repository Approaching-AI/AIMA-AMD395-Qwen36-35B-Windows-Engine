#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_karatsuba_core.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using Bf16x16 = unsigned short __attribute__((ext_vector_type(16)));
using F16x16 = _Float16 __attribute__((ext_vector_type(16)));
using F32x8 = float __attribute__((ext_vector_type(8)));
constexpr unsigned guard = 64u, tiles = 1024u, cells = tiles * 256u;

__global__ void encode_cores(uint16_t* encoded) {
    const unsigned core = blockIdx.x * blockDim.x + threadIdx.x;
    if (core >= 65536u) return;
    const auto d = qrt_sm121_karatsuba::split(uint16_t(core));
    encoded[core * 3u] = qrt_sm121_karatsuba::small_bf16(d.high);
    encoded[core * 3u + 1u] = qrt_sm121_karatsuba::small_bf16(d.low);
    encoded[core * 3u + 2u] = qrt_sm121_karatsuba::small_bf16(d.high + d.low);
}

// Each launch contains one matrix datatype only. Inputs come from immutable
// CPU-produced BF16 words, so matrix tests do not depend on GPU core encoding.
template<unsigned Kind>
__global__ void compare_contract(const uint16_t* input, float* output) {
    const unsigned lane = threadIdx.x, base = blockIdx.x * 512u;
    if constexpr (Kind < 2u) {
        F32x8 result{};
        if constexpr (Kind == 0u) {
            Bf16x16 a{}, b{};
            for (unsigned i = 0u; i < 16u; ++i) {
                a[i] = input[base + lane % 16u * 16u + i];
                b[i] = input[base + 256u + lane % 16u * 16u + i];
            }
            result = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, result);
        } else {
            F16x16 a{}, b{};
            for (unsigned i = 0u; i < 16u; ++i) {
                a[i] = _Float16(__uint_as_float(uint32_t(input[base + lane % 16u * 16u + i]) << 16u));
                b[i] = _Float16(__uint_as_float(uint32_t(input[base + 256u + lane % 16u * 16u + i]) << 16u));
            }
            result = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, result);
        }
        for (unsigned i = 0u; i < 8u; ++i)
            output[blockIdx.x * 256u + (2u * i + lane / 16u) * 16u + lane % 16u] = result[i];
    } else {
        for (unsigned cell = lane; cell < 256u; cell += 32u) {
            const unsigned row = cell / 16u, column = cell % 16u;
            float sum = 0.0f;
            for (unsigned i = 0u; i < 16u; i += 2u) {
                const uint16_t a0 = input[base + row * 16u + i], a1 = input[base + row * 16u + i + 1u];
                const uint16_t b0 = input[base + 256u + column * 16u + i], b1 = input[base + 256u + column * 16u + i + 1u];
                if constexpr (Kind == 2u) {
                    const uint32_t a = uint32_t(a0) | (uint32_t(a1) << 16u), b = uint32_t(b0) | (uint32_t(b1) << 16u);
                    asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(sum) : "v"(a), "v"(b));
                } else {
                    sum = __fadd_rn(sum, __fmul_rn(__uint_as_float(uint32_t(a0) << 16u), __uint_as_float(uint32_t(b0) << 16u)));
                    sum = __fadd_rn(sum, __fmul_rn(__uint_as_float(uint32_t(a1) << 16u), __uint_as_float(uint32_t(b1) << 16u)));
                }
            }
            output[blockIdx.x * 256u + cell] = sum;
        }
    }
}

void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
template<class T> struct Guarded {
    T* base = nullptr; size_t size;
    explicit Guarded(size_t n): size(n) {
        check(hipMalloc(reinterpret_cast<void**>(&base), (n + 2u * guard) * sizeof(T)));
        check(hipMemset(base, 0xa5, (n + 2u * guard) * sizeof(T)));
    }
    ~Guarded() { if (base && hipFree(base) != hipSuccess) std::abort(); }
    T* data() { return base + guard; }
    std::vector<T> read() { std::vector<T> result(size); check(hipMemcpy(result.data(), data(), size * sizeof(T), hipMemcpyDeviceToHost)); return result; }
    bool guards() {
        std::vector<unsigned char> a(guard * sizeof(T)), b(a.size());
        check(hipMemcpy(a.data(), base, a.size(), hipMemcpyDeviceToHost));
        check(hipMemcpy(b.data(), data() + size, b.size(), hipMemcpyDeviceToHost));
        return std::all_of(a.begin(), a.end(), [](unsigned char x) { return x == 0xa5; }) &&
               std::all_of(b.begin(), b.end(), [](unsigned char x) { return x == 0xa5; });
    }
};
uint32_t seed = 0x3958192u;
uint32_t next() { seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u; return seed; }

int main() try {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    Guarded<uint16_t> encoded(65536u * 3u);
    hipLaunchKernelGGL(encode_cores, dim3(256u), dim3(256u), 0u, nullptr, encoded.data());
    check(hipGetLastError()); check(hipDeviceSynchronize()); const auto code = encoded.read();
    unsigned encoding_bad = 0u;
    for (unsigned core = 0u; core < 65536u; ++core) {
        const auto d = qrt_sm121_karatsuba::split(uint16_t(core));
        const int values[] = {d.high, d.low, d.high + d.low};
        for (unsigned j = 0u; j < 3u; ++j) {
            uint32_t bits = uint32_t(code[core * 3u + j]) << 16u; float value; std::memcpy(&value, &bits, 4u);
            encoding_bad += unsigned(value != float(values[j]));
        }
    }
    const bool encoding_guard = encoded.guards();
    std::printf("{\"kind\":\"integer_matrix_encoding\",\"core_encodings\":65536,\"values\":196608,\"encoding_mismatches\":%u,\"redzones_pass\":%s,\"diagnostic_only\":true}\n", encoding_bad, encoding_guard ? "true" : "false");
    if (encoding_bad || !encoding_guard) return 2;
    for (unsigned mode = 0u; mode < 6u; ++mode) {
        std::vector<int> integers(tiles * 512u); std::vector<uint16_t> inputs(integers.size());
        for (unsigned i = 0u; i < integers.size(); ++i) {
            const int value = mode == 0u ? int(next() % 513u) - 256 : mode == 1u ? int(next() % 257u) :
                mode == 2u ? int(next() % 257u) * (i / 256u % 2u ? -1 : 1) :
                mode == 3u ? ((i % 16u == i / 512u % 16u) ? int(next() % 513u) - 256 : 0) :
                mode == 4u ? (i / 256u % 2u ? ((i & 1u) ? -255 : 255) : 255) : int(i / 512u % 257u);
            integers[i] = value; inputs[i] = qrt_sm121_karatsuba::small_bf16(value);
        }
        std::vector<int64_t> reference(cells);
        for (unsigned cell = 0u; cell < cells; ++cell) {
            const unsigned base = cell / 256u * 512u, row = cell / 16u % 16u, column = cell % 16u;
            for (unsigned i = 0u; i < 16u; ++i) reference[cell] += int64_t(integers[base + row * 16u + i]) * integers[base + 256u + column * 16u + i];
        }
        Guarded<uint16_t> input(inputs.size()); Guarded<float> output(cells);
        check(hipMemcpy(input.data(), inputs.data(), inputs.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        for (unsigned kind = 0u; kind < 4u; ++kind) {
            if (kind == 0u) hipLaunchKernelGGL(HIP_KERNEL_NAME(compare_contract<0u>), dim3(tiles), dim3(32u), 0u, nullptr, input.data(), output.data());
            else if (kind == 1u) hipLaunchKernelGGL(HIP_KERNEL_NAME(compare_contract<1u>), dim3(tiles), dim3(32u), 0u, nullptr, input.data(), output.data());
            else if (kind == 2u) hipLaunchKernelGGL(HIP_KERNEL_NAME(compare_contract<2u>), dim3(tiles), dim3(32u), 0u, nullptr, input.data(), output.data());
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(compare_contract<3u>), dim3(tiles), dim3(32u), 0u, nullptr, input.data(), output.data());
            check(hipGetLastError()); check(hipDeviceSynchronize()); const auto result = output.read(); const auto after = input.read();
            unsigned raw_bad = 0u, nearest_bad = 0u, nonfinite = 0u; double max_error = 0.0;
            for (unsigned i = 0u; i < cells; ++i) {
                if (!std::isfinite(result[i])) { ++nonfinite; continue; }
                raw_bad += unsigned(double(result[i]) != double(reference[i]));
                nearest_bad += unsigned(std::round(double(result[i])) != double(reference[i]));
                max_error = std::max(max_error, std::abs(double(result[i]) - double(reference[i])));
            }
            const bool stable = after == inputs, guards_pass = input.guards() && output.guards();
            std::printf("{\"kind\":\"integer_matrix_contract\",\"operand_mode\":%u,\"instruction_kind\":%u,\"cells\":%u,\"raw_mismatches\":%u,\"nearest_integer_mismatches\":%u,\"maximum_absolute_error\":%.9g,\"nonfinite\":%u,\"redzones_pass\":%s,\"immutable_inputs\":%s,\"diagnostic_only\":true,\"numerical_acceptance\":false,\"inference_acceptance\":false}\n", mode, kind, cells, raw_bad, nearest_bad, max_error, nonfinite, guards_pass ? "true" : "false", stable ? "true" : "false");
            if (nonfinite || !guards_pass || !stable || (kind == 3u && raw_bad)) return 2;
        }
    }
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "integer_wmma_contract_error=%s\n", error.what()); return 1; }
