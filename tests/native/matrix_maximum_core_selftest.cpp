#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_matrix_maximum.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace matrix = qrt_sm121_matrix_maximum;
constexpr unsigned tiles = 4096u, guard = 64u, inputs = tiles * 512u, outputs = tiles * 256u;
using B16 = uint16_t __attribute__((ext_vector_type(16)));
using F8 = float __attribute__((ext_vector_type(8)));
struct Result { float native; int maximum; };
__global__ void probe(const uint16_t* input, uint16_t* encoded, Result* output) {
    const unsigned tile = blockIdx.x, lane = threadIdx.x, source = lane % 16u;
    const size_t left = size_t(tile) * 512u + source * 16u, right = left + 256u;
    B16 a{}, b{}; unsigned left_valid = 1u, right_valid = 1u;
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) {
        const uint16_t x = matrix::encode(input[left + i]), y = matrix::encode(input[right + i]);
        left_valid &= unsigned(!(x & 0x8000u)); right_valid &= unsigned(!(y & 0x8000u));
        a[i] = x & 0x8000u ? 0u : x; b[i] = y & 0x8000u ? 0u : y;
        if (lane < 16u) { encoded[left + i] = x; encoded[right + i] = y; }
    }
    const F8 zero{};
    const auto product = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, zero);
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) {
        const unsigned row = 2u * i + lane / 16u, cell = row * 16u + source;
        output[size_t(tile) * 256u + cell] = {product[i], __shfl(left_valid, row, 32u) && right_valid
            ? matrix::recover(product[i]) : matrix::invalid_maximum};
    }
}
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void check(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
int main() try {
    hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0));
    require(!std::strncmp(device.gcnArchName, "gfx1151", 7u), "requires gfx1151");
    std::vector<uint16_t> source(inputs + 2u * guard, 0xa5a5u), expected_encoding(source.size(), 0xa5a5u);
    uint32_t random = 0x5a121u;
    auto next = [&] { random ^= random << 13u; random ^= random >> 17u; random ^= random << 5u; return random; };
    for (unsigned tile = 0u; tile < tiles; ++tile) for (unsigned side = 0u; side < 2u; ++side)
        for (unsigned row = 0u; row < 16u; ++row) for (unsigned i = 0u; i < 16u; ++i) {
            uint16_t x = uint16_t((next() & 0x807fu) | ((115u + next() % 25u) << 7u));
            const unsigned mode = tile % 8u;
            if (mode == 1u) x &= 0x8000u;
            if (mode == 2u) x = uint16_t((139u << 7u) | (next() & 0x807fu));
            if (mode == 3u) x = uint16_t((115u << 7u) | (next() & 0x807fu));
            if (mode == 4u && i != row) x &= 0x8000u;
            if (mode == 5u) x = uint16_t(((i == tile % 16u ? 139u : 115u) << 7u) | (next() & 0x807fu));
            if (mode == 6u && i == row && row % 3u == 0u) {
                const uint16_t edges[] = {uint16_t(114u << 7u), uint16_t(140u << 7u), 1u, 0x7f80u, 0x7fc1u};
                x = edges[(tile / 8u + side + row) % 5u];
            }
            if (mode == 7u && next() % 4u == 0u) x &= 0x8000u;
            const size_t index = guard + size_t(tile) * 512u + side * 256u + row * 16u + i;
            source[index] = x;
            const int e = int((x >> 7u) & 255u) - 127;
            expected_encoding[index] = !(x & 0x7fffu) ? 0u : e >= -12 && e <= 12
                ? uint16_t((127 + 5 * e) << 7u) : 0x8000u;
        }
    uint16_t *input = nullptr, *encoded = nullptr; Result* output = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&input), source.size() * 2u));
    check(hipMalloc(reinterpret_cast<void**>(&encoded), source.size() * 2u));
    check(hipMalloc(reinterpret_cast<void**>(&output), (outputs + 2u * guard) * sizeof(Result)));
    check(hipMemcpy(input, source.data(), source.size() * 2u, hipMemcpyHostToDevice));
    check(hipMemset(encoded, 0xa5, source.size() * 2u));
    check(hipMemset(output, 0xa5, (outputs + 2u * guard) * sizeof(Result)));
    hipLaunchKernelGGL(probe, dim3(tiles), dim3(32u), 0u, nullptr, input + guard, encoded + guard, output + guard);
    check(hipGetLastError());
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event); if (status == hipSuccess) break;
        require(status == hipErrorNotReady, "matrix maximum completion error");
        require(std::chrono::steady_clock::now() < deadline, "matrix maximum deadline"); std::this_thread::yield();
    }
    check(hipEventDestroy(event));
    std::vector<Result> actual(outputs + 2u * guard);
    std::vector<uint16_t> encoding(source.size()), after(source.size());
    check(hipMemcpy(actual.data(), output, actual.size() * sizeof(Result), hipMemcpyDeviceToHost));
    check(hipMemcpy(encoding.data(), encoded, encoding.size() * 2u, hipMemcpyDeviceToHost));
    check(hipMemcpy(after.data(), input, after.size() * 2u, hipMemcpyDeviceToHost));
    check(hipFree(output)); check(hipFree(encoded)); check(hipFree(input));
    require(source == after && encoding == expected_encoding, "input/encoding/guard mismatch");
    const auto* bytes = reinterpret_cast<const unsigned char*>(actual.data());
    for (size_t i = 0u; i < guard * sizeof(Result); ++i)
        require(bytes[i] == 0xa5u && bytes[(outputs + guard) * sizeof(Result) + i] == 0xa5u, "output guard mismatch");
    unsigned mismatches = 0u, valid_cells = 0u, fallback_cells = 0u, zeros = 0u, violations = 0u;
    double maximum_relative_error = 0.0;
    for (unsigned tile = 0u; tile < tiles; ++tile) for (unsigned row = 0u; row < 16u; ++row)
        for (unsigned column = 0u; column < 16u; ++column) {
            bool valid = true; int expected = -133; double positive = 0.0;
            for (unsigned i = 0u; i < 16u; ++i) {
                const uint16_t a = source[guard + size_t(tile) * 512u + row * 16u + i];
                const uint16_t b = source[guard + size_t(tile) * 512u + 256u + column * 16u + i];
                const int ae = int((a >> 7u) & 255u) - 127, be = int((b >> 7u) & 255u) - 127;
                valid &= !(a & 0x7fffu) || (ae >= -12 && ae <= 12);
                valid &= !(b & 0x7fffu) || (be >= -12 && be <= 12);
                if ((a & 0x7fffu) && (b & 0x7fffu)) { expected = std::max(expected, ae + be); positive += std::ldexp(1.0, 5 * (ae + be)); }
            }
            const auto result = actual[guard + size_t(tile) * 256u + row * 16u + column];
            mismatches += result.maximum != (valid ? expected : matrix::invalid_maximum);
            if (valid) {
                ++valid_cells;
                if (!positive) { ++zeros; violations += result.native != 0.0f; }
                else {
                    const double relative = std::fabs(double(result.native) - positive) / positive;
                    maximum_relative_error = std::max(maximum_relative_error, relative);
                    violations += !std::isfinite(relative) || relative > 0.25;
                }
            } else ++fallback_cells;
        }
    std::printf("{\"kind\":\"matrix_maximum_native\",\"matrix_output_cells\":%u,\"exact_gpu_encodings_checked\":%u,\"valid_cells\":%u,\"fallback_cells\":%u,\"zero_cells\":%u,\"maximum_relative_error\":%.9g,\"maximum_mismatches\":%u,\"conditional_bound_violations\":%u,\"native_relative_error_condition\":0.25,\"redzones_pass\":true,\"immutable_inputs\":true,\"hardware_error_bound_proven\":false,\"inference_acceptance\":false}\n",
        outputs, inputs, valid_cells, fallback_cells, zeros, maximum_relative_error, mismatches, violations);
    return mismatches || violations ? 2 : 0;
} catch (const std::exception& e) { std::fprintf(stderr, "matrix_maximum_error=%s\n", e.what()); return 1; }
