"""Execute both real kernel mappings; validate every issued WMMA operand.

The CPU stand-in checks transport and ordered calls, not AMD matrix arithmetic.
Native replay and the frozen real-model matrix supply numerical qualification.
"""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class WmmaOperandStagingTests(unittest.TestCase):
    def test_real_fragments_tails_reuse_and_endpoints(self):
        provider = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        kernels = '\n'.join(function(provider, 'void ' + name + '(') for name in (
            'selected_bf16_projection_wmma_k16_m64_kernel',
            'selected_bf16_projection_wmma_k16_m64_lds_kernel'))
        source = r'''
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <cstdio>
constexpr unsigned QRT_QWEN36_HIDDEN_SIZE = 2048u;
using SelectedProjectionWmmaBf16x16 = std::array<uint16_t,16>;
using SelectedProjectionWmmaF32x8 = std::array<float,8>;
#define __shared__ static
struct Dim { unsigned x = 0, y = 0; } blockIdx;
thread_local Dim threadIdx;
thread_local unsigned call;
std::mutex lock;
std::condition_variable changed;
unsigned arrived = 0, generation = 0;
std::atomic<bool> bad{false};
const uint16_t *original_weights, *original_inputs;
unsigned actual_rows, actual_tokens, abs_mode;
void __syncthreads() {
    std::unique_lock<std::mutex> guard(lock);
    const unsigned current = generation;
    if (++arrived == 256u) { arrived = 0; ++generation; changed.notify_all(); }
    else changed.wait(guard, [&] { return generation != current; });
}
float device_bf16_round_to_float(float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4);
    bits = (bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u;
    std::memcpy(&value, &bits, 4); return value;
}
SelectedProjectionWmmaF32x8 simulated_wmma(
    const SelectedProjectionWmmaBf16x16& input,
    const SelectedProjectionWmmaBf16x16& weight,
    SelectedProjectionWmmaF32x8 result) {
    const unsigned fragment = call % 4u, k = call / 4u * 16u;
    const unsigned source = threadIdx.x % 16u;
    const unsigned row = blockIdx.x * 128u + threadIdx.x / 32u * 16u + source;
    const unsigned token = blockIdx.y * 64u + fragment * 16u + source;
    const unsigned mask = abs_mode ? 0x7fffu : 0xffffu;
    unsigned fingerprint = 0;
    if (k >= QRT_QWEN36_HIDDEN_SIZE) bad = true;
    for (unsigned e = 0; e < 16; ++e) {
        const uint16_t expected_input = token < actual_tokens
            ? original_inputs[size_t(token) * 2048u + k + e] & mask : 0u;
        const uint16_t expected_weight = row < actual_rows
            ? original_weights[size_t(row) * 2048u + k + e] & mask : 0u;
        if (input[e] != expected_input || weight[e] != expected_weight) bad = true;
        fingerprint += ((input[e] * 17u) ^ (weight[e] * 31u)) & 0x3ffu;
    }
    for (unsigned e = 0; e < 8; ++e) result[e] += float(fingerprint + e);
    ++call;
    return result;
}
#define __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 simulated_wmma
''' + kernels + r'''
int main() {
    constexpr unsigned guard = 17u; // Also exercises BF16 pointers offset by two bytes.
    unsigned seed = 0x8191395u;
    for (auto shape : {std::array<unsigned,4>{129,65,0,0},
                       std::array<unsigned,4>{17,7,1,1},
                       std::array<unsigned,4>{128,64,1,0},
                       std::array<unsigned,4>{1,1,0,1}}) {
        actual_rows = shape[0]; actual_tokens = shape[1]; abs_mode = shape[3];
        const size_t cells = size_t(actual_rows) * actual_tokens;
        std::vector<uint16_t> weights(size_t(actual_rows) * 2048u + guard * 2u, 0xdead);
        std::vector<uint16_t> inputs(size_t(actual_tokens) * 2048u + guard * 2u, 0xdead);
        for (auto* operand : {&weights, &inputs})
            for (size_t i = guard; i + guard < operand->size(); ++i) {
                seed = seed * 1664525u + 1013904223u;
                (*operand)[i] = uint16_t((seed >> 16u) & 0xffffu);
            }
        const auto saved_weights = weights, saved_inputs = inputs;
        original_weights = weights.data() + guard;
        original_inputs = inputs.data() + guard;
        std::vector<float> legacy(cells + guard * 2u, -12345.0f), staged = legacy;
        for (unsigned by = 0; by < (actual_tokens + 63u) / 64u; ++by)
            for (unsigned bx = 0; bx < (actual_rows + 127u) / 128u; ++bx) {
                blockIdx = {bx, by};
                for (unsigned mode = 0; mode < 2u; ++mode) {
                    std::vector<std::thread> threads;
                    for (unsigned thread = 0; thread < 256u; ++thread)
                        threads.emplace_back([&, thread, mode] {
                            threadIdx = {thread, 0}; call = 0;
                            auto kernel = mode ? selected_bf16_projection_wmma_k16_m64_lds_kernel
                                               : selected_bf16_projection_wmma_k16_m64_kernel;
                            kernel(original_weights, original_inputs,
                                (mode ? staged : legacy).data() + guard,
                                actual_rows, actual_tokens, shape[2], shape[3]);
                            if (call != 512u) bad = true;
                        });
                    for (auto& thread : threads) thread.join();
                }
            }
        if (bad || weights != saved_weights || inputs != saved_inputs) return 1;
        if (legacy != staged) return 2;
        for (size_t i = 0; i < legacy.size(); ++i) {
            const bool outside = i < guard || i >= cells + guard;
            if ((legacy[i] == -12345.0f) != outside) return 3;
        }
    }
    std::puts("staged_wmma_operands_order_tails_redzones=pass");
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-wmma-staging-') as tmp:
            exe = str(Path(tmp) / 'check')
            compiled = subprocess.run(
                [os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                 '-fsanitize=address,undefined', '-x', 'c++', '-', '-o', exe],
                input=source, text=True, capture_output=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([exe], text=True, capture_output=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('staged_wmma_operands_order_tails_redzones=pass', result.stdout)


if __name__ == '__main__':
    unittest.main()
