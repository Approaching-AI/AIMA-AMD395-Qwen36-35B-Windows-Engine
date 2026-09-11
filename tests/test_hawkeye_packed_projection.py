"""Execute actual packed candidate math and transpose, with a scalar oracle."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class HawkeyePackedProjectionTests(unittest.TestCase):
    def test_sparse_candidates_and_partial_transpose_against_original_scalar(self):
        provider = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        wave = (ROOT / 'native/providers/moe_accumulator/sm121_wave16.h').read_text()
        normalizer = function(wave, '__device__ __forceinline__ qrt_q1_moe_hawkeye::Value\nnormalize(')
        transpose = function(provider, '__global__ void selected_hawkeye_transpose_weights_kernel(')
        packed = function(provider, 'void selected_bf16_projection_hawkeye_packed_correction_kernel(')
        source = r'''
#include "native/providers/moe_accumulator/sm121_group16_modulo.h"
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <cstring>
#include <cstdio>
#define __device__
#define __forceinline__ inline
#define __global__
#define __shared__ static
#define __clz __builtin_clz
struct Dim { unsigned x = 0, y = 0; } blockIdx, blockDim;
thread_local Dim threadIdx;
std::mutex lock;
std::condition_variable changed;
unsigned arrived = 0, generation = 0;
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
namespace qrt_sm121_wave16 {
constexpr int16_t kZeroExponent = -133;
''' + normalizer + '\n}\n' + transpose + '\n' + packed + r'''
uint32_t random_state = 0x395a8192u;
uint16_t operand() {
    random_state = random_state * 1664525u + 1013904223u;
    return uint16_t(((random_state >> 16u) & 0x8000u) | (0x3800u + (random_state & 0x7ffu)));
}
int main() {
    // Run the actual shared-memory transpose with real block barriers. These
    // rectangles cover simultaneous row/column tails and the minimum K16.
    for (const auto shape : {std::pair{35u, 48u}, std::pair{17u, 16u}}) {
        const unsigned rows = shape.first, k = shape.second;
        std::vector<uint16_t> weights(size_t(rows) * k + 64u, 0xdeadu);
        std::vector<uint16_t> transposed(weights.size(), 0xdeadu);
        for (size_t i = 32u; i + 32u < weights.size(); ++i) weights[i] = operand();
        const auto original = weights;
        for (unsigned by = 0u; by < (rows + 31u) / 32u; ++by)
            for (unsigned bx = 0u; bx < (k + 31u) / 32u; ++bx) {
                blockIdx = {bx, by}; blockDim = {32u, 8u};
                std::vector<std::thread> threads;
                for (unsigned t = 0; t < 256u; ++t) threads.emplace_back([&, t] {
                    threadIdx = {t % 32u, t / 32u};
                    selected_hawkeye_transpose_weights_kernel(weights.data() + 32u,
                        transposed.data() + 32u, rows, k);
                });
                for (auto& thread : threads) thread.join();
            }
        if (weights != original) return 1;
        for (unsigned r = 0; r < rows; ++r) for (unsigned c = 0; c < k; ++c)
            if (transposed[32u + size_t(c) * rows + r] != weights[32u + size_t(r) * k + c]) return 2;
        for (unsigned i = 0; i < 32u; ++i)
            if (transposed[i] != 0xdeadu || transposed[transposed.size() - 1u - i] != 0xdeadu) return 3;
    }
    for (unsigned k : {16u, 48u, 2048u, 4096u}) {
        constexpr unsigned rows = 35u, tokens = 13u, guard = 32u;
        std::vector<uint16_t> weights(size_t(rows) * k), inputs(size_t(tokens) * k);
        for (auto& x : weights) x = operand();
        for (auto& x : inputs) x = operand();
        // Include zero signs and subnormals in the actual product stream.
        for (unsigned i = 0; i < 8; ++i) { inputs[i] = uint16_t(i & 1u ? 0x8000u : i); weights[i] = 1u; }
        const auto original_weights = weights, original_inputs = inputs;
        std::vector<uint16_t> transposed(weights.size());
        for (unsigned r = 0; r < rows; ++r) for (unsigned c = 0; c < k; ++c)
            transposed[size_t(c) * rows + r] = weights[size_t(r) * k + c];
        const auto original_transposed = transposed;
        std::vector<unsigned> indices(400u);
        for (unsigned i = 0; i < indices.size(); ++i) indices[i] = (i * 71u) % (rows * tokens);
        std::vector<float> expected(rows * tokens + 2u * guard, -12345.f);
        for (unsigned i = 5u; i < 379u; ++i) {
            const unsigned index = indices[i], token = index / rows, row = index % rows;
            expected[guard + index] = device_bf16_round_to_float(
                qrt_q1_moe_hawkeye::dot_bf16_impl<26, 16, -133>(
                    inputs.data() + size_t(token) * k, weights.data() + size_t(row) * k, k));
        }
        for (unsigned quantum : {16u, 128u, 300u, 400u}) {
            std::vector<float> output(expected.size(), -12345.f);
            blockDim = {256u, 1u};
            for (unsigned offset = 5u; offset < 379u; offset += quantum) {
                const unsigned end = std::min(379u, offset + quantum);
                for (unsigned block = 0; block < (end - offset + 255u) / 256u; ++block) {
                    blockIdx = {block, 0u};
                    for (unsigned lane = 0; lane < 256u; ++lane) {
                        threadIdx = {lane, 0u};
                        selected_bf16_projection_hawkeye_packed_correction_kernel(
                            transposed.data(), inputs.data(), output.data() + guard,
                            rows, k, indices.data(), offset, end);
                    }
                }
            }
            if (std::memcmp(output.data(), expected.data(), expected.size() * sizeof(float))) return 4;
            if (weights != original_weights || inputs != original_inputs || transposed != original_transposed) return 5;
        }
    }
    std::puts("packed_candidates_and_transpose=pass");
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-packed-projection-') as temporary:
            exe = str(Path(temporary) / 'check')
            compiled = subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                            '-ffp-contract=off', '-fsanitize=address,undefined',
                            '-I', str(ROOT), '-x', 'c++', '-', '-o', exe],
                           input=source, text=True, capture_output=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([exe], text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('packed_candidates_and_transpose=pass', result.stdout)
