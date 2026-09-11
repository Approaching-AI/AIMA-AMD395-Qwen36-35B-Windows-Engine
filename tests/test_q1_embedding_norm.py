"""Reproduce the original embedding norm at two real long-decode origins."""

from pathlib import Path
import json
import os
import struct
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class Q1EmbeddingNormTests(unittest.TestCase):
    def test_actual_kernel_resident_host_and_device_top1_inputs(self):
        fixture = json.loads((ROOT / 'tests/fixtures/q1_embedding_norm.json').read_text())
        provider = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        reduction = function(provider[provider.index("// TorchInductor's q8192 Triton kernel"):],
                             '__device__ __forceinline__ float vllm_triton_reduce_sumsq(')
        kernel = function(provider, '__global__ void layer0_input_rmsnorm_sm121_kernel(')
        # Run the actual kernel and block reduction with CPU block barriers.
        # Only the device-specific rsqrt instruction is supplied by its real
        # captured variance/inverse pair; all input selection and math is live.
        source = r'''
#include "native/providers/gdn/sm121_q1_math.h"
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>
#include <cstdlib>
constexpr unsigned kThreads = 256, QRT_QWEN36_HIDDEN_SIZE = 2048;
constexpr unsigned QRT_QWEN36_VOCAB_SIZE = 248320;
constexpr float QRT_QWEN36_RMS_NORM_EPSILON = 1.e-6f;
#define __device__
#define __forceinline__ inline
#define __global__
#define __shared__ static
thread_local struct { unsigned x; } threadIdx;
std::mutex mutex;
std::condition_variable changed;
unsigned arrived = 0, generation = 0;
void __syncthreads() {
    std::unique_lock<std::mutex> lock(mutex);
    unsigned current = generation;
    if (++arrived == kThreads) {
        arrived = 0; ++generation; changed.notify_all();
    } else changed.wait(lock, [&] { return generation != current; });
}
float device_add_separate(float a, float b) { return qrt_sm121_q1::add(a, b); }
float device_bf16_to_float(uint16_t x) { return qrt_sm121_q1::widen(x); }
uint32_t variance_bits, inverse_bits;
std::atomic<bool> variance_ok{true};
float device_sm121_rsqrt_from_gfx1151(float v, const uint8_t*) {
    if (qrt_sm121_exp2::bits(v) != variance_bits) variance_ok = false;
    return qrt_sm121_exp2::value(inverse_bits);
}
''' + reduction + '\n' + kernel + r'''
std::vector<uint16_t> read(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint16_t> result(2048);
    f.read(reinterpret_cast<char*>(result.data()), 4096);
    if (!f || f.peek() != std::char_traits<char>::eof()) std::abort();
    return result;
}
int main(int argc, char** argv) {
    if (argc != 7) return 1;
    const auto weights = read(argv[1]), input = read(argv[2]), expected = read(argv[3]);
    const uint32_t id = std::stoul(argv[4]);
    variance_bits = std::stoul(argv[5]); inverse_bits = std::stoul(argv[6]);
    std::vector<uint16_t> embeddings((size_t(id) + 1) * 2048 + 64, 0xdead);
    std::copy(input.begin(), input.end(), embeddings.begin() + size_t(id) * 2048 + 32);
    const auto original_embeddings = embeddings;
    for (unsigned mode = 0; mode < 3; ++mode) {
        constexpr float guard = -12345.f;
        std::vector<float> residual(2112, guard), output(2112, guard);
        for (unsigned i = 0; i < 2048; ++i) residual[i + 32] = device_bf16_to_float(input[i]);
        const uint8_t correction = 0;
        std::vector<std::thread> block;
        for (unsigned lane = 0; lane < 256; ++lane) block.emplace_back([&, lane] {
            threadIdx.x = lane;
            layer0_input_rmsnorm_sm121_kernel(
                mode == 0 ? input.data() : mode == 2 ? embeddings.data() + 32 : nullptr,
                mode == 1 ? residual.data() + 32 : nullptr, weights.data(),
                residual.data() + 32, output.data() + 32, &correction,
                mode == 2 ? &id : nullptr);
        });
        for (auto& thread : block) thread.join();
        if (!variance_ok || embeddings != original_embeddings) return 2;
        for (unsigned i = 0; i < 2048; ++i)
            if (output[i + 32] != device_bf16_to_float(expected[i]) ||
                residual[i + 32] != device_bf16_to_float(input[i])) return 3;
        for (unsigned i = 0; i < 32; ++i)
            if (residual[i] != guard || residual[i + 2080] != guard ||
                output[i] != guard || output[i + 2080] != guard) return 4;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-q1-embedding-kernel-') as directory:
            tmp = Path(directory)
            executable = str(tmp / 'embedding-kernel')
            subprocess.run([os.getenv('CXX', 'c++'), '-std=c++17', '-O2', '-pthread',
                            '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
                            '-Wno-unknown-pragmas', '-I', str(ROOT), '-x', 'c++',
                            '-', '-o', executable], input=source, text=True,
                           check=True, timeout=30)
            (tmp / 'weights').write_bytes(struct.pack('<2048H', *fixture['weight_bf16']))
            for case in fixture['cases']:
                (tmp / 'input').write_bytes(struct.pack('<2048H', *case['embedding_bf16']))
                (tmp / 'expected').write_bytes(struct.pack('<2048H', *case['expected_norm_bf16']))
                subprocess.run([executable, str(tmp / 'weights'), str(tmp / 'input'),
                                str(tmp / 'expected'), str(case['token_id']),
                                str(case['expected_variance_bits']), str(case['original_inverse_bits'])],
                               check=True, timeout=15)

    def test_original_variance_and_all_bf16_outputs(self):
        fixture = json.loads((ROOT / 'tests/fixtures/q1_embedding_norm.json').read_text())
        source = r'''
#include "native/providers/gdn/sm121_q1_math.h"
using namespace qrt_sm121_q1;
int main() {
'''
        source += 'const uint16_t weights[2048] = {'
        source += ','.join(map(str, fixture['weight_bf16'])) + '};\n'
        for index, case in enumerate(fixture['cases']):
            source += '{\nconst uint16_t inputs[2048] = {'
            source += ','.join(map(str, case['embedding_bf16'])) + '};\n'
            source += 'const uint16_t expected[2048] = {'
            source += ','.join(map(str, case['expected_norm_bf16'])) + '};\n'
            source += r'''
float warps[8];
for (unsigned w = 0; w < 8; ++w) {
    float partial[32];
    for (unsigned lane = 0; lane < 32; ++lane) {
        float values[8];
        for (unsigned i = 0; i < 8; ++i)
            values[i] = widen(inputs[(w * 32 + lane) * 8 + i]);
        partial[lane] = embedding_lane_sumsq(values);
    }
    for (unsigned step = 16; step; step >>= 1)
        for (unsigned i = 0; i < step; ++i)
            partial[i] = add(partial[i], partial[i + step]);
    warps[w] = partial[0];
}
for (unsigned step = 4; step; step >>= 1)
    for (unsigned i = 0; i < step; ++i)
        warps[i] = add(warps[i], warps[i + step]);
const float variance = add(multiply(warps[0], 1.f / 2048.f), 1.e-6f);
'''
            source += f'''
if (qrt_sm121_exp2::bits(warps[0]) != {case['expected_sumsq_bits']}u) return {index * 3 + 1};
if (qrt_sm121_exp2::bits(variance) != {case['expected_variance_bits']}u) return {index * 3 + 2};
// The inverse is captured from original CUDA at this measured variance;
// table/instruction emulation has its own independent domain tests.
const float inverse = qrt_sm121_exp2::value({case['original_inverse_bits']}u);
for (unsigned i = 0; i < 2048; ++i)
    if (bf16(embedding_norm_value(widen(inputs[i]), inverse, weights[i])) != expected[i])
        return {index * 3 + 3};
}}
'''
        source += 'return 0;\n}\n'
        with tempfile.TemporaryDirectory(prefix='qrt-q1-embedding-norm-') as directory:
            executable = str(Path(directory) / 'embedding-norm')
            subprocess.run([os.getenv('CXX', 'c++'), '-std=c++17', '-O2',
                            '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT), '-x', 'c++', '-', '-o', executable],
                           input=source, text=True, check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=5)


if __name__ == '__main__':
    unittest.main()
