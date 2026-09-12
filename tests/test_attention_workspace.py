"""Exercise the actual provider workspace ownership and partial-submit cleanup."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def function(source, signature):
    begin = source.index(signature)
    opening = source.index("{", begin)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


class AttentionWorkspaceTests(unittest.TestCase):
    def test_atomic_prepare_release_and_failed_submission_drain(self):
        source = (ROOT / "native/providers/ck_fmha/qrt_ck_fmha_q8192_provider.cpp").read_text()
        globals_ = source.split("std::mutex g_sm121_mutex;", 1)[1].split(
            "bool sm121_attention_enabled", 1
        )[0]
        actual = "\n".join(function(source, name) for name in (
            "bool sm121_attention_enabled(",
            "int prepare_sm121_attention_locked()",
            "int launch_sm121_attention(",
            "QRT_CK_EXPORT int qrt_ck_fmha_q8192_release()",
        ))
        harness = r'''
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
using hipStream_t = void*;
enum hipError_t { hipSuccess, hipErrorUnknown, hipErrorInvalidValue, hipErrorLaunchTimeOut };
constexpr unsigned kQueryHeads = 16, kKvHeads = 2, kHeadDim = 256;
namespace qrt_blackwell_attention { constexpr unsigned kSplitMaxTokens = 16384; }
std::mutex g_sm121_mutex, g_state_mutex;
struct ProviderState { void* q = nullptr; void* k = nullptr; void* v = nullptr; } g_state;
#define QRT_CK_EXPORT
#define QRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL 1
''' + globals_ + r'''
namespace qrt_sm121_exp2 {
constexpr size_t table_bytes = 64;
const unsigned char sha256[32]{};
bool valid_layout(const unsigned char*, size_t) { return true; }
}
namespace qrt_sm121_attention_rcp {
constexpr size_t table_bytes = 96;
const unsigned char sha256[32]{};
bool valid_layout(const unsigned char*, size_t) { return true; }
}
unsigned allocations = 0, fail_allocation = 0, transposes = 0, queries = 0, syncs = 0;
unsigned fail_query = 0;
unsigned observed_layout = 0, largest_batch = 0;
bool fail_transpose = false;
std::set<void*> live;
hipError_t hipMalloc(void** pointer, size_t) {
    ++allocations;
    if (allocations == fail_allocation) { *pointer = nullptr; return hipErrorUnknown; }
    *pointer = reinterpret_cast<void*>(uintptr_t(0x1000u + allocations * 16u));
    if (!live.insert(*pointer).second) std::abort();
    return hipSuccess;
}
hipError_t hipFree(void* pointer) {
    if (pointer && live.erase(pointer) != 1u) std::abort();
    return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) { ++syncs; return hipSuccess; }
template<class Validate>
hipError_t load_sm121_table(const char*, size_t bytes, const unsigned char*, Validate,
                           unsigned char** output) {
    return hipMalloc(reinterpret_cast<void**>(output), bytes);
}
namespace qrt_blackwell_attention {
namespace exp2_backend = qrt_sm121_exp2;
int transpose_keys(const uint16_t*, uint16_t* prepared, size_t elements,
                   unsigned tokens, hipStream_t) {
    ++transposes;
    if (prepared != g_sm121_transposed_keys || elements < size_t(tokens) * 512u)
        std::abort();
    return fail_transpose ? hipErrorUnknown : hipSuccess;
}
int launch_queries(const uint16_t*, const uint16_t*, const uint16_t*, float*, hipStream_t,
                   unsigned start, unsigned count, unsigned, const unsigned char*,
                   float*, float*, bool, const unsigned char*, unsigned layout,
                   float* scores, size_t elements, void*, void*, const uint16_t* prepared,
                   unsigned key_stride, bool = false) {
    ++queries;
    observed_layout = layout; largest_batch = std::max(largest_batch, count);
    const bool matrix = layout == 6u || layout == 7u;
    const bool expanded = layout >= 5u && layout <= 7u;
    if (!count || count > (matrix ? 32u : 8u) || scores != (expanded ? g_sm121_mantissa_scores : g_sm121_scores) ||
        elements < size_t(count) * 16u * (start + count)) std::abort();
    if (layout >= 4u && layout <= 7u) {
        if (transposes != 1u || prepared != g_sm121_transposed_keys || key_stride < start + count)
            std::abort();
        if (expanded && (elements != kSm121MantissaElements ||
            elements < size_t(count) * 16u * (start + count) * 3u / 2u +
                size_t(count) * 16u * (((start + count + 31u) / 32u) + 1u))) std::abort();
    } else if (layout != 2u || prepared || transposes) std::abort();
    return queries == fail_query ? hipErrorUnknown : hipSuccess;
}
}
''' + actual + r'''
bool empty() {
    return live.empty() && !g_sm121_exp2 && !g_sm121_rcp && !g_sm121_scores &&
           !g_sm121_transposed_keys && !g_sm121_mantissa_scores;
}
void reset() {
    qrt_ck_fmha_q8192_release();
    if (!empty()) std::abort();
    allocations = fail_allocation = transposes = queries = syncs = fail_query = 0;
    observed_layout = largest_batch = 0;
    fail_transpose = false;
}
int main() {
    for (unsigned failure = 1; failure <= 4; ++failure) {
        reset(); fail_allocation = failure;
        if (prepare_sm121_attention_locked() != hipErrorUnknown || !empty()) return 1;
    }
    reset();
    if (prepare_sm121_attention_locked() != hipSuccess || allocations != 4u || live.size() != 4u)
        return 2;
    if (prepare_sm121_attention_locked() != hipSuccess || allocations != 4u) return 3;
    reset();
    uint16_t operand = 0; float output = 0;
    auto launch = [&](unsigned start, unsigned count) {
        return launch_sm121_attention(&operand, &operand, &operand, &output, nullptr,
                                      start, count, 0);
    };
    if (launch(0, 0) != hipErrorInvalidValue || launch(16383, 2) != hipErrorInvalidValue ||
        launch(0xffffffffu, 2) != hipErrorInvalidValue || allocations)
        return 4;
    if (launch(0, 17) != hipSuccess || transposes != 1u || queries != 3u || syncs != 3u)
        return 5;
    reset();
    if (launch(7168, 1) != hipSuccess || transposes || queries != 1u || syncs != 1u) return 6;
    reset(); fail_transpose = true;
    if (launch(0, 17) != hipErrorUnknown || transposes != 1u || queries || syncs != 1u) return 7;
    reset(); fail_query = 2u;
    if (launch(0, 17) != hipErrorUnknown || transposes != 1u || queries != 2u || syncs != 2u)
        return 8;
    reset();
    setenv("QRT_CK_SM121_MANTISSA_WMMA", "1", 1);
    fail_allocation = 5u;
    if (launch(0, 17) != hipErrorUnknown || live.size() != 4u ||
        g_sm121_mantissa_scores || transposes || queries || syncs) return 9;
    reset();
    if (launch(0, 17) != hipSuccess || live.size() != 5u || allocations != 5u ||
        !g_sm121_mantissa_scores || transposes != 1u || queries != 3u || syncs != 3u) return 10;
    transposes = queries = syncs = 0u;
    if (launch(0, 17) != hipSuccess || allocations != 5u || queries != 3u) return 11;
    reset(); fail_query = 2u;
    if (launch(0, 17) != hipErrorUnknown || queries != 2u || syncs != 2u) return 12;
    reset();
    if (launch(7168, 1) != hipSuccess || live.size() != 4u || g_sm121_mantissa_scores || transposes)
        return 13;
    reset(); unsetenv("QRT_CK_SM121_MANTISSA_WMMA");
    for (const unsigned tokens : {1u, 7169u, 8191u, 8192u, 8193u, 16383u, 16384u}) {
        setenv("QRT_CK_FMHA_SM121_FULL_PREFIX", "0", 1);
        if (sm121_attention_enabled(tokens)) return 14;
        setenv("QRT_CK_FMHA_SM121_FULL_PREFIX", "1", 1);
        if (!sm121_attention_enabled(tokens)) return 15;
        reset();
        const unsigned batches = (tokens + 7u) / 8u;
        if (launch(0, tokens) != hipSuccess || queries != batches || syncs != batches ||
            transposes != unsigned(tokens > 1u)) return 16;
        reset();
        if (launch(tokens - 1u, 1u) != hipSuccess || queries != 1u || transposes) return 17;
    }
    if (sm121_attention_enabled(0u) || sm121_attention_enabled(16385u) ||
        sm121_attention_enabled(0xffffffffu)) return 18;
    reset();
    unsetenv("QRT_CK_FMHA_SM121_FULL_PREFIX");
    for (const char* mode : {"1", "2"}) {
        setenv("QRT_CK_SM121_NATIVE_BF16_MATRIX", mode, 1);
        reset(); fail_allocation = 5u;
        if (launch(0, 65) != hipErrorUnknown || transposes || queries ||
            g_sm121_mantissa_scores || live.size() != 4u) return 19;
        reset();
        if (launch(0, 65) != hipSuccess || queries != 3u || syncs != 3u ||
            largest_batch != 32u || observed_layout != unsigned(5 + mode[0] - '0')) return 20;
        reset(); fail_query = 2u;
        if (launch(0, 65) != hipErrorUnknown || queries != 2u || syncs != 2u) return 21;
        reset();
        if (launch(8192, 1) != hipSuccess || observed_layout != 2u || transposes ||
            g_sm121_mantissa_scores || largest_batch != 1u) return 22;
        reset();
        if (launch(16352, 32) != hipSuccess || observed_layout != unsigned(5 + mode[0] - '0') ||
            queries != 1u || largest_batch != 32u) return 23;
        for (const char* conflict : {"QRT_CK_SM121_MANTISSA_WMMA", "QRT_CK_SM121_NATIVE_PRODUCTS"}) {
            reset(); setenv(conflict, "1", 1);
            if (launch(0, 65) != hipErrorInvalidValue || transposes || queries) return 24;
            unsetenv(conflict);
        }
    }
    for (const char* bad : {"3", "-1", "true", "1junk", "20", " 2"}) {
        reset(); setenv("QRT_CK_SM121_NATIVE_BF16_MATRIX", bad, 1);
        if (launch(0, 65) != hipErrorInvalidValue || allocations || transposes || queries) return 25;
    }
    reset(); unsetenv("QRT_CK_SM121_NATIVE_BF16_MATRIX");
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-attention-workspace-") as tmp:
            executable = str(Path(tmp) / "workspace")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                 "-Werror", "-x", "c++", "-", "-o", executable],
                input=harness, text=True, check=True, timeout=30,
            )
            subprocess.run([executable], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
