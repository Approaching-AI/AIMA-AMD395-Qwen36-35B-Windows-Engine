"""Exercise the actual provider workspace ownership and partial-submit cleanup."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def attention_capacity():
    return (ROOT / "native/providers/sm121_attention_capacity.h").read_text().replace("#pragma once", "")


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
        header = (ROOT / "native/providers/ck_fmha/blackwell_attention.h").read_text()
        maximum = "constexpr unsigned int kSplitMaxTokens" + header.split(
            "constexpr unsigned int kSplitMaxTokens", 1)[1].split(";", 1)[0] + ";"
        globals_ = source.split("std::mutex g_sm121_mutex;", 1)[1].split(
            "bool sm121_attention_enabled", 1
        )[0]
        actual = "\n".join(function(source, name) for name in (
            "bool sm121_attention_enabled(",
            "int prepare_sm121_attention_locked()",
            "int launch_sm121_attention(",
            "QRT_CK_EXPORT int qrt_ck_fmha_q8192_release()",
        ))
        actual = actual.replace("std::chrono::steady_clock::now()", "mock_now()")
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
namespace qrt_blackwell_attention {
''' + attention_capacity() + maximum + r'''
}
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
unsigned value_transposes = 0;
bool fail_value_transpose = false;
unsigned preparations = 0;
unsigned clock_ms = 0, sync_ms = 0;
std::chrono::steady_clock::time_point mock_now() {
    return std::chrono::steady_clock::time_point(std::chrono::milliseconds(clock_ms));
}
bool fail_preparation = false;
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
hipError_t hipStreamSynchronize(hipStream_t) { ++syncs; clock_ms += sync_ms; return hipSuccess; }
template<class Validate>
hipError_t load_sm121_table(const char*, size_t bytes, const unsigned char*, Validate,
                           unsigned char** output) {
    return hipMalloc(reinterpret_cast<void**>(output), bytes);
}
namespace qrt_blackwell_attention {
namespace exp2_backend = qrt_sm121_exp2;
int prepare_value_encoding(const uint16_t*, uint32_t* output, size_t elements,
                          unsigned tokens, hipStream_t) {
    ++preparations;
    if (output != g_sm121_prepared_values || elements != kSm121KeyElements ||
        elements < size_t(tokens) * 512u) std::abort();
    return fail_preparation ? hipErrorUnknown : hipSuccess;
}
int transpose_keys(const uint16_t*, uint16_t* prepared, size_t elements,
                   unsigned tokens, hipStream_t) {
    if (prepared == g_sm121_transposed_values && prepared) {
        ++value_transposes;
        if(elements != kSm121TransposedValueElements || tokens>8192u) std::abort();
        return fail_value_transpose ? hipErrorUnknown : hipSuccess;
    }
    ++transposes;
    if (prepared != g_sm121_transposed_keys || elements < size_t(tokens) * 512u)
        std::abort();
    return fail_transpose ? hipErrorUnknown : hipSuccess;
}
int launch_queries(const uint16_t*, const uint16_t*, const uint16_t*, float*, hipStream_t,
                   unsigned start, unsigned count, unsigned, const unsigned char*,
                   float*, float*, bool, const unsigned char*, unsigned layout,
                   float* scores, size_t elements, void*, void*, const uint16_t* prepared,
                   unsigned key_stride, bool = false, const void* = nullptr,
                   const uint32_t* wide = nullptr, unsigned wide_tokens = 0u,
                   const void* = nullptr, const void* = nullptr,
                   const uint16_t* transposed_value = nullptr, unsigned value_tokens = 0u) {
    ++queries;
    observed_layout = layout; largest_batch = std::max(largest_batch, count);
    const bool matrix = layout == 6u || layout == 7u || ((layout >= 13u && layout <= 17u) || layout == 22u || layout == 23u || layout == 24u);
    const bool expanded = (layout >= 5u && layout <= 7u) || ((layout >= 13u && layout <= 17u) || layout == 22u || layout == 23u || layout == 24u);
    if(transposed_value) {
        if(transposed_value!=g_sm121_transposed_values || value_tokens!=key_stride ||
           !value_transposes || value_tokens>8192u || (layout!=22u && layout!=24u)) std::abort();
    } else if(value_tokens || value_transposes) std::abort();
    if (layout == 17u) {
        if (!preparations || wide != g_sm121_prepared_values || wide_tokens != key_stride) std::abort();
    } else if (wide || wide_tokens) std::abort();
    if (!count || count > ((layout == 22u || layout == 24u) && key_stride <= 8192u ? 128u : matrix ? 32u : 8u) || scores != (expanded ? g_sm121_mantissa_scores : g_sm121_scores) ||
        elements < size_t(count) * 16u * (start + count)) std::abort();
    if ((layout >= 4u && layout <= 7u) || ((layout >= 13u && layout <= 17u) || layout == 22u || layout == 23u || layout == 24u)) {
        if (transposes != 1u || prepared != g_sm121_transposed_keys || key_stride < start + count)
            std::abort();
        if (expanded && (elements != kSm121MantissaElements ||
            elements < size_t(count) * 16u * (start + count) * 3u / 2u +
                size_t(count) * 16u * (((start + count + 31u) / 32u) + 1u +
                    ((layout == 22u || layout == 24u) ? 512u : (layout == 13u || layout == 23u) ? 256u : 0u)))) std::abort();
    } else if (layout != 2u || prepared || transposes) std::abort();
    return queries == fail_query ? hipErrorUnknown : hipSuccess;
}
}
''' + actual + r'''
bool empty() {
    return live.empty() && !g_sm121_exp2 && !g_sm121_rcp && !g_sm121_scores &&
           !g_sm121_transposed_keys && !g_sm121_transposed_values && !g_sm121_mantissa_scores && !g_sm121_prepared_values;
}
void reset() {
    qrt_ck_fmha_q8192_release();
    if (!empty()) std::abort();
    allocations = fail_allocation = transposes = queries = syncs = fail_query = 0;
    observed_layout = largest_batch = 0;
    fail_transpose = false;
    preparations = 0; fail_preparation = false;
    value_transposes = 0; fail_value_transpose = false;
    clock_ms = sync_ms = 0;
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
    if (launch(0, 0) != hipErrorInvalidValue || launch(kSm121MaxTokens - 1u, 2) != hipErrorInvalidValue ||
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
    for (const unsigned tokens : {1u, 7169u, 8191u, 8192u, 8193u, 16383u, 16384u,
                                 16385u, 17408u, 17920u, 32768u, 33792u, 34304u,
                                 kSm121MaxTokens}) {
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
    if (sm121_attention_enabled(0u) || sm121_attention_enabled(kSm121MaxTokens + 1u) ||
        sm121_attention_enabled(0xffffffffu)) return 18;
    reset();
    unsetenv("QRT_CK_FMHA_SM121_FULL_PREFIX");
    for (const char* mode : {"1", "2", "3", "4"}) {
        const unsigned layout = mode[0] >= '3' ? unsigned(10 + mode[0] - '0') : unsigned(5 + mode[0] - '0');
        setenv("QRT_CK_SM121_NATIVE_BF16_MATRIX", mode, 1);
        reset(); fail_allocation = 5u;
        if (launch(0, 65) != hipErrorUnknown || transposes || queries ||
            g_sm121_mantissa_scores || live.size() != 4u) return 19;
        reset();
        if (launch(0, 65) != hipSuccess || queries != 3u || syncs != 3u ||
            largest_batch != 32u || observed_layout != layout) return 20;
        reset(); fail_query = 2u;
        if (launch(0, 65) != hipErrorUnknown || queries != 2u || syncs != 2u) return 21;
        reset();
        if (launch(8192, 1) != hipSuccess || observed_layout != 2u || transposes ||
            g_sm121_mantissa_scores || largest_batch != 1u) return 22;
        reset();
        if (launch(16352, 32) != hipSuccess || observed_layout != layout ||
            queries != 1u || largest_batch != 32u) return 23;
        for (const char* conflict : {"QRT_CK_SM121_MANTISSA_WMMA", "QRT_CK_SM121_NATIVE_PRODUCTS"}) {
            reset(); setenv(conflict, "1", 1);
            if (launch(0, 65) != hipErrorInvalidValue || transposes || queries) return 24;
            unsetenv(conflict);
        }
    }
    for (const char* bad : {"5", "-1", "true", "1junk", "20", " 2"}) {
        reset(); setenv("QRT_CK_SM121_NATIVE_BF16_MATRIX", bad, 1);
        if (launch(0, 65) != hipErrorInvalidValue || allocations || transposes || queries) return 25;
    }
    reset(); unsetenv("QRT_CK_SM121_NATIVE_BF16_MATRIX");
    setenv("QRT_CK_SM121_TILED_EXACT_QK","1",1);
    reset(); fail_allocation=5u;
    if(launch(0,65)!=hipErrorUnknown || queries || transposes || live.size()!=4u) return 26;
    reset();
    if(launch(0,65)!=hipSuccess || observed_layout!=15u || queries!=3u || largest_batch!=32u) return 27;
    reset();fail_query=2u;
    if(launch(0,65)!=hipErrorUnknown || queries!=2u || syncs!=2u) return 28;
    reset();
    if(launch(7168,1)!=hipSuccess || observed_layout!=2u || transposes || g_sm121_mantissa_scores) return 29;
    reset();
    if(launch(16352,32)!=hipSuccess || observed_layout!=15u || queries!=1u) return 30;
    // Every tile must retain the complete new key extent, including the
    // registered suffix and the final decode position beyond the old bound.
    for (const unsigned tokens : {17408u, 17920u, kSm121MaxTokens}) {
        reset();
        if (launch(0, tokens) != hipSuccess || observed_layout != 15u ||
            queries != (tokens + 31u) / 32u || transposes != 1u) return 49;
        reset();
        if (launch(tokens - 1u, 1u) != hipSuccess || observed_layout != 2u ||
            queries != 1u || transposes) return 50;
    }
    for(const char* conflict : {"QRT_CK_SM121_NATIVE_BF16_MATRIX","QRT_CK_SM121_NATIVE_PRODUCTS","QRT_CK_SM121_MANTISSA_WMMA"}) {
        reset();setenv(conflict,"1",1);
        if(launch(0,65)!=hipErrorInvalidValue || queries || transposes) return 31;
        unsetenv(conflict);
    }
    for(const char* bad : {"2","true","1junk"}) {
        reset();setenv("QRT_CK_SM121_TILED_EXACT_QK",bad,1);
        if(launch(0,65)!=hipErrorInvalidValue || queries || allocations) return 32;
    }
    reset();setenv("QRT_CK_SM121_TILED_EXACT_QK","1",1);
    setenv("QRT_CK_SM121_WARP_SOFTMAX","1",1);
    if(launch(0,65)!=hipSuccess || observed_layout!=16u || queries!=3u) return 33;
    reset();fail_query=2u;
    if(launch(0,65)!=hipErrorUnknown || queries!=2u || syncs!=2u) return 34;
    reset();
    if(launch(7168,1)!=hipSuccess || observed_layout!=2u || transposes) return 35;
    reset();unsetenv("QRT_CK_SM121_TILED_EXACT_QK");
    if(launch(0,65)!=hipErrorInvalidValue || queries || allocations) return 36;
    for(const char* bad : {"2","true","1junk"}) {
        reset();setenv("QRT_CK_SM121_WARP_SOFTMAX",bad,1);
        if(launch(0,65)!=hipErrorInvalidValue || queries || allocations) return 37;
    }
    reset();unsetenv("QRT_CK_SM121_WARP_SOFTMAX");
    setenv("QRT_CK_SM121_PREPARED_VALUE","1",1);
    if(launch(0,65)!=hipErrorInvalidValue || allocations || preparations) return 38;
    setenv("QRT_CK_SM121_TILED_EXACT_QK","1",1);
    for(unsigned failed : {5u,6u}) {
        reset();fail_allocation=failed;
        if(launch(0,65)!=hipErrorUnknown || g_sm121_prepared_values || queries ||
           transposes || preparations || live.size()!=failed-1u) return 39;
    }
    reset();fail_preparation=true;
    if(launch(0,65)!=hipErrorUnknown || preparations!=1u || transposes || queries || syncs!=1u) return 40;
    reset();fail_transpose=true;
    if(launch(0,65)!=hipErrorUnknown || preparations!=1u || transposes!=1u || queries || syncs!=1u) return 41;
    reset();
    if(launch(0,65)!=hipSuccess || observed_layout!=17u || preparations!=1u ||
       queries!=3u || syncs!=3u || allocations!=6u) return 42;
    queries=transposes=syncs=0u;
    if(launch(0,8193)!=hipSuccess || observed_layout!=17u || preparations!=2u ||
       queries!=257u || allocations!=6u) return 43;
    reset();fail_query=2u;
    if(launch(0,65)!=hipErrorUnknown || preparations!=1u || queries!=2u || syncs!=2u) return 44;
    reset();
    if(launch(7168,1)!=hipSuccess || preparations || g_sm121_prepared_values || observed_layout!=2u) return 45;
    reset();
    if(launch(16352,32)!=hipSuccess || preparations!=1u || queries!=1u || observed_layout!=17u) return 46;
    for(const char* conflict : {"QRT_CK_SM121_WARP_SOFTMAX","QRT_CK_SM121_NATIVE_BF16_MATRIX",
                               "QRT_CK_SM121_NATIVE_PRODUCTS","QRT_CK_SM121_MANTISSA_WMMA"}) {
        reset();setenv(conflict,"1",1);
        if(launch(0,65)!=hipErrorInvalidValue || preparations || queries || transposes) return 47;
        unsetenv(conflict);
    }
    for(const char* bad : {"2","true","1junk"}) {
        reset();setenv("QRT_CK_SM121_PREPARED_VALUE",bad,1);
        if(launch(0,65)!=hipErrorInvalidValue || allocations || preparations || queries) return 48;
    }
    reset();unsetenv("QRT_CK_SM121_PREPARED_VALUE");unsetenv("QRT_CK_SM121_TILED_EXACT_QK");
    setenv("QRT_CK_SM121_TILED_EXACT_QK","1",1);
    // A healthy 32k call can exceed twenty seconds in total, while every
    // completed 8192-query window remains bounded. No real waiting here.
    reset();sync_ms=35;
    if(launch(0,32768)!=hipSuccess || clock_ms!=35840 || queries!=1024u || syncs!=queries) return 49;
    reset();sync_ms=80;
    if(launch(0,8192)!=hipErrorLaunchTimeOut || queries!=251u || syncs!=queries) return 50;
    // A larger total allowance must not mask stalled first-window progress.
    reset();sync_ms=80;
    if(launch(0,32768)!=hipErrorLaunchTimeOut || queries!=251u || syncs!=queries) return 51;
    reset();sync_ms=20000;
    if(launch(32768,1)!=hipSuccess || queries!=1u || syncs!=1u) return 52;
    reset();sync_ms=20001;
    if(launch(32768,1)!=hipErrorLaunchTimeOut || queries!=1u || syncs!=1u) return 53;
    reset();
    if(launch(32768,1024)!=hipSuccess || queries!=32u || syncs!=queries) return 54;
    for(const char* mode : {"1","2","3"}) {
        reset();setenv("QRT_CK_SM121_COMPACT_PV_REPLAY",mode,1);
        if(launch(0,65)!=hipSuccess || observed_layout!=21u+unsigned(*mode-'0') || queries!=3u || transposes!=1u) return 55;
        reset();fail_query=2u;
        if(launch(0,65)!=hipErrorUnknown || queries!=2u || syncs!=2u) return 56;
        reset();
        if(launch(7168,1)!=hipSuccess || observed_layout!=2u || transposes) return 57;
    }
    reset();setenv("QRT_CK_SM121_COMPACT_PV_REPLAY","1",1);
    for(const char* conflict : {"QRT_CK_SM121_PREPARED_VALUE","QRT_CK_SM121_WARP_SOFTMAX"}) {
        setenv(conflict,"1",1);
        if(launch(0,65)!=hipErrorInvalidValue || queries || allocations) return 58;
        unsetenv(conflict);
    }
    unsetenv("QRT_CK_SM121_TILED_EXACT_QK");
    if(launch(0,65)!=hipErrorInvalidValue || queries || allocations) return 59;
    setenv("QRT_CK_SM121_TILED_EXACT_QK","1",1);
    for(const char* bad : {"4","true","1junk"}) {
        setenv("QRT_CK_SM121_COMPACT_PV_REPLAY",bad,1);
        if(launch(0,65)!=hipErrorInvalidValue || queries || allocations) return 60;
    }
    setenv("QRT_CK_SM121_COMPACT_PV_REPLAY","1",1);
    for(const char* batch : {"32","64","128"}) {
        setenv("QRT_CK_SM121_PREFILL_QUERY_BATCH",batch,1);
        const unsigned n=unsigned(std::atoi(batch));
        reset();
        if(launch(0,7169)!=hipSuccess || queries!=(7169u+n-1u)/n || syncs!=queries ||
           largest_batch!=n || allocations!=5u) return 61;
        reset();
        if(launch(0,8192)!=hipSuccess || queries!=8192u/n || largest_batch!=n) return 62;
        reset();
        if(launch(1,8192)!=hipSuccess || queries!=256u || largest_batch!=32u) return 63;
        reset();
        if(launch(7168,1)!=hipSuccess || observed_layout!=2u || largest_batch!=1u) return 64;
        reset();fail_query=2u;
        if(launch(0,7169)!=hipErrorUnknown || queries!=2u || syncs!=2u) return 65;
        reset();fail_allocation=5u;
        if(launch(0,7169)!=hipErrorUnknown || queries || transposes || live.size()!=4u) return 66;
    }
    for(const char* bad : {"0","16","33","256","-1","128junk"," 128","0128"}) {
        reset();setenv("QRT_CK_SM121_PREFILL_QUERY_BATCH",bad,1);
        if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries || transposes) return 67;
    }
    setenv("QRT_CK_SM121_PREFILL_QUERY_BATCH","128",1);
    setenv("QRT_CK_SM121_COMPACT_PV_TRANSPOSE_VALUE","1",1);
    reset();fail_allocation=6u;
    if(launch(0,8192)!=hipErrorUnknown || g_sm121_transposed_values || queries ||
       transposes || value_transposes || live.size()!=5u) return 68;
    reset();fail_value_transpose=true;
    if(launch(0,8192)!=hipErrorUnknown || transposes!=1u || value_transposes!=1u || queries || syncs!=1u) return 69;
    reset();
    if(launch(0,8192)!=hipSuccess || allocations!=6u || live.size()!=6u ||
       value_transposes!=1u || queries!=64u || syncs!=queries) return 70;
    transposes=value_transposes=queries=syncs=0u;
    if(launch(0,7169)!=hipSuccess || allocations!=6u || value_transposes!=1u || queries!=57u) return 71;
    reset();fail_query=2u;
    if(launch(0,8192)!=hipErrorUnknown || queries!=2u || syncs!=2u || value_transposes!=1u) return 72;
    for(unsigned tokens : {1u,8193u,17408u}) {
        reset();
        if(launch(0,tokens)!=hipSuccess || value_transposes || g_sm121_transposed_values ||
           allocations!=(tokens==1u ? 4u : 5u)) return 73;
    }
    for(const char* bad : {"2","-1","true","1junk"," 1"}) {
        reset();setenv("QRT_CK_SM121_COMPACT_PV_TRANSPOSE_VALUE",bad,1);
        if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries || value_transposes) return 74;
    }
    unsetenv("QRT_CK_SM121_COMPACT_PV_TRANSPOSE_VALUE");
    unsetenv("QRT_CK_SM121_PREFILL_QUERY_BATCH");
    unsetenv("QRT_CK_SM121_COMPACT_PV_REPLAY");
    reset();unsetenv("QRT_CK_SM121_TILED_EXACT_QK");
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
