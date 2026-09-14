"""Exercise the actual provider workspace ownership and partial-submit cleanup."""

from pathlib import Path
import os
import re
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
        deadline = (ROOT / "native/providers/ck_fmha/attention_deadline.h").read_text().replace("#pragma once", "")
        maximum = "constexpr unsigned int kSplitMaxTokens" + header.split(
            "constexpr unsigned int kSplitMaxTokens", 1)[1].split(";", 1)[0] + ";"
        globals_ = source.split("std::mutex g_sm121_mutex;", 1)[1].split(
            "bool sm121_attention_enabled", 1
        )[0]
        actual = "\n".join(function(source, name) for name in (
            "bool sm121_attention_enabled(",
            "bool supported_dynamic_tokens(",
            "int prepare_sm121_attention_locked()",
            "int prepare_sm121_extended_attention_locked()",
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
#include <array>
#include <vector>
using hipStream_t = void*;
enum hipError_t { hipSuccess, hipErrorUnknown, hipErrorInvalidValue, hipErrorLaunchTimeOut };
constexpr unsigned kQueryHeads = 16, kKvHeads = 2, kHeadDim = 256;
constexpr unsigned kQ262144Tokens = 262144;
''' + attention_capacity() + r'''
namespace qrt_blackwell_attention {
''' + maximum + r'''
}
std::mutex g_sm121_mutex, g_state_mutex;
struct ProviderState { void* q = nullptr; void* k = nullptr; void* v = nullptr; } g_state;
#define QRT_CK_EXPORT
#define QRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL 1
#include "''' + str(ROOT / 'native/providers/ck_fmha/prepared_decoded_qk_workspace.h') + r'''"
''' + globals_ + deadline + r'''
using AttentionBudget = qrt_sm121_attention_deadline::Budget;
static_assert(AttentionBudget{0,8192,kSm121MaxTokens}.call_limit_seconds()==20.0);
static_assert(AttentionBudget{0,32768,kSm121MaxTokens}.window_limit_seconds(8192)==20.0);
static_assert(AttentionBudget{0,32768,kSm121MaxTokens}.window_limit_seconds(8193)==40.0);
static_assert(AttentionBudget{0,32768,kSm121MaxTokens}.call_limit_seconds()==200.0);
static_assert(AttentionBudget{24576,8192,kSm121MaxTokens}.window_limit_seconds(8128)==80.0);
static_assert(AttentionBudget{65536,1024,kSm121MaxTokens}.call_limit_seconds()==40.0);
static_assert(AttentionBudget{131071,1,kSm121MaxTokens}.call_limit_seconds()==20.0);
static_assert(AttentionBudget{0,131072,kSm121MaxTokens}.call_limit_seconds()==2720.0);
static_assert(AttentionBudget{~0u,1,kSm121MaxTokens}.call_limit_seconds()==0.0);
static_assert(AttentionBudget{1,~0u,kSm121MaxTokens}.call_limit_seconds()==0.0);
static_assert(AttentionBudget{0,0,kSm121MaxTokens}.window_limit_seconds(1)==0.0);
static_assert(AttentionBudget{0,8192,kSm121MaxTokens}.window_limit_seconds(8193)==0.0);
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
unsigned fail_sync = 0, profile_observations = 0;
unsigned observed_layout = 0, largest_batch = 0;
unsigned final_bound_queries = 0;
unsigned direct_pv_queries = 0;
unsigned float_alignment_queries = 0;
unsigned decoded_preparations=0, decoded_queries=0, fail_decoded_prepare=0;
unsigned selective_qk_queries = 0;
unsigned value_transposes = 0;
bool fail_value_transpose = false;
unsigned preparations = 0;
unsigned clock_ms = 0, sync_ms = 0;
bool track_submissions = false;
unsigned pending_submissions = 0, maximum_pending = 0, submit_ms = 0;
std::vector<std::array<unsigned,3>> submitted_ranges;
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
    if (track_submissions && pending_submissions) std::abort();
    if (pointer && live.erase(pointer) != 1u) std::abort();
    return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) {
    pending_submissions = 0u;
    ++syncs; clock_ms += sync_ms;
    return syncs == fail_sync ? hipErrorUnknown : hipSuccess;
}
template<class Validate>
hipError_t load_sm121_table(const char*, size_t bytes, const unsigned char*, Validate,
                           unsigned char** output) {
    return hipMalloc(reinterpret_cast<void**>(output), bytes);
}
namespace qrt_prepared_decoded_qk {
int prepare_workspace(const uint16_t*,const uint16_t*,uint16_t* transposed,const Workspace& workspace,hipStream_t) {
    ++decoded_preparations;
    if(!valid(workspace) || workspace.words!=g_sm121_prepared_decoded_qk || transposed!=g_sm121_transposed_keys)
        std::abort();
    if(fail_decoded_prepare) return hipErrorUnknown;
    ++transposes;return hipSuccess;
}
int launch_workspace(const void*,const uint16_t*,const uint16_t*,float*,hipStream_t,unsigned,unsigned,unsigned,unsigned) {
    return hipSuccess;
}
}
namespace qrt_blackwell_attention {
namespace exp2_backend = qrt_sm121_exp2;
struct SplitCompletionObserver {
    void* state;
    int (*observe)(void*, unsigned, hipStream_t);
};
struct SplitQkProducer {
    const void* state;
    int (*launch)(const void*,const uint16_t*,const uint16_t*,float*,hipStream_t,unsigned,unsigned,unsigned,unsigned);
};
int prepare_value_encoding(const uint16_t*, uint32_t* output, size_t elements,
                          unsigned tokens, hipStream_t) {
    ++preparations;
    const bool extended=tokens>kSm121InitialTokens;
    if (output != (extended?g_sm121_extended.prepared_values:g_sm121_prepared_values) ||
        elements != (extended?kSm121ExtendedKeyElements:kSm121KeyElements) ||
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
    if (prepared != (tokens>kSm121InitialTokens?g_sm121_extended.transposed_keys:g_sm121_transposed_keys) || elements < size_t(tokens) * 512u)
        std::abort();
    return fail_transpose ? hipErrorUnknown : hipSuccess;
}
int launch_queries(const uint16_t*, const uint16_t*, const uint16_t*, float*, hipStream_t stream,
                   unsigned start, unsigned count, unsigned output_start, const unsigned char*,
                   float*, float*, bool, const unsigned char*, unsigned layout,
                   float* scores, size_t elements, void*, void*, const uint16_t* prepared,
                   unsigned key_stride, bool = false, const void* = nullptr,
                   const uint32_t* wide = nullptr, unsigned wide_tokens = 0u,
                   const void* = nullptr, SplitCompletionObserver* observer = nullptr,
                   const uint16_t* transposed_value = nullptr, unsigned value_tokens = 0u,
                   unsigned = 1u, unsigned = 1u, bool final_pv_bound = false,
                   bool direct_pv_operands = false, bool float_alignment_qk = false,
                   unsigned = 0u, bool = false, const SplitQkProducer* producer = nullptr) {
    ++queries;
    if (track_submissions) {
        ++pending_submissions;
        maximum_pending = std::max(maximum_pending,pending_submissions);
        submitted_ranges.push_back({start,count,output_start});
        clock_ms += submit_ms;
    }
    if(producer) {
        const auto* workspace=static_cast<const qrt_prepared_decoded_qk::Workspace*>(producer->state);
        if(!workspace || !qrt_prepared_decoded_qk::valid(*workspace) ||
           workspace->words!=g_sm121_prepared_decoded_qk || workspace->tokens!=key_stride ||
           producer->launch!=qrt_prepared_decoded_qk::launch_workspace || !decoded_preparations ||
           !float_alignment_qk || (layout!=22u && layout!=24u)) std::abort();
        ++decoded_queries;
    }
    if(float_alignment_qk) {
        if(layout!=15u && layout!=16u && layout!=17u && layout!=22u && layout!=23u && layout!=24u)
            std::abort();
        ++float_alignment_queries;
    }
    if(final_pv_bound) {
        if((layout!=22u && layout!=24u) || start+count>8192u) std::abort();
        ++final_bound_queries;
    }
    if(direct_pv_operands) {
        if((layout!=22u && layout!=24u) || start+count>kSm121MaxTokens) std::abort();
        ++direct_pv_queries;
    }
    observed_layout = layout; largest_batch = std::max(largest_batch, count);
    const bool matrix = layout == 6u || layout == 7u || ((layout >= 13u && layout <= 17u) || layout == 22u || layout == 23u || layout == 24u);
    const bool expanded = (layout >= 5u && layout <= 7u) || ((layout >= 13u && layout <= 17u) || layout == 22u || layout == 23u || layout == 24u);
    const bool extended = key_stride>kSm121InitialTokens;
    if(transposed_value) {
        if(transposed_value!=g_sm121_transposed_values || value_tokens!=key_stride ||
           !value_transposes || value_tokens>8192u || (layout!=22u && layout!=24u)) std::abort();
    } else if(value_tokens || value_transposes) std::abort();
    if (layout == 17u) {
        if (!preparations || wide != (extended?g_sm121_extended.prepared_values:g_sm121_prepared_values) || wide_tokens != key_stride) std::abort();
    } else if (wide || wide_tokens) std::abort();
    const auto* wanted_scores=extended?(expanded?g_sm121_extended.mantissa_scores:g_sm121_extended.scores)
        :(expanded?g_sm121_mantissa_scores:g_sm121_scores);
    if (!count || count > ((layout == 22u || layout == 24u) && key_stride <= 8192u ? 128u : matrix ? 32u : 8u) || scores != wanted_scores ||
        elements < size_t(count) * 16u * (start + count)) std::abort();
    if ((layout >= 4u && layout <= 7u) || ((layout >= 13u && layout <= 17u) || layout == 22u || layout == 23u || layout == 24u)) {
        if (transposes != 1u || prepared != (extended?g_sm121_extended.transposed_keys:g_sm121_transposed_keys) || key_stride < start + count)
            std::abort();
        if (expanded && (elements != (extended?kSm121ExtendedMantissaElements:kSm121MantissaElements) ||
            elements < size_t(count) * 16u * (start + count) * 3u / 2u +
                size_t(count) * 16u * (((start + count + 31u) / 32u) + 1u +
                    ((layout == 22u || layout == 24u) ? 512u : (layout == 13u || layout == 23u) ? 256u : 0u)))) std::abort();
    } else if (layout != 2u || prepared || transposes) std::abort();
    if (queries == fail_query) return hipErrorUnknown;
    if (observer) {
        if (layout != 22u && layout != 24u) std::abort();
        for (unsigned stage = 0u; stage < 5u; ++stage) {
            ++profile_observations;
            const int status = observer->observe(observer->state, stage, stream);
            if (status != hipSuccess) return status;
        }
    }
    return hipSuccess;
}
}
namespace qrt_selective_qk {
int launch_probability_attention(const uint16_t* q, const uint16_t* k, const uint16_t* v,
    float* output, hipStream_t stream, unsigned start, unsigned count, unsigned output_start,
    const unsigned char* exp2, const unsigned char* reciprocal, unsigned layout,
    float* scratch, size_t elements, float* work, size_t work_elements,
    const uint16_t* transposed_key, unsigned key_stride, const uint16_t* transposed_value,
    unsigned value_stride, bool final_bound, bool direct_pv) {
    ++selective_qk_queries;
    if(!work || work!=g_sm121_selective_qk || work_elements!=kSm121SelectiveQkElements ||
       work_elements < size_t(count)*16u*(2u*(start+count)+(start+count+31u)/32u)+1u ||
       (layout!=22u && layout!=24u) || start+count>8192u) std::abort();
    return qrt_blackwell_attention::launch_queries(q,k,v,output,stream,start,count,output_start,
        exp2,nullptr,nullptr,true,reciprocal,layout,scratch,elements,nullptr,nullptr,transposed_key,key_stride,
        false,nullptr,nullptr,0u,nullptr,nullptr,transposed_value,value_stride,1u,1u,final_bound,direct_pv);
}
}
''' + actual + r'''
bool empty() {
    return live.empty() && !g_sm121_exp2 && !g_sm121_rcp && !g_sm121_scores &&
           !g_sm121_transposed_keys && !g_sm121_transposed_values && !g_sm121_mantissa_scores && !g_sm121_prepared_values && !g_sm121_selective_qk && !g_sm121_prepared_decoded_qk &&
           !g_sm121_extended.scores && !g_sm121_extended.transposed_keys && !g_sm121_extended.mantissa_scores && !g_sm121_extended.prepared_values;
}
void reset() {
    qrt_ck_fmha_q8192_release();
    if (!empty()) std::abort();
    allocations = fail_allocation = transposes = queries = syncs = fail_query = 0;
    fail_sync = profile_observations = 0;
    observed_layout = largest_batch = final_bound_queries = direct_pv_queries = selective_qk_queries = 0;
    float_alignment_queries = 0;decoded_preparations=decoded_queries=fail_decoded_prepare=0;
    fail_transpose = false;
    preparations = 0; fail_preparation = false;
    value_transposes = 0; fail_value_transpose = false;
    clock_ms = sync_ms = 0;
    track_submissions = false;
    pending_submissions = maximum_pending = submit_ms = 0u;
    submitted_ranges.clear();
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
                                 65535u, 65536u, 65537u, 66560u, 67072u, 68097u,
                                 131072u, 131073u, 132096u, 132608u, 262144u, 263168u, 263680u, 264705u,
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
    // The real long-prefix failure had drained8128/8192 queries at20.081793s.
    // Its24k history needs a larger work allowance; the original q8192 bound
    // and stalled first-window checks above still apply unchanged.
    reset();sync_ms=79;
    if(launch(24576,8192)!=hipSuccess || queries!=256u || syncs!=queries || clock_ms!=20224u) return 120;
    reset();sync_ms=313;
    if(launch(24576,8192)!=hipErrorLaunchTimeOut || queries!=256u || syncs!=queries) return 121;
    reset();sync_ms=1250;
    if(launch(65536,1024)!=hipSuccess || queries!=32u || syncs!=queries || clock_ms!=40000u) return 122;
    reset();sync_ms=1251;
    if(launch(65536,1024)!=hipErrorLaunchTimeOut || queries!=32u || syncs!=queries) return 123;
    reset();
    if(launch(32768,1024)!=hipSuccess || queries!=32u || syncs!=queries) return 54;
    reset();
    if(launch(65536,1024)!=hipSuccess || queries!=32u || syncs!=queries) return 101;
    reset();
    if(launch(66560,512)!=hipSuccess || queries!=16u || syncs!=queries) return 102;
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
    for(const char* bad : {"2","-1","true","1junk"," 1"}) {
        reset();setenv("QRT_CK_SM121_FINAL_PV_BOUND",bad,1);
        if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries) return 75;
    }
    setenv("QRT_CK_SM121_FINAL_PV_BOUND","1",1);
    for(const char* mode : {"1","2","3"}) {
        reset();setenv("QRT_CK_SM121_COMPACT_PV_REPLAY",mode,1);
        if(launch(0,8192)!=hipSuccess || final_bound_queries!=(*mode=='2' ? 0u : queries)) return 76;
        reset();fail_query=2u;
        if(launch(0,8192)!=hipErrorUnknown || queries!=2u || syncs!=2u) return 77;
        for(unsigned tokens : {1u,8193u,17408u}) {
            reset();if(launch(0,tokens)!=hipSuccess || final_bound_queries) return 78;
        }
    }
    unsetenv("QRT_CK_SM121_FINAL_PV_BOUND");
    for(const char* bad : {"2","-1","true","1junk"," 1"}) {
        reset();setenv("QRT_CK_SM121_DIRECT_PV_OPERANDS",bad,1);
        if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries) return 79;
    }
    setenv("QRT_CK_SM121_DIRECT_PV_OPERANDS","1",1);
    for(const char* mode : {"1","2","3"}) for(const char* final : {"0","1"}) {
        reset();setenv("QRT_CK_SM121_COMPACT_PV_REPLAY",mode,1);
        setenv("QRT_CK_SM121_FINAL_PV_BOUND",final,1);
        if(launch(0,8192)!=hipSuccess || direct_pv_queries!=(*mode=='2' ? 0u : queries) ||
           final_bound_queries!=(*mode=='2' || *final=='0' ? 0u : queries) || allocations!=5u) return 80;
        reset();fail_query=2u;
        if(launch(0,8192)!=hipErrorUnknown || queries!=2u || syncs!=2u) return 81;
        for(unsigned tokens : {1u,8193u,17408u}) {
            reset();if(launch(0,tokens)!=hipSuccess || direct_pv_queries || final_bound_queries) return 82;
        }
    }
    for(const char* bad : {"2","-1","true","1junk"," 1"}) {
        reset();setenv("QRT_CK_SM121_LONG_DIRECT_PV_OPERANDS",bad,1);
        if(launch(65536,1024)!=hipErrorInvalidValue || allocations || queries || syncs) return 124;
    }
    setenv("QRT_CK_SM121_LONG_DIRECT_PV_OPERANDS","1",1);
    for(const char* mode : {"1","2","3"}) {
        setenv("QRT_CK_SM121_COMPACT_PV_REPLAY",mode,1);
        for(unsigned start : {8192u,16384u,32768u,65536u,131072u,262144u,kSm121MaxTokens-1024u}) {
            reset();
            if(launch(start,1024)!=hipSuccess || queries!=32u || syncs!=queries || allocations!=(start+1024u>kSm121InitialTokens?7u:5u) ||
               direct_pv_queries!=(*mode=='2' ? 0u : queries) || final_bound_queries) return 125;
            reset();fail_query=2u;
            if(launch(start,1024)!=hipErrorUnknown || queries!=2u || syncs!=queries) return 126;
        }
        reset();setenv("QRT_CK_SM121_DIRECT_PV_OPERANDS","0",1);
        if(launch(65536,1024)!=hipSuccess || direct_pv_queries || final_bound_queries) return 127;
        setenv("QRT_CK_SM121_DIRECT_PV_OPERANDS","1",1);
    }
    unsetenv("QRT_CK_SM121_LONG_DIRECT_PV_OPERANDS");
    unsetenv("QRT_CK_SM121_DIRECT_PV_OPERANDS");
    unsetenv("QRT_CK_SM121_FINAL_PV_BOUND");
    for(const char* bad : {"2","-1","true","1junk"," 1"}) {
        reset();setenv("QRT_CK_SM121_SELECTIVE_QK_PROBABILITY",bad,1);
        if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries) return 83;
    }
    setenv("QRT_CK_SM121_SELECTIVE_QK_PROBABILITY","1",1);
    for(const char* mode : {"1","2","3"}) {
        reset();setenv("QRT_CK_SM121_COMPACT_PV_REPLAY",mode,1);
        if(launch(0,8192)!=hipSuccess || selective_qk_queries!=(*mode=='2' ? 0u : queries) ||
           allocations!=(*mode=='2' ? 5u : 6u)) return 84;
        reset();fail_query=2u;
        if(launch(0,8192)!=hipErrorUnknown || queries!=2u || syncs!=2u) return 85;
        for(unsigned tokens : {1u,8193u,17408u}) {
            reset();if(launch(0,tokens)!=hipSuccess || selective_qk_queries || g_sm121_selective_qk ||
               allocations!=(tokens==1u ? 4u : 5u)) return 86;
        }
    }
    reset();fail_allocation=6u;
    if(launch(0,8192)!=hipErrorUnknown || g_sm121_selective_qk || transposes || queries || syncs || live.size()!=5u) return 87;
    reset();
    if(launch(0,8192)!=hipSuccess || allocations!=6u || selective_qk_queries!=64u) return 88;
    transposes=queries=syncs=selective_qk_queries=0u;
    if(launch(0,7169)!=hipSuccess || allocations!=6u || selective_qk_queries!=57u) return 89;
    unsetenv("QRT_CK_SM121_SELECTIVE_QK_PROBABILITY");
    for(const char* bad : {"2","-1","true","1junk"," 1"}) {
        reset();setenv("QRT_CK_SM121_FLOAT_ALIGNMENT_QK",bad,1);
        if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries) return 90;
    }
    setenv("QRT_CK_SM121_FLOAT_ALIGNMENT_QK","1",1);
    for(const char* mode : {"1","2","3"}) {
        setenv("QRT_CK_SM121_COMPACT_PV_REPLAY",mode,1);
        for(unsigned tokens : {1u,7169u,8192u,8193u,17408u}) {
            reset();
            if(launch(0,tokens)!=hipSuccess || float_alignment_queries!=(tokens==1u ? 0u : queries) ||
               allocations!=(tokens==1u ? 4u : 5u) || selective_qk_queries) return 91;
        }
        reset();fail_query=2u;
        if(launch(0,8192)!=hipErrorUnknown || queries!=2u || syncs!=2u || float_alignment_queries!=2u)
            return 92;
    }
    reset();setenv("QRT_CK_SM121_SELECTIVE_QK_PROBABILITY","1",1);
    if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries || syncs) return 93;
    unsetenv("QRT_CK_SM121_SELECTIVE_QK_PROBABILITY");
    setenv("QRT_CK_SM121_COMPACT_PV_REPLAY","1",1);
    for(const char* bad : {"2","-1","true","1junk"," 1"}) {
        reset();setenv("QRT_CK_SM121_PROFILE_COMPLETED_STAGES",bad,1);
        if(launch(0,129)!=hipErrorInvalidValue || allocations || queries || syncs) return 128;
    }
    reset();setenv("QRT_CK_SM121_PROFILE_COMPLETED_STAGES","1",1);sync_ms=1u;
    if(launch(0,129)!=hipSuccess || queries!=2u || syncs!=14u ||
       profile_observations!=10u || clock_ms!=14u || allocations!=5u) return 129;
    // A failed completion at any observed stage must stop dependent stages and
    // batches, while the provider drains submitted work before releasing its lock.
    for(unsigned failure=3u;failure<=7u;++failure) {
        reset();fail_sync=failure;
        if(launch(0,129)!=hipErrorUnknown || queries!=1u || syncs!=failure+1u ||
           profile_observations!=failure-2u) return 130;
    }
    reset();sync_ms=3000u;
    if(launch(0,128)!=hipErrorLaunchTimeOut || queries!=1u ||
       profile_observations!=5u || syncs!=8u) return 131;
    reset();
    if(launch(7168,1)!=hipSuccess || queries!=1u || syncs!=1u || profile_observations) return 132;
    reset();setenv("QRT_CK_SM121_COMPACT_PV_REPLAY","2",1);
    if(launch(0,129)!=hipErrorInvalidValue || allocations || queries || syncs) return 133;
    unsetenv("QRT_CK_SM121_PROFILE_COMPLETED_STAGES");
    unsetenv("QRT_CK_SM121_TILED_EXACT_QK");
    unsetenv("QRT_CK_SM121_COMPACT_PV_REPLAY");
    reset();
    if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries || syncs) return 94;
    setenv("QRT_CK_SM121_TILED_EXACT_QK","1",1);
    setenv("QRT_CK_SM121_COMPACT_PV_REPLAY","1",1);
    for(const char* bad : {"2","-1","true","1junk"," 1"}) {
        reset();setenv("QRT_CK_SM121_PREPARED_DECODED_QK",bad,1);
        if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries || syncs) return 134;
    }
    setenv("QRT_CK_SM121_PREPARED_DECODED_QK","1",1);
    for(unsigned tokens : {2u,7169u,8192u}) {
        reset();
        if(launch(0,tokens)!=hipSuccess || allocations!=6u || decoded_preparations!=1u ||
           decoded_queries!=queries || transposes!=1u) return 135;
    }
    transposes=queries=syncs=decoded_queries=0u;
    if(launch(0,7169)!=hipSuccess || allocations!=6u || decoded_preparations!=2u ||
       decoded_queries!=57u || transposes!=1u) return 136;
    for(unsigned tokens : {1u,8193u,17408u}) {
        reset();
        if(launch(0,tokens)!=hipSuccess || decoded_preparations || decoded_queries ||
           g_sm121_prepared_decoded_qk || allocations!=(tokens==1u?4u:5u)) return 137;
    }
    reset();
    if(launch(128u,128u)!=hipSuccess || decoded_preparations || decoded_queries ||
       g_sm121_prepared_decoded_qk || allocations!=5u) return 138;
    reset();fail_allocation=6u;
    if(launch(0,8192)!=hipErrorUnknown || g_sm121_prepared_decoded_qk || transposes ||
       queries || syncs || live.size()!=5u) return 139;
    reset();fail_decoded_prepare=1u;
    if(launch(0,8192)!=hipErrorUnknown || decoded_preparations!=1u || transposes ||
       queries || syncs!=1u || live.size()!=6u) return 140;
    reset();fail_query=2u;
    if(launch(0,8192)!=hipErrorUnknown || queries!=2u || syncs!=2u || decoded_queries!=2u) return 141;
    for(const char* mode : {"0","2"}) {
        reset();setenv("QRT_CK_SM121_COMPACT_PV_REPLAY",mode,1);
        if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries || syncs) return 142;
    }
    setenv("QRT_CK_SM121_COMPACT_PV_REPLAY","1",1);
    reset();setenv("QRT_CK_SM121_FLOAT_ALIGNMENT_QK","0",1);
    if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries || syncs) return 143;
    unsetenv("QRT_CK_SM121_PREPARED_DECODED_QK");
    unsetenv("QRT_CK_SM121_FLOAT_ALIGNMENT_QK");
    // Exercise actual provider scheduling with original kernel arguments and
    // workspace ownership, including tails and both exact score producers.
    setenv("QRT_CK_SM121_FLOAT_ALIGNMENT_QK","1",1);
    for(const char* bad : {"0","2","7","65","-1","true","8junk"," 8"}) {
        reset();setenv("QRT_CK_SM121_SUBMIT_SLABS",bad,1);
        if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries || syncs) return 144;
    }
    for(const char* mode : {"1","3"}) for(const char* decoded : {"0","1"}) {
        setenv("QRT_CK_SM121_COMPACT_PV_REPLAY",mode,1);
        setenv("QRT_CK_SM121_PREPARED_DECODED_QK",decoded,1);
        for(const char* batch : {"32","64","128"}) for(const char* cadence : {"1","8","64"}) {
            setenv("QRT_CK_SM121_PREFILL_QUERY_BATCH",batch,1);
            setenv("QRT_CK_SM121_SUBMIT_SLABS",cadence,1);
            const unsigned batch_size=unsigned(std::atoi(batch)), group=unsigned(std::atoi(cadence));
            for(unsigned tokens : {2u,128u,129u,1023u,1024u,1025u,7169u,8192u}) {
                reset();track_submissions=true;
                const unsigned batches=(tokens+batch_size-1u)/batch_size;
                if(launch(0,tokens)!=hipSuccess || queries!=batches ||
                   syncs!=(batches+group-1u)/group || maximum_pending!=std::min(group,batches) ||
                   pending_submissions || allocations!=(*decoded=='1'?6u:5u) ||
                   decoded_queries!=(*decoded=='1'?batches:0u)) return 145;
                unsigned offset=0u;
                for(const auto& range:submitted_ranges) {
                    const unsigned count=std::min(batch_size,tokens-offset);
                    if(range!=std::array<unsigned,3>{offset,count,offset}) return 146;
                    offset+=count;
                }
                if(offset!=tokens) return 147;
            }
        }
    }
    setenv("QRT_CK_SM121_PREFILL_QUERY_BATCH","128",1);
    setenv("QRT_CK_SM121_COMPACT_PV_REPLAY","1",1);
    setenv("QRT_CK_SM121_SUBMIT_SLABS","8",1);
    for(unsigned failure=1u;failure<=64u;++failure) {
        reset();track_submissions=true;fail_query=failure;
        if(launch(0,8192)!=hipErrorUnknown || queries!=failure || syncs!=(failure+7u)/8u ||
           pending_submissions || maximum_pending>8u) return 148;
    }
    for(unsigned failure=1u;failure<=8u;++failure) {
        reset();track_submissions=true;fail_sync=failure;
        if(launch(0,8192)!=hipErrorUnknown || queries!=failure*8u || syncs!=failure ||
           pending_submissions || maximum_pending!=8u) return 149;
    }
    reset();track_submissions=true;fail_decoded_prepare=1u;
    if(launch(0,8192)!=hipErrorUnknown || queries || syncs!=1u || pending_submissions) return 150;
    reset();track_submissions=true;submit_ms=3000u;
    if(launch(0,8192)!=hipErrorLaunchTimeOut || queries!=7u || syncs!=1u ||
       pending_submissions || clock_ms!=21000u) return 151;
    reset();track_submissions=true;sync_ms=20001u;
    if(launch(0,8192)!=hipErrorLaunchTimeOut || queries!=8u || syncs!=1u || pending_submissions) return 152;
    reset();track_submissions=true;sync_ms=20000u;
    if(launch(0,1024)!=hipSuccess || queries!=8u || syncs!=1u || pending_submissions) return 153;
    // New batching is inert for decode, suffix and larger calls.
    setenv("QRT_CK_SM121_SUBMIT_SLABS","64",1);
    for(const auto& shape : {std::array<unsigned,2>{0,1},{0,8193},{0,17408},{128,128},{65536,1024}}) {
        reset();track_submissions=true;
        if(launch(shape[0],shape[1])!=hipSuccess || queries!=syncs || maximum_pending!=1u ||
           pending_submissions || decoded_queries) return 154;
    }
    reset();setenv("QRT_CK_SM121_PROFILE_COMPLETED_STAGES","1",1);
    if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries || syncs) return 155;
    unsetenv("QRT_CK_SM121_PROFILE_COMPLETED_STAGES");
    unsetenv("QRT_CK_SM121_PREPARED_DECODED_QK");
    for(const char* mode : {"0","2"}) {
        reset();setenv("QRT_CK_SM121_COMPACT_PV_REPLAY",mode,1);
        if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries || syncs) return 156;
    }
    unsetenv("QRT_CK_SM121_FLOAT_ALIGNMENT_QK");
    setenv("QRT_CK_SM121_COMPACT_PV_REPLAY","1",1);
    reset();setenv("QRT_CK_SM121_SELECTIVE_QK_PROBABILITY","1",1);
    if(launch(0,8192)!=hipErrorInvalidValue || allocations || queries || syncs) return 157;
    unsetenv("QRT_CK_SM121_SELECTIVE_QK_PROBABILITY");
    unsetenv("QRT_CK_SM121_SUBMIT_SLABS");
    unsetenv("QRT_CK_SM121_PREFILL_QUERY_BATCH");
    unsetenv("QRT_CK_SM121_COMPACT_PV_REPLAY");
    reset();unsetenv("QRT_CK_SM121_TILED_EXACT_QK");
    // Extended storage is separate and lazy. Failed allocations must leave
    // the existing short-context owner intact and submit no consumers.
    setenv("QRT_CK_SM121_TILED_EXACT_QK","1",1);
    for(unsigned failure:{5u,6u,7u}) {
        reset();fail_allocation=failure;
        if(launch(131072u,1024u)!=hipErrorUnknown || queries || transposes || syncs ||
           live.size()!=(failure==7u?6u:4u) || g_sm121_extended.mantissa_scores ||
           (failure<7u && (g_sm121_extended.scores || g_sm121_extended.transposed_keys))) return 180;
        auto* short_scores=g_sm121_scores;auto* short_keys=g_sm121_transposed_keys;
        fail_allocation=0u;
        if(launch(131072u,1024u)!=hipSuccess || live.size()!=7u ||
           !g_sm121_extended.scores || !g_sm121_extended.transposed_keys || !g_sm121_extended.mantissa_scores ||
           g_sm121_scores!=short_scores || g_sm121_transposed_keys!=short_keys ||
           g_sm121_extended.scores==short_scores || g_sm121_extended.transposed_keys==short_keys) return 181;
        auto* long_scores=g_sm121_extended.mantissa_scores;
        transposes=queries=syncs=0u;
        if(launch(0u,8192u)!=hipSuccess || !g_sm121_mantissa_scores || live.size()!=8u ||
           g_sm121_extended.mantissa_scores!=long_scores || g_sm121_mantissa_scores==long_scores) return 182;
    }
    for(unsigned failure:{1u,2u,16u}) {
        reset();track_submissions=true;fail_query=failure;
        if(launch(262144u,1024u)!=hipErrorUnknown || queries!=failure || syncs!=failure ||
           pending_submissions || maximum_pending!=1u) return 183;
    }
    setenv("QRT_CK_SM121_PREPARED_VALUE","1",1);
    reset();fail_allocation=8u;
    if(launch(262144u,1024u)!=hipErrorUnknown || live.size()!=7u ||
       g_sm121_extended.prepared_values || preparations || transposes || queries) return 184;
    fail_allocation=0u;
    if(launch(262144u,1024u)!=hipSuccess || live.size()!=8u || !g_sm121_extended.prepared_values ||
       g_sm121_prepared_values || preparations!=1u || transposes!=1u || queries!=32u) return 185;
    transposes=queries=syncs=preparations=0u;
    if(launch(263679u,1u)!=hipSuccess || preparations || transposes || queries!=1u || live.size()!=8u) return 186;
    reset();unsetenv("QRT_CK_SM121_PREPARED_VALUE");unsetenv("QRT_CK_SM121_TILED_EXACT_QK");
    for(unsigned output_start:{kSm121MaxTokens-1u,kSm121MaxTokens,0xffffffffu}) {
        if(launch_sm121_attention(&operand,&operand,&operand,&output,nullptr,0u,2u,output_start)!=hipErrorInvalidValue || allocations) return 187;
    }
    if(launch_sm121_attention(&operand,&operand,&operand,&output,nullptr,kSm121MaxTokens-1u,1u,kSm121MaxTokens-1u)!=hipSuccess) return 188;
    reset();
    for(unsigned tokens:{1u,131072u,262144u,262145u,263168u,264705u,kSm121MaxTokens,kSm121MaxTokens+1u,0xffffffffu}) {
        setenv("QRT_CK_FMHA_SM121_FULL_PREFIX","0",1);
        if(supported_dynamic_tokens(tokens)!=(tokens<=262144u)) return 189;
        setenv("QRT_CK_FMHA_SM121_FULL_PREFIX","1",1);
        if(supported_dynamic_tokens(tokens)!=(tokens<=kSm121MaxTokens)) return 190;
    }
    if(supported_dynamic_tokens(0u)) return 191;
    unsetenv("QRT_CK_FMHA_SM121_FULL_PREFIX");
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
            result = subprocess.run([executable], check=True, timeout=5, capture_output=True, text=True)
            profiles = [dict(re.findall(r"(\w+)=([^ ]+)", line))
                        for line in result.stderr.splitlines()
                        if line.startswith("SM121_COMPLETED_STAGE_PROFILE ")]
            self.assertEqual(len(profiles), 1)
            row = profiles[0]
            self.assertEqual((row["query_count"], row["batches"], row["completed_stages"]),
                             ("129", "2", "10"))
            for field in ("qk_ms", "probability_ms", "approximate_pv_ms", "collect_pv_ms", "exact_pv_ms"):
                self.assertEqual(float(row[field]), 2.0)
            self.assertEqual(float(row["entry_wait_ms"]), 1.0)
            self.assertEqual(float(row["preparation_ms"]), 1.0)
            self.assertEqual(float(row["dispatch_remainder_ms"]), 2.0)
            self.assertEqual(float(row["total_ms"]), 14.0)


if __name__ == "__main__":
    unittest.main()
