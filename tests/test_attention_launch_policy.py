"""Check actual split-attention bounds and failure ordering without GPU work."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import attention_capacity

ROOT = Path(__file__).resolve().parents[1]


class AttentionLaunchPolicyTests(unittest.TestCase):
    def test_scratch_span_is_checked_and_failed_scores_do_not_submit_pv(self):
        header = (ROOT / "native/providers/ck_fmha/blackwell_attention.h").read_text()
        launch = "constexpr unsigned int kSplitMaxTokens" + header.split(
            "constexpr unsigned int kSplitMaxTokens", 1
        )[1].split("} // namespace qrt_blackwell_attention", 1)[0]
        row = 'struct IntegerOperandRow' + header.split('struct IntegerOperandRow', 1)[1].split(
            '__device__ __forceinline__ void blackwell_prepare_integer_row', 1)[0]
        packed = 'struct PrepackedIntegerWorkspace' + header.split('struct PrepackedIntegerWorkspace', 1)[1].split(
            'template<IntegerRowKind Kind>', 1)[0]
        source = r'''
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#define __host__
#define __device__
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
using hipStream_t = void*;
using hipEvent_t = void*;
struct dim3 { unsigned x, y, z; explicit dim3(unsigned a, unsigned b = 1u, unsigned c = 1u):x(a),y(b),z(c) {} };
namespace qrt_sm121_strided_pair { constexpr unsigned kCellsPer256Threads=128u; }
namespace qrt_sm121_integer_core { struct Row {}; }
constexpr unsigned kQueryHeads = 16, kHeadDim = 256, kThreads = 256;
constexpr unsigned kKvHeads = 2, kIntegerMatrixColumns = 128;
constexpr unsigned kBlackwellSubgroups = 16;
constexpr unsigned kCooperativeColumns = 64;
constexpr unsigned kTiledExactQueries = 8, kTiledExactKeys = 32;
constexpr unsigned kExactTileTokens = 32;
''' + attention_capacity() + row + packed + r'''
void blackwell_exact_scores_kernel() {}
void blackwell_cooperative_scores_kernel() {}
void blackwell_cooperative_value_kernel() {}
void blackwell_transpose_keys_kernel() {}
void blackwell_strided_scores_kernel() {}
void blackwell_tiled_exact_scores_kernel() {}
void blackwell_prepare_value_encoding_kernel() {}
template<bool SparseCore = false> void blackwell_cell_parallel_integer_scores_kernel() {}
template<bool NativeProducts = false> void blackwell_transposed_scores_kernel() {}
void blackwell_online_probability_kernel() {}
void blackwell_parallel_probability_kernel() {}
void blackwell_probability_value_kernel() {}
void blackwell_collect_pv_replay_kernel() {}
template<bool TransposedValue = false> void blackwell_compacted_pv_replay_kernel() {}
template<bool NativeMma = false, bool Prepacked = false, bool SparseCore = false> void blackwell_mantissa_scores_kernel() {}
template<bool NativeMma = false, bool Prepacked = false, bool BoundError = false> void blackwell_mantissa_value_kernel() {}
template<IntegerRowKind Kind> void blackwell_prepare_integer_rows_kernel() {}
template<IntegerRowKind Kind> void blackwell_prepare_integer_core_rows_kernel() {}
template<bool SerialValue, bool PrecomputedScores = false, bool SplitDecodeValue = false,
         bool NativeProducts = false, bool StridedValue = false, bool WarpSoftmax = false,
         bool PreparedValue = false>
void blackwell_exact_attention_kernel() {}
unsigned launches = 0, error_queries = 0, memsets = 0;
bool fail_memset = false;
hipError_t hipMemsetAsync(void*, int, size_t bytes, hipStream_t) {
    ++memsets; return fail_memset || bytes != sizeof(unsigned) ? hipErrorUnknown : hipSuccess;
}
unsigned fail_launch = 0;
bool fail_scores = false, fail_probability = false;
bool fail_event = false;
unsigned events = 0;
hipError_t hipEventRecord(hipEvent_t, hipStream_t) {
    ++events;
    return fail_event ? hipErrorUnknown : hipSuccess;
}
const char* launch_names[64]{};
unsigned launch_threads[64]{}, launch_grids[64]{};
template<class Kernel, class... T> void record_launch(const char* name, Kernel, dim3 grid, dim3 threads, T...) {
    launch_grids[launches]=grid.x; launch_threads[launches]=threads.x; launch_names[launches++] = name;
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel, ...) record_launch(#kernel, kernel, __VA_ARGS__)
hipError_t hipGetLastError() {
    ++error_queries;
    return ((fail_launch && launches == fail_launch) ||
        (fail_scores && launches == 1u) || (fail_probability && launches == 2u))
        ? hipErrorUnknown : hipSuccess;
}
''' + launch + r'''
int main() {
    uint16_t operand = 0u; float output = 0.0f, scratch = 0.0f;
    auto split = [&](unsigned start, unsigned count, float* workspace,
                     size_t elements, unsigned layout = 2u) {
        return launch_queries(&operand, &operand, &operand, &output, nullptr,
            start, count, 0u, nullptr, nullptr, nullptr, true, nullptr,
            layout, workspace, elements);
    };
    if (split(0, 8, nullptr, 1024) != hipErrorInvalidValue) return 1;
    if (split(0, 8, &scratch, 1023) != hipErrorInvalidValue) return 2;
    if (split(7160, 8, &scratch, 8u * 16u * 8u) != hipErrorInvalidValue) return 3;
    if (split(kSplitMaxTokens - 1u, 2, &scratch, SIZE_MAX) != hipErrorInvalidValue) return 4;
    if (split(0, 33, &scratch, SIZE_MAX) != hipErrorInvalidValue) return 5;
    if (split(0, 0, &scratch, SIZE_MAX) != hipErrorInvalidValue) return 6;
    if (split(UINT32_MAX, 8, &scratch, SIZE_MAX) != hipErrorInvalidValue) return 7;
    if (split(0, 8, &scratch, SIZE_MAX, 9) != hipErrorInvalidValue) return 8;
    if (launches || error_queries) return 9;
    if (split(0, 8, &scratch, 1791, 3) != hipErrorInvalidValue || launches || error_queries)
        return 13;
    if (split(7160, 8, &scratch, 8u * 16u * 7168u) != hipSuccess ||
        launches != 2u || error_queries != 2u) return 10;
    launches = error_queries = 0u; fail_scores = true;
    if (split(0, 8, &scratch, 1024) != hipErrorUnknown ||
        launches != 1u || error_queries != 1u) return 11;
    launches = error_queries = 0u; fail_scores = false;
    if (split(262143, 1, nullptr, 0, 1) != hipSuccess || launches != 1u)
        return 12;  // Original long terminal route needs no new scratch.
    launches = error_queries = 0u;
    if (split(0, 8, &scratch, 1792, 3) != hipSuccess || launches != 3u || error_queries != 3u)
        return 14;
    launches = error_queries = 0u; fail_probability = true;
    if (split(0, 8, &scratch, 1792, 3) != hipErrorUnknown || launches != 2u || error_queries != 2u)
        return 15;
    if (events) return 16;  // No instrumentation calls in the provider default.
    launches = error_queries = 0u; fail_probability = false; fail_event = true;
    if (launch_queries(&operand, &operand, &operand, &output, nullptr,
        0, 8, 0, nullptr, nullptr, nullptr, true, nullptr, 2, &scratch, 1024,
        &operand) != hipErrorUnknown || launches != 1u || events != 1u)
        return 17;  // A failed timing event also prevents dependent PV work.
    launches = error_queries = events = 0u; fail_event = false;
    if (transpose_keys(nullptr, &operand, 512u, 1u, nullptr) != hipErrorInvalidValue ||
        transpose_keys(&operand, &operand, 511u, 1u, nullptr) != hipErrorInvalidValue ||
        transpose_keys(&operand, &operand, SIZE_MAX, (kSplitMaxTokens + 1u), nullptr) != hipErrorInvalidValue ||
        transpose_keys(&operand, &operand, SIZE_MAX, 0u, nullptr) != hipErrorInvalidValue || launches)
        return 18;
    if (transpose_keys(&operand, &operand, 16384u * 512u, 16384u, nullptr) != hipSuccess ||
        launches != 1u || error_queries != 1u) return 19;
    launches = error_queries = 0u;
    auto transposed = [&](const uint16_t* prepared, unsigned stride) {
        return launch_queries(&operand, &operand, &operand, &output, nullptr,
            7160, 8, 0, nullptr, nullptr, nullptr, true, nullptr, 4, &scratch,
            8u * 16u * 7168u, nullptr, nullptr, prepared, stride);
    };
    if (transposed(nullptr, 7168u) != hipErrorInvalidValue ||
        transposed(&operand, 7167u) != hipErrorInvalidValue ||
        transposed(&operand, (kSplitMaxTokens + 1u)) != hipErrorInvalidValue || launches) return 20;
    if (transposed(&operand, 7169u) != hipSuccess || launches != 2u || error_queries != 2u)
        return 21;
    launches = error_queries = 0u; fail_scores = true;
    if (transposed(&operand, 7169u) != hipErrorUnknown || launches != 1u || error_queries != 1u)
        return 22;
    launches = error_queries = 0u; fail_scores = false;
    const size_t continuation_cells = 16u * 8197u;
    if (split(8196, 1, &scratch, continuation_cells - 1u) != hipErrorInvalidValue || launches)
        return 23;
    if (split(8196, 1, &scratch, continuation_cells) != hipSuccess || launches != 2u)
        return 24;
    launches = error_queries = 0u;
    if (split(16383, 1, &scratch, 16u * 16384u) != hipSuccess || launches != 2u)
        return 25;
    if (split_scratch_elements(1u, (kSplitMaxTokens + 1u), 2u) != 0u) return 26;
    for (const unsigned tokens : {17408u, 17920u, kSplitMaxTokens}) {
        launches = error_queries = 0u;
        const size_t elements = 16u * tokens;
        if (split(tokens - 1u, 1u, &scratch, elements - 1u) != hipErrorInvalidValue || launches)
            return 73;
        if (split(tokens - 1u, 1u, &scratch, elements) != hipSuccess || launches != 2u)
            return 74;
    }
    launches = error_queries = 0u;
    if (launch_queries(&operand, &operand, &operand, &output, nullptr,
        0, 8, 0, nullptr, nullptr, nullptr, true, nullptr, 2, &scratch, 1024,
        nullptr, nullptr, &operand, 8u, true) != hipErrorInvalidValue || launches)
        return 27;
    if (launch_queries(&operand, &operand, &operand, &output, nullptr,
        0, 8, 0, nullptr, nullptr, nullptr, true, nullptr, 4, &scratch, 1024,
        nullptr, nullptr, &operand, 8u, true) != hipSuccess || launches != 2u)
        return 28;
    launches = error_queries = 0u; fail_scores = true;
    if (launch_queries(&operand, &operand, &operand, &output, nullptr,
        0, 8, 0, nullptr, nullptr, nullptr, true, nullptr, 4, &scratch, 1024,
        nullptr, nullptr, &operand, 8u, true) != hipErrorUnknown || launches != 1u)
        return 29;
    launches = error_queries = 0u; fail_scores = false;
    auto matrix = [&](size_t elements, const uint16_t* prepared,
                      unsigned stride = 17u, bool native = false) {
        return launch_queries(&operand, &operand, &operand, &output, nullptr,
            1, 16, 0, nullptr, nullptr, nullptr, true, nullptr, 5, &scratch,
            elements, nullptr, nullptr, prepared, stride, native);
    };
    const size_t matrix_elements = split_scratch_elements(16u, 17u, 5u);
    if (matrix_elements != 7040u || matrix(matrix_elements - 1u, &operand) != hipErrorInvalidValue ||
        matrix(matrix_elements, nullptr) != hipErrorInvalidValue ||
        matrix(matrix_elements, &operand, 16u) != hipErrorInvalidValue ||
        matrix(matrix_elements, &operand, (kSplitMaxTokens + 1u)) != hipErrorInvalidValue ||
        matrix(matrix_elements, &operand, 17u, true) != hipErrorInvalidValue || launches)
        return 30;
    if (matrix(matrix_elements, &operand) != hipSuccess || launches != 3u || error_queries != 3u) return 31;
    launches = error_queries = 0u; fail_scores = true;
    if (matrix(matrix_elements, &operand) != hipErrorUnknown || launches != 1u) return 32;
    launches = error_queries = 0u; fail_scores = false; fail_probability = true;
    if (matrix(matrix_elements, &operand) != hipErrorUnknown || launches != 2u) return 33;
    fail_probability = false;
    for (unsigned layout : {6u, 7u}) {
        launches = error_queries = 0u;
        if (launch_queries(&operand, &operand, &operand, &output, nullptr,
            1, 16, 0, nullptr, nullptr, nullptr, true, nullptr, layout, &scratch,
            matrix_elements, nullptr, nullptr, &operand, 17u) != hipSuccess || launches != 3u) return 34;
        const char* expected = layout == 6u ? "blackwell_transposed_scores_kernel<false>" : "blackwell_mantissa_scores_kernel<true>";
        if (!std::strstr(launch_names[0], expected) ||
            !std::strstr(launch_names[1], "blackwell_online_probability_kernel") ||
            !std::strstr(launch_names[2], "blackwell_mantissa_value_kernel<true>")) return 35;
    }
    launches = error_queries = 0u;
    if (split(1, 16, &scratch, matrix_elements - 1u, 8u) != hipErrorInvalidValue || launches) return 36;
    if (split(1, 16, &scratch, matrix_elements, 8u) != hipSuccess || launches != 3u ||
        !std::strstr(launch_names[0], "blackwell_cooperative_scores_kernel") ||
        !std::strstr(launch_names[1], "blackwell_online_probability_kernel") ||
        !std::strstr(launch_names[2], "blackwell_cooperative_value_kernel")) return 37;
    launches = error_queries = 0u; fail_scores = true;
    if (split(1, 16, &scratch, matrix_elements, 8u) != hipErrorUnknown || launches != 1u) return 38;
    launches = error_queries = 0u; fail_scores = false; fail_probability = true;
    if (split(1, 16, &scratch, matrix_elements, 8u) != hipErrorUnknown || launches != 2u) return 39;
    launches = error_queries = 0u; fail_probability = false;
    IntegerOperandRow encoded{};
    PrepackedIntegerWorkspace prepared{&encoded,&encoded,&encoded,&encoded,17u,16u};
    auto prepacked = [&](const PrepackedIntegerWorkspace* p, size_t elements) {
        return launch_queries(&operand, &operand, &operand, &output, nullptr,
            1, 16, 0, nullptr, nullptr, nullptr, true, nullptr, 9, &scratch,
            elements, nullptr, nullptr, nullptr, 0u, false, p);
    };
    if (prepacked(nullptr,matrix_elements) != hipErrorInvalidValue ||
        prepacked(&prepared,matrix_elements-1) != hipErrorInvalidValue || launches) return 40;
    prepared.tokens=16;
    if (prepacked(&prepared,matrix_elements) != hipErrorInvalidValue || launches) return 41;
    prepared.tokens=17;prepared.queries=15;
    if (prepacked(&prepared,matrix_elements) != hipErrorInvalidValue || launches) return 42;
    prepared.queries=16;prepared.probability=nullptr;
    if (prepacked(&prepared,matrix_elements) != hipErrorInvalidValue || launches) return 43;
    prepared.probability=&encoded;
    if (prepacked(&prepared,matrix_elements) != hipSuccess || launches!=5 || error_queries!=5 ||
        !std::strstr(launch_names[0],"blackwell_prepare_integer_rows_kernel") ||
        !std::strstr(launch_names[1],"blackwell_mantissa_scores_kernel<false, true>") ||
        !std::strstr(launch_names[2],"blackwell_online_probability_kernel") ||
        !std::strstr(launch_names[3],"blackwell_prepare_integer_rows_kernel") ||
        !std::strstr(launch_names[4],"blackwell_mantissa_value_kernel<false, true>")) return 44;
    for(unsigned failed=1;failed<=5;++failed){
        launches=error_queries=0;fail_launch=failed;
        if(prepacked(&prepared,matrix_elements)!=hipErrorUnknown || launches!=failed || error_queries!=failed) return 45;
    }
    fail_launch=0;
    for (unsigned layout : {10u,11u,12u}) {
        launches=error_queries=0;
        const size_t elements=16u*16u*17u;
        if (split_scratch_elements(16u,17u,layout)!=elements || !split_transposed_keys(layout) ||
            split_separate_probability(layout)) return 46;
        auto strided=[&](const uint16_t* key, unsigned stride, size_t span) {
            return launch_queries(&operand,&operand,&operand,&output,nullptr,1,16,0,
                nullptr,nullptr,nullptr,true,nullptr,layout,&scratch,span,nullptr,nullptr,key,stride);
        };
        if (strided(nullptr,17,elements)!=hipErrorInvalidValue ||
            strided(&operand,16,elements)!=hipErrorInvalidValue ||
            strided(&operand,(kSplitMaxTokens + 1u),elements)!=hipErrorInvalidValue ||
            strided(&operand,17,elements-1u)!=hipErrorInvalidValue || launches) return 47;
        if (strided(&operand,17,elements)!=hipSuccess || launches!=2u || error_queries!=2u ||
            launch_threads[0]!=256u || launch_threads[1]!=(layout==11u ? 256u : 512u)) return 48;
        const char* score=layout==12u ? "blackwell_transposed_scores_kernel<false>" : "blackwell_strided_scores_kernel";
        const char* value=layout==11u ? "blackwell_exact_attention_kernel<true, true>" :
            "blackwell_exact_attention_kernel<true, true, false, false, true>";
        if (!std::strstr(launch_names[0],score) || !std::strstr(launch_names[1],value)) return 49;
        for (unsigned failed=1;failed<=2;++failed) {
            launches=error_queries=0;fail_launch=failed;
            if(strided(&operand,17,elements)!=hipErrorUnknown || launches!=failed || error_queries!=failed) return 50;
        }
        fail_launch=0;
    }
    launches=error_queries=0;
    if (split(0,8,&scratch,SIZE_MAX,22u)!=hipErrorInvalidValue || launches ||
        split_scratch_elements(8u,8u,25u)!=0u) return 51;
    const size_t selective_elements=split_scratch_elements(16u,17u,13u);
    if(selective_elements!=matrix_elements+16u*16u*256u) return 52;
    auto selective=[&](size_t elements, const unsigned char* rcp, bool sum=true) {
        return launch_queries(&operand,&operand,&operand,&output,nullptr,1,16,0,
            nullptr,nullptr,nullptr,sum,rcp,13u,&scratch,elements,nullptr,nullptr,&operand,17u);
    };
    const auto* rcp=reinterpret_cast<const unsigned char*>(&operand);
    if(selective(selective_elements,nullptr)!=hipErrorInvalidValue ||
       selective(selective_elements,rcp,false)!=hipErrorInvalidValue ||
       selective(selective_elements-1u,rcp)!=hipErrorInvalidValue || launches) return 53;
    if(selective(selective_elements,rcp)!=hipSuccess || launches!=4u || error_queries!=4u ||
       !std::strstr(launch_names[0],"blackwell_transposed_scores_kernel<false>") ||
       !std::strstr(launch_names[2],"blackwell_mantissa_value_kernel<true, false, true>") ||
       !std::strstr(launch_names[3],"blackwell_probability_value_kernel")) return 54;
    for(unsigned failed=1;failed<=4;++failed) {
        launches=error_queries=0;fail_launch=failed;
        if(selective(selective_elements,rcp)!=hipErrorUnknown || launches!=failed || error_queries!=failed)
            return 55;
    }
    launches=error_queries=fail_launch=0u;
    auto native_qk=[&](size_t elements,const uint16_t* key) {
        return launch_queries(&operand,&operand,&operand,&output,nullptr,1,16,0,
            nullptr,nullptr,nullptr,true,rcp,14u,&scratch,elements,nullptr,nullptr,key,17u);
    };
    if(split_scratch_elements(16u,17u,14u)!=matrix_elements ||
       native_qk(matrix_elements-1u,&operand)!=hipErrorInvalidValue ||
       native_qk(matrix_elements,nullptr)!=hipErrorInvalidValue || launches) return 56;
    if(native_qk(matrix_elements,&operand)!=hipSuccess || launches!=3u ||
       !std::strstr(launch_names[0],"blackwell_mantissa_scores_kernel<true>") ||
       !std::strstr(launch_names[1],"blackwell_online_probability_kernel") ||
       !std::strstr(launch_names[2],"blackwell_probability_value_kernel")) return 57;
    for(unsigned failed=1;failed<=3;++failed) {
        launches=error_queries=0;fail_launch=failed;
        if(native_qk(matrix_elements,&operand)!=hipErrorUnknown || launches!=failed) return 58;
    }
    launches=error_queries=fail_launch=0u;
    auto tiled=[&](size_t elements,const uint16_t* key) {
        return launch_queries(&operand,&operand,&operand,&output,nullptr,7110,32,0,
            nullptr,nullptr,nullptr,true,rcp,15u,&scratch,elements,nullptr,nullptr,key,7169u);
    };
    const size_t tiled_cells=32u*16u*7142u;
    if(split_scratch_elements(32u,7142u,15u)!=tiled_cells || split_separate_probability(15u) ||
       tiled(tiled_cells-1u,&operand)!=hipErrorInvalidValue ||
       tiled(tiled_cells,nullptr)!=hipErrorInvalidValue || launches) return 59;
    if(tiled(tiled_cells,&operand)!=hipSuccess || launches!=2u ||
       !std::strstr(launch_names[0],"blackwell_tiled_exact_scores_kernel") ||
       !std::strstr(launch_names[1],"blackwell_exact_attention_kernel<true, true>")) return 60;
    for(unsigned failed=1;failed<=2;++failed) {
        launches=error_queries=0;fail_launch=failed;
        if(tiled(tiled_cells,&operand)!=hipErrorUnknown || launches!=failed) return 61;
    }
    launches=error_queries=fail_launch=0u;
    auto warp=[&](size_t elements,const uint16_t* key) {
        return launch_queries(&operand,&operand,&operand,&output,nullptr,7110,32,0,
            nullptr,nullptr,nullptr,true,rcp,16u,&scratch,elements,nullptr,nullptr,key,7169u);
    };
    if(split_scratch_elements(32u,7142u,16u)!=tiled_cells || split_separate_probability(16u) ||
       warp(tiled_cells-1u,&operand)!=hipErrorInvalidValue ||
       warp(tiled_cells,nullptr)!=hipErrorInvalidValue || launches) return 62;
    if(warp(tiled_cells,&operand)!=hipSuccess || launches!=2u ||
       !std::strstr(launch_names[0],"blackwell_tiled_exact_scores_kernel") ||
       !std::strstr(launch_names[1],"blackwell_exact_attention_kernel<true, true, false, false, false, true>")) return 63;
    for(unsigned failed=1;failed<=2;++failed) {
        launches=error_queries=0;fail_launch=failed;
        if(warp(tiled_cells,&operand)!=hipErrorUnknown || launches!=failed) return 64;
    }
    launches=error_queries=fail_launch=0u;
    uint32_t wide_value=0u;
    auto wide=[&](size_t cells,const uint32_t* values,unsigned tokens=7169u) {
        return launch_queries(&operand,&operand,&operand,&output,nullptr,7110,32,0,
            nullptr,nullptr,nullptr,true,rcp,17u,&scratch,cells,nullptr,nullptr,&operand,7169u,
            false,nullptr,values,tokens);
    };
    if(split_scratch_elements(32u,7142u,17u)!=tiled_cells || split_separate_probability(17u) ||
       wide(tiled_cells-1u,&wide_value)!=hipErrorInvalidValue || wide(tiled_cells,nullptr)!=hipErrorInvalidValue ||
       wide(tiled_cells,&wide_value,7141u)!=hipErrorInvalidValue ||
       wide(tiled_cells,&wide_value,(kSplitMaxTokens + 1u))!=hipErrorInvalidValue || launches) return 65;
    if(wide(tiled_cells,&wide_value)!=hipSuccess || launches!=2u ||
       !std::strstr(launch_names[0],"blackwell_tiled_exact_scores_kernel") ||
       !std::strstr(launch_names[1],"blackwell_exact_attention_kernel<true, true, false, false, false, false, true>")) return 66;
    for(unsigned failed=1;failed<=2;++failed) {
        launches=error_queries=0;fail_launch=failed;
        if(wide(tiled_cells,&wide_value)!=hipErrorUnknown || launches!=failed) return 67;
    }
    launches=error_queries=fail_launch=0u;
    if(prepare_value_encoding(nullptr,&wide_value,512u,1u,nullptr)!=hipErrorInvalidValue ||
       prepare_value_encoding(&operand,nullptr,512u,1u,nullptr)!=hipErrorInvalidValue ||
       prepare_value_encoding(&operand,&wide_value,511u,1u,nullptr)!=hipErrorInvalidValue ||
       prepare_value_encoding(&operand,&wide_value,SIZE_MAX,0u,nullptr)!=hipErrorInvalidValue ||
       prepare_value_encoding(&operand,&wide_value,SIZE_MAX,(kSplitMaxTokens + 1u),nullptr)!=hipErrorInvalidValue || launches) return 68;
    if(prepare_value_encoding(&operand,&wide_value,16384u*512u,16384u,nullptr)!=hipSuccess ||
       launches!=1u || !std::strstr(launch_names[0],"blackwell_prepare_value_encoding_kernel")) return 69;
    launches=error_queries=0u;fail_launch=1u;
    if(prepare_value_encoding(&operand,&wide_value,512u,1u,nullptr)!=hipErrorUnknown || launches!=1u) return 70;
    launches=error_queries=fail_launch=0u;
    auto cells=[&](size_t span,const uint32_t* values,unsigned layout) {
        return launch_queries(&operand,&operand,&operand,&output,nullptr,7110,32,0,
            nullptr,nullptr,nullptr,true,rcp,layout,&scratch,span,nullptr,nullptr,&operand,7169u,
            false,nullptr,values,7169u);
    };
    for(unsigned layout:{18u,19u,20u}) {
        launches=error_queries=fail_launch=0u;
        if(split_scratch_elements(32u,7142u,layout)!=tiled_cells || split_separate_probability(layout) ||
           cells(tiled_cells-1u,&wide_value,layout)!=hipErrorInvalidValue ||
           cells(tiled_cells,nullptr,layout)!=hipErrorInvalidValue || launches) return 71;
        const char* expected=layout==18u ? "blackwell_cell_parallel_integer_scores_kernel<false>" :
            layout==19u ? "blackwell_mantissa_scores_kernel<false, false, true>" :
                         "blackwell_cell_parallel_integer_scores_kernel<true>";
        if(cells(tiled_cells,&wide_value,layout)!=hipSuccess || launches!=2u ||
           !std::strstr(launch_names[0],expected) ||
           !std::strstr(launch_names[1],"blackwell_exact_attention_kernel<true, true, false, false, false, false, true>")) return 72;
        for(unsigned failed=1;failed<=2;++failed) {
            launches=error_queries=0;fail_launch=failed;
            if(cells(tiled_cells,&wide_value,layout)!=hipErrorUnknown || launches!=failed) return 73;
        }
    }
    launches=error_queries=fail_launch=0u;
    if(split_scratch_elements(32u,7142u,25u)!=0u ||
       cells(tiled_cells,&wide_value,22u)!=hipErrorInvalidValue || launches) return 75;
    qrt_sm121_integer_core::Row core_row;
    CoreIntegerWorkspace core{&core_row,&core_row,7169u,32u};
    auto core_call=[&](const CoreIntegerWorkspace* rows,size_t span) {
        return launch_queries(&operand,&operand,&operand,&output,nullptr,7110,32,0,
            nullptr,nullptr,nullptr,true,rcp,21u,&scratch,span,nullptr,nullptr,nullptr,0u,
            false,nullptr,&wide_value,7169u,rows);
    };
    if(split_scratch_elements(32u,7142u,21u)!=tiled_cells || split_separate_probability(21u) ||
       split_transposed_keys(21u) || core_call(nullptr,tiled_cells)!=hipErrorInvalidValue ||
       core_call(&core,tiled_cells-1u)!=hipErrorInvalidValue || launches) return 76;
    core.tokens=7141u;if(core_call(&core,tiled_cells)!=hipErrorInvalidValue || launches) return 77;
    core.tokens=7169u;core.queries=31u;if(core_call(&core,tiled_cells)!=hipErrorInvalidValue || launches) return 78;
    core.queries=32u;core.key=nullptr;if(core_call(&core,tiled_cells)!=hipErrorInvalidValue || launches) return 79;
    core.key=&core_row;core.query=nullptr;if(core_call(&core,tiled_cells)!=hipErrorInvalidValue || launches) return 80;
    core.query=&core_row;
    if(core_call(&core,tiled_cells)!=hipSuccess || launches!=3u ||
       !std::strstr(launch_names[0],"blackwell_prepare_integer_core_rows_kernel") ||
       !std::strstr(launch_names[1],"blackwell_mantissa_scores_kernel<false, true, true>") ||
       !std::strstr(launch_names[2],"blackwell_exact_attention_kernel<true, true, false, false, false, false, true>")) return 81;
    for(unsigned failed=1;failed<=3;++failed) {
        launches=error_queries=0;fail_launch=failed;
        if(core_call(&core,tiled_cells)!=hipErrorUnknown || launches!=failed || error_queries!=failed) return 82;
    }
    fail_launch=0;fail_scores=fail_probability=fail_event=false;
    for(unsigned layout : {22u,23u,24u}) {
        launches=error_queries=memsets=0;
        const size_t span=split_scratch_elements(16u,17u,layout);
        if(span != matrix_elements + ((layout==22u || layout==24u) ? 2u*16u*16u*256u+1u : 16u*16u*256u) ||
           !split_transposed_keys(layout) || !split_separate_probability(layout)) return 83;
        auto compact_call=[&](size_t bytes) {
            return launch_queries(&operand,&operand,&operand,&output,nullptr,1u,16u,0u,
                nullptr,nullptr,nullptr,true,rcp,layout,&scratch,bytes,nullptr,nullptr,&operand,17u);
        };
        if(compact_call(span-1u)!=hipErrorInvalidValue || launches || memsets) return 84;
        if(compact_call(span)!=hipSuccess || launches!=((layout==22u || layout==24u)?5u:4u) ||
           memsets!=((layout==22u || layout==24u)?1u:0u) || !std::strstr(launch_names[0],"blackwell_tiled_exact_scores_kernel")) return 85;
        if((layout==22u || layout==24u) && (launch_grids[3]!=256u || launch_grids[4]!=1024u ||
            !std::strstr(launch_names[3],"blackwell_collect_pv_replay_kernel") ||
            !std::strstr(launch_names[4],"blackwell_compacted_pv_replay_kernel"))) return 86;
        if(launch_threads[1]!=(layout==24u?256u:32u) ||
           !std::strstr(launch_names[1],layout==24u ? "blackwell_parallel_probability_kernel" : "blackwell_online_probability_kernel")) return 89;
        for(unsigned failure=1u;failure<=((layout==22u || layout==24u)?5u:4u);++failure) {
            launches=error_queries=memsets=0;fail_launch=failure;
            if(compact_call(span)!=hipErrorUnknown || launches!=failure || error_queries!=failure) return 87;
        }
        fail_launch=0;
        if(layout==22u || layout==24u) {
            launches=error_queries=memsets=0;fail_memset=true;
            if(compact_call(span)!=hipErrorUnknown || launches!=3u || memsets!=1u) return 88;
            fail_memset=false;
        }
    }
    for(unsigned layout : {22u,24u}) {
        for(unsigned count : {33u,64u,65u,127u,128u}) {
            launches=error_queries=memsets=fail_launch=0u;
            const unsigned start=8192u-count;
            const size_t span=split_scratch_elements(count,8192u,layout);
            const size_t expected=size_t(count)*16u*(8192u+4096u+257u+512u)+1u;
            auto slab=[&](size_t elements) {
                return launch_queries(&operand,&operand,&operand,&output,nullptr,start,count,3u,
                    nullptr,nullptr,nullptr,true,rcp,layout,&scratch,elements,nullptr,nullptr,&operand,8192u);
            };
            if(span!=expected || slab(span-1u)!=hipErrorInvalidValue || launches || memsets) return 90;
            if(slab(span)!=hipSuccess || launches!=5u || memsets!=1u || launch_grids[4]!=1024u) return 91;
            for(unsigned failure=1u;failure<=5u;++failure) {
                launches=error_queries=memsets=0u;fail_launch=failure;
                if(slab(span)!=hipErrorUnknown || launches!=failure || error_queries!=failure) return 92;
            }
        }
        if(split_scratch_elements(129u,8192u,layout) || split_scratch_elements(33u,8193u,layout)) return 93;
    }
    if(split_scratch_elements(33u,8192u,23u) || split_scratch_elements(33u,8192u,15u)) return 94;
    struct Observed { unsigned next=0u, fail=5u; } observed;
    SplitCompletionObserver observer{&observed, [](void* state, unsigned stage, hipStream_t)->int {
        auto& o=*static_cast<Observed*>(state);
        if(stage!=o.next++) return int(hipErrorInvalidValue);
        return stage==o.fail ? int(hipErrorUnknown) : int(hipSuccess);
    }};
    const size_t observed_span=split_scratch_elements(16u,17u,22u);
    for(unsigned failure=0u;failure<=5u;++failure) {
        observed={0u,failure};launches=error_queries=memsets=fail_launch=0u;
        const int status=launch_queries(&operand,&operand,&operand,&output,nullptr,1u,16u,0u,
            nullptr,nullptr,nullptr,true,rcp,22u,&scratch,observed_span,nullptr,nullptr,&operand,17u,
            false,nullptr,nullptr,0u,nullptr,&observer);
        if(status!=(failure==5u ? hipSuccess : hipErrorUnknown) ||
           launches!=(failure==5u ? 5u : failure+1u) || observed.next!=launches) return 95;
    }
    auto transposed_pv=[&](const uint16_t* values,unsigned stride,unsigned layout=22u) {
        return launch_queries(&operand,&operand,&operand,&output,nullptr,1u,16u,0u,
            nullptr,nullptr,nullptr,true,rcp,layout,&scratch,observed_span,nullptr,nullptr,&operand,17u,
            false,nullptr,nullptr,0u,nullptr,nullptr,values,stride);
    };
    launches=error_queries=memsets=fail_launch=0u;
    if(transposed_pv(nullptr,17u)!=hipErrorInvalidValue || transposed_pv(&operand,16u)!=hipErrorInvalidValue ||
       transposed_pv(&operand,kSplitMaxTokens+1u)!=hipErrorInvalidValue ||
       transposed_pv(&operand,17u,23u)!=hipErrorInvalidValue || launches || memsets) return 96;
    if(transposed_pv(&operand,19u)!=hipSuccess || launches!=5u ||
       !std::strstr(launch_names[4],"blackwell_compacted_pv_replay_kernel<true>")) return 97;
    for(unsigned failure=1u;failure<=5u;++failure) {
        launches=error_queries=memsets=0u;fail_launch=failure;
        if(transposed_pv(&operand,17u)!=hipErrorUnknown || launches!=failure || error_queries!=failure) return 98;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-attention-guard-") as tmp:
            executable = str(Path(tmp) / "launch-guard")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                 "-Werror", "-x", "c++", "-", "-o", executable],
                input=source, text=True, check=True, timeout=30,
            )
            subprocess.run([executable], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
