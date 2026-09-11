#include "blackwell_state.h"
#include "blackwell_accumulator.h"
#include "blackwell_cooperative.h"
#include "sm121_exp2_table.h"
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

namespace qrt_fla_blackwell_state {
namespace {
using namespace qrt_fla_blackwell;
unsigned char* exp2_table = nullptr;

__device__ __forceinline__ float gate_exp(float value, const unsigned char* table) {
    const float scaled = value * 1.4426950408889634074f;
    if (table) return qrt_sm121_exp2::evaluate(table, scaled);
    float result;
    asm("v_exp_f32 %0, %1" : "=v"(result) : "v"(scaled));
    return result;
}
__device__ __forceinline__ float finish(qrt_q1_moe_hawkeye::Value accumulator) {
    accumulator = qrt_q1_moe_hawkeye::group_sum<26, kZeroExponent>(&accumulator, 1);
    return qrt_q1_moe_hawkeye::value_to_float(accumulator);
}

// A subgroup owns one (token,head,value) projection. Only token zero writes
// the BF16 chunk checkpoint; other tokens independently round the F32 seed.
// No subgroup consumes another subgroup's checkpoint output.
__global__ void project_kernel(const uint16_t* w, const uint16_t* u, const float* g,
                               const float* initial, uint16_t* h, uint16_t* v_new,
                               uint16_t* residual, unsigned count, const unsigned char* table) {
    const unsigned cell = blockIdx.x * (kThreads / kGroup) + threadIdx.x / kGroup;
    const unsigned token = cell / 128u, v = cell % 128u, head = blockIdx.y, lane = threadIdx.x % kGroup;
    if (token >= 64u || head >= 32u) return;
    const unsigned output = (token * 32u + head) * 128u + v;
    if (token >= count) { if (lane == 0) residual[output] = 0; return; }
    qrt_q1_moe_hawkeye::Value accumulator{0u, kZeroExponent, false};
    for (unsigned base = 0; base < 128u; base += kGroup) {
        const unsigned key = base + lane, state_index = (head * 128u + v) * 128u + key;
        const uint16_t state = to_bf16(initial[state_index]);
        if (token == 0) h[state_index] = state;
        accumulator = accumulate(accumulator, w[(token * 32u + head) * 128u + key], state, lane);
    }
    if (lane == 0) {
        const float current = from_bf16(u[output]) - finish(accumulator);
        v_new[output] = to_bf16(current);
        const float gate = gate_exp(g[(count - 1u) * 32u + head] - g[token * 32u + head], table);
        residual[output] = to_bf16(current * gate);
    }
}

// A subgroup owns one final (head,value,key) state cell. The K64 dot starts
// at zero, then an explicit IEEE FMA applies decay to the untouched F32 seed.
__global__ void update_kernel(const uint16_t* k, const uint16_t* residual, const float* g,
                              const float* initial, float* final, unsigned count, const unsigned char* table) {
    const unsigned cell = blockIdx.x * (kThreads / kGroup) + threadIdx.x / kGroup;
    const unsigned v = cell / 128u, key = cell % 128u, head = blockIdx.y, lane = threadIdx.x % kGroup;
    if (v >= 128u || head >= 32u) return;
    qrt_q1_moe_hawkeye::Value accumulator{0u, kZeroExponent, false};
    for (unsigned base = 0; base < 64u; base += kGroup) {
        const unsigned token = base + lane;
        const uint16_t left = token < count ? k[(token * 16u + head / 2u) * 128u + key] : 0;
        const uint16_t right = token < count ? residual[(token * 32u + head) * 128u + v] : 0;
        accumulator = accumulate(accumulator, left, right, lane);
    }
    if (lane == 0) {
        const unsigned index = (head * 128u + v) * 128u + key;
        final[index] = fmaf(initial[index], gate_exp(g[(count - 1u) * 32u + head], table), finish(accumulator));
    }
}

// Preserve each original K128 projection, K64 update and final IEEE FMA,
// while carrying a disjoint state tile through up to sixteen chunks. No CTA
// reads a state cell owned by another CTA and no cross-CTA barrier is needed.
__global__ void segment_kernel(const uint16_t* k, const uint16_t* u,
                               const uint16_t* w, const float* g, uint16_t* h,
                               uint16_t* v_new, float* state, unsigned count,
                               const unsigned char* table) {
    constexpr unsigned columns = 4u;
    __shared__ float current_state[columns * 128u];
    __shared__ uint16_t rounded_state[columns * 128u];
    __shared__ uint16_t residual[64u * columns];
    const unsigned head = blockIdx.y, column_base = blockIdx.x * columns;
    const unsigned lane = threadIdx.x % kGroup, group = threadIdx.x / kGroup;
    for (unsigned cell = threadIdx.x; cell < columns * 128u; cell += kThreads) {
        const unsigned index = (head * 128u + column_base + cell / 128u) * 128u + cell % 128u;
        current_state[cell] = state[index];
    }
    __syncthreads();
    for (unsigned offset = 0u; offset < count; offset += 64u) {
        const unsigned valid = count - offset < 64u ? count - offset : 64u;
        const float* gate = g + size_t(offset) * 32u;
        for (unsigned cell = threadIdx.x; cell < columns * 128u; cell += kThreads) {
            const uint16_t rounded = to_bf16(current_state[cell]);
            rounded_state[cell] = rounded;
            const size_t checkpoint = size_t(offset / 64u) * 524288u +
                (head * 128u + column_base + cell / 128u) * 128u + cell % 128u;
            h[checkpoint] = rounded;
        }
        __syncthreads();
        for (unsigned cell = group; cell < 64u * columns; cell += kThreads / kGroup) {
            const unsigned token = cell / columns, column = cell % columns;
            if (token >= valid) { if (lane == 0u) residual[cell] = 0u; continue; }
            qrt_q1_moe_hawkeye::Value sum{0u, kZeroExponent, false};
            for (unsigned base = 0u; base < 128u; base += kGroup) {
                const unsigned key = base + lane;
                sum = accumulate(sum, w[(size_t(offset + token) * 32u + head) * 128u + key],
                                 rounded_state[column * 128u + key], lane);
            }
            if (lane == 0u) {
                const size_t index = (size_t(offset + token) * 32u + head) * 128u + column_base + column;
                const float value = from_bf16(u[index]) - finish(sum);
                v_new[index] = to_bf16(value);
                residual[cell] = to_bf16(value * gate_exp(
                    gate[(valid - 1u) * 32u + head] - gate[token * 32u + head], table));
            }
        }
        __syncthreads();
        for (unsigned cell = group; cell < columns * 128u; cell += kThreads / kGroup) {
            const unsigned column = cell / 128u, key = cell % 128u;
            qrt_q1_moe_hawkeye::Value sum{0u, kZeroExponent, false};
            for (unsigned base = 0u; base < 64u; base += kGroup) {
                const unsigned token = base + lane;
                const uint16_t left = token < valid
                    ? k[(size_t(offset + token) * 16u + head / 2u) * 128u + key] : 0u;
                const uint16_t right = token < valid ? residual[token * columns + column] : 0u;
                sum = accumulate(sum, left, right, lane);
            }
            if (lane == 0u) current_state[cell] = fmaf(current_state[cell],
                gate_exp(gate[(valid - 1u) * 32u + head], table), finish(sum));
        }
        __syncthreads();
    }
    for (unsigned cell = threadIdx.x; cell < columns * 128u; cell += kThreads) {
        const unsigned index = (head * 128u + column_base + cell / 128u) * 128u + cell % 128u;
        state[index] = current_state[cell];
    }
}
}

hipError_t prepare_exp2_table() {
    const char* path = std::getenv("QRT_FLA_GDN_SM121_EXP2_TABLE");
    if (!path || !*path || exp2_table) return hipSuccess;
#if defined(_WIN32)
    try {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() != static_cast<std::streamoff>(qrt_sm121_exp2::table_bytes)) return hipErrorInvalidValue;
        std::vector<unsigned char> data(qrt_sm121_exp2::table_bytes);
        file.seekg(0); file.read(reinterpret_cast<char*>(data.data()), data.size());
        if (!file || !qrt_sm121_exp2::valid_layout(data.data(), data.size())) return hipErrorInvalidValue;
        BCRYPT_ALG_HANDLE algorithm = nullptr;
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) return hipErrorInvalidValue;
        unsigned char digest[32]{};
        const NTSTATUS hash_status = BCryptHash(algorithm, nullptr, 0, data.data(), static_cast<ULONG>(data.size()), digest, sizeof(digest));
        BCryptCloseAlgorithmProvider(algorithm, 0);
        if (hash_status < 0 || std::memcmp(digest, qrt_sm121_exp2::sha256, sizeof(digest))) return hipErrorInvalidValue;
        size_t available = 0, total = 0;
        hipError_t status = hipMemGetInfo(&available, &total);
        if (status != hipSuccess) return status;
        if (available < data.size() + (512u << 20)) return hipErrorOutOfMemory;
        status = hipMalloc(reinterpret_cast<void**>(&exp2_table), data.size());
        if (status == hipSuccess) status = hipMemcpy(exp2_table, data.data(), data.size(), hipMemcpyHostToDevice);
        if (status != hipSuccess) { release_exp2_table(); return status; }
        std::fprintf(stderr, "SM121_EXP2_TABLE bytes=%llu sha256=f490940df2bd80421159a96424c3e922330b7ca120d5ae7b629a973b9183730b model_independent=1\n",
                     static_cast<unsigned long long>(data.size()));
        return hipSuccess;
    } catch (...) { return hipErrorOutOfMemory; }
#else
    return hipErrorInvalidValue;  // This runtime loader is native Windows only.
#endif
}
void release_exp2_table() { if (exp2_table) { (void)hipFree(exp2_table); exp2_table = nullptr; } }
uint64_t exp2_table_storage_bytes() { return exp2_table ? qrt_sm121_exp2::table_bytes : 0u; }
const unsigned char* exp2_table_device() { return exp2_table; }

hipError_t project(const uint16_t* w, const uint16_t* u, const float* g,
                   const float* initial, uint16_t* h, uint16_t* v_new,
                   uint16_t* residual, unsigned count, hipStream_t stream) {
    if (!valid_project(w, u, g, initial, h, v_new, residual, count)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(project_kernel, dim3(512u, 32u), dim3(256u), 0, stream,
                       w, u, g, initial, h, v_new, residual, count, exp2_table);
    return hipGetLastError();
}
hipError_t update(const uint16_t* k, const uint16_t* residual, const float* g,
                  const float* initial, float* final, unsigned count, hipStream_t stream) {
    if (!valid_update(k, residual, g, initial, final, count)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(update_kernel, dim3(1024u, 32u), dim3(256u), 0, stream,
                       k, residual, g, initial, final, count, exp2_table);
    return hipGetLastError();
}
hipError_t segment(const uint16_t* k, const uint16_t* u, const uint16_t* w,
                   const float* g, uint16_t* h, uint16_t* v_new, float* state,
                   unsigned count, hipStream_t stream) {
    if (!k || !u || !w || !g || !h || !v_new || !state ||
        !count || count > 1024u || !exp2_table) return hipErrorInvalidValue;
    const void* inputs[] = {k, u, w, g, state};
    for (const void* input : inputs) if (static_cast<void*>(h) == input || static_cast<void*>(v_new) == input)
        return hipErrorInvalidValue;
    if (h == v_new || static_cast<void*>(state) == k || static_cast<void*>(state) == u ||
        static_cast<void*>(state) == w || state == g) return hipErrorInvalidValue;
    if (qrt_fla_blackwell_cooperative::enabled())
        return qrt_fla_blackwell_cooperative::state(k, u, w, g, h, v_new, state, count, exp2_table, stream);
    hipLaunchKernelGGL(segment_kernel, dim3(32u, 32u), dim3(256u), 0, stream,
                       k, u, w, g, h, v_new, state, count, exp2_table);
    return hipGetLastError();
}
}
