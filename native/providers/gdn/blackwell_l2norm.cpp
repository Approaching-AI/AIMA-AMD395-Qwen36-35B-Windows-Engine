#include "blackwell_l2norm.h"
#include "blackwell_accumulator.h"
#include "sm121_rsqrt_table.h"
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

namespace qrt_fla_blackwell_norm {
namespace {
unsigned char* rsqrt_table = nullptr;
__global__ void norm_kernel(const float* raw, uint16_t* q, uint16_t* k, unsigned tokens,
                            const unsigned char* table) {
    const unsigned row = blockIdx.x * 16u + threadIdx.x / 16u, lane = threadIdx.x % 16u;
    if (row >= tokens * 16u) return;
    const unsigned token = row / 16u, head = row % 16u;
    float qv[8], kv[8];
    #pragma unroll
    for (unsigned i = 0; i < 8u; ++i) {
        const unsigned offset = token * 8192u + head * 128u + lane * 8u + i;
        qv[i] = raw[offset]; kv[i] = raw[offset + 2048u];
    }
    // SM121 uses eight contiguous dimensions per lane and a 16-lane XOR
    // reduction. Other algebraically equivalent sum trees move BF16 ties.
    float qs = fmaf(qv[0], qv[0], qv[1] * qv[1]);
    float ks = fmaf(kv[0], kv[0], kv[1] * kv[1]);
    #pragma unroll
    for (unsigned i = 2; i < 8u; ++i) {
        qs = fmaf(qv[i], qv[i], qs); ks = fmaf(kv[i], kv[i], ks);
    }
    #pragma unroll
    for (unsigned delta = 8u; delta; delta >>= 1u) {
        qs += __shfl_xor(qs, delta, 16); ks += __shfl_xor(ks, delta, 16);
    }
    const float qr = qrt_sm121_rsqrt::evaluate(table, qs + 1.0e-6f);
    const float kr = qrt_sm121_rsqrt::evaluate(table, ks + 1.0e-6f);
    #pragma unroll
    for (unsigned i = 0; i < 8u; ++i) {
        const unsigned offset = row * 128u + lane * 8u + i;
        q[offset] = qrt_fla_blackwell::to_bf16(qv[i] * qr);
        k[offset] = qrt_fla_blackwell::to_bf16(kv[i] * kr);
    }
}
}

hipError_t prepare_table() {
    if (rsqrt_table) return hipSuccess;
    const char* path = std::getenv("QRT_FLA_GDN_SM121_RSQRT_TABLE");
    if (!path || !*path) return hipErrorInvalidValue;
#if defined(_WIN32)
    try {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() != static_cast<std::streamoff>(qrt_sm121_rsqrt::table_bytes)) return hipErrorInvalidValue;
        std::vector<unsigned char> data(qrt_sm121_rsqrt::table_bytes);
        file.seekg(0); file.read(reinterpret_cast<char*>(data.data()), data.size());
        if (!file || !qrt_sm121_rsqrt::valid_layout(data.data(), data.size())) return hipErrorInvalidValue;
        BCRYPT_ALG_HANDLE algorithm = nullptr;
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) return hipErrorInvalidValue;
        unsigned char digest[32]{};
        const NTSTATUS hash_status = BCryptHash(algorithm, nullptr, 0, data.data(), static_cast<ULONG>(data.size()), digest, sizeof(digest));
        BCryptCloseAlgorithmProvider(algorithm, 0);
        if (hash_status < 0 || std::memcmp(digest, qrt_sm121_rsqrt::sha256, sizeof(digest))) return hipErrorInvalidValue;
        size_t available = 0, total = 0;
        hipError_t status = hipMemGetInfo(&available, &total);
        if (status != hipSuccess) return status;
        if (available < data.size() + (512u << 20)) return hipErrorOutOfMemory;
        status = hipMalloc(reinterpret_cast<void**>(&rsqrt_table), data.size());
        if (status == hipSuccess) status = hipMemcpy(rsqrt_table, data.data(), data.size(), hipMemcpyHostToDevice);
        if (status != hipSuccess) { release_table(); return status; }
        std::fprintf(stderr, "SM121_RSQRT_TABLE bytes=%llu sha256=ca0230a8bae9bd101ac368f8a7c34007cda637df6513dbe4714253c36b940850 model_independent=1\n",
                     static_cast<unsigned long long>(data.size()));
        return hipSuccess;
    } catch (...) { return hipErrorOutOfMemory; }
#else
    return hipErrorInvalidValue;
#endif
}
void release_table() { if (rsqrt_table) { (void)hipFree(rsqrt_table); rsqrt_table = nullptr; } }
uint64_t table_storage_bytes() { return rsqrt_table ? qrt_sm121_rsqrt::table_bytes : 0u; }
const unsigned char* table_device() { return rsqrt_table; }
hipError_t normalize(const float* raw, uint16_t* q, uint16_t* k, unsigned tokens, hipStream_t stream) {
    if (!valid_normalize(raw, q, k, tokens) || !rsqrt_table) return hipErrorInvalidValue;
    hipLaunchKernelGGL(norm_kernel, dim3(tokens), dim3(256u), 0, stream, raw, q, k, tokens, rsqrt_table);
    return hipGetLastError();
}
}
