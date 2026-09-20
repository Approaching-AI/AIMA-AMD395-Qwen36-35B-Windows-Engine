#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>
#include "sm121_q1_full_runtime.h"
#include "sm121_q1_moe_runtime.h"
#include "sm121_q1_attention_runtime.h"
#include "gdn/sm121_mtp_drafter.h"

namespace qrt_sm121_mtp_runtime {
// This is the original BF16-domain/BF16-output sigmoid. The target GDN beta
// table contains FP32 values and must not be reinterpreted as this table.
inline constexpr size_t sigmoid_bytes = 65536u * sizeof(uint16_t);
inline constexpr unsigned char sigmoid_sha256[32] = {
    0x32,0x92,0x3b,0x94,0xec,0xa9,0x38,0xcd,0x0f,0x96,0x6f,0x39,0xef,0xb5,0xfc,0xbd,
    0xa4,0x0f,0x2c,0x2b,0xb7,0x48,0xcd,0xd9,0x0b,0xfd,0x3d,0xde,0xff,0x9e,0x8f,0x97};

inline hipError_t prepare_sigmoid(const char* path, const uint16_t** output) {
    if (!output) return hipErrorInvalidValue;
    *output = nullptr;
    if (!path || !*path) return hipErrorInvalidValue;
#if defined(_WIN32)
    struct State {
        const uint16_t* ready = nullptr;
        void* pending_device = nullptr;
        void* pending_host = nullptr;
        hipError_t terminal = hipSuccess;
        std::string path;
    };
    static std::mutex mutex;
    static State resident;
    std::lock_guard<std::mutex> lock(mutex);
    if (resident.terminal != hipSuccess) return resident.terminal;
    if (resident.ready) {
        if (resident.path != path) return hipErrorInvalidValue;
        *output = resident.ready;
        return hipSuccess;
    }
    unsigned char* host = nullptr;
    uint16_t* device = nullptr;
    const auto cleanup = [&]() {
        if (device) (void)hipFree(device);
        if (host) (void)hipHostFree(host);
    };
    try {
        std::string requested_path(path);
        std::ifstream file(requested_path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() != std::streamoff(sigmoid_bytes)) return hipErrorInvalidValue;
        hipError_t status = hipHostMalloc(reinterpret_cast<void**>(&host), sigmoid_bytes);
        if (status != hipSuccess) return status;
        file.seekg(0); file.read(reinterpret_cast<char*>(host), sigmoid_bytes);
        if (!file) { cleanup(); return hipErrorInvalidValue; }
        BCRYPT_ALG_HANDLE algorithm = nullptr;
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) {
            cleanup(); return hipErrorInvalidValue;
        }
        unsigned char digest[32]{};
        const NTSTATUS hashed = BCryptHash(algorithm, nullptr, 0, host,
            static_cast<ULONG>(sigmoid_bytes), digest, sizeof(digest));
        BCryptCloseAlgorithmProvider(algorithm, 0);
        if (hashed < 0 || std::memcmp(digest, sigmoid_sha256, sizeof(digest))) {
            cleanup(); return hipErrorInvalidValue;
        }
        status = hipMalloc(reinterpret_cast<void**>(&device), sigmoid_bytes);
        if (status != hipSuccess) { cleanup(); return status; }
        status = hipMemcpyAsync(device, host, sigmoid_bytes, hipMemcpyHostToDevice, nullptr);
        const hipError_t completed = hipStreamSynchronize(nullptr);
        if (completed != hipSuccess) {
            // These buffers are already allocated. Keep both static roots
            // after an unknown completion, including a failed enqueue.
            resident.pending_device = device; resident.pending_host = host;
            resident.terminal = completed;
            return completed;
        }
        if (status != hipSuccess) { cleanup(); return status; }
        (void)hipHostFree(host); host = nullptr;
        resident.path = std::move(requested_path);
        resident.ready = device;
        *output = resident.ready;
        std::fprintf(stderr, "SM121_MTP_SIGMOID_TABLE bytes=%zu bf16_domain=1 bf16_output=1 model_independent=1\n", sigmoid_bytes);
        return hipSuccess;
    } catch (...) { cleanup(); return hipErrorOutOfMemory; }
#else
    return hipErrorInvalidValue;
#endif
}

inline hipError_t prepare(qrt_sm121_mtp::DrafterTables* output, unsigned last_position) {
    if (!output) return hipErrorInvalidValue;
    *output = {};
    if (last_position >= 262144u) return hipErrorInvalidValue;
    qrt_sm121_q1_full_runtime::Tables full;
    hipError_t status = qrt_sm121_q1_full_runtime::prepare(&full, last_position);
    if (status != hipSuccess) return status;
    qrt_sm121_q1_moe_runtime::Tables moe;
    status = qrt_sm121_q1_moe_runtime::prepare(&moe);
    if (status != hipSuccess) return status;
    const unsigned char* reciprocal = nullptr;
    status = qrt_sm121_q1_attention_runtime::prepare(&reciprocal);
    if (status != hipSuccess) return status;
    if (!full.core.rsqrt || !full.core.exp2 || full.core.rsqrt != moe.core.rsqrt ||
        full.core.exp2 != moe.core.exp2 || !full.rope || full.rope_rows <= last_position ||
        !reciprocal || !moe.router || !moe.silu)
        return hipErrorInvalidValue;
    const uint16_t* sigmoid = nullptr;
    status = prepare_sigmoid(std::getenv("QRT_QWEN36_MTP_BF16_SIGMOID_TABLE"), &sigmoid);
    if (status != hipSuccess) return status;
    // The target owns a verified table that may cover positions beyond the
    // drafter limit. Borrow its supported prefix without expanding MTP's
    // logical context or rejecting the shared long-context allocation.
    const unsigned rope_rows = static_cast<unsigned>(
        full.rope_rows < 262144u ? full.rope_rows : 262144u);
    *output = {full.core.rsqrt, full.rope, rope_rows,
        full.core.exp2, reciprocal, {moe.silu, sigmoid, moe.router}};
    return hipSuccess;
}
} // namespace qrt_sm121_mtp_runtime
