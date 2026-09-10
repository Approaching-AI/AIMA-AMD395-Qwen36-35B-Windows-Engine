#ifndef QRT_SM121_SILU_RUNTIME_H
#define QRT_SM121_SILU_RUNTIME_H
#include "gdn/sm121_silu_table.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

namespace qrt_sm121_silu_runtime {
struct Store {
    std::mutex mutex;
    std::string path;
    unsigned char *device = nullptr;
};

// This immutable, model-independent table is shared for the provider process.
// Loading is opt-in and never changes an established arithmetic route silently.
inline hipError_t prepare(const char *path, const unsigned char **output) {
    if (!path || !*path || !output) return hipErrorInvalidValue;
    *output = nullptr;
#if defined(_WIN32)
    static Store store;
    std::lock_guard<std::mutex> lock(store.mutex);
    if (store.device) {
        if (store.path != path) return hipErrorInvalidValue;
        *output = store.device;
        return hipSuccess;
    }
    try {
        std::string requested_path(path);
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() != static_cast<std::streamoff>(qrt_sm121_silu::table_bytes)) return hipErrorInvalidValue;
        std::vector<unsigned char> data(qrt_sm121_silu::table_bytes);
        file.seekg(0); file.read(reinterpret_cast<char *>(data.data()), data.size());
        if (!file || !qrt_sm121_silu::valid_layout(data.data(), data.size())) return hipErrorInvalidValue;
        BCRYPT_ALG_HANDLE algorithm = nullptr;
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) return hipErrorInvalidValue;
        unsigned char digest[32]{};
        const NTSTATUS hashed = BCryptHash(algorithm, nullptr, 0, data.data(), static_cast<ULONG>(data.size()), digest, sizeof(digest));
        BCryptCloseAlgorithmProvider(algorithm, 0);
        if (hashed < 0 || std::memcmp(digest, qrt_sm121_silu::sha256, sizeof(digest))) return hipErrorInvalidValue;
        size_t available = 0, total = 0;
        hipError_t status = hipMemGetInfo(&available, &total);
        if (status != hipSuccess) return status;
        if (available < data.size() + (512u << 20)) return hipErrorOutOfMemory;
        unsigned char *device = nullptr;
        status = hipMalloc(reinterpret_cast<void **>(&device), data.size());
        if (status == hipSuccess) status = hipMemcpy(device, data.data(), data.size(), hipMemcpyHostToDevice);
        if (status != hipSuccess) { if (device) (void)hipFree(device); return status; }
        store.path = std::move(requested_path);
        store.device = device;
        *output = device;
        std::fprintf(stderr, "SM121_SILU_TABLE bytes=648036 sha256=673f8dd1280700578c1e8743afd2e3b4da134b1fbd463c890527e1c4d9f796b8 scope=process model_independent=1\n");
        return hipSuccess;
    } catch (...) { return hipErrorOutOfMemory; }
#else
    return hipErrorInvalidValue;
#endif
}
} // namespace qrt_sm121_silu_runtime
#endif
