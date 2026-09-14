#ifndef QRT_SM121_Q1_FULL_RUNTIME_H
#define QRT_SM121_Q1_FULL_RUNTIME_H
#include "sm121_q1_runtime.h"
#include "sm121_rope_cache.h"
#include "gdn/sm121_q1_full.h"

namespace qrt_sm121_q1_full_runtime {
struct Tables {
    qrt_sm121_q1_runtime::Tables core;
    const uint16_t *rope = nullptr;
    size_t rope_rows = 0u;
};
inline hipError_t prepare(Tables *output, size_t position) {
    if (!output) return hipErrorInvalidValue;
    *output = Tables{};
#if defined(_WIN32)
    static std::mutex mutex;
    static Tables resident;
    static std::string resident_path;
    const char *path = std::getenv("QRT_QWEN36_Q1_SM121_ROPE_TABLE");
    if (!path || !*path) return hipErrorInvalidValue;
    std::lock_guard<std::mutex> lock(mutex);
    qrt_sm121_q1_runtime::Tables core;
    hipError_t status = qrt_sm121_q1_runtime::prepare(&core);
    if (status != hipSuccess) return status;
    if (resident.rope) {
        if (resident_path != path ||
            !qrt_sm121_rope_cache::position_supported(resident.rope_rows, position))
            return hipErrorInvalidValue;
        *output = resident; return hipSuccess;
    }
    uint16_t *device = nullptr;
    try {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0) return hipErrorInvalidValue;
        const auto *layout = qrt_sm121_rope_cache::layout_for_bytes(
            static_cast<size_t>(file.tellg()));
        if (!layout || !qrt_sm121_rope_cache::position_supported(layout->rows, position))
            return hipErrorInvalidValue;
        const size_t bytes = layout->bytes;
        size_t available = 0u, total = 0u;
        status = hipMemGetInfo(&available, &total);
        if (status != hipSuccess) return status;
        if (available < bytes + (128u << 20u)) return hipErrorOutOfMemory;
        std::vector<unsigned char> data(bytes); file.seekg(0);
        file.read(reinterpret_cast<char *>(data.data()), bytes);
        if (!file) return hipErrorInvalidValue;
        BCRYPT_ALG_HANDLE algorithm = nullptr;
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
            return hipErrorInvalidValue;
        unsigned char digest[32]{};
        const NTSTATUS hashed = BCryptHash(algorithm, nullptr, 0, data.data(),
                                           static_cast<ULONG>(bytes), digest, 32);
        BCryptCloseAlgorithmProvider(algorithm, 0);
        if (hashed < 0 || std::memcmp(digest, layout->sha256, 32)) return hipErrorInvalidValue;
        status = hipMalloc(reinterpret_cast<void **>(&device), bytes);
        if (status == hipSuccess) status = hipMemcpy(device, data.data(), bytes, hipMemcpyHostToDevice);
        if (status != hipSuccess) { if (device) (void)hipFree(device); return status; }
        resident_path = path; resident = {core, device, layout->rows}; *output = resident;
        std::fprintf(stderr, "SM121_Q1_FULL_TABLE rope_bytes=%zu rope_rows=%zu model_parameter_only=1\n",
                     bytes, layout->rows);
        return hipSuccess;
    } catch (...) { if (device) (void)hipFree(device); return hipErrorOutOfMemory; }
#else
    (void)position;
    return hipErrorInvalidValue;
#endif
}
}  // namespace qrt_sm121_q1_full_runtime
#endif
