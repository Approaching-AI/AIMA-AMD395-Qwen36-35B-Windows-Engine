#ifndef QRT_SM121_Q1_FULL_RUNTIME_H
#define QRT_SM121_Q1_FULL_RUNTIME_H
#include "sm121_q1_runtime.h"
#include "gdn/sm121_q1_full.h"

namespace qrt_sm121_q1_full_runtime {
struct Tables {
    qrt_sm121_q1_runtime::Tables core;
    const uint16_t *rope = nullptr;
};
inline hipError_t prepare(Tables *output) {
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
        if (resident_path != path) return hipErrorInvalidValue;
        *output = resident; return hipSuccess;
    }
    constexpr size_t bytes = 262144u * 64u * 2u;
    const unsigned char expected[32] = {
        0xba,0x12,0xce,0x21,0x83,0x27,0xd4,0xcf,0x23,0xaa,0xc7,0xdf,0xac,0xd8,0xe9,0xef,
        0xbc,0x99,0xfd,0x20,0x76,0x11,0xa8,0x46,0x62,0x27,0x08,0x98,0x38,0xef,0x0e,0x80};
    uint16_t *device = nullptr;
    try {
        size_t available = 0u, total = 0u;
        status = hipMemGetInfo(&available, &total);
        if (status != hipSuccess) return status;
        if (available < bytes + (128u << 20u)) return hipErrorOutOfMemory;
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() != static_cast<std::streamoff>(bytes)) return hipErrorInvalidValue;
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
        if (hashed < 0 || std::memcmp(digest, expected, 32)) return hipErrorInvalidValue;
        status = hipMalloc(reinterpret_cast<void **>(&device), bytes);
        if (status == hipSuccess) status = hipMemcpy(device, data.data(), bytes, hipMemcpyHostToDevice);
        if (status != hipSuccess) { if (device) (void)hipFree(device); return status; }
        resident_path = path; resident = {core, device}; *output = resident;
        std::fprintf(stderr, "SM121_Q1_FULL_TABLE rope_bytes=%zu model_parameter_only=1\n", bytes);
        return hipSuccess;
    } catch (...) { if (device) (void)hipFree(device); return hipErrorOutOfMemory; }
#else
    return hipErrorInvalidValue;
#endif
}
}  // namespace qrt_sm121_q1_full_runtime
#endif
