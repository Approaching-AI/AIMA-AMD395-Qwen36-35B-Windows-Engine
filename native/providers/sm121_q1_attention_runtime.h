#ifndef QRT_SM121_Q1_ATTENTION_RUNTIME_H
#define QRT_SM121_Q1_ATTENTION_RUNTIME_H
#include "sm121_q1_runtime.h"
#include "gdn/sm121_q1_attention.h"

namespace qrt_sm121_q1_attention_runtime {
inline hipError_t prepare(const unsigned char **output) {
    if (!output) return hipErrorInvalidValue;
    *output = nullptr;
#if defined(_WIN32)
    static std::mutex mutex;
    static unsigned char *resident = nullptr;
    static std::string resident_path;
    const char *path = std::getenv("QRT_QWEN36_Q1_SM121_RCP_TABLE");
    if (!path || !*path) return hipErrorInvalidValue;
    std::lock_guard<std::mutex> lock(mutex);
    if (resident) {
        if (resident_path != path) return hipErrorInvalidValue;
        *output = resident; return hipSuccess;
    }
    unsigned char *device = nullptr;
    try {
        constexpr size_t bytes = qrt_sm121_attention_rcp::table_bytes;
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() != static_cast<std::streamoff>(bytes)) return hipErrorInvalidValue;
        std::vector<unsigned char> data(bytes); file.seekg(0);
        file.read(reinterpret_cast<char *>(data.data()), bytes);
        if (!file || !qrt_sm121_attention_rcp::valid_layout(data.data(), bytes)) return hipErrorInvalidValue;
        BCRYPT_ALG_HANDLE algorithm = nullptr;
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) return hipErrorInvalidValue;
        unsigned char digest[32]{};
        const NTSTATUS hashed = BCryptHash(algorithm, nullptr, 0, data.data(),
                                           static_cast<ULONG>(bytes), digest, 32);
        BCryptCloseAlgorithmProvider(algorithm, 0);
        if (hashed < 0 || std::memcmp(digest, qrt_sm121_attention_rcp::sha256, 32)) return hipErrorInvalidValue;
        hipError_t status = hipMalloc(reinterpret_cast<void **>(&device), bytes);
        if (status == hipSuccess) status = hipMemcpy(device, data.data(), bytes, hipMemcpyHostToDevice);
        if (status != hipSuccess) { if (device) (void)hipFree(device); return status; }
        resident_path = path; resident = device; *output = resident;
        std::fprintf(stderr, "SM121_Q1_ATTENTION_TABLE rcp_bytes=%zu model_independent=1\n", bytes);
        return hipSuccess;
    } catch (...) { if (device) (void)hipFree(device); return hipErrorOutOfMemory; }
#else
    return hipErrorInvalidValue;
#endif
}
}  // namespace qrt_sm121_q1_attention_runtime
#endif
