#ifndef QRT_SM121_Q1_PACKED_RUNTIME_H
#define QRT_SM121_Q1_PACKED_RUNTIME_H
#include "sm121_q1_attention_runtime.h"
#include "native_mtp_target_stream.h"

namespace qrt_sm121_q1_packed_runtime {
struct Tables {
    const unsigned char* sqrt = nullptr;
    const unsigned char* reciprocal = nullptr;
};

// The reference target allows extended context, while its MTP drafter retains
// the model's 262144-token limit. Requests starting at that limit or beyond
// use the original non-speculative packed recurrence for their continuation.
// A request that began earlier switches when its actual MTP checkpoint retires;
// the scheduled two-row extent may retire it after accepting only the first row.
constexpr size_t reference_draft_context_limit = 262144u;
inline bool applies_to_prefix(size_t prefix_tokens) {
    return prefix_tokens >= reference_draft_context_limit ||
        Qwen36NativeTargetOnly::single_row_reference();
}

inline hipError_t prepare(Tables* output) {
    if (!output) return hipErrorInvalidValue;
    *output = Tables{};
#if defined(_WIN32)
    static std::mutex mutex;
    static unsigned char* resident = nullptr;
    static std::string resident_path;
    const char* path = std::getenv("QRT_QWEN36_Q1_SM121_SQRT_TABLE");
    if (!path || !*path) return hipErrorInvalidValue;
    const unsigned char* reciprocal = nullptr;
    hipError_t status = qrt_sm121_q1_attention_runtime::prepare(&reciprocal);
    if (status != hipSuccess) return status;
    std::lock_guard<std::mutex> lock(mutex);
    if (resident) {
        if (resident_path != path) return hipErrorInvalidValue;
        *output = {resident, reciprocal}; return hipSuccess;
    }
    unsigned char* device = nullptr;
    try {
        constexpr size_t bytes = qrt_sm121_sqrt::table_bytes;
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() != static_cast<std::streamoff>(bytes)) return hipErrorInvalidValue;
        std::vector<unsigned char> data(bytes); file.seekg(0);
        file.read(reinterpret_cast<char*>(data.data()), bytes);
        if (!file || !qrt_sm121_sqrt::valid_layout(data.data(), bytes)) return hipErrorInvalidValue;
        BCRYPT_ALG_HANDLE algorithm = nullptr;
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) return hipErrorInvalidValue;
        unsigned char digest[32]{};
        const NTSTATUS hashed = BCryptHash(algorithm, nullptr, 0, data.data(), static_cast<ULONG>(bytes), digest, 32);
        BCryptCloseAlgorithmProvider(algorithm, 0);
        if (hashed < 0 || std::memcmp(digest, qrt_sm121_sqrt::sha256, 32)) return hipErrorInvalidValue;
        status = hipMalloc(reinterpret_cast<void**>(&device), bytes);
        if (status == hipSuccess) status = hipMemcpy(device, data.data(), bytes, hipMemcpyHostToDevice);
        if (status != hipSuccess) { if (device) (void)hipFree(device); return status; }
        resident_path = path; resident = device; *output = {resident, reciprocal};
        std::fprintf(stderr, "SM121_Q1_PACKED_TABLE sqrt_bytes=%zu model_independent=1\n", bytes);
        return hipSuccess;
    } catch (...) { if (device) (void)hipFree(device); return hipErrorOutOfMemory; }
#else
    return hipErrorInvalidValue;
#endif
}
} // namespace qrt_sm121_q1_packed_runtime
#endif
