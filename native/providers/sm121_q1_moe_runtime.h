#ifndef QRT_SM121_Q1_MOE_RUNTIME_H
#define QRT_SM121_Q1_MOE_RUNTIME_H
#include "sm121_q1_runtime.h"

namespace qrt_sm121_q1_moe_runtime {
struct Tables {
    qrt_sm121_q1_runtime::Tables core;
    const uint32_t *router = nullptr;
    const uint16_t *silu = nullptr;
};
inline hipError_t prepare(Tables *output) {
    if (!output) return hipErrorInvalidValue;
    *output = Tables{};
#if defined(_WIN32)
    static std::mutex mutex;
    static Tables resident;
    static std::array<std::string, 2> resident_paths;
    const char *names[] = {"QRT_QWEN36_Q1_SM121_ROUTER_TABLE", "QRT_QWEN36_Q1_SM121_MOE_SILU_TABLE"};
    std::array<std::string, 2> paths;
    for (unsigned i = 0; i < 2; ++i) {
        const char *p = std::getenv(names[i]); if (!p || !*p) return hipErrorInvalidValue;
        paths[i] = p;
    }
    std::lock_guard<std::mutex> lock(mutex);
    qrt_sm121_q1_runtime::Tables core;
    hipError_t status = qrt_sm121_q1_runtime::prepare(&core);
    if (status != hipSuccess) return status;
    if (resident.router) {
        if (paths != resident_paths) return hipErrorInvalidValue;
        *output = resident; return hipSuccess;
    }
    constexpr size_t sizes[] = {33554432u, 131096u};
    const unsigned char hashes[2][32] = {
        {0xb2,0xa4,0x2c,0x4a,0x62,0x64,0x69,0xc9,0x86,0xe3,0x3e,0x43,0xf1,0x6f,0x41,0xbd,0xe9,0xd8,0x4d,0xe9,0x43,0x47,0xcd,0x9c,0xea,0xa1,0xbd,0x68,0xd3,0x6b,0xcd,0xf0},
        {0x97,0xa2,0xa7,0x29,0x26,0x6b,0xb0,0x68,0x19,0x83,0xaa,0x5b,0x2c,0x6d,0xdf,0xfa,0xfe,0xaa,0xb1,0xba,0xb9,0x9c,0x76,0x58,0xa0,0x52,0x4b,0x09,0x33,0x1a,0x11,0xac}};
    unsigned char *device[2]{};
    auto cleanup = [&]() { for (auto p : device) if (p) (void)hipFree(p); };
    try {
        size_t available = 0, total = 0;
        status = hipMemGetInfo(&available, &total);
        if (status != hipSuccess) return status;
        if (available < sizes[0] + sizes[1] + (128u << 20)) return hipErrorOutOfMemory;
        for (unsigned i = 0; i < 2; ++i) {
            std::ifstream file(paths[i], std::ios::binary | std::ios::ate);
            if (!file || file.tellg() != std::streamoff(sizes[i])) { cleanup(); return hipErrorInvalidValue; }
            std::vector<unsigned char> data(sizes[i]); file.seekg(0);
            file.read(reinterpret_cast<char *>(data.data()), data.size());
            if (!file) { cleanup(); return hipErrorInvalidValue; }
            BCRYPT_ALG_HANDLE algorithm = nullptr;
            if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) { cleanup(); return hipErrorInvalidValue; }
            unsigned char digest[32]{};
            const NTSTATUS hashed = BCryptHash(algorithm, nullptr, 0, data.data(), ULONG(data.size()), digest, 32);
            BCryptCloseAlgorithmProvider(algorithm, 0);
            if (hashed < 0 || std::memcmp(digest, hashes[i], 32)) { cleanup(); return hipErrorInvalidValue; }
            status = hipMalloc(reinterpret_cast<void **>(&device[i]), sizes[i]);
            if (status == hipSuccess) status = hipMemcpy(device[i], data.data(), sizes[i], hipMemcpyHostToDevice);
            if (status != hipSuccess) { cleanup(); return status; }
        }
        resident_paths = paths;
        resident = {core, reinterpret_cast<const uint32_t *>(device[0]), reinterpret_cast<const uint16_t *>(device[1] + 24u)};
        *output = resident;
        std::fprintf(stderr, "SM121_Q1_MOE_TABLES router_bytes=%zu silu_bytes=%zu model_independent=1\n", sizes[0], sizes[1]);
        return hipSuccess;
    } catch (...) { cleanup(); return hipErrorOutOfMemory; }
#else
    return hipErrorInvalidValue;
#endif
}
}  // namespace qrt_sm121_q1_moe_runtime
#endif
