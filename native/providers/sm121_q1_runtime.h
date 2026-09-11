#ifndef QRT_SM121_Q1_RUNTIME_H
#define QRT_SM121_Q1_RUNTIME_H
#include "sm121_silu_runtime.h"
#include "gdn/sm121_q1_gdn.h"
#include <array>
#include <cstdlib>

namespace qrt_sm121_q1_runtime {
struct Tables {
    const unsigned char *exp2 = nullptr, *rsqrt = nullptr, *silu = nullptr;
    const float *beta = nullptr;
};

inline hipError_t prepare(Tables *output) {
    if (!output) return hipErrorInvalidValue;
    *output = Tables{};
#if defined(_WIN32)
    static std::mutex mutex;
    static Tables resident;
    static std::array<std::string, 5> resident_paths;
    const char *names[] = {"QRT_QWEN36_Q1_SM121_EXP2_TABLE", "QRT_QWEN36_Q1_SM121_RSQRT_TABLE",
                          "QRT_QWEN36_Q1_SM121_BETA_TABLE", "QRT_QWEN36_Q1_SM121_GATE_LUT_DIR",
                          "QRT_QWEN36_Q1_SM121_SILU_TABLE"};
    std::array<std::string, 5> paths;
    for (unsigned int i = 0; i < 5; ++i) {
        const char *path = std::getenv(names[i]);
        if (!path || !*path) return hipErrorInvalidValue;
        paths[i] = path;
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (resident.exp2) {
        if (paths != resident_paths) return hipErrorInvalidValue;
        *output = resident;
        return hipSuccess;
    }
    const unsigned char beta_sha[] = {
        0xca,0xfc,0x4d,0xc1,0x01,0x2d,0x53,0x4a,0x6a,0x99,0x73,0x9f,0x9c,0x4a,0x32,0x7b,
        0xc4,0x05,0x2a,0xd2,0xd3,0x44,0x1a,0x79,0xdf,0xf1,0x8f,0x9a,0x2e,0xa7,0xeb,0xc5};
    const size_t sizes[] = {qrt_sm121_exp2::table_bytes, qrt_sm121_rsqrt::table_bytes, 65536u * 4u};
    const unsigned char *hashes[] = {qrt_sm121_exp2::sha256, qrt_sm121_rsqrt::sha256, beta_sha};
    unsigned char *device[3] = {};
    auto cleanup = [&]() { for (auto p : device) if (p) (void)hipFree(p); };
    try {
        size_t available = 0, total = 0;
        hipError_t status = hipMemGetInfo(&available, &total);
        if (status != hipSuccess) return status;
        if (available < sizes[0] + sizes[1] + sizes[2] + (512u << 20)) return hipErrorOutOfMemory;
        for (unsigned int i = 0; i < 3; ++i) {
            std::ifstream file(paths[i], std::ios::binary | std::ios::ate);
            if (!file || file.tellg() != static_cast<std::streamoff>(sizes[i])) {
                cleanup(); return hipErrorInvalidValue;
            }
            std::vector<unsigned char> data(sizes[i]);
            file.seekg(0); file.read(reinterpret_cast<char *>(data.data()), data.size());
            if (!file || (i == 0 && !qrt_sm121_exp2::valid_layout(data.data(), data.size())) ||
                (i == 1 && !qrt_sm121_rsqrt::valid_layout(data.data(), data.size()))) {
                cleanup(); return hipErrorInvalidValue;
            }
            BCRYPT_ALG_HANDLE algorithm = nullptr;
            if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) {
                cleanup(); return hipErrorInvalidValue;
            }
            unsigned char digest[32]{};
            const NTSTATUS hashed = BCryptHash(algorithm, nullptr, 0, data.data(),
                static_cast<ULONG>(data.size()), digest, sizeof(digest));
            BCryptCloseAlgorithmProvider(algorithm, 0);
            if (hashed < 0 || std::memcmp(digest, hashes[i], sizeof(digest))) {
                cleanup(); return hipErrorInvalidValue;
            }
            status = hipMalloc(reinterpret_cast<void **>(&device[i]), sizes[i]);
            if (status == hipSuccess)
                status = hipMemcpy(device[i], data.data(), sizes[i], hipMemcpyHostToDevice);
            if (status != hipSuccess) { cleanup(); return status; }
        }
        const unsigned char *silu = nullptr;
        status = qrt_sm121_silu_runtime::prepare(std::getenv("QRT_QWEN36_Q1_SM121_SILU_TABLE"), &silu);
        if (status != hipSuccess) { cleanup(); return status; }
        resident_paths = paths;
        resident = {device[0], device[1], silu, reinterpret_cast<const float *>(device[2])};
        *output = resident;
        std::fprintf(stderr, "SM121_Q1_TABLES exp2_bytes=%zu rsqrt_bytes=%zu beta_bytes=%zu model_independent=1\n",
                     sizes[0], sizes[1], sizes[2]);
        return hipSuccess;
    } catch (...) { cleanup(); return hipErrorOutOfMemory; }
#else
    return hipErrorInvalidValue;
#endif
}
} // namespace qrt_sm121_q1_runtime
#endif
