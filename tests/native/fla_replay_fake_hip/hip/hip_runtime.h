// CPU-only launch-marshalling test double. Not a numerical GPU implementation.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
using hipError_t = int;
constexpr int hipSuccess = 0, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2;
using hipEvent_t = void*;
using hipStream_t = void*;
using hipModule_t = void*;
using hipFunction_t = const char*;
struct hipDeviceProp_t { char gcnArchName[32] = "gfx1151"; };
inline std::vector<std::pair<void*, size_t>> fake_allocations;
inline bool fake_range(void* p, size_t n) {
    for (const auto& item : fake_allocations) {
        const auto start = reinterpret_cast<uintptr_t>(item.first), address = reinterpret_cast<uintptr_t>(p);
        if (address >= start && address - start <= item.second && n <= item.second - (address - start)) return true;
    }
    return false;
}
inline const char* hipGetErrorString(int) { return "fake HIP range or ABI failure"; }
inline int hipSetDevice(int) { std::cerr << "FAKE_HIP preflight\n"; return 0; }
inline int hipGetDeviceProperties(hipDeviceProp_t*, int) { return 0; }
inline int hipMemGetInfo(size_t* free, size_t* total) { *free = *total = size_t(2) << 30; return 0; }
inline int hipMalloc(void** p, size_t bytes) {
    *p = std::calloc(1, bytes); if (!*p) return 1;
    fake_allocations.emplace_back(*p, bytes); return 0;
}
inline int hipFree(void* p) {
    fake_allocations.erase(std::remove_if(fake_allocations.begin(), fake_allocations.end(),
        [p](const auto& entry) { return entry.first == p; }), fake_allocations.end());
    std::free(p); return 0;
}
inline int hipMemcpy(void* to, const void* from, size_t bytes, int kind) {
    if (!fake_range(kind == hipMemcpyHostToDevice ? to : const_cast<void*>(from), bytes)) return 1;
    std::memcpy(to, from, bytes); return 0;
}
inline int hipMemset(void* p, int x, size_t bytes) { if (!fake_range(p, bytes)) return 1; std::memset(p, x, bytes); return 0; }
inline int hipStreamCreate(hipStream_t* p) { *p = reinterpret_cast<void*>(1); return 0; }
inline int hipStreamDestroy(hipStream_t) { return 0; }
inline int hipEventCreate(hipEvent_t* p) { *p = reinterpret_cast<void*>(1); return 0; }
inline int hipEventDestroy(hipEvent_t) { return 0; }
inline int hipEventRecord(hipEvent_t, hipStream_t) { return 0; }
inline int hipEventSynchronize(hipEvent_t) { return 0; }
inline int hipEventElapsedTime(float* ms, hipEvent_t, hipEvent_t) {
    const char* value = std::getenv("QRT_TEST_FAKE_DISPATCH_MS"); *ms = value ? std::strtof(value, nullptr) : 1.0f; return 0;
}
inline int hipModuleLoad(hipModule_t* p, const char*) { *p = reinterpret_cast<void*>(1); return 0; }
inline int hipModuleUnload(hipModule_t) { return 0; }
inline int hipModuleGetFunction(hipFunction_t* p, hipModule_t, const char* name) { *p = name; return 0; }
inline int hipModuleLaunchKernel(hipFunction_t function, unsigned x, unsigned y, unsigned z,
                                unsigned, unsigned, unsigned, unsigned, hipStream_t, void** args, void*) {
    const std::string name(function);
    const bool solve = name == "_fla_solve_tril_64_kernel", wu = name == "_fla_recompute_w_u_kernel";
    if (!solve && !wu && name != "_fla_chunk_state_kernel") return 1;
    const size_t pointers = solve ? 2 : wu ? 7 : 8, source = pointers + 1;
    const int32_t tokens = *static_cast<int32_t*>(args[pointers]);
    if (tokens <= 0 || tokens > 1024 || tokens % 64 || y != 32 || z != 1 || x != (solve || wu ? unsigned(tokens / 64) : 8)) return 1;
    if (*static_cast<void**>(args[source]) || *static_cast<void**>(args[source + 1])) return 1;
    const size_t t = static_cast<size_t>(tokens), state = 32u * 128u * 128u;
    const std::vector<size_t> bytes = solve ? std::vector<size_t>{t * 2048 * 4, t * 2048 * 2} :
        wu ? std::vector<size_t>{t * 2048 * 2, t * 4096 * 2, t * 32 * 2, t * 4096 * 2,
                                t * 4096 * 2, t * 2048 * 2, t * 32 * 4} :
             std::vector<size_t>{t * 2048 * 2, t * 4096 * 2, t * 4096 * 2, t * 4096 * 2,
                                t * 32 * 4, state * 4, t / 64 * state * 2, state * 4};
    for (size_t i = 0; i < pointers; ++i) if (!fake_range(*static_cast<void**>(args[i]), bytes[i])) return 1;
    if (!solve && !wu && *static_cast<void**>(args[5]) == *static_cast<void**>(args[7])) return 1;
    std::cerr << "FAKE_HIP launch slots=" << source + 2 << " tokens=" << tokens << '\n';
    return 0;
}
