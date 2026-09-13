#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

// Component diagnostic only. All wait modes submit the same kernel and event
// on the default stream; no other stream or model is active in this process.
// Blocking API calls are additionally bounded by the native process guard.
namespace {
using Clock = std::chrono::steady_clock;
constexpr unsigned count = 1024u, guard = 16u, repeats = 32u;
constexpr uint32_t sentinel = 0x5a5a5a5au;
void require(bool ok, const char* stage) { if (!ok) throw std::runtime_error(stage); }
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
__host__ __device__ uint32_t advance(uint32_t value) {
    value ^= value << 13u; value ^= value >> 17u; value ^= value << 5u;
    return value;
}
__global__ void work(uint32_t* output, unsigned loops, unsigned ticket) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t value = 0x3958192u ^ (index * 65537u) ^ ticket;
#pragma unroll 1
    for (unsigned i = 0u; i < loops; ++i) value = advance(value);
    output[index] = value;
}
void complete(unsigned method, hipEvent_t event) {
    if (method == 0u) { check(hipStreamSynchronize(nullptr)); return; }
    if (method == 1u) { check(hipDeviceSynchronize()); return; }
    if (method == 2u) { check(hipEventSynchronize(event)); return; }
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) return;
        require(status == hipErrorNotReady, "event query failed");
        require(Clock::now() < deadline, "event query exceeded five seconds");
        if (method == 3u) std::this_thread::yield();
    }
}
}

int main(int argc, char** argv) try {
    require(argc == 2 && (!std::strcmp(argv[1], "auto") || !std::strcmp(argv[1], "spin")),
            "expected auto or spin device schedule");
    const unsigned requested = !std::strcmp(argv[1], "spin") ? hipDeviceScheduleSpin : hipDeviceScheduleAuto;
    check(hipSetDeviceFlags(requested));
    unsigned actual = 0u; check(hipGetDeviceFlags(&actual));
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    require(!std::strncmp(properties.gcnArchName, "gfx1151", 7u), "requires gfx1151");
    int runtime = 0, driver = 0;
    check(hipRuntimeGetVersion(&runtime)); check(hipDriverGetVersion(&driver));
    std::printf("{\"type\":\"wait_configuration\",\"schedule\":\"%s\",\"requested_flags\":%u,\"actual_flags\":%u,\"runtime_version\":%d,\"driver_version\":%d,\"architecture\":\"%s\",\"model_loaded\":false}\n",
        argv[1], requested, actual, runtime, driver, properties.gcnArchName);
    std::vector<uint32_t> host(count + 2u * guard, sentinel);
    uint32_t* device = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(uint32_t)));
    check(hipMemcpy(device, host.data(), host.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    hipEvent_t event = nullptr; check(hipEventCreateWithFlags(&event, hipEventDisableTiming));
    const char* methods[] = {"stream_sync", "device_sync", "event_sync", "event_query_yield", "event_query_spin"};
    for (unsigned loops : {1u, 1024u, 32768u}) {
        for (unsigned method = 0u; method < 5u; ++method) {
            std::array<double, repeats> intervals{};
            for (unsigned iteration = 0u; iteration < repeats; ++iteration) {
                const auto start = Clock::now();
                hipLaunchKernelGGL(work, dim3(count / 256u), dim3(256u), 0u, nullptr,
                    device + guard, loops, iteration);
                check(hipGetLastError()); check(hipEventRecord(event, nullptr));
                complete(method, event);
                intervals[iteration] = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
            }
            check(hipMemcpy(host.data(), device, host.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
            for (unsigned i = 0u; i < guard; ++i)
                require(host[i] == sentinel && host[guard + count + i] == sentinel, "output redzone changed");
            for (unsigned index : {0u, 1u, 17u, 511u, 1023u}) {
                uint32_t expected = 0x3958192u ^ (index * 65537u) ^ (repeats - 1u);
                for (unsigned i = 0u; i < loops; ++i) expected = advance(expected);
                require(host[guard + index] == expected, "completed output differs from independent CPU recurrence");
            }
            auto sorted = intervals; std::sort(sorted.begin(), sorted.end());
            std::printf("{\"type\":\"wait_case\",\"schedule\":\"%s\",\"method\":\"%s\",\"loops\":%u,\"repeats\":%u,\"first_us\":%.6f,\"minimum_us\":%.6f,\"median_us\":%.6f,\"p95_us\":%.6f,\"maximum_us\":%.6f,\"redzones_pass\":true,\"cpu_samples\":5,\"inference_acceptance\":false,\"completed_launch_and_wait_us\":[",
                argv[1], methods[method], loops, repeats, intervals[0], sorted.front(),
                (sorted[15] + sorted[16]) * 0.5, sorted[30], sorted.back());
            for (unsigned i = 0u; i < repeats; ++i) std::printf("%s%.6f", i ? "," : "", intervals[i]);
            std::printf("]}\n"); std::fflush(stdout);
        }
    }
    check(hipEventDestroy(event)); check(hipFree(device));
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "hip_completion_wait_probe_error=%s\n", error.what());
    return 2;
}
