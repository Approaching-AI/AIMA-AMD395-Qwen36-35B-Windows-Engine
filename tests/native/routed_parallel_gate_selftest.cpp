// Compare both actual WMMA schedules, including the original overflow
// descriptors. This synthetic operator check is not model acceptance.
#define QRT_TRITON_MOE_BATCHED_HAWKEYE 1
#define QRT_TRITON_MOE_NATIVE_WMMA_GATE 1
#define QRT_TRITON_MOE_NATIVE_WMMA_DOWN 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SPLIT_GATE_PASSES 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SERIAL_GATE_N32 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SERIAL_DOWN_N32 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_LOAD_THREADS 192
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_FUSED_OVERFLOW32 1
#define QRT_TRITON_MOE_NATIVE_WMMA_K_STAGE 32
#define QRT_TRITON_MOE_GROUP_M 1
#include "../../native/providers/triton_moe/qrt_triton_moe_q8192_provider.cpp"
#include <chrono>
#include <stdexcept>

namespace parallel_gate_test {
constexpr size_t guard = 128;
constexpr float sentinel = 12345.25f;
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void hip_ok(hipError_t status, const char *message) { require(status == hipSuccess, message); }
template<class T> struct Device {
    T *pointer = nullptr;
    explicit Device(const std::vector<T> &v) {
        hip_ok(hipMalloc(reinterpret_cast<void **>(&pointer), v.size() * sizeof(T)), "allocate");
        write(v);
    }
    ~Device() { if (pointer) (void)hipFree(pointer); }
    T *data() { return pointer + guard; }
    void write(const std::vector<T> &v) {
        hip_ok(hipMemcpy(pointer, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    std::vector<T> read(size_t n) {
        std::vector<T> v(n);
        hip_ok(hipMemcpy(v.data(), pointer, n * sizeof(T), hipMemcpyDeviceToHost), "download");
        return v;
    }
};
uint32_t seed = 0x3958192;
uint32_t next() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; }
uint16_t operand() { return uint16_t((next() & 0x807f) | ((119 + next() % 8) << 7)); }
void run() {
    const std::vector<unsigned> counts{0,1,15,16,17,31,32,33,63,64,65,80,96,97,127,128,129,257,1024};
    unsigned route_count = 0, padded_count = 0;
    for (auto n : counts) { route_count += n; padded_count += (n + 63) / 64 * 64; }
    std::vector<int32_t> sorted(size_t(padded_count) + 2 * guard, int32_t(kRoutes));
    std::vector<int32_t> experts(size_t(padded_count / 64) + 2 * guard, -123456);
    std::vector<int32_t> padded(1 + 2 * guard, -123456); padded[guard] = int32_t(padded_count);
    unsigned first = 0, route = 0, merged = 0;
    for (unsigned expert = 0; expert < counts.size(); ++expert) {
        const unsigned count = counts[expert], blocks = (count + 63) / 64;
        const bool fused = blocks >= 2 && count % 64 >= 1 && count % 64 <= 32;
        merged += fused;
        for (unsigned j = 0; j < blocks; ++j)
            experts[guard + first / 64 + j] = fused && j + 2 == blocks ? -int(expert)-1 :
                fused && j + 1 == blocks ? -int(256+expert)-1 : int(expert);
        // Reverse real route indices to exercise scattered token ownership.
        for (unsigned j = 0; j < count; ++j) sorted[guard + first + j] = int32_t(route_count - 1 - route++);
        first += blocks * 64;
    }
    require(route == route_count && merged >= 5, "fixture misses original overflow descriptors");
    const unsigned tokens = (route_count + kTopK - 1) / kTopK;
    std::vector<uint16_t> input(size_t(tokens) * kHidden + 2 * guard, 0x5a5a);
    std::vector<uint16_t> weights(counts.size() * 2 * kIntermediate * kHidden + 2 * guard, 0x5a5a);
    for (size_t i = guard; i < input.size() - guard; ++i) input[i] = operand();
    for (size_t i = guard; i < weights.size() - guard; ++i) weights[i] = operand();
    std::vector<float> blank(2 * kActivatedElements + 2 * guard, sentinel);
    Device<uint16_t> di(input), dw(weights);
    Device<int32_t> ds(sorted), de(experts), dp(padded);
    Device<float> dout(blank);
    hipStream_t stream = nullptr;
    hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    std::vector<float> reference;
    double times[3]{};
    for (unsigned trial = 0; trial < 3; ++trial) {
        dout.write(blank);
        const auto begin = std::chrono::steady_clock::now();
        if (trial == 1) {
            hipLaunchKernelGGL(qrt_routed_parallel_gate::matrix,
                dim3((padded_count / 64 + 3) * qrt_routed_parallel_gate::kColumnBlocks), dim3(256),
                0, stream, di.data(), dw.data(), ds.data(), de.data(), dp.data(), dout.data(), route_count);
        } else {
            hipLaunchKernelGGL(native_wmma_gate_up_silu_lds_b_split_passes_kernel,
                dim3((padded_count / 64 + 3) * kNativeWmmaLdsBGateGridN), dim3(kNativeWmmaLdsBGateThreads),
                0, stream, di.data(), dw.data(), ds.data(), de.data(), dp.data(),
                static_cast<uint16_t *>(nullptr), dout.data(), static_cast<const uint16_t *>(nullptr), 0u, 0u);
        }
        hip_ok(hipGetLastError(), "matrix launch");
        hipEvent_t end = nullptr;
        hip_ok(hipEventCreateWithFlags(&end, hipEventDisableTiming), "completion event");
        hip_ok(hipEventRecord(end, stream), "record completion");
        const auto deadline = begin + std::chrono::seconds(10);
        for (;;) {
            const hipError_t status = hipEventQuery(end);
            if (status == hipSuccess) break;
            require(status == hipErrorNotReady && std::chrono::steady_clock::now() < deadline, "matrix completion deadline");
            std::this_thread::yield();
        }
        times[trial] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        hip_ok(hipEventDestroy(end), "destroy completion");
        auto actual = dout.read(blank.size());
        size_t writes = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const bool in_surface = i >= guard && i < guard + 2 * kActivatedElements;
            const bool real = in_surface && (i - guard) % kActivatedElements < size_t(route_count) * kIntermediate;
            if (real) { require(std::isfinite(actual[i]) && actual[i] != sentinel, "real output missing or nonfinite"); ++writes; }
            else require(actual[i] == sentinel, "padding or redzone overwritten");
        }
        require(writes == size_t(route_count) * 2 * kIntermediate, "output coverage mismatch");
        if (trial == 0) reference = std::move(actual);
        else require(std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)) == 0,
                     "parallel matrix or repeated control differs from original FP32 values");
    }
    require(di.read(input.size()) == input && dw.read(weights.size()) == weights &&
            ds.read(sorted.size()) == sorted && de.read(experts.size()) == experts && dp.read(padded.size()) == padded,
            "immutable input changed");
    hip_ok(hipStreamDestroy(stream), "destroy stream");
    std::printf("{\"kind\":\"routed_parallel_gate\",\"logical_routes\":%u,\"padded_routes\":%u,"
        "\"expert_shapes\":%zu,\"fused_overflow_pairs\":%u,\"output_cells\":%zu,\"full_surface_cells_checked\":%zu,"
        "\"control_first_ms\":%.6f,\"parallel_ms\":%.6f,\"control_last_ms\":%.6f,"
        "\"raw_fp32_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
        route_count, padded_count, counts.size(), merged, size_t(route_count) * 2 * kIntermediate,
        blank.size(), times[0], times[1], times[2]);
}
}
int main() {
    try { parallel_gate_test::run(); return 0; }
    catch (const std::exception &error) { std::fprintf(stderr, "parallel gate test failed: %s\n", error.what()); return 1; }
}
