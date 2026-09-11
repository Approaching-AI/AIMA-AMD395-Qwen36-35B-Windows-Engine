// Exercise the actual provider selector, exact dot and host batch launcher.
// Synthetic dense/sparse controls are safety evidence, not model acceptance.
#define QRT_TRITON_MOE_BATCHED_HAWKEYE 1
#include "../../native/providers/triton_moe/qrt_triton_moe_q8192_provider.cpp"

#include <chrono>
#include <stdexcept>

namespace moe_batch_test {
constexpr size_t kGuard = 128u;
constexpr uint16_t kSentinel = UINT16_C(0x5a5a);

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void hip_ok(hipError_t status, const char* stage) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(stage) + ": " + hipGetErrorString(status));
}
uint16_t bf16(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}
template<class T> struct Device {
    T* pointer = nullptr;
    explicit Device(const std::vector<T>& values) {
        hip_ok(hipMalloc(reinterpret_cast<void**>(&pointer), values.size() * sizeof(T)), "allocate");
        write(values);
    }
    ~Device() { if (pointer) hipFree(pointer); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    T* data() { return pointer + kGuard; }
    void write(const std::vector<T>& values) {
        hip_ok(hipMemcpy(pointer, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    std::vector<T> read(size_t size) {
        std::vector<T> result(size);
        hip_ok(hipMemcpy(result.data(), pointer, size * sizeof(T), hipMemcpyDeviceToHost), "download");
        return result;
    }
};

template<uint32_t MaximumBlocks>
void run_case(uint32_t tokens, bool dense) {
    const size_t elements = static_cast<size_t>(tokens) * kIntermediate;
    std::vector<uint16_t> input(static_cast<size_t>(tokens) * kHidden + 2u * kGuard, kSentinel);
    std::vector<uint16_t> weights(static_cast<size_t>(kIntermediate) * kHidden + 2u * kGuard, kSentinel);
    for (uint32_t token = 0; token < tokens; ++token)
        for (uint32_t k = 0; k < kHidden; ++k)
            input[kGuard + static_cast<size_t>(token) * kHidden + k] = bf16(
                static_cast<float>(static_cast<int>((token % 7u * 13u + k * 3u) % 61u) - 30) / 32.0f);
    for (uint32_t row = 0; row < kIntermediate; ++row)
        for (uint32_t k = 0; k < kHidden; ++k)
            weights[kGuard + static_cast<size_t>(row) * kHidden + k] = bf16(
                static_cast<float>(static_cast<int>((row % 11u * 7u + k * 5u) % 47u) - 23) / 64.0f);
    uint16_t reference[7][11]{};
    for (uint32_t token = 0; token < 7u; ++token)
        for (uint32_t row = 0; row < 11u; ++row)
            reference[token][row] = bf16(qrt_q1_moe_hawkeye::dot_bf16_hopper(
                input.data() + kGuard + static_cast<size_t>(token) * kHidden,
                weights.data() + kGuard + static_cast<size_t>(row) * kHidden, kHidden));

    std::vector<float> native(elements + 2u * kGuard, 12345.25f);
    std::vector<uint16_t> output(elements + 2u * kGuard, kSentinel);
    for (size_t i = 0; i < elements; ++i) {
        const uint16_t expected = reference[(i / kIntermediate) % 7u][(i % kIntermediate) % 11u];
        uint32_t bits = static_cast<uint32_t>(expected) << 16u;
        if (dense || i % 17u == 0u) bits = UINT32_C(0x3f808000);  // Exact BF16 midpoint.
        std::memcpy(&native[kGuard + i], &bits, sizeof(bits));
    }
    // Allocated zero L2 surfaces keep the bounded launcher active while the
    // midpoint selector controls the deliberately dense/sparse safety cases.
    std::vector<float> input_norm(tokens + 2u * kGuard, 0.0f);
    std::vector<float> weight_norm(kIntermediate + 2u * kGuard, 0.0f);
    Device<uint16_t> di(input), dw(weights), dout(output);
    Device<float> dn(native), din(input_norm), dwn(weight_norm);
    g_state.sm121_moe_absolute_error_ppb = 1000u;
    g_state.moe_l2[static_cast<size_t>(MoeL2::SharedInput)] = din.data();
    g_state.moe_l2[static_cast<size_t>(MoeL2::SharedGate)] = dwn.data();
    hipEvent_t begin = nullptr, end = nullptr;
    hip_ok(hipEventCreate(&begin), "begin event");
    hip_ok(hipEventCreate(&end), "end event");
    hip_ok(hipEventRecord(begin, nullptr), "record begin");
    const uint32_t blocks = static_cast<uint32_t>((elements + kNativeThreads - 1u) / kNativeThreads);
    hip_ok(launch_moe_correction<MaximumBlocks>(
        shared_projection_hawkeye_midpoint_correction_kernel, blocks, nullptr,
        MoeL2::SharedInput, MoeL2::SharedGate, dn.data(), di.data(), dw.data(),
        dout.data(), tokens, dense ? 32768u : 0u), "actual correction launcher");
    hip_ok(hipEventRecord(end, nullptr), "record end");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        const hipError_t status = hipEventQuery(end);
        if (status == hipSuccess) break;
        require(status == hipErrorNotReady, "completion event failed");
        require(std::chrono::steady_clock::now() < deadline, "bounded correction timed out");
        std::this_thread::yield();
    }
    float milliseconds = 0.0f;
    hip_ok(hipEventElapsedTime(&milliseconds, begin, end), "elapsed time");
    hip_ok(hipEventDestroy(end), "destroy end");
    hip_ok(hipEventDestroy(begin), "destroy begin");
    require(milliseconds < 100.0f, "dense correction sequence exceeded 100 ms");
    const auto actual = dout.read(output.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        uint16_t expected = kSentinel;
        if (i >= kGuard && i < kGuard + elements) {
            const size_t cell = i - kGuard;
            expected = reference[(cell / kIntermediate) % 7u][(cell % kIntermediate) % 11u];
        }
        require(actual[i] == expected, "exact result or output redzone changed");
    }
    require(di.read(input.size()) == input && dw.read(weights.size()) == weights &&
            dn.read(native.size()) == native && din.read(input_norm.size()) == input_norm &&
            dwn.read(weight_norm.size()) == weight_norm, "immutable input changed");
    g_state.moe_l2.fill(nullptr);
    std::printf("{\"kind\":\"moe_correction_batch_safety\",\"tokens\":%u,"
                "\"elements\":%zu,\"dense\":%s,\"maximum_blocks\":%u,"
                "\"interval_kind\":\"correction_sequence\",\"interval_ms\":%.6f,"
                "\"cpu_reference_mismatches\":0,\"redzones\":\"pass\","
                "\"immutable_inputs\":\"pass\",\"inference_acceptance\":false}\n",
                tokens, elements, dense ? "true" : "false", MaximumBlocks, milliseconds);
}
}

int main() {
    try {
        moe_batch_test::run_case<64u>(512u, true);
        moe_batch_test::run_case<kMaximumMoeCorrectionBlocks>(512u, true);
        moe_batch_test::run_case<kMaximumMoeCorrectionBlocks>(513u, true);
        moe_batch_test::run_case<kMaximumMoeCorrectionBlocks>(513u, false);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MoE correction batch safety failed: %s\n", error.what());
        return 1;
    }
}
