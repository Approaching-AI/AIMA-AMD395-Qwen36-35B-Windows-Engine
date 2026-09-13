// Exercise the actual provider selector, exact dot and host batch launcher.
// Synthetic dense/sparse controls are safety evidence, not model acceptance.
#define QRT_TRITON_MOE_BATCHED_HAWKEYE 1
#define QRT_TRITON_MOE_NATIVE_WMMA_GATE 1
#define QRT_TRITON_MOE_NATIVE_WMMA_DOWN 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SPLIT_GATE_PASSES 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SERIAL_GATE_N32 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SERIAL_DOWN_N32 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_LOAD_THREADS 192
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_FUSED_OVERFLOW32 1
#include "../../native/providers/triton_moe/qrt_triton_moe_q8192_provider.cpp"

#include <chrono>
#include <stdexcept>
#include "scaled_l2_reference.h"

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
    ~Device() { if (pointer) (void)hipFree(pointer); }
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
        shared_projection_hawkeye_midpoint_correction_kernel<false>, blocks, nullptr,
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

void compare_routed_compaction(uint32_t tokens, uint32_t mode,
                               uint32_t window_blocks = kMoeCompactionBlocks, bool prepared = false, bool float_replay = false) {
    require(window_blocks >= kMoeCompactionBlocks && window_blocks <= kMaximumMoeCompactionBlocks,
            "invalid test compaction window");
    g_state.moe_compaction_blocks = window_blocks;
    const uint32_t routes = tokens * kTopK;
    const size_t elements = static_cast<size_t>(routes) * kIntermediate;
    const size_t down_elements = static_cast<size_t>(routes) * kHidden;
    std::vector<uint16_t> input(static_cast<size_t>(tokens) * kHidden + 2u * kGuard, kSentinel);
    std::vector<uint16_t> weights(4u * kIntermediate * kHidden + 2u * kGuard, kSentinel);
    std::vector<uint16_t> down_weights(2u * kHidden * kIntermediate + 2u * kGuard, kSentinel);
    for (auto *values : {&input, &weights, &down_weights}) {
        for (size_t i = kGuard; i < values->size() - kGuard; ++i) {
            (*values)[i] = bf16(static_cast<float>(static_cast<int>((i * 17u) % 127u) - 63) / 64.0f);
        }
    }
    if (mode == 4u) {
        // Normal operands outside the compact range and subnormal rows must
        // use the original dot; ordinary rows continue through preparation.
        for (size_t i = kGuard; i + kGuard < input.size(); i += 3u * kHidden)
            input[i] = 0x0001u;
        for (size_t i = kGuard; i + kGuard < weights.size(); i += 7u * kHidden)
            weights[i] = uint16_t(63u << 7u | 19u);
        for (size_t i = kGuard; i + kGuard < down_weights.size(); i += 11u * kIntermediate)
            down_weights[i] = uint16_t(192u << 7u | 11u);
    }
    std::vector<int32_t> ids(routes + 2u * kGuard, -1234567);
    std::vector<float> topk(routes + 2u * kGuard, 12345.25f);
    for (uint32_t i = 0u; i < routes; ++i) {
        ids[kGuard + i] = static_cast<int32_t>((i / 3u) % 2u);
        topk[kGuard + i] = (i % 2u ? -0.125f : 0.25f);
    }
    std::vector<float> native(kActivatedElements + elements + 2u * kGuard, 12345.25f);
    std::vector<float> down(down_elements + 2u * kGuard, 12345.25f);
    for (size_t i = 0u; i < elements; ++i) {
        const uint32_t gate_bits = UINT32_C(0x3d800000) + static_cast<uint32_t>((i * 17777u) % UINT32_C(0x2000000));
        const uint32_t up_bits = UINT32_C(0xbd800000) + static_cast<uint32_t>((i * 13717u) % UINT32_C(0x2000000));
        std::memcpy(&native[kGuard + i], &gate_bits, 4u);
        std::memcpy(&native[kGuard + kActivatedElements + i], &up_bits, 4u);
    }
    for (size_t i = 0u; i < down_elements; ++i) {
        const uint32_t bits = UINT32_C(0x3e800000) + static_cast<uint32_t>((i * 7171u) % UINT32_C(0x2000000));
        std::memcpy(&down[kGuard + i], &bits, 4u);
    }
    std::vector<uint16_t> activated(elements + 2u * kGuard, kSentinel);
    std::vector<uint16_t> silu(65536u + 2u * kGuard, kSentinel);
    for (uint32_t i = 0u; i < 65536u; ++i) {
        uint32_t bits = i << 16u; float x; std::memcpy(&x, &bits, 4u);
        silu[kGuard + i] = bf16(!std::isfinite(x) || x < -80.0f ? 0.0f :
                               x > 80.0f ? x : x / (1.0f + std::exp(-x)));
    }
    std::vector<float> input_norm(routes + 2u * kGuard, mode == 3u ? 1000.0f : 0.0f);
    std::vector<float> weight_norm(4u * kHidden + 2u * kGuard, mode == 3u ? 1000.0f : 0.0f);
    std::vector<uint32_t> index(size_t(window_blocks) * kNativeThreads + 2u * kGuard, UINT32_C(0x5a5a5a5a));
    std::vector<uint32_t> count(1u + 2u * kGuard, UINT32_C(0x5a5a5a5a));
    Device<uint16_t> di(input), dw(weights), ddw(down_weights), da(activated), dl(silu);
    Device<int32_t> did(ids);
    Device<float> dt(topk), dn(native), dd(down), din(input_norm), dwn(weight_norm);
    Device<uint32_t> dix(index), dc(count);
    std::vector<uint16_t> encoded_input(std::max(input.size(), activated.size()), kSentinel);
    std::vector<uint16_t> encoded_weights(std::max(weights.size(), down_weights.size()), kSentinel);
    std::vector<uint32_t> encoded_input_flags(routes + 2u * kGuard, 0x5a5a5a5au);
    std::vector<uint32_t> encoded_weight_flags(2u * kHidden + 2u * kGuard, 0x5a5a5a5au);
    std::vector<float> prepared_norm(std::max(size_t(routes), size_t(2u * kHidden)) + 2u * kGuard, 12345.25f);
    Device<uint16_t> dei(encoded_input), dew(encoded_weights);
    Device<uint32_t> defi(encoded_input_flags), defw(encoded_weight_flags);
    Device<float> depn(prepared_norm);
    g_state.moe_compacted_indices = dix.data(); g_state.moe_compacted_count = dc.data();
    g_state.sm121_moe_absolute_error_ppb = 1000u;
    g_state.moe_l2[static_cast<size_t>(MoeL2::Input)] = din.data();
    g_state.moe_l2[static_cast<size_t>(MoeL2::RoutedActivated)] = din.data();
    g_state.moe_l2[static_cast<size_t>(MoeL2::RoutedGateUp)] = dwn.data();
    g_state.moe_l2[static_cast<size_t>(MoeL2::RoutedDown)] = dwn.data();
    hipStream_t stream = nullptr;
    hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "compaction stream");
    hipEvent_t begin = nullptr, end = nullptr;
    hip_ok(hipEventCreate(&begin), "compaction begin"); hip_ok(hipEventCreate(&end), "compaction end");
    std::vector<float> expected_native, expected_down;
    std::vector<uint16_t> expected_activated;
    float times[4]{};
    auto prepare_view = [&](const uint16_t* raw, uint16_t* encoded, uint32_t* flags,
                            uint32_t rows, uint32_t columns) {
        for (uint32_t first = 0u; first < rows; first += 4096u) {
            hipLaunchKernelGGL(moe_bf16_row_l2_prepared_kernel,
                dim3(std::min(4096u, rows - first)), dim3(kNativeThreads), 0, stream,
                raw, depn.data(), encoded, flags, rows, columns, first);
            hip_ok(hipGetLastError(), "prepare routed replay view");
        }
    };
    for (uint32_t compact = 0u; compact < (float_replay ? 4u : prepared ? 3u : 2u); ++compact) {
        dn.write(native); dd.write(down); da.write(activated);
        g_state.compact_routed_hawkeye = compact != 0u;
        g_state.prepared_replay_active = compact == 2u;
        g_state.float_replay_active = compact == 3u;
        g_state.prepared_replay_inputs = dei.data(); g_state.prepared_replay_weights = dew.data();
        g_state.prepared_replay_input_rows = defi.data(); g_state.prepared_replay_weight_rows = defw.data();
        const uint32_t radius = mode == 1u || mode == 4u ? 32768u : mode == 2u ? 128u : 0u;
        const uint32_t exponent = mode == 2u ? 124u : 0u;
        const uint32_t blocks = static_cast<uint32_t>((elements + kNativeThreads - 1u) / kNativeThreads);
        using P = MoeCorrectionPhase;
        hip_ok(hipEventRecord(begin, stream), "compaction timing begin");
        if (compact == 2u) {
            prepare_view(di.data(), dei.data(), defi.data(), tokens, kHidden);
            prepare_view(dw.data(), dew.data(), defw.data(), 4u * kIntermediate, kHidden);
        }
        hip_ok(launch_moe_routed_correction<false>(
            routed_gate_batched_hawkeye_correction_kernel<P::Local>, routed_gate_batched_hawkeye_correction_kernel<P::Collect>,
            routed_gate_batched_hawkeye_correction_kernel<P::Replay>, routed_gate_batched_hawkeye_correction_kernel<P::Local>,
            blocks, stream, MoeL2::Input, MoeL2::RoutedGateUp, dn.data(), di.data(), dw.data(), did.data(), da.data(), dl.data(),
            routes, radius, exponent), "compaction gate");
        hip_ok(launch_moe_routed_correction<true>(
            routed_up_batched_hawkeye_correction_activation_kernel<P::Local>, routed_up_batched_hawkeye_correction_activation_kernel<P::Collect>,
            routed_up_batched_hawkeye_correction_activation_kernel<P::Replay>, routed_up_batched_hawkeye_correction_activation_kernel<P::Finalize>,
            blocks, stream, MoeL2::Input, MoeL2::RoutedGateUp, dn.data(), di.data(), dw.data(), did.data(), da.data(), dl.data(),
            routes, radius, exponent), "compaction up");
        if (compact == 2u) {
            prepare_view(da.data(), dei.data(), defi.data(), routes, kIntermediate);
            prepare_view(ddw.data(), dew.data(), defw.data(), 2u * kHidden, kIntermediate);
        }
        hip_ok(launch_moe_routed_correction<false>(
            routed_down_batched_hawkeye_correction_kernel<P::Local>, routed_down_batched_hawkeye_correction_kernel<P::Collect>,
            routed_down_batched_hawkeye_correction_kernel<P::Replay>, routed_down_batched_hawkeye_correction_kernel<P::Local>,
            static_cast<uint32_t>((down_elements + kNativeThreads - 1u) / kNativeThreads), stream,
            MoeL2::RoutedActivated, MoeL2::RoutedDown, dd.data(), dt.data(), did.data(), da.data(), ddw.data(), routes, radius, exponent), "compaction down");
        hip_ok(hipEventRecord(end, stream), "compaction timing end");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        for (;;) {
            const hipError_t status = hipEventQuery(end);
            if (status == hipSuccess) break;
            require(status == hipErrorNotReady, "compaction completion failed");
            require(std::chrono::steady_clock::now() < deadline, "compaction sequence deadline");
            std::this_thread::yield();
        }
        hip_ok(hipEventElapsedTime(&times[compact], begin, end), "compaction interval");
        auto actual_native = dn.read(native.size()), actual_down = dd.read(down.size());
        auto actual_activated = da.read(activated.size());
        if (compact == 0u) {
            expected_native = std::move(actual_native); expected_down = std::move(actual_down);
            expected_activated = std::move(actual_activated);
        } else {
            require(std::memcmp(actual_native.data(), expected_native.data(), native.size() * sizeof(float)) == 0 &&
                    std::memcmp(actual_down.data(), expected_down.data(), down.size() * sizeof(float)) == 0 &&
                    actual_activated == expected_activated, "routed compaction differs from original kernel");
        }
    }
    if (prepared) {
        for (auto pair : {std::make_pair(dei.read(encoded_input.size()), encoded_input),
                          std::make_pair(dew.read(encoded_weights.size()), encoded_weights)})
            for (size_t i = 0u; i < kGuard; ++i)
                require(pair.first[i] == pair.second[i] && pair.first[pair.first.size()-1u-i] == pair.second.back(),
                        "prepared operand redzone changed");
        for (auto pair : {std::make_pair(defi.read(encoded_input_flags.size()), encoded_input_flags),
                          std::make_pair(defw.read(encoded_weight_flags.size()), encoded_weight_flags)})
            for (size_t i = 0u; i < kGuard; ++i)
                require(pair.first[i] == pair.second[i] && pair.first[pair.first.size()-1u-i] == pair.second.back(),
                        "prepared eligibility redzone changed");
    }
    require(std::memcmp(expected_native.data(), native.data(), (kGuard + kActivatedElements) * sizeof(float)) == 0,
            "gate accumulator or unused projection gap changed");
    for (size_t i = 0u; i < kGuard; ++i) {
        require(expected_native[i] == native[i] && expected_native[native.size()-1u-i] == native.back() &&
                expected_down[i] == down[i] && expected_down[down.size()-1u-i] == down.back() &&
                expected_activated[i] == kSentinel && expected_activated[activated.size()-1u-i] == kSentinel,
                "routed output redzone changed");
    }
    auto actual_indices = dix.read(index.size()), actual_count = dc.read(count.size());
    for (size_t i = 0u; i < kGuard; ++i) {
        require(actual_indices[i] == index[i] && actual_indices[index.size()-1u-i] == index.back() &&
                actual_count[i] == count[i] && actual_count[count.size()-1u-i] == count.back(), "compaction scratch redzone changed");
    }
    require(di.read(input.size()) == input && dw.read(weights.size()) == weights && ddw.read(down_weights.size()) == down_weights &&
            did.read(ids.size()) == ids && dt.read(topk.size()) == topk && dl.read(silu.size()) == silu &&
            din.read(input_norm.size()) == input_norm && dwn.read(weight_norm.size()) == weight_norm, "routed input changed");
    hip_ok(hipEventDestroy(end), "compaction end destroy"); hip_ok(hipEventDestroy(begin), "compaction begin destroy");
    hip_ok(hipStreamDestroy(stream), "compaction stream destroy");
    g_state.moe_l2.fill(nullptr); g_state.moe_compacted_indices = nullptr; g_state.moe_compacted_count = nullptr;
    g_state.compact_routed_hawkeye = false;
    g_state.prepared_replay_active = false;
    g_state.float_replay_active = false;
    g_state.prepared_replay_inputs = g_state.prepared_replay_weights = nullptr;
    g_state.prepared_replay_input_rows = g_state.prepared_replay_weight_rows = nullptr;
    g_state.moe_compaction_blocks = kMoeCompactionBlocks;
    std::printf("{\"kind\":\"routed_compaction_comparison\",\"tokens\":%u,\"mode\":%u,\"projection_elements\":%zu,"
                "\"down_elements\":%zu,\"local_ms\":%.6f,\"compact_ms\":%.6f,\"raw_bit_mismatches\":0,"
                "\"prepared_replay_checked\":%s,\"prepared_sequence_ms\":%.6f,"
                "\"float_replay_checked\":%s,\"float_sequence_ms\":%.6f,"
                "\"replay_lanes\":%u,\"window_blocks\":%u,\"maximum_replay_blocks\":%u,"
                "\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
                tokens, mode, elements, down_elements, times[0], times[1], prepared ? "true" : "false", times[2], float_replay ? "true" : "false", times[3], unsigned(QRT_MOE_ROUTED_REPLAY_LANES),
                window_blocks, kMoeCompactionBlocks);
}

void compare_prepared_norm(unsigned columns) {
    constexpr unsigned first = 3u, tested_rows = 259u, rows = first + tested_rows;
    const auto fixture = scaled_l2_test::fixture(tested_rows, columns);
    std::vector<uint16_t> input(size_t(rows) * columns + 2u * kGuard, kSentinel);
    std::copy(fixture.begin(), fixture.end(), input.begin() + kGuard + size_t(first) * columns);
    std::vector<uint16_t> encoded(input.size(), kSentinel), expected(encoded);
    std::vector<uint32_t> flags(rows + 2u * kGuard, 0x5a5a5a5au), expected_flags(flags);
    std::vector<float> norms(rows + 2u * kGuard, 12345.25f);
    Device<uint16_t> di(input), de(encoded);
    Device<uint32_t> df(flags);
    Device<float> original(norms), candidate(norms);
    hipLaunchKernelGGL(moe_bf16_row_l2_kernel, dim3(tested_rows), dim3(kNativeThreads), 0, nullptr,
        di.data(), original.data(), rows, columns, first);
    hip_ok(hipGetLastError(), "original prepared norm control");
    hipLaunchKernelGGL(moe_bf16_row_l2_prepared_kernel, dim3(tested_rows), dim3(kNativeThreads), 0, nullptr,
        di.data(), candidate.data(), de.data(), df.data(), rows, columns, first);
    hip_ok(hipGetLastError(), "fused prepared norm");
    const auto a = original.read(norms.size()), b = candidate.read(norms.size());
    require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
            "prepared scan changed original FP64 norm bits");
    unsigned eligible_rows = 0u;
    for (unsigned row = first; row < rows; ++row) {
        bool valid_row = true;
        for (unsigned k = 0; k < columns; ++k) {
            const size_t index = kGuard + size_t(row) * columns + k;
            const bool valid = qrt_sm121_prepared_bf16::eligible(input[index]);
            valid_row &= valid;
            expected[index] = valid ? qrt_sm121_prepared_bf16::encode(input[index]) : 0u;
        }
        expected_flags[kGuard + row] = valid_row;
        eligible_rows += valid_row;
    }
    require(de.read(encoded.size()) == expected && df.read(flags.size()) == expected_flags,
            "prepared values, eligibility or redzones differ");
    require(di.read(input.size()) == input, "prepared scan changed original operands");
    for (size_t i = 0; i < norms.size(); ++i)
        if (i < kGuard + first || i >= kGuard + rows)
            require(a[i] == norms[i] && b[i] == norms[i], "prepared norm redzone changed");
    std::printf("{\"kind\":\"prepared_moe_norm_scan\",\"rows\":%u,\"columns\":%u,"
        "\"encoded_cells\":%zu,\"eligible_rows\":%u,\"norm_bit_differences\":0,"
        "\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
        tested_rows, columns, size_t(tested_rows) * columns, eligible_rows);
}

void compare_scaled_l2(unsigned columns) {
    constexpr unsigned first = 3u, tested_rows = 259u, rows = first + tested_rows;
    const auto fixture = scaled_l2_test::fixture(tested_rows, columns);
    std::vector<uint16_t> input(size_t(rows) * columns + 2u * kGuard, kSentinel);
    std::copy(fixture.begin(), fixture.end(), input.begin() + kGuard + size_t(first) * columns);
    std::vector<float> output(rows + 2u * kGuard, 12345.25f);
    Device<uint16_t> di(input);
    Device<float> original(output), scaled(output);
    hipLaunchKernelGGL(moe_bf16_row_l2_kernel, dim3(tested_rows), dim3(kNativeThreads), 0, nullptr,
                      di.data(), original.data(), rows, columns, first);
    hip_ok(hipGetLastError(), "original row norm launch");
    hipLaunchKernelGGL(moe_bf16_scaled_row_l2_kernel, dim3(tested_rows), dim3(kNativeThreads), 0, nullptr,
                      di.data(), scaled.data(), rows, columns, first);
    hip_ok(hipGetLastError(), "scaled row norm launch");
    const auto control = original.read(output.size());
    const auto candidate = scaled.read(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        if (i < kGuard + first || i >= kGuard + rows) {
            require(control[i] == output[i] && candidate[i] == output[i], "norm output redzone changed");
            continue;
        }
        const size_t row = i - kGuard - first;
        const long double expected = scaled_l2_test::norm(fixture.data() + row * columns, columns);
        if (!std::isfinite(expected)) {
            require(std::isinf(candidate[i]), "nonfinite operand did not select conservative infinity");
        } else {
            require(static_cast<long double>(candidate[i]) >= expected * static_cast<long double>(1.00002f),
                    "scaled GPU norm underestimates inflated reference");
            require(candidate[i] >= control[i], "scaled GPU norm below original metadata");
            if (std::isfinite(candidate[i]))
                require(static_cast<long double>(candidate[i]) <= expected * 1.0001L, "scaled GPU norm inflation too large");
        }
    }
    require(di.read(input.size()) == input, "norm input changed");
    std::printf("{\"kind\":\"scaled_l2_bound_comparison\",\"rows\":%u,\"columns\":%u,"
                "\"first_row\":%u,\"underestimates\":0,\"redzones_pass\":true,"
                "\"immutable_inputs\":true,\"inference_acceptance\":false}\n", tested_rows, columns, first);
}
}

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--float-replay") == 0) {
            moe_batch_test::compare_routed_compaction(1u, 0u, kMaximumMoeCompactionBlocks, true, true);
            moe_batch_test::compare_routed_compaction(65u, 2u, kMaximumMoeCompactionBlocks, true, true);
            moe_batch_test::compare_routed_compaction(129u, 4u, kMaximumMoeCompactionBlocks, true, true);
            moe_batch_test::compare_routed_compaction(1025u, 3u, kMaximumMoeCompactionBlocks, true, true);
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--prepared-replay") == 0) {
            moe_batch_test::compare_prepared_norm(512u);
            moe_batch_test::compare_prepared_norm(2048u);
            moe_batch_test::compare_routed_compaction(1u, 0u, kMaximumMoeCompactionBlocks, true);
            moe_batch_test::compare_routed_compaction(65u, 2u, kMaximumMoeCompactionBlocks, true);
            moe_batch_test::compare_routed_compaction(129u, 4u, kMaximumMoeCompactionBlocks, true);
            moe_batch_test::compare_routed_compaction(1025u, 3u, kMaximumMoeCompactionBlocks, true);
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--wide-compaction") == 0) {
            moe_batch_test::compare_routed_compaction(1u, 0u, kMaximumMoeCompactionBlocks);
            moe_batch_test::compare_routed_compaction(65u, 2u, kMaximumMoeCompactionBlocks);
            // Gate crosses one complete four-million-cell window; down
            // crosses four. Dense mode also exercises every persistent slot.
            moe_batch_test::compare_routed_compaction(1025u, 1u, kMaximumMoeCompactionBlocks);
            moe_batch_test::compare_routed_compaction(1025u, 3u, kMaximumMoeCompactionBlocks);
            return 0;
        }
        moe_batch_test::require(argc == 1 || (argc == 2 && std::strcmp(argv[1], "--selftest") == 0),
                               "unexpected safety test arguments");
        moe_batch_test::compare_scaled_l2(512u);
        moe_batch_test::compare_scaled_l2(2048u);
        moe_batch_test::run_case<64u>(512u, true);
        moe_batch_test::run_case<kMaximumMoeCorrectionBlocks>(512u, true);
        moe_batch_test::run_case<kMaximumMoeCorrectionBlocks>(513u, true);
        moe_batch_test::run_case<kMaximumMoeCorrectionBlocks>(513u, false);
        moe_batch_test::compare_routed_compaction(1u, 0u);
        moe_batch_test::compare_routed_compaction(65u, 1u);
        moe_batch_test::compare_routed_compaction(65u, 2u);
        moe_batch_test::compare_routed_compaction(65u, 3u);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "MoE correction batch safety failed: %s\n", error.what());
        return 1;
    }
}
