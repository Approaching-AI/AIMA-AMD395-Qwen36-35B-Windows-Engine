// Compare the actual producer epilogue with the retained matrix plus global
// expert-ordered correction. Generated operands are not model acceptance.
#define QRT_TRITON_MOE_BATCHED_HAWKEYE 1
#define QRT_TRITON_MOE_NATIVE_WMMA_GATE 1
#define QRT_TRITON_MOE_NATIVE_WMMA_DOWN 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SPLIT_GATE_PASSES 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SERIAL_GATE_N32 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SERIAL_DOWN_N32 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_LOAD_THREADS 192
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_FUSED_OVERFLOW32 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SKIP_INACTIVE_A_STORES 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_PALETTE 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE 1
#define QRT_TRITON_MOE_NATIVE_WMMA_K_STAGE 32
#define QRT_MOE_ROUTED_REPLAY_LANES 4
#define QRT_TRITON_MOE_ROUTED_PROJECTION_DEBUG 1
#include "../../native/providers/triton_moe/qrt_triton_moe_q8192_provider.cpp"
#include <chrono>
#include <stdexcept>

namespace producer_gate_test {
constexpr size_t guard = 128u;
constexpr float sentinel = 12345.25f;
constexpr uint16_t sentinel16 = 0x5a5au;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void hip_ok(hipError_t status, const char* message) {
    if (status != hipSuccess) throw std::runtime_error(std::string(message) + ": " + hipGetErrorString(status));
}
template<class T> struct Device {
    T* pointer = nullptr;
    explicit Device(const std::vector<T>& v) {
        hip_ok(hipMalloc(reinterpret_cast<void**>(&pointer), v.size() * sizeof(T)), "allocate"); write(v);
    }
    ~Device() { if (pointer) (void)hipFree(pointer); }
    T* data() { return pointer + guard; }
    void write(const std::vector<T>& v) {
        hip_ok(hipMemcpy(pointer, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    std::vector<T> read(size_t n) {
        std::vector<T> v(n);
        hip_ok(hipMemcpy(v.data(), pointer, n * sizeof(T), hipMemcpyDeviceToHost), "download"); return v;
    }
};
uint32_t seed = 0x3958192u;
uint32_t next() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; }
uint16_t operand() { return uint16_t((next() & 0x807fu) | ((119u + next() % 8u) << 7u)); }
uint16_t bf16(float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4u);
    return uint16_t((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}
template<class T> void same(const std::vector<T>& a, const std::vector<T>& b, const char* message) {
    require(a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(T)), message);
}
void run(bool full, unsigned mode) {
    std::vector<unsigned> counts{0,1,15,16,17,31,32,33,63,64,65,80,96,97,127,128,129,257};
    if (full) {
        counts.resize(256u, 256u);
        unsigned sum = 0u; for (auto n : counts) sum += n;
        counts.back() += kRoutes - sum;
    }
    unsigned route_count = 0u, padded_count = 0u;
    for (auto n : counts) { route_count += n; padded_count += (n + 63u) / 64u * 64u; }
    const unsigned tokens = (route_count + 7u) / 8u;
    const size_t elements = size_t(route_count) * kIntermediate;
    const unsigned weight_rows = unsigned(counts.size()) * 2u * kIntermediate;
    std::vector<int32_t> sorted(padded_count + 2u * guard, int32_t(kRoutes));
    std::vector<int32_t> experts(padded_count / 64u + 2u * guard, -123456);
    std::vector<int32_t> padded(1u + 2u * guard, -123456), ids(route_count + 2u * guard, -123456);
    padded[guard] = int32_t(padded_count);
    unsigned first = 0u, route = 0u, merged = 0u;
    for (unsigned expert = 0u; expert < counts.size(); ++expert) {
        const unsigned count = counts[expert], blocks = (count + 63u) / 64u;
        const bool fused = blocks >= 2u && count % 64u >= 1u && count % 64u <= 32u;
        merged += fused;
        for (unsigned j = 0u; j < blocks; ++j)
            experts[guard + first / 64u + j] = fused && j + 2u == blocks ? -int(expert)-1 :
                fused && j + 1u == blocks ? -int(256u+expert)-1 : int(expert);
        for (unsigned j = 0u; j < count; ++j) {
            const unsigned id = route_count - 1u - route++;
            sorted[guard + first + j] = int32_t(id); ids[guard + id] = int32_t(expert);
        }
        first += blocks * 64u;
    }
    require(route == route_count && merged >= 5u, "missing overflow/scattered descriptors");
    std::vector<uint16_t> input(size_t(tokens) * kHidden + 2u * guard, sentinel16);
    std::vector<uint16_t> weights(size_t(weight_rows) * kHidden + 2u * guard, sentinel16);
    for (auto* v : {&input, &weights}) {
        for (size_t i = guard; i + guard < v->size(); ++i) (*v)[i] = operand();
        if (mode == 4u) for (size_t i = guard; i + guard < v->size(); i += 3071u)
            (*v)[i] = i % 3u == 0u ? 0x0001u : i % 3u == 1u ? 0u : uint16_t(90u << 7u);
    }
    std::vector<uint16_t> lut(65536u + 2u * guard, sentinel16);
    for (uint32_t i = 0u; i < 65536u; ++i) {
        uint32_t bits = i << 16u; float x; std::memcpy(&x, &bits, 4u);
        lut[guard+i] = bf16(!std::isfinite(x) || x < -80.0f ? 0.0f : x > 80.0f ? x : x / (1.0f + std::exp(-x)));
    }
    // The generated norm controls exercise empty, dense and sparse selection;
    // they are not claimed as conservative bounds for model inference.
    std::vector<float> inorm(tokens + 2u * guard, 1000.0f), wnorm(weight_rows + 2u * guard, 1000.0f);
    std::vector<float> blank(kActivatedElements + elements + 2u * guard, sentinel);
    std::vector<uint16_t> act(elements + 2u * guard, sentinel16);
    std::vector<uint16_t> debug(8u * kIntermediate + 2u * guard, sentinel16);
    std::vector<float> fdebug(debug.size(), sentinel);
    std::vector<uint32_t> counter(1u + 2u * guard, 0x5a5a5a5au); counter[guard] = 0u;
    const uint32_t capacity = 16384u * kNativeThreads;
    std::vector<uint32_t> scratch(capacity + 2u * guard, 0x5a5a5a5au);
    std::vector<uint32_t> order(qrt_moe_expert_order::bytes(capacity) / 4u + 2u * guard, 0x5a5a5a5au);
    Device<uint16_t> di(input), dw(weights), dl(lut), da(act), dg(debug), du(debug);
    Device<int32_t> ds(sorted), de(experts), dp(padded), did(ids);
    Device<float> dn(blank), din(inorm), dwn(wnorm), dgf(fdebug), duf(fdebug);
    Device<uint32_t> dc(counter), dcount(counter), dix(scratch), dorder(order);
    using Row = qrt_sm121_staged_half_projection::Row;
    Row row_sentinel{}; std::memset(&row_sentinel, 0x5a, sizeof(Row));
    std::vector<Row> pin(size_t(tokens) * kHidden / 16u + 2u * guard, row_sentinel);
    std::vector<Row> pweight(size_t(weight_rows) * kHidden / 16u + 2u * guard, row_sentinel);
    Device<Row> dpi(pin), dpw(pweight);
    hipStream_t stream = nullptr; hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    auto prepare = [&](const uint16_t* raw, Row* out, uint32_t rows) {
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,
            dim3((size_t(rows) * (kHidden / 16u) + 255u) / 256u), dim3(256), 0, stream, raw, out, rows, kHidden);
        hip_ok(hipGetLastError(), "prepare lossless operands");
    };
    prepare(di.data(), dpi.data(), tokens); prepare(dw.data(), dpw.data(), weight_rows);
    g_state.compact_routed_hawkeye = true; g_state.moe_compaction_blocks = 16384u;
    g_state.moe_expert_order_active = true; g_state.prevalidated_float_active = true;
    g_state.staged_half_replay_active = true; g_state.topk_ids = did.data();
    g_state.moe_compacted_indices = dix.data(); g_state.moe_compacted_count = dcount.data();
    g_state.moe_expert_order_storage = dorder.data();
    g_state.moe_l2[size_t(MoeL2::Input)] = din.data(); g_state.moe_l2[size_t(MoeL2::RoutedGateUp)] = dwn.data();
    g_state.prepared_replay_inputs = reinterpret_cast<uint16_t*>(dpi.data());
    g_state.prepared_replay_weights = reinterpret_cast<uint16_t*>(dpw.data());
    g_state.sm121_moe_absolute_error_ppb = mode == 3u ? 1000000000u : mode == 4u ? 512u : 0u;
    MoeCorrectionBounds bounds{din.data(), dwn.data(), float(g_state.sm121_moe_absolute_error_ppb) * 1.0e-9f, 0u};
    bounds.prepared_input = g_state.prepared_replay_inputs; bounds.prepared_weights = g_state.prepared_replay_weights;
    bounds.staged_half_replay = true; bounds.prevalidated_float = true;
    const uint32_t radius = mode == 1u ? 32768u : mode == 4u ? 512u : 0u;
    const uint32_t exponent = mode == 2u ? 255u : 0u;
    const uint32_t debug_token = tokens / 2u;
    std::vector<float> reference, reference_gf, reference_uf;
    std::vector<uint16_t> reference_act, reference_g, reference_u;
    std::vector<uint32_t> reference_count;
    float times[3]{};
    for (unsigned trial = 0u; trial < 3u; ++trial) {
        dn.write(blank); da.write(act); dg.write(debug); du.write(debug);
        dgf.write(fdebug); duf.write(fdebug); dc.write(counter);
        hipEvent_t begin = nullptr, end = nullptr;
        hip_ok(hipEventCreate(&begin), "begin event"); hip_ok(hipEventCreate(&end), "end event");
        hip_ok(hipEventRecord(begin, stream), "begin record");
        const auto kernel = trial == 1u ? native_wmma_gate_up_silu_lds_b_split_passes_kernel<true> :
            native_wmma_gate_up_silu_lds_b_split_passes_kernel<false>;
        hipLaunchKernelGGL(kernel,
            dim3((padded_count / 64u + 3u) * kNativeWmmaLdsBGateGridN), dim3(kNativeWmmaLdsBGateThreads),
            0, stream, di.data(), dw.data(), static_cast<const uint8_t*>(nullptr),
            static_cast<const uint32_t*>(nullptr), static_cast<const uint16_t*>(nullptr),
            ds.data(), de.data(), dp.data(), da.data(), dn.data(), dl.data(), radius, radius,
            dg.data(), du.data(), dgf.data(), duf.data(), dc.data(), debug_token, bounds, exponent, exponent);
        hip_ok(hipGetLastError(), "matrix launch");
        if (trial != 1u) {
            using P = MoeCorrectionPhase;
            const unsigned blocks = unsigned((elements + 255u) / 256u);
            hip_ok(launch_moe_routed_correction<false>(
                routed_gate_batched_hawkeye_correction_kernel<P::Local>, routed_gate_batched_hawkeye_correction_kernel<P::Collect>,
                routed_gate_batched_hawkeye_correction_kernel<P::Replay>, routed_gate_batched_hawkeye_correction_kernel<P::Local>,
                blocks, stream, MoeL2::Input, MoeL2::RoutedGateUp, nullptr, nullptr, nullptr,
                dn.data(), di.data(), dw.data(), did.data(), da.data(), dl.data(), route_count, radius, exponent,
                dg.data(), dgf.data(), dc.data(), debug_token), "original gate correction");
            hip_ok(launch_moe_routed_correction<true>(
                routed_up_batched_hawkeye_correction_activation_kernel<P::Local>, routed_up_batched_hawkeye_correction_activation_kernel<P::Collect>,
                routed_up_batched_hawkeye_correction_activation_kernel<P::Replay>, routed_up_batched_hawkeye_correction_activation_kernel<P::Finalize>,
                blocks, stream, MoeL2::Input, MoeL2::RoutedGateUp, nullptr, nullptr, nullptr,
                dn.data(), di.data(), dw.data(), did.data(), da.data(), dl.data(), route_count, radius, exponent,
                du.data(), duf.data(), dc.data(), debug_token), "original up correction");
        }
        hip_ok(hipEventRecord(end, stream), "end record");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (;;) {
            const auto status = hipEventQuery(end); if (status == hipSuccess) break;
            require(status == hipErrorNotReady && std::chrono::steady_clock::now() < deadline, "completion deadline");
            std::this_thread::yield();
        }
        hip_ok(hipEventElapsedTime(&times[trial], begin, end), "elapsed");
        hip_ok(hipEventDestroy(begin), "destroy begin"); hip_ok(hipEventDestroy(end), "destroy end");
        auto actual = dn.read(blank.size()), gf = dgf.read(fdebug.size()), uf = duf.read(fdebug.size());
        auto activated = da.read(act.size()), gate = dg.read(debug.size()), up = du.read(debug.size());
        auto selected = dc.read(counter.size());
        for (size_t i = 0u; i < actual.size(); ++i) {
            const bool live = i >= guard && i < guard + kActivatedElements + elements &&
                (i - guard) % kActivatedElements < elements;
            require(live ? std::isfinite(actual[i]) && actual[i] != sentinel : actual[i] == sentinel,
                "projection coverage/gap/guard changed");
        }
        for (size_t i = 0u; i < guard; ++i)
            require(activated[i] == sentinel16 && activated[activated.size()-1u-i] == sentinel16 &&
                selected[i] == counter[i] && selected[selected.size()-1u-i] == counter.back(), "activation/count redzone");
        if (mode == 0u) require(selected[guard] == 0u, "empty selector was not empty");
        else require(selected[guard] > 0u, "nonempty selector was empty");
        if (mode == 3u) require(selected[guard] == 2u * elements, "dense local queue was not full");
        if (trial == 0u) {
            reference = std::move(actual); reference_act = std::move(activated); reference_count = std::move(selected);
            reference_g = std::move(gate); reference_u = std::move(up); reference_gf = std::move(gf); reference_uf = std::move(uf);
        } else {
            same(actual, reference, "native or replayed raw FP32 mismatch"); same(activated, reference_act, "activation mismatch");
            same(selected, reference_count, "selected counts mismatch"); same(gate, reference_g, "gate debug mismatch");
            same(up, reference_u, "up debug mismatch"); same(gf, reference_gf, "native gate debug mismatch");
            same(uf, reference_uf, "native up debug mismatch");
        }
    }
    same(di.read(input.size()), input, "input changed"); same(dw.read(weights.size()), weights, "weights changed");
    same(dl.read(lut.size()), lut, "LUT changed"); same(ds.read(sorted.size()), sorted, "routes changed");
    same(de.read(experts.size()), experts, "descriptors changed"); same(dp.read(padded.size()), padded, "padding changed");
    same(did.read(ids.size()), ids, "topk ids changed"); same(din.read(inorm.size()), inorm, "input norm changed");
    same(dwn.read(wnorm.size()), wnorm, "weight norm changed");
    size_t supported = 0u, fallback = 0u;
    auto check_prepared = [&](Device<Row>& device, size_t n, const std::vector<uint16_t>& raw) {
        auto actual = device.read(n);
        for (size_t i = 0u; i < n; ++i) {
            if (i < guard || i + guard >= n) require(!std::memcmp(&actual[i], &row_sentinel, sizeof(Row)), "prepared redzone");
            else {
                const auto& row = actual[i];
                (qrt_sm121_scaled_half_products::unit(row) == -32768 ? fallback : supported)++;
                for (unsigned j = 0u; j < 16u; ++j)
                    require(qrt_sm121_scaled_half_products::original(row,j) == raw[guard + (i-guard)*16u+j], "prepared word changed");
            }
        }
    };
    check_prepared(dpi, pin.size(), input); check_prepared(dpw, pweight.size(), weights);
    require(supported && (mode != 4u || fallback), "missing supported/fallback groups");
    for (auto pair : {std::make_pair(dix.read(scratch.size()), scratch), std::make_pair(dorder.read(order.size()), order)})
        for (size_t i = 0u; i < guard; ++i) require(pair.first[i] == pair.second[i] &&
            pair.first[pair.first.size()-1u-i] == pair.second.back(), "global control scratch guard");
    hip_ok(hipStreamDestroy(stream), "destroy stream");
    g_state.moe_l2.fill(nullptr); g_state.topk_ids = nullptr; g_state.moe_compacted_indices = nullptr;
    g_state.moe_compacted_count = nullptr; g_state.moe_expert_order_storage = nullptr;
    g_state.prepared_replay_inputs = nullptr; g_state.prepared_replay_weights = nullptr;
    std::printf("{\"kind\":\"moe_producer_gate_safety\",\"tokens\":%u,\"logical_routes\":%u,\"padded_routes\":%u,"
        "\"mode\":%u,\"expert_shapes\":%zu,\"fused_overflow_pairs\":%u,\"projection_cells\":%zu,"
        "\"selected\":%u,\"supported_groups\":%zu,\"fallback_groups\":%zu,\"control_first_ms\":%.6f,"
        "\"producer_ms\":%.6f,\"control_last_ms\":%.6f,\"raw_fp32_mismatches\":0,\"bf16_mismatches\":0,"
        "\"debug_parity\":true,\"candidate_count_parity\":true,\"redzones_pass\":true,"
        "\"immutable_inputs\":true,\"all_prepared_words_checked\":true,\"inference_acceptance\":false}\n",
        tokens, route_count, padded_count, mode, counts.size(), merged, 2u * elements,
        reference_count[guard], supported, fallback, times[0], times[1], times[2]);
    std::fflush(stdout);
}
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && !std::strcmp(argv[1], "--q8192")) producer_gate_test::run(true, 4u);
        else { for (unsigned mode = 0u; mode < 5u; ++mode) producer_gate_test::run(false, mode); }
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "producer gate test failed: %s\n", e.what()); return 1; }
}
