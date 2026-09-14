#pragma once
#include "q8192_out_l1_policy.h"

// Included after the projection selector and the matrix helper declaration.
// All candidate work owns private storage. Only the existing corrected control
// is consumed by inference; a failed empirical envelope is reported, not used.
namespace qrt_out_l1_shadow {
constexpr unsigned guard = 128u, common = 4u;
constexpr unsigned counters = common + 3u * qrt_out_l1_policy::variants;

__global__ __launch_bounds__(256) void compare(
    const float* control, const float* before, const float* magnitude,
    const float* input_norm, const float* weight_norm,
    const unsigned* wf, const unsigned* xf, unsigned radius, unsigned ppb,
    bool exact_terminal, unsigned* stats) {
    constexpr unsigned rows = 2048u, tokens = 8192u, k = 4096u;
    // First-index slots are updated separately; reduce the remaining counters
    // per block so the audit does not serialize millions of atomic additions.
    unsigned local[counters]{};
    __shared__ unsigned reduction[256];
    for (unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
         index < rows * tokens; index += gridDim.x * blockDim.x) {
        const unsigned row = index % rows, token = index / rows;
        const float raw = before[index], bound_raw = magnitude[index];
        const bool terminal = exact_terminal && token == tokens - 1u;
        const bool original = terminal || selected_bf16_projection_hawkeye_candidate(
            raw, index, rows, radius, 0u, ppb, nullptr, input_norm, weight_norm);
        const bool different = __float_as_uint(device_bf16_round_to_float(raw)) !=
            __float_as_uint(device_bf16_round_to_float(control[index]));
        local[0] += original;
        local[1] += different && !original;
        local[2] += !isfinite(raw) || !isfinite(control[index]);
        local[3] += isnan(bound_raw) || bound_raw < 0.0f || wf[row] > 1u || xf[token] > 1u;
        const float upper = wf[row] && xf[token]
            ? qrt_bf16_positive_sum_bound::finish(bound_raw, k)
            : qrt_bf16_positive_sum_bound::value(0x7f800000u);
        const float cauchy = input_norm[token] * weight_norm[row];
        #pragma unroll
        for (unsigned variant = 0u; variant < qrt_out_l1_policy::variants; ++variant) {
            const unsigned offset = common + 3u * variant;
            const float proposed = qrt_out_l1_policy::capped(
                upper, cauchy, qrt_out_l1_policy::multiplier(variant));
            const bool selected = original && (terminal ||
                selected_bf16_projection_hawkeye_candidate(raw, index, rows,
                    radius, 0u, ppb, &proposed, nullptr, nullptr, index));
            local[offset] += selected;
            const bool miss = original && !selected && different;
            local[offset + 1u] += miss;
            if (miss) atomicMin(stats + offset + 2u, index);
        }
    }
    #pragma unroll
    for (unsigned field = 0u; field < counters; ++field) {
        if (field >= common && (field - common) % 3u == 2u) continue;
        reduction[threadIdx.x] = local[field];
        __syncthreads();
        for (unsigned stride = 128u; stride; stride >>= 1u) {
            if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            __syncthreads();
        }
        if (!threadIdx.x) atomicAdd(stats + field, reduction[0]);
        __syncthreads();
    }
}

inline hipError_t run(
    const uint16_t* weights, const uint16_t* inputs, const float* control,
    const float* input_norm, const float* weight_norm,
    unsigned rows, unsigned tokens, unsigned k, unsigned radius, unsigned ppb,
    hipStream_t stream, const std::string& stage, bool exact_terminal = false) {
    const char* setting = std::getenv("QRT_QWEN36_Q8192_OUT_L1_SHADOW_AUDIT");
    if (!setting || !*setting || !std::strcmp(setting, "0")) return hipSuccess;
    if (std::strcmp(setting, "1")) return hipErrorInvalidValue;
    if (rows != 2048u || tokens != 8192u || k != 4096u) return hipSuccess;
    unsigned choice = 99u;
    if (!qrt_q8192_matrix_producer::resolve(
            std::getenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_ALGORITHM"), rows, k,
            tokens, true, &choice, std::getenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE")) ||
        choice != 0u || !weights || !inputs || !control || !input_norm || !weight_norm || !ppb)
        return hipErrorInvalidValue;
#if !defined(QRT_ENABLE_HIPBLASLT_RESIDENT_MATRIX_PROVIDER)
    return hipErrorInvalidConfiguration;
#else
    const unsigned cells = rows * tokens;
    const size_t surface = size_t(cells) + 2u * guard;
    const size_t wsize = size_t(rows) * k + 2u * guard, xsize = size_t(tokens) * k + 2u * guard;
    const size_t msize = counters + rows + tokens + 6u * guard;
    std::array<void*, 4> owned{};
    const std::array<size_t, 4> bytes = {2u * surface * sizeof(float),
        wsize * sizeof(uint16_t), xsize * sizeof(uint16_t), msize * sizeof(unsigned)};
    std::array<unsigned, counters> host{};
    for (unsigned v = 0u; v < qrt_out_l1_policy::variants; ++v) host[common + 3u * v + 2u] = UINT_MAX;
    const auto started = std::chrono::steady_clock::now();
    auto complete = [&]() -> hipError_t {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (;;) {
            const hipError_t status = hipStreamQuery(stream);
            if (status != hipErrorNotReady) return status;
            if (std::chrono::steady_clock::now() >= deadline) return hipErrorLaunchTimeOut;
            std::this_thread::yield();
        }
    };
    const hipError_t status = [&]() -> hipError_t {
        hipError_t result;
        for (unsigned i = 0u; i < owned.size(); ++i)
            if ((result = hipMalloc(&owned[i], bytes[i])) != hipSuccess) return result;
        for (unsigned i = 0u; i < owned.size(); ++i)
            if ((result = hipMemsetAsync(owned[i], 0xa5, bytes[i], stream)) != hipSuccess) return result;
        float* before = static_cast<float*>(owned[0]) + guard;
        float* magnitude = static_cast<float*>(owned[0]) + surface + guard;
        uint16_t* mw = static_cast<uint16_t*>(owned[1]) + guard;
        uint16_t* mx = static_cast<uint16_t*>(owned[2]) + guard;
        unsigned* stats = static_cast<unsigned*>(owned[3]) + guard;
        unsigned* wf = stats + counters + 2u * guard;
        unsigned* xf = wf + rows + 2u * guard;
        if ((result = hipMemcpyAsync(stats, host.data(), sizeof(host), hipMemcpyHostToDevice, stream)) != hipSuccess)
            return result;
        hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,
            dim3(rows), dim3(256u), 0u, stream, weights, wf, rows, k);
        if ((result = hipGetLastError()) != hipSuccess) return result;
        hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,
            dim3(tokens), dim3(256u), 0u, stream, inputs, xf, tokens, k);
        if ((result = hipGetLastError()) != hipSuccess) return result;
        hipLaunchKernelGGL(qrt_bf16_absolute_product_views::prepare_rows_kernel,
            dim3(rows), dim3(256u), 0u, stream, weights, wf, mw, rows, k);
        if ((result = hipGetLastError()) != hipSuccess) return result;
        hipLaunchKernelGGL(qrt_bf16_absolute_product_views::prepare_rows_kernel,
            dim3(tokens), dim3(256u), 0u, stream, inputs, xf, mx, tokens, k);
        if ((result = hipGetLastError()) != hipSuccess || (result = complete()) != hipSuccess) return result;
        const auto prepared = std::chrono::steady_clock::now();
        std::string failed_stage, failure;
        for (unsigned which = 0u; which < 2u; ++which) {
            if (!resident_bf16_matrix_matmul_f32_output_with_heuristic_index(
                    which ? mw : weights, which ? mx : inputs, which ? magnitude : before,
                    rows, k, tokens, which ? 4u : 0u, stream,
                    stage + (which ? "_l1_magnitude4" : "_l1_before0"), &failed_stage, &failure)) {
                std::fprintf(stderr, "BATCH_MARK out_l1_shadow_failed stage=%s detail=%s\n",
                    failed_stage.c_str(), failure.c_str());
                return hipErrorInvalidConfiguration;
            }
        }
        if ((result = complete()) != hipSuccess) return result;
        const auto produced = std::chrono::steady_clock::now();
        hipLaunchKernelGGL(compare, dim3(4096u), dim3(256u), 0u, stream,
            control, before, magnitude, input_norm, weight_norm, wf, xf,
            radius, ppb, exact_terminal, stats);
        if ((result = hipGetLastError()) != hipSuccess || (result = complete()) != hipSuccess ||
            (result = hipMemcpy(host.data(), stats, sizeof(host), hipMemcpyDeviceToHost)) != hipSuccess) return result;
        const auto compared = std::chrono::steady_clock::now();
        // Check each typed redzone at its actual byte width, including both
        // sides of all three independently guarded metadata views.
        for (unsigned i = 0u; i < owned.size(); ++i) {
            const size_t unit = i == 1u || i == 2u ? sizeof(uint16_t) : sizeof(unsigned);
            const auto* base = static_cast<const unsigned char*>(owned[i]);
            std::vector<size_t> offsets = {0u, bytes[i] - guard * unit};
            if (i == 0u) { offsets.push_back((surface - guard) * unit); offsets.push_back(surface * unit); }
            if (i == 3u) for (size_t n : {size_t(counters + guard), size_t(counters + 2u * guard),
                    size_t(counters + rows + 3u * guard), size_t(counters + rows + 4u * guard)}) offsets.push_back(n * unit);
            std::array<unsigned char, guard * sizeof(unsigned)> redzone{};
            for (size_t offset : offsets) {
                if ((result = hipMemcpy(redzone.data(), base + offset, guard * unit, hipMemcpyDeviceToHost)) != hipSuccess) return result;
                for (size_t j = 0u; j < guard * unit; ++j) if (redzone[j] != 0xa5u) return hipErrorInvalidValue;
            }
        }
        for (unsigned variant = 0u; variant < qrt_out_l1_policy::variants; ++variant) {
            const unsigned offset = common + 3u * variant, index = host[offset + 2u];
            std::array<float, 5> detail{};
            unsigned canonical_bits = 0u, cpu_endpoint_matches = 0u;
            double exact_l1 = 0.0;
            if (index != UINT_MAX) {
                const unsigned row = index % rows, token = index / rows;
                const std::array<const float*, 5> sources = {before + index, control + index,
                    magnitude + index, input_norm + token, weight_norm + row};
                for (unsigned j = 0u; j < sources.size(); ++j)
                    if ((result = hipMemcpy(&detail[j], sources[j], sizeof(float), hipMemcpyDeviceToHost)) != hipSuccess) return result;
                std::vector<uint16_t> left(k), right(k);
                if ((result = hipMemcpy(left.data(), inputs + size_t(token) * k, k * sizeof(uint16_t), hipMemcpyDeviceToHost)) != hipSuccess ||
                    (result = hipMemcpy(right.data(), weights + size_t(row) * k, k * sizeof(uint16_t), hipMemcpyDeviceToHost)) != hipSuccess) return result;
                const float cpu = qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f, left.data(), right.data(), k);
                canonical_bits = qrt_bf16_positive_sum_bound::bits(cpu);
                auto endpoint = [](float x) {
                    const auto bits = qrt_bf16_positive_sum_bound::bits(x);
                    return (bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u;
                };
                cpu_endpoint_matches = endpoint(cpu) == endpoint(detail[1]);
                for (unsigned j = 0u; j < k; ++j)
                    exact_l1 += std::abs(double(qrt_bf16_positive_sum_bound::value(unsigned(left[j]) << 16u)) *
                        double(qrt_bf16_positive_sum_bound::value(unsigned(right[j]) << 16u)));
                if (!cpu_endpoint_matches || exact_l1 > qrt_bf16_positive_sum_bound::finish(detail[2], k)) return hipErrorInvalidValue;
            }
            std::fprintf(stderr,
                "BATCH_MARK out_l1_shadow stage=%s rows=%u tokens=%u k=%u cells=%u multiplier=%u radius=%u ppb=%u "
                "original_candidates=%u candidates=%u omitted=%u omitted_bf16_errors=%u first_index=%u "
                "baseline_undercoverage=%u nonfinite=%u invalid_magnitude_or_flags=%u exact_terminal=%u "
                "first_before_bits=%08x first_control_bits=%08x first_magnitude_bits=%08x input_norm_bits=%08x weight_norm_bits=%08x "
                "first_cpu_canonical_bits=%08x first_cpu_endpoint_matches=%u first_exact_l1=%.17g "
                "redzones_pass=1 control_output_read_only=1 diagnostic_only=1 performance_acceptance=0\n",
                stage.c_str(), rows, tokens, k, cells, qrt_out_l1_policy::multiplier(variant), radius, ppb,
                host[0], host[offset], host[0] - host[offset], host[offset + 1u], index, host[1], host[2], host[3],
                unsigned(exact_terminal), qrt_bf16_positive_sum_bound::bits(detail[0]), qrt_bf16_positive_sum_bound::bits(detail[1]),
                qrt_bf16_positive_sum_bound::bits(detail[2]), qrt_bf16_positive_sum_bound::bits(detail[3]),
                qrt_bf16_positive_sum_bound::bits(detail[4]), canonical_bits, cpu_endpoint_matches, exact_l1);
        }
        std::fprintf(stderr,
            "BATCH_MARK out_l1_shadow_setup stage=%s prepare_ms=%.6f matrices_ms=%.6f compare_ms=%.6f private_bytes=%zu variants=%u completion_deadline_ms=30000 diagnostic_only=1 performance_acceptance=0\n",
            stage.c_str(), std::chrono::duration<double, std::milli>(prepared - started).count(),
            std::chrono::duration<double, std::milli>(produced - prepared).count(),
            std::chrono::duration<double, std::milli>(compared - produced).count(),
            std::accumulate(bytes.begin(), bytes.end(), size_t(0)), qrt_out_l1_policy::variants);
        std::fflush(stderr);
        return host[1] || host[2] || host[3] ? hipErrorInvalidValue : hipSuccess;
    }();
    // Even a submission failure can leave prior work outstanding. Drain before
    // freeing owned storage; the native process has an independent outer guard.
    if (status != hipSuccess) (void)hipStreamSynchronize(stream);
    hipError_t released = hipSuccess;
    for (void* p : owned) if (p) {
        const auto result = hipFree(p);
        if (released == hipSuccess) released = result;
    }
    return status != hipSuccess ? status : released;
#endif
}
} // namespace qrt_out_l1_shadow
