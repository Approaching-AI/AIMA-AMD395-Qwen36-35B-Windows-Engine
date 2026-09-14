#pragma once

// Included after the original correction implementation. The qualified OUT
// output is read-only; a separate algorithm-4 surface never drives inference.
#if defined(QRT_ENABLE_HIPBLASLT_RESIDENT_MATRIX_PROVIDER)
bool resident_bf16_matrix_matmul_f32_output_with_heuristic_index(
    const uint16_t*, const uint16_t*, float*, unsigned, unsigned, unsigned,
    unsigned, hipStream_t, const std::string&, std::string*, std::string*);
#endif

namespace qrt_out_matrix_shadow {
constexpr unsigned guard = 128u, capacity = 4096u, counters = 16u;

__global__ __launch_bounds__(256) void compare(
    const float* control, const float* shadow, unsigned cells,
    unsigned* stats, unsigned* indices) {
    __shared__ unsigned raw_counts[256], finite_counts[256];
    unsigned raw = 0, nonfinite = 0;
    for (unsigned cell = blockIdx.x * blockDim.x + threadIdx.x; cell < cells;
         cell += blockDim.x * gridDim.x) {
        const float a = control[cell], b = shadow[cell];
        raw += __float_as_uint(a) != __float_as_uint(b);
        nonfinite += !isfinite(a) || !isfinite(b);
        if (__float_as_uint(device_bf16_round_to_float(a)) !=
            __float_as_uint(device_bf16_round_to_float(b))) {
            const unsigned slot = atomicAdd(stats + 1, 1u);
            atomicMin(stats + 3, cell);
            if (slot < capacity) indices[slot] = cell;
        }
    }
    raw_counts[threadIdx.x] = raw; finite_counts[threadIdx.x] = nonfinite;
    __syncthreads();
    for (unsigned stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            raw_counts[threadIdx.x] += raw_counts[threadIdx.x + stride];
            finite_counts[threadIdx.x] += finite_counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (!threadIdx.x) { atomicAdd(stats, raw_counts[0]); atomicAdd(stats + 2, finite_counts[0]); }
}

// Independent original integer wave16 arithmetic classifies every stored
// difference against both endpoints and the algorithm-4 admission predicate.
__global__ __launch_bounds__(256) void classify(
    const uint16_t* weights, const uint16_t* inputs,
    const float* control, const float* shadow, const float* before,
    const float* input_norm, const float* weight_norm,
    const unsigned* indices, unsigned count, unsigned radius, unsigned ppb,
    unsigned* stats) {
    const unsigned slot = blockIdx.x * 16u + threadIdx.x / 16u;
    if (slot >= count) return;
    const unsigned index = indices[slot], token = index / 2048u, row = index % 2048u;
    const float exact = selected_hawkeye_wave16_dot_bf16_hopper(
        inputs + size_t(token) * 4096u, weights + size_t(row) * 4096u, 4096u);
    if (threadIdx.x & 15u) return;
    const auto endpoint = __float_as_uint(device_bf16_round_to_float(exact));
    const bool base_bad = __float_as_uint(device_bf16_round_to_float(control[index])) != endpoint;
    const bool shadow_bad = __float_as_uint(device_bf16_round_to_float(shadow[index])) != endpoint;
    const bool selected = selected_bf16_projection_hawkeye_candidate(
        before[index], index, 2048u, radius, 0u, ppb, nullptr, input_norm, weight_norm);
    atomicAdd(stats + 4, unsigned(base_bad));
    atomicAdd(stats + 5, unsigned(shadow_bad));
    atomicAdd(stats + 6, unsigned(shadow_bad && !selected));
    if (index == stats[3]) {
        stats[7] = selected;
        stats[8] = __float_as_uint(before[index]);
        stats[9] = __float_as_uint(shadow[index]);
        stats[10] = __float_as_uint(control[index]);
        stats[11] = __float_as_uint(exact);
        stats[12] = __float_as_uint(input_norm[token]);
        stats[13] = __float_as_uint(weight_norm[row]);
        stats[14] = 1u;
    }
}

inline hipError_t run(
    const uint16_t* weights, const uint16_t* inputs, const float* control,
    const float* input_norm, const float* weight_norm,
    unsigned rows, unsigned tokens, unsigned k, unsigned radius, unsigned ppb,
    unsigned maximum_blocks, hipStream_t stream, const std::string& stage) {
    const char* setting = std::getenv("QRT_QWEN36_Q8192_OUT_MATRIX_SHADOW_AUDIT");
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
    float* storage = nullptr; unsigned* metadata = nullptr;
    hipError_t status = hipMalloc(reinterpret_cast<void**>(&storage), 2u * surface * sizeof(float));
    if (status != hipSuccess) return status;
    status = hipMalloc(reinterpret_cast<void**>(&metadata), (counters + capacity + 2u * guard) * sizeof(unsigned));
    if (status != hipSuccess) { (void)hipFree(storage); return status; }
    float* shadow = storage + guard;
    float* before = storage + surface + guard;
    unsigned* stats = metadata + guard;
    unsigned* indices = stats + counters;
    std::array<unsigned, counters> host{}; host[3] = UINT_MAX;
    status = [&]() -> hipError_t {
        hipError_t result;
        if ((result = hipMemsetAsync(storage, 0xa5, 2u * surface * sizeof(float), stream)) != hipSuccess ||
            (result = hipMemsetAsync(metadata, 0xa5, (counters + capacity + 2u * guard) * sizeof(unsigned), stream)) != hipSuccess ||
            (result = hipMemcpyAsync(stats, host.data(), sizeof(host), hipMemcpyHostToDevice, stream)) != hipSuccess)
            return result;
        std::string failed_stage, failure;
        if (!resident_bf16_matrix_matmul_f32_output_with_heuristic_index(
                weights, inputs, shadow, rows, k, tokens, 4u, stream,
                stage + "_shadow_algorithm4", &failed_stage, &failure)) {
            std::fprintf(stderr, "BATCH_MARK out_matrix_shadow_failed stage=%s detail=%s\n",
                failed_stage.c_str(), failure.c_str());
            return hipErrorInvalidConfiguration;
        }
        if ((result = hipMemcpyAsync(before, shadow, size_t(cells) * sizeof(float), hipMemcpyDeviceToDevice, stream)) != hipSuccess)
            return result;
        if ((result = launch_selected_bf16_projection_hawkeye_midpoint_correction(
                weights, inputs, nullptr, input_norm, weight_norm, shadow,
                rows, tokens, k, radius, 0u, ppb, maximum_blocks, stream)) != hipSuccess)
            return result;
        hipLaunchKernelGGL(compare, dim3(4096u), dim3(256u), 0u, stream,
            control, shadow, cells, stats, indices);
        if ((result = hipGetLastError()) != hipSuccess ||
            (result = hipStreamSynchronize(stream)) != hipSuccess ||
            (result = hipMemcpy(host.data(), stats, sizeof(host), hipMemcpyDeviceToHost)) != hipSuccess)
            return result;
        const unsigned count = (std::min)(host[1], capacity);
        if (count) {
            hipLaunchKernelGGL(classify, dim3((count + 15u) / 16u), dim3(256u), 0u, stream,
                weights, inputs, control, shadow, before, input_norm, weight_norm,
                indices, count, radius, ppb, stats);
            if ((result = hipGetLastError()) != hipSuccess ||
                (result = hipStreamSynchronize(stream)) != hipSuccess ||
                (result = hipMemcpy(host.data(), stats, sizeof(host), hipMemcpyDeviceToHost)) != hipSuccess)
                return result;
        }
        if (host[14]) {
            std::vector<uint16_t> left(k), right(k);
            const size_t token = host[3] / rows, row = host[3] % rows;
            if ((result = hipMemcpy(left.data(), inputs + token * k, k * sizeof(uint16_t), hipMemcpyDeviceToHost)) != hipSuccess ||
                (result = hipMemcpy(right.data(), weights + row * k, k * sizeof(uint16_t), hipMemcpyDeviceToHost)) != hipSuccess)
                return result;
            const float cpu = qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(
                0.0f, left.data(), right.data(), k);
            std::memcpy(&host[15], &cpu, sizeof(cpu));
            if (host[15] != host[11]) return hipErrorInvalidValue;
        }
        std::array<unsigned, guard> redzone{};
        for (const void* address : {static_cast<const void*>(storage),
                static_cast<const void*>(shadow + cells), static_cast<const void*>(storage + surface),
                static_cast<const void*>(before + cells), static_cast<const void*>(metadata),
                static_cast<const void*>(indices + capacity)}) {
            if ((result = hipMemcpy(redzone.data(), address, sizeof(redzone), hipMemcpyDeviceToHost)) != hipSuccess)
                return result;
            for (auto word : redzone) if (word != 0xa5a5a5a5u) return hipErrorInvalidValue;
        }
        std::fprintf(stderr,
            "BATCH_MARK out_matrix_shadow stage=%s rows=%u tokens=%u k=%u control_algorithm=0 shadow_algorithm=4 "
            "cells=%u raw_bit_differences=%u bf16_differences=%u nonfinite=%u first_index=%u "
            "classified=%u classification_complete=%u control_canonical_errors=%u shadow_canonical_errors=%u "
            "shadow_admission_misses=%u first_selected=%u first_before_bits=%08x first_after_bits=%08x "
            "first_control_bits=%08x first_canonical_bits=%08x input_norm_bits=%08x weight_norm_bits=%08x "
            "first_detail_available=%u first_cpu_canonical_bits=%08x "
            "radius=%u ppb=%u redzones_pass=1 control_output_read_only=1 diagnostic_only=1 performance_acceptance=0\n",
            stage.c_str(), rows, tokens, k, cells, host[0], host[1], host[2], host[3], count,
            host[1] <= capacity, host[4], host[5], host[6], host[7], host[8], host[9],
            host[10], host[11], host[12], host[13], host[14], host[15], radius, ppb);
        std::fflush(stderr);
        return host[2] ? hipErrorInvalidValue : hipSuccess;
    }();
    if (status != hipSuccess) (void)hipStreamSynchronize(stream);
    const auto metadata_free = hipFree(metadata), storage_free = hipFree(storage);
    return status != hipSuccess ? status : metadata_free != hipSuccess ? metadata_free : storage_free;
#endif
}
} // namespace qrt_out_matrix_shadow
