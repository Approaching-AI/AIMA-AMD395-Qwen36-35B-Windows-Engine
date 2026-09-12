// Included after the resident session types and provider loaders. This scope
// borrows only the transaction's mutable shadow; the resident owner is never
// replaced by the temporary prefill session used for suffix orchestration.
#pragma once

__global__ void qwen36_prefix_suffix_halo_kernel(
    const float *qkv, const float *ring_f32, const uint16_t *ring_bf16,
    float *halo, unsigned prefix, unsigned tokens
) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(tokens + 3u) * 8192u) return;
    const unsigned row = unsigned(i / 8192u), channel = unsigned(i % 8192u);
    if (row >= 3u) halo[i] = qkv[i - 3u * 8192u];
    else {
        const size_t source = size_t((prefix - 3u + row) % 4u) * 8192u + channel;
        halo[i] = ring_bf16 ? device_bf16_to_float(ring_bf16[source]) : ring_f32[source];
    }
}

__global__ void qwen36_prefix_suffix_ring_kernel(
    const float *qkv, float *ring_f32, uint16_t *ring_bf16,
    unsigned prefix, unsigned tokens
) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 4u * 8192u) return;
    const unsigned row = tokens - 4u + i / 8192u, channel = i % 8192u;
    const size_t destination = size_t((prefix + row) % 4u) * 8192u + channel;
    const float value = qkv[size_t(row) * 8192u + channel];
    if (ring_bf16) ring_bf16[destination] = device_float_to_bf16(value);
    else ring_f32[destination] = value;
}

__global__ void qwen36_prefix_suffix_canonical_state_kernel(
    const float *source, float *destination, bool key_major
) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 32u * 128u * 128u) return;
    const unsigned base = i / 16384u * 16384u;
    destination[i] = source[key_major ? base + (i % 128u) * 128u + (i % 16384u) / 128u : i];
}

struct ScopedQwen36PrefixBatchSuffix {
    inline static thread_local ScopedQwen36PrefixBatchSuffix *active = nullptr;
    ScopedQwen36PrefixBatchSuffix *previous;
    Qwen36ResidentSessionState *session;
    unsigned prefix, tokens;
    uint64_t convolution_layers = 0u, recurrent_layers = 0u, attention_layers = 0u;
    float *halo = nullptr;
    uint32_t *indices = nullptr;
    std::array<uint32_t, 2> terminal_ids{};
    std::array<float, 2> terminal_logits{};
    bool terminal_valid = false;
    std::string failure;

    ScopedQwen36PrefixBatchSuffix(Qwen36ResidentSessionState *owner, unsigned before, unsigned count)
        : previous(active), session(owner), prefix(before), tokens(count) { active = this; }
    ~ScopedQwen36PrefixBatchSuffix() {
        // The caller synchronizes before publishing counters. Drain even a
        // partial enqueue before freeing the shared convolution workspace.
        (void)hipStreamSynchronize(nullptr);
        if (indices) (void)hipFree(indices);
        if (halo) (void)hipFree(halo);
        active = previous;
    }
    ScopedQwen36PrefixBatchSuffix(const ScopedQwen36PrefixBatchSuffix&) = delete;
    ScopedQwen36PrefixBatchSuffix& operator=(const ScopedQwen36PrefixBatchSuffix&) = delete;

    bool reject(const char *reason) { failure = reason; return false; }
    bool check(hipError_t status) {
        if (status == hipSuccess) return true;
        failure = hipGetErrorString(status); return false;
    }
    bool validate() {
        if (previous || !session || !session->valid || !session->owner_engine ||
            session->prefix_tokens != prefix || session->committed_decode_token_count ||
            prefix < 8192u || prefix % 8192u || prefix > 31744u || tokens != 1024u)
            return reject("batch suffix requires an untouched aligned prefix shadow and 1024 actual inputs");
        for (unsigned i = 0; i < 40u; ++i) {
            if (i % 4u != 3u) {
                const auto &layer = session->linear_layers[i];
                const bool bf16 = layer.qkv_element_kind == Qwen36ResidentSessionElementKind::kBf16;
                if (!layer.valid || !layer.device_recurrent_state || !layer.device_qkv_ring ||
                    layer.prefix_tokens != prefix || layer.decode_qkv_token_count ||
                    layer.decode_recurrent_token_count || layer.recurrent_state_bytes != 524288u * 4u ||
                    (!bf16 && layer.qkv_element_kind != Qwen36ResidentSessionElementKind::kF32) ||
                    layer.qkv_ring_bytes != 4u * 8192u * (bf16 ? 2u : 4u))
                    return reject("batch suffix linear state or convolution ring has incompatible layout");
            } else {
                const auto &layer = session->full_attention_layers[i];
                if (!layer.valid || !layer.device_k || !layer.device_v ||
                    !layer.device_decode_tail_k || !layer.device_decode_tail_v ||
                    layer.element_kind != Qwen36ResidentSessionElementKind::kBf16 ||
                    layer.history_tokens != prefix || layer.decode_tail_token_count ||
                    layer.decode_tail_capacity_tokens < tokens ||
                    layer.k_bytes < size_t(prefix) * 1024u || layer.v_bytes < size_t(prefix) * 1024u ||
                    layer.decode_tail_k_bytes < size_t(tokens) * 1024u ||
                    layer.decode_tail_v_bytes < size_t(tokens) * 1024u)
                    return reject("batch suffix attention requires BF16 prefix KV and a separate writable tail");
            }
        }
        return true;
    }

    bool convolution(unsigned layer_index, const float *qkv, const uint16_t *weights,
        float *output, unsigned count, unsigned arithmetic, const uint32_t *silu_keys,
        const uint16_t *silu_values, unsigned silu_probe, const unsigned char *silu_table) {
        if (layer_index >= 40u || layer_index % 4u == 3u || count != tokens ||
            !qkv || !weights || !output || !silu_table ||
            (convolution_layers & (UINT64_C(1) << layer_index)))
            return reject("batch suffix convolution escaped its original-input layer scope");
        auto &layer = session->linear_layers[layer_index];
        if (!halo) {
            if (!check(hipMalloc(reinterpret_cast<void **>(&halo), size_t(tokens + 3u) * 8192u * 4u)) ||
                !check(hipMalloc(reinterpret_cast<void **>(&indices), size_t(tokens) * 4u * 4u))) return false;
            std::array<uint32_t, 4096> host_indices{};
            for (unsigned row = 0; row < tokens; ++row)
                for (unsigned tap = 0; tap < 4u; ++tap) host_indices[row * 4u + tap] = row + tap;
            if (!check(hipMemcpy(indices, host_indices.data(), host_indices.size() * 4u, hipMemcpyHostToDevice))) return false;
        }
        const bool bf16 = layer.qkv_element_kind == Qwen36ResidentSessionElementKind::kBf16;
        auto *ring_f32 = bf16 ? nullptr : static_cast<float *>(layer.device_qkv_ring);
        auto *ring_bf16 = bf16 ? static_cast<uint16_t *>(layer.device_qkv_ring) : nullptr;
        hipLaunchKernelGGL(qwen36_prefix_suffix_halo_kernel,
            dim3((size_t(tokens + 3u) * 8192u + 255u) / 256u), dim3(256), 0, 0,
            qkv, ring_f32, ring_bf16, halo, prefix, tokens);
        if (!check(hipGetLastError())) return false;
        hipLaunchKernelGGL(selected_conv_qkv_window_kernel, dim3(8192u / 256u, tokens), dim3(256), 0, 0,
            halo, weights, indices, output, tokens, arithmetic, silu_keys, silu_values, silu_probe, silu_table);
        if (!check(hipGetLastError())) return false;
        hipLaunchKernelGGL(qwen36_prefix_suffix_ring_kernel, dim3(4u * 8192u / 256u), dim3(256), 0, 0,
            qkv, ring_f32, ring_bf16, prefix, tokens);
        if (!check(hipGetLastError())) return false;
        layer.decode_qkv_token_count = tokens;
        convolution_layers |= UINT64_C(1) << layer_index;
        return true;
    }

    int recurrent(unsigned layer_index, const float *raw, const float *gates,
        float *output, float *canonical_state, int decay, int count) {
        if (layer_index >= 40u || layer_index % 4u == 3u || count != int(tokens) || decay ||
            !raw || !gates || !output || !(convolution_layers & (UINT64_C(1) << layer_index)) ||
            (recurrent_layers & (UINT64_C(1) << layer_index)))
            return reject("batch suffix FLA requires one matching convolution and raw log gates");
#ifndef _WIN32
        return reject("batch suffix FLA requires Windows");
#else
        auto &layer = session->linear_layers[layer_index];
        qrt_fla_checkpoint::SeededLaunch launch = nullptr;
        {
            auto &provider = fla_chunk_gdn_dynamic_provider_state();
            std::lock_guard<std::mutex> lock(provider.mutex);
            launch = layer.recurrent_state_key_major ? provider.seeded_key_major_launch : provider.seeded_launch;
        }
        if (!launch) return reject("FLA provider lacks the original FP32 seeded state interface");
        if (!launch(raw, gates, output, layer.device_recurrent_state, 0, nullptr, count)) {
            failure = fla_chunk_gdn_dynamic_provider_last_error(); return 0;
        }
        if (canonical_state) {
            hipLaunchKernelGGL(qwen36_prefix_suffix_canonical_state_kernel,
                dim3(524288u / 256u), dim3(256), 0, 0,
                layer.device_recurrent_state, canonical_state, layer.recurrent_state_key_major);
            if (!check(hipGetLastError())) return 0;
        }
        layer.decode_recurrent_token_count = tokens;
        recurrent_layers |= UINT64_C(1) << layer_index;
        return 1;
#endif
    }

    hipError_t attention(unsigned layer_index, const uint16_t *q, const uint16_t *k,
        const uint16_t *v, float *output, unsigned count) {
        if (layer_index >= 40u || layer_index % 4u != 3u || count != tokens ||
            !q || !k || !v || !output || (attention_layers & (UINT64_C(1) << layer_index))) {
            reject("batch suffix attention escaped its original-input layer scope"); return hipErrorInvalidValue;
        }
#ifndef _WIN32
        reject("batch suffix attention requires Windows"); return hipErrorNotSupported;
#else
        using Launch = int (*)(const uint16_t *, const uint16_t *, const uint16_t *,
            const uint16_t *, const uint16_t *, float *, void *, unsigned, unsigned);
        Launch launch = nullptr;
        {
            auto &provider = ck_fmha_provider_state();
            std::lock_guard<std::mutex> lock(provider.mutex);
            if (provider.module && provider.prepared)
                launch = reinterpret_cast<Launch>(GetProcAddress(provider.module, "qrt_ck_fmha_sm121_suffix_bf16_v1"));
        }
        if (!launch) { reject("CK provider lacks exact compact suffix attention"); return hipErrorNotSupported; }
        auto &layer = session->full_attention_layers[layer_index];
        hipError_t status = hipMemcpyAsync(layer.device_decode_tail_k, k, size_t(tokens) * 1024u,
            hipMemcpyDeviceToDevice, nullptr);
        if (status == hipSuccess) status = hipMemcpyAsync(layer.device_decode_tail_v, v, size_t(tokens) * 1024u,
            hipMemcpyDeviceToDevice, nullptr);
        if (status == hipSuccess) status = hipError_t(launch(q,
            static_cast<const uint16_t *>(layer.device_k), static_cast<const uint16_t *>(layer.device_v),
            k, v, output, nullptr, prefix, tokens));
        if (status != hipSuccess) { (void)hipStreamSynchronize(nullptr); check(status); return status; }
        layer.decode_tail_token_count = tokens;
        attention_layers |= UINT64_C(1) << layer_index;
        return hipSuccess;
#endif
    }

    bool complete() const {
        return failure.empty() && convolution_layers == UINT64_C(0x7777777777) &&
            recurrent_layers == UINT64_C(0x7777777777) && attention_layers == UINT64_C(0x8888888888) &&
            terminal_valid;
    }
};
