// Included after the CK loader. This experiment preserves full-prefix QKV
// production and resident KV capture. Only layer39 attention queries whose
// outputs have no consumer are removed; all later row-local operators retain
// their current implementation and allocation shape.
#pragma once

namespace qrt_final_query_liveness {

inline thread_local bool active = false;

struct Scope {
    const bool previous;
    explicit Scope(bool enabled) : previous(active) { active = enabled; }
    ~Scope() { active = previous; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

inline hipError_t launch(const uint16_t *q, const uint16_t *k,
    const uint16_t *v, float *output, unsigned tokens) {
    if (!active || tokens != 8192u || !q || !k || !v || !output)
        return hipErrorInvalidValue;
#ifndef _WIN32
    return hipErrorNotSupported;
#else
    using Launch = int (__cdecl *)(const uint16_t *, const uint16_t *,
        const uint16_t *, const uint16_t *, const uint16_t *, float *,
        void *, unsigned, unsigned);
    Launch suffix_launch = nullptr;
    {
        auto &provider = ck_fmha_provider_state();
        std::lock_guard<std::mutex> lock(provider.mutex);
        if (provider.module)
            suffix_launch = reinterpret_cast<Launch>(GetProcAddress(
                provider.module, "qrt_ck_fmha_sm121_suffix_bf16_v1"));
    }
    if (!suffix_launch) return hipErrorNotSupported;
    constexpr size_t query_features = 4096u;
    constexpr size_t kv_features = 512u;
    const size_t last = tokens - 1u;
    // Defined zero contexts keep the unchanged full-shape gate/OUT/MoE
    // consumers from reading uninitialized storage. Their earlier rows have
    // no model-output consumer in this scope and are not valid hidden states.
    auto status = hipMemsetAsync(output, 0,
        size_t(tokens) * query_features * sizeof(float), nullptr);
    if (status != hipSuccess) return status;
    status = static_cast<hipError_t>(suffix_launch(
        q + last * query_features, k, v,
        k + last * kv_features, v + last * kv_features,
        output + last * query_features, nullptr, tokens - 1u, 1u));
    if (status == hipSuccess)
        std::cerr << "BATCH_MARK final_query_liveness layer=39 tokens=8192"
            << " query_start=8191 query_count=1 full_prefix_qkv=1"
            << " original_kv_capture=1 unused_contexts_zeroed=1"
            << " downstream_full_shape=1 suffix_workspace_bytes=83886080"
            << " gb10_boundary_required=1" << std::endl;
    return status;
#endif
}

} // namespace qrt_final_query_liveness
