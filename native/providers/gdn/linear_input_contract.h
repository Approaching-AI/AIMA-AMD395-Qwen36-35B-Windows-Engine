#pragma once
#include "sm121_silu_table.h"
#include "sm121_rsqrt_table.h"

namespace qrt_linear_input_preparation {
constexpr unsigned features = 8192u, qk_features = 2048u, v_features = 4096u;
constexpr unsigned threads = 256u, maximum_tokens = 8192u;

struct Inputs {
    const float* projected;
    size_t projected_cells;
    const uint16_t* weights;
    size_t weight_cells;
    const unsigned char* silu;
    size_t silu_bytes;
    const unsigned char* rsqrt;
    size_t rsqrt_bytes;
    unsigned tokens;
};
struct Outputs {
    uint16_t* q;
    uint16_t* k;
    uint16_t* v;
    size_t q_cells, k_cells, v_cells;
    float* raw = nullptr; // Optional diagnostics, never required by the owner.
    size_t raw_cells = 0u;
};

inline bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    const auto av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
    return av <= bv ? bv - av < an : av - bv < bn;
}

inline bool valid(const Inputs& in, const Outputs& out, unsigned first, unsigned count) {
    if (!in.projected || !in.weights || !in.silu || !in.rsqrt ||
        !out.q || !out.k || !out.v || !in.tokens || in.tokens > maximum_tokens ||
        !count || first >= in.tokens || count > in.tokens - first ||
        in.projected_cells < size_t(in.tokens) * features ||
        in.weight_cells < size_t(features) * 4u ||
        in.silu_bytes < qrt_sm121_silu::table_bytes ||
        in.rsqrt_bytes < qrt_sm121_rsqrt::table_bytes ||
        out.q_cells < size_t(count) * qk_features || out.k_cells < size_t(count) * qk_features ||
        out.v_cells < size_t(count) * v_features ||
        (out.raw && out.raw_cells < size_t(count) * features) || (!out.raw && out.raw_cells)) return false;
    const void* inputs[] = {in.projected, in.weights, in.silu, in.rsqrt};
    const size_t input_bytes[] = {size_t(in.tokens) * features * 4u, size_t(features) * 4u * 2u,
        qrt_sm121_silu::table_bytes, qrt_sm121_rsqrt::table_bytes};
    const unsigned input_alignment[] = {4u, 2u, 4u, 4u};
    const void* outputs[] = {out.q, out.k, out.v, out.raw};
    const size_t output_bytes[] = {size_t(count) * qk_features * 2u,
        size_t(count) * qk_features * 2u, size_t(count) * v_features * 2u,
        out.raw ? size_t(count) * features * 4u : 0u};
    for (unsigned i = 0u; i < 4u; ++i) {
        if (reinterpret_cast<uintptr_t>(inputs[i]) % input_alignment[i]) return false;
        if (!outputs[i]) continue;
        if (reinterpret_cast<uintptr_t>(outputs[i]) % (i == 3u ? 4u : 2u)) return false;
        for (unsigned j = 0u; j < 4u; ++j)
            if (overlaps(outputs[i], output_bytes[i], inputs[j], input_bytes[j])) return false;
        for (unsigned j = i + 1u; j < 4u; ++j)
            if (outputs[j] && overlaps(outputs[i], output_bytes[i], outputs[j], output_bytes[j])) return false;
    }
    return true;
}

} // namespace qrt_linear_input_preparation
