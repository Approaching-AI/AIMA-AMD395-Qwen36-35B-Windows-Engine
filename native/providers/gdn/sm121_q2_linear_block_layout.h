#pragma once
#include "sm121_q2_linear_layout.h"

namespace qrt_sm121_q2 {
template<class Element> struct LinearBlockViews {
    const uint16_t* normalized_input = nullptr; // [2,2048]
    const uint16_t* qkv_weights = nullptr;      // [8192,2048]
    const uint16_t* z_weights = nullptr;        // [4096,2048]
    const uint16_t* a_weights = nullptr;        // [32,2048]
    const uint16_t* b_weights = nullptr;        // [32,2048]
    const uint16_t* output_weights = nullptr;   // [2048,4096]
    const uint16_t* norm_weights = nullptr;     // [128]
    uint16_t* qkv = nullptr;                    // [2,8192]
    uint16_t* z = nullptr;                      // [2,4096]
    uint16_t* a = nullptr;                      // [2,32]
    uint16_t* b = nullptr;                      // [2,32]
    uint16_t* gated = nullptr;                  // [2,4096]
    uint16_t* output = nullptr;                 // [2,2048]
    ConvolutionViews<Element> convolution;
    RecurrentViews recurrent;
};

struct LinearBlockTables {
    RecurrentTables recurrent;
    const float* gated_silu = nullptr; // [65536], original BF16-domain FP32 SiLU.
};

template<class Element> inline bool valid_linear_block(const LinearBlockViews<Element>& v,
                                                       const LinearBlockTables& t) {
    if (!valid_linear_views(v.convolution, v.recurrent, t.recurrent) ||
        v.convolution.qkv != v.qkv || v.recurrent.a != v.a || v.recurrent.b != v.b)
        return false;
    using recurrent_detail::Span;
    const Span writes[] = {
        {v.qkv, 2u*8192u*2u, 2u}, {v.z, 2u*4096u*2u, 2u},
        {v.a, 2u*32u*2u, 2u}, {v.b, 2u*32u*2u, 2u},
        {v.gated, 2u*4096u*2u, 2u}, {v.output, 2u*2048u*2u, 2u},
        {v.convolution.staged_rings, 2u*ring_elements*sizeof(Element), alignof(Element)},
        {v.convolution.staged_convolution, 2u*8192u*2u, 2u},
        {v.recurrent.staged_states, staged_state_bytes, alignof(float)},
        {v.recurrent.staged_core, staged_core_bytes, 2u}
    };
    const Span reads[] = {
        {v.normalized_input, 2u*2048u*2u, 2u},
        {v.qkv_weights, 8192u*2048u*2u, 2u}, {v.z_weights, 4096u*2048u*2u, 2u},
        {v.a_weights, 32u*2048u*2u, 2u}, {v.b_weights, 32u*2048u*2u, 2u},
        {v.output_weights, 2048u*4096u*2u, 2u}, {v.norm_weights, 128u*2u, 2u},
        {v.convolution.initial_ring, ring_elements*sizeof(Element), alignof(Element)},
        {v.convolution.weights, 8192u*4u*2u, 2u},
        {v.convolution.silu, qrt_sm121_silu::table_bytes, 1u},
        {v.recurrent.initial_state, state_elements*sizeof(float), alignof(float)},
        {t.recurrent.g, 32u*65536u*sizeof(float), alignof(float)},
        {t.recurrent.beta, 65536u*sizeof(float), alignof(float)},
        {t.recurrent.exp2, qrt_sm121_exp2::table_bytes, 1u},
        {t.recurrent.rsqrt, qrt_sm121_rsqrt::table_bytes, 1u},
        {t.gated_silu, 65536u*sizeof(float), alignof(float)}
    };
    for (size_t i = 0; i < sizeof(writes)/sizeof(writes[0]); ++i) {
        for (size_t j = 0; j < i; ++j)
            if (!recurrent_detail::disjoint(writes[i], writes[j])) return false;
        for (const auto& read : reads)
            if (!recurrent_detail::disjoint(writes[i], read)) return false;
    }
    return true;
}
} // namespace qrt_sm121_q2
