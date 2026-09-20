#pragma once
#include "sm121_q2_recurrent_layout.h"
#include "sm121_q2_convolution_math.h"

namespace qrt_sm121_q2 {
template<class Element> struct ConvolutionViews {
    const uint16_t* qkv = nullptr;          // [2,8192], BF16 projection output.
    const Element* initial_ring = nullptr; // [4,8192], absolute-position modulo.
    const uint16_t* weights = nullptr;     // [8192,4], original model tensor.
    const unsigned char* silu = nullptr;
    Element* staged_rings = nullptr;       // [2,4,8192], same element kind.
    uint16_t* staged_convolution = nullptr;// [2,8192], consumed by recurrence.
    size_t first_position = 0;
};

template<class Element> inline bool valid_convolution_views(const ConvolutionViews<Element>& view) {
    static_assert(std::is_same<Element,float>::value || std::is_same<Element,uint16_t>::value,"native ring element");
    using recurrent_detail::Span;
    const Span writes[] = {{view.staged_rings,2u*ring_elements*sizeof(Element),alignof(Element)},
                           {view.staged_convolution,2u*8192u*sizeof(uint16_t),alignof(uint16_t)}};
    const Span reads[] = {{view.qkv,2u*8192u*sizeof(uint16_t),alignof(uint16_t)},
        {view.initial_ring,ring_elements*sizeof(Element),alignof(Element)},
        {view.weights,8192u*4u*sizeof(uint16_t),alignof(uint16_t)},
        {view.silu,qrt_sm121_silu::table_bytes,1u}};
    if(view.first_position>(std::numeric_limits<size_t>::max)()-2u ||
        !recurrent_detail::disjoint(writes[0],writes[1]))return false;
    for(const auto& write:writes)for(const auto& read:reads)
        if(!recurrent_detail::disjoint(write,read))return false;
    return true;
}

template<class Element> inline bool valid_linear_views(const ConvolutionViews<Element>& conv,
    const RecurrentViews& recurrence,const RecurrentTables& tables) {
    if(!valid_convolution_views(conv) || !valid_recurrent_views(recurrence,tables) ||
        conv.staged_convolution!=recurrence.convolution)return false;
    using recurrent_detail::Span;
    const Span conv_writes[]={{conv.staged_rings,2u*ring_elements*sizeof(Element),alignof(Element)},
        {conv.staged_convolution,2u*8192u*sizeof(uint16_t),alignof(uint16_t)}};
    const Span recurrent_writes[]={{recurrence.staged_states,staged_state_bytes,alignof(float)},
        {recurrence.staged_core,staged_core_bytes,alignof(uint16_t)}};
    const Span recurrent_reads[]={{recurrence.a,2u*32u*sizeof(uint16_t),alignof(uint16_t)},
        {recurrence.b,2u*32u*sizeof(uint16_t),alignof(uint16_t)},
        {recurrence.initial_state,state_elements*sizeof(float),alignof(float)},
        {tables.g,32u*65536u*sizeof(float),alignof(float)},
        {tables.beta,65536u*sizeof(float),alignof(float)},
        {tables.exp2,qrt_sm121_exp2::table_bytes,1u},{tables.rsqrt,qrt_sm121_rsqrt::table_bytes,1u}};
    const Span conv_reads[]={{conv.qkv,2u*8192u*sizeof(uint16_t),alignof(uint16_t)},
        {conv.initial_ring,ring_elements*sizeof(Element),alignof(Element)},
        {conv.weights,8192u*4u*sizeof(uint16_t),alignof(uint16_t)},
        {conv.silu,qrt_sm121_silu::table_bytes,1u}};
    for(const auto& write:conv_writes) {
        for(const auto& read:recurrent_reads)if(!recurrent_detail::disjoint(write,read))return false;
        for(const auto& other:recurrent_writes)if(!recurrent_detail::disjoint(write,other))return false;
    }
    for(const auto& write:recurrent_writes)for(const auto& read:conv_reads)
        if(!recurrent_detail::disjoint(write,read))return false;
    return true;
}

template<class Element> struct LinearSelection {
    const float* state = nullptr;
    const Element* ring = nullptr;
    unsigned rows = 0;
};

// A borrowed selection after caller-verified completion. No copy or resident
// publication occurs here; the enclosing target transaction supplies extent.
template<class Element> inline LinearSelection<Element> accepted_linear(
    const ConvolutionViews<Element>& conv,const RecurrentViews& recurrence,unsigned rows) {
    const auto* state=accepted_state(recurrence,rows);
    if(!state || !conv.staged_rings || conv.staged_convolution!=recurrence.convolution)return {};
    return {state,conv.staged_rings+size_t(rows-1u)*ring_elements,rows};
}
} // namespace qrt_sm121_q2
