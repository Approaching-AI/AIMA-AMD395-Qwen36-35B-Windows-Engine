#pragma once
#include "sm121_q1_math.h"
#include "sm121_silu_table.h"
#include <type_traits>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_Q2_CONV_INLINE __host__ __device__ __forceinline__
#else
#define QRT_Q2_CONV_INLINE inline
#endif

namespace qrt_sm121_q2 {
constexpr size_t ring_elements = 4u * 8192u;
QRT_Q2_CONV_INLINE float ring_float(float value) { return value; }
QRT_Q2_CONV_INLINE float ring_float(uint16_t value) { return qrt_sm121_q1::widen(value); }
template<class Element> QRT_Q2_CONV_INLINE Element ring_element(uint16_t value) {
    static_assert(std::is_same<Element,float>::value || std::is_same<Element,uint16_t>::value,"native ring element");
    if constexpr(std::is_same<Element,float>::value) return qrt_sm121_q1::widen(value);
    else return value;
}

// The native ring always uses absolute-position modulo four. Rejected draft
// rows occupy only the private second result; the input ring is never changed.
template<class Element> QRT_Q2_CONV_INLINE void convolution_feature(
    const uint16_t* qkv, const Element* initial_ring, const uint16_t* weights,
    const unsigned char* silu, size_t first_position, unsigned feature,
    Element* staged_rings, uint16_t* staged_convolution) {
    using namespace qrt_sm121_q1;
    Element history[4];
    for(unsigned slot=0;slot<4u;++slot) history[slot]=initial_ring[slot*8192u+feature];
    for(unsigned row=0;row<2u;++row) {
        const size_t position=first_position+row;
        float sum=0;
        for(unsigned tap=0;tap<4u;++tap) {
            const unsigned age=3u-tap;
            if(position<age) continue;
            const size_t source=position-age;
            const float x=source==position ? widen(qkv[row*8192u+feature]) : ring_float(history[source%4u]);
            sum=add(sum,widen(bf16(multiply(widen(bf16(x)),widen(weights[feature*4u+tap])))));
        }
        staged_convolution[row*8192u+feature]=qrt_sm121_silu::evaluate(silu,sum);
        history[position%4u]=ring_element<Element>(qkv[row*8192u+feature]);
        for(unsigned slot=0;slot<4u;++slot)
            staged_rings[(row*4u+slot)*8192u+feature]=history[slot];
    }
}
} // namespace qrt_sm121_q2
#undef QRT_Q2_CONV_INLINE
