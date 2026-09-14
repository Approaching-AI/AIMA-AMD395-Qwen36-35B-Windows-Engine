#ifndef QRT_DECODED_WINDOW_QK_H
#define QRT_DECODED_WINDOW_QK_H
#include "blackwell_attention.h"
#include "../moe_accumulator/sm121_decoded_bf16.h"

// Component experiment: the product dispatcher does not include this header.
namespace qrt_decoded_window_qk {
namespace decoded = qrt_sm121_decoded_bf16;

// An exceptional staged tile or carried value restarts the complete original
// integer dot. Keep its temporaries outside the fast window's register scope.
__device__ __attribute__((noinline)) float raw_dot(const uint16_t* query,
    const uint16_t* key, unsigned key_stride) {
    qrt_q1_moe_hawkeye::Value carry{0u,qrt_blackwell_attention::kBlackwellZeroExponent,false};
    for (unsigned base=0u;base<qrt_blackwell_attention::kHeadDim;base+=16u) {
        uint32_t products[16];
#pragma unroll
        for (unsigned i=0u;i<16u;++i)
            products[i]=qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(
                query[base+i],key[size_t(base+i)*key_stride],qrt_blackwell_attention::kBlackwellZeroExponent));
        const auto sum=qrt_sm121_group16::sum_packed(carry,products);
        carry=qrt_sm121_wave16::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
    }
    return qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry))*qrt_blackwell_attention::kExactScale;
}

template<unsigned Window, unsigned QueryRows=16u, unsigned KeyColumns=16u>
__global__ void scores(const uint16_t* query,const uint16_t* transposed_key,float* output,
    unsigned query_start,unsigned query_count,unsigned stride,unsigned key_stride) {
    static_assert(Window && Window%16u==0u && qrt_blackwell_attention::kHeadDim%Window==0u);
    static_assert(QueryRows*KeyColumns==qrt_blackwell_attention::kThreads);
    __shared__ float qvalues[QueryRows][Window],kvalues[Window][KeyColumns];
    __shared__ int16_t qexponents[QueryRows][Window],kexponents[Window][KeyColumns];
    __shared__ unsigned invalid_tile;
    const unsigned head=blockIdx.y,kv_head=head/(qrt_blackwell_attention::kQueryHeads/qrt_blackwell_attention::kKvHeads);
    const unsigned query_tile=blockIdx.z*QueryRows,key_tile=blockIdx.x*KeyColumns;
    const unsigned qr=threadIdx.x/KeyColumns,kc=threadIdx.x%KeyColumns;
    const unsigned row=query_tile+qr,key=key_tile+kc;
    const bool live=row<query_count && key<stride;
    const bool active=live && key<=query_start+row;
    const size_t output_cell=(size_t(row)*qrt_blackwell_attention::kQueryHeads+head)*stride+key;
    const unsigned last_query=query_start+min(query_tile+QueryRows,query_count)-1u;
    if (key_tile>last_query) {
        if (live) output[output_cell]=-INFINITY;
        return;
    }
    if (!threadIdx.x) invalid_tile=0u;
    __syncthreads();
    qrt_q1_moe_hawkeye::Value carry{0u,qrt_blackwell_attention::kBlackwellZeroExponent,false};
    bool fallback=false;
    for (unsigned window=0u;window<qrt_blackwell_attention::kHeadDim;window+=Window) {
        bool invalid=false;
        for (unsigned cell=threadIdx.x;cell<QueryRows*Window;cell+=qrt_blackwell_attention::kThreads) {
            const unsigned r=cell/Window,c=cell%Window;
            const uint16_t x=query_tile+r<query_count
                ? query[(size_t(query_start+query_tile+r)*qrt_blackwell_attention::kQueryHeads+head)*qrt_blackwell_attention::kHeadDim+window+c] : 0u;
            invalid=invalid || !qrt_sm121_float_alignment::eligible(x);
            qvalues[r][c]=decoded::value(x);qexponents[r][c]=decoded::exponent(x);
        }
        for (unsigned cell=threadIdx.x;cell<Window*KeyColumns;cell+=qrt_blackwell_attention::kThreads) {
            const unsigned r=cell/KeyColumns,c=cell%KeyColumns;
            const uint16_t x=key_tile+c<stride
                ? transposed_key[(size_t(kv_head)*qrt_blackwell_attention::kHeadDim+window+r)*key_stride+key_tile+c] : 0u;
            invalid=invalid || !qrt_sm121_float_alignment::eligible(x);
            kvalues[r][c]=decoded::value(x);kexponents[r][c]=decoded::exponent(x);
        }
        if (invalid) atomicOr(&invalid_tile,1u);
        __syncthreads();
        // This branch is uniform for the whole CTA; inactive cells have not
        // left any earlier barrier. Late invalid windows discard partial work.
        if (invalid_tile) {
            if (live) output[output_cell]=active ? raw_dot(
                query+(size_t(query_start+row)*qrt_blackwell_attention::kQueryHeads+head)*qrt_blackwell_attention::kHeadDim,
                transposed_key+size_t(kv_head)*qrt_blackwell_attention::kHeadDim*key_stride+key,key_stride) : -INFINITY;
            return;
        }
        if (active && !fallback) {
            for (unsigned base=0u;base<Window;base+=16u) {
                qrt_sm121_float_alignment::Group group;
#pragma unroll
                for (unsigned i=0u;i<16u;++i)
                    decoded::set(group,i,qvalues[qr][base+i],kvalues[base+i][kc],
                        qexponents[qr][base+i],kexponents[base+i][kc]);
                qrt_sm121_group16::AlignedSum sum;
                if (!qrt_sm121_float_alignment::sum(carry,group,&sum)) { fallback=true;break; }
                carry=qrt_sm121_wave16::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
            }
        }
        __syncthreads();
    }
    if (live) output[output_cell]=!active ? -INFINITY : fallback ? raw_dot(
        query+(size_t(query_start+row)*qrt_blackwell_attention::kQueryHeads+head)*qrt_blackwell_attention::kHeadDim,
        transposed_key+size_t(kv_head)*qrt_blackwell_attention::kHeadDim*key_stride+key,key_stride)
        : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry))*qrt_blackwell_attention::kExactScale;
}
} // namespace qrt_decoded_window_qk
#endif
