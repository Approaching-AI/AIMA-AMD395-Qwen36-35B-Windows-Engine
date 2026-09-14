#pragma once
#include "sm121_scalar_projection.h"
#include "sm121_scaled_half_products.h"

// Component-only candidate replay. Each K16 operand group is independently
// lossless; one unsupported group does not reject a complete projection row.
namespace qrt_sm121_scaled_half_projection {
namespace half=qrt_sm121_scaled_half_products;
using Row=half::Row;
using Value=qrt_q1_moe_hawkeye::Value;
struct Stats { unsigned transformed=0u,original=0u; };

__global__ void prepare_rows(const uint16_t* input,Row* output,unsigned rows,unsigned width) {
    const size_t group=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(group>=size_t(rows)*(width/16u))return;
    uint16_t original[16];
    for(unsigned i=0u;i<16u;++i)original[i]=input[group*16u+i];
    output[group]=half::prepare(original);
}
__device__ __forceinline__ Value original_group(Value carry,const Row& left,const Row& right) {
    const unsigned first=(threadIdx.x&3u)*4u;uint32_t products[4];
#pragma unroll
    for(unsigned i=0u;i<4u;++i)products[i]=qrt_sm121_group16::pack_product(
        qrt_q1_moe_hawkeye::multiply_bf16(half::original(left,first+i),half::original(right,first+i),-133));
    return qrt_sm121_subgroup::accumulate_products<4u>(carry,products);
}
template<bool AllNonzero>
__device__ __forceinline__ Value transformed_group(Value carry,const Row& left,const Row& right,unsigned active) {
    const unsigned lane=threadIdx.x&3u;float products[4];uint32_t paired_maximum=0u;
#pragma unroll
    for(unsigned i=0u;i<2u;++i) {
        const unsigned pair=lane*2u+i;const uint32_t a=left.pairs[pair],b=right.pairs[pair];
        products[2u*i]=half::product<false>(a,b);products[2u*i+1u]=half::product<true>(a,b);
        uint32_t exponents=((a>>10u)&0x001f001fu)+((b>>10u)&0x001f001fu);
        if constexpr(!AllNonzero) {
            const unsigned bits=(active>>(2u*pair))&3u;
            exponents&=((bits&1u)|((bits&2u)<<15u))*65535u;
        }
        paired_maximum=qrt_sm121_prepared_integer_pairs::maximum_pair(paired_maximum,exponents);
    }
    int maximum=int((paired_maximum&65535u)>(paired_maximum>>16u)?paired_maximum&65535u:paired_maximum>>16u);
    maximum=qrt_sm121_lane_reduce::maximum<4u>(maximum);
    const int combined_unit=half::unit(left)+half::unit(right);
    maximum=maximum-30+combined_unit;
    maximum=maximum>carry.exponent?maximum:carry.exponent;maximum=maximum>-133?maximum:-133;
    const int power=25-maximum+combined_unit;
    uint32_t total=0u;
    // For active normal FP16 operands,maxHalf>=2 implies power<=53.
    // Products below the scale floor all align to integer zero exactly.
    if(power>=-126) {
        const float scale=qrt_sm121_float_alignment::from_bits(unsigned(127+power)<<23u);
#pragma unroll
        for(unsigned i=0u;i<4u;++i)total+=uint32_t(int32_t(products[i]*scale));
    }
    total=qrt_sm121_lane_reduce::sum<4u>(total);
    const unsigned shift=unsigned(maximum-carry.exponent);
    const uint32_t aligned=shift>=32u?0u:(carry.significand<<2u)>>shift;
    total+=carry.negative?0u-aligned:aligned;
    // Any product resolves the bounded modulo overlap: reaching that range
    // requires all16 original products to have the same sign.
    const bool negative=((left.pairs[lane*2u]^right.pairs[lane*2u])&0x8000u)!=0u;
    const auto sum=qrt_sm121_group16::decode_modulo_sum(total,negative);
    return qrt_sm121_canonical::normalize(sum.magnitude,sum.negative,maximum);
}
__device__ __forceinline__ Value accumulate_four(Value carry,const Row& left,const Row& right,bool* transformed=nullptr) {
    if(half::unit(left)==-32768 || half::unit(right)==-32768) {
        if(transformed)*transformed=false;
        return original_group(carry,left,right);
    }
    if(transformed)*transformed=true;
    const unsigned active=(left.control&right.control)>>16u;
    if(!active) {
        const int maximum=carry.exponent>-133?carry.exponent:-133;
        const unsigned shift=unsigned(maximum-carry.exponent);
        const uint32_t magnitude=shift>=32u?0u:(carry.significand<<2u)>>shift;
        return qrt_sm121_canonical::normalize(magnitude,magnitude && carry.negative,maximum);
    }
    return active==65535u?transformed_group<true>(carry,left,right,active):transformed_group<false>(carry,left,right,active);
}
template<unsigned Lanes,bool Audit=false>
__device__ __forceinline__ float dot(const Row* left,const Row* right,unsigned width,
    uint32_t* raw_trace=nullptr,Stats* stats=nullptr) {
    static_assert(Lanes==1u || Lanes==4u);
    const unsigned lane=threadIdx.x&(Lanes-1u);Value carry{0u,-133,false};Stats counts;
#pragma unroll 1
    for(unsigned group=0u;group<width/16u;++group) {
        bool transformed;
        if constexpr(Lanes==1u) {
            unsigned path=0u;carry=half::accumulate<true>(carry,left[group],right[group],Audit?&path:nullptr);
            if constexpr(Audit)transformed=path!=0u;
        }else carry=accumulate_four(carry,left[group],right[group],Audit?&transformed:nullptr);
        if constexpr(Audit) {
            counts.transformed+=transformed;counts.original+=!transformed;
            if(!lane && raw_trace) {
                raw_trace[3u*group]=carry.significand;
                raw_trace[3u*group+1u]=uint32_t(int32_t(carry.exponent));
                raw_trace[3u*group+2u]=unsigned(carry.negative);
            }
        }
    }
    if constexpr(Audit)if(!lane && stats)*stats=counts;
    return lane?0.0f:qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
} // namespace qrt_sm121_scaled_half_projection
