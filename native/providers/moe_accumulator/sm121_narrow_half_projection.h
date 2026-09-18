#pragma once
#include "sm121_staged_half_projection.h"
#include "sm121_narrow_half_carry.h"

namespace qrt_sm121_narrow_half_projection {
namespace staged=qrt_sm121_staged_half_projection;
namespace half=qrt_sm121_scaled_half_products;
namespace carry_math=qrt_sm121_narrow_half_carry;
namespace f32=qrt_sm121_f32_carry;
using Row=staged::Row;
struct Stats { unsigned accepted_groups=0u;bool restarted=false; };

// One128-thread CTA owns a complete original row. Flags are published only
// after checking every raw operand and every prepared K16 control. Launch
// this after lossless preparation on the same stream; the extra read is timed.
__global__ void classify_rows(const uint16_t* raw,const Row* prepared,unsigned* flags,
    unsigned rows,unsigned width) {
    const unsigned row=blockIdx.x;
    if(row>=rows)return;
    unsigned invalid=unsigned(!width || width>8192u || width%16u);
    for(unsigned i=threadIdx.x;i<width;i+=128u)
        invalid|=!carry_math::narrow::eligible(raw[size_t(row)*width+i]);
    for(unsigned g=threadIdx.x;g<width/16u;g+=128u)
        invalid|=int16_t(prepared[size_t(row)*(width/16u)+g].control)==-32768;
    __shared__ unsigned rejected[4];
    const unsigned wave=threadIdx.x/32u;
    const unsigned mask=__ballot(invalid!=0u);
    if(!(threadIdx.x&31u))rejected[wave]=mask;
    __syncthreads();
    if(!threadIdx.x)flags[row]=!(rejected[0]|rejected[1]|rejected[2]|rejected[3]);
}

template<bool AllNonzero>
__device__ __forceinline__ float accumulate(float carry,const staged::LaneOperands& p,unsigned active) {
    const unsigned lane=threadIdx.x&3u;
    float products[4];uint32_t paired_maximum=0u;
#pragma unroll
    for(unsigned i=0u;i<2u;++i){
        const uint32_t a=p.left[i],b=p.right[i];
        products[2u*i]=half::product<false>(a,b);products[2u*i+1u]=half::product<true>(a,b);
        uint32_t exponents=((a>>10u)&0x001f001fu)+((b>>10u)&0x001f001fu);
        if constexpr(!AllNonzero){
            const unsigned live=(active>>(lane*4u+2u*i))&3u;
            exponents&=((live&1u)|((live&2u)<<15u))*65535u;
        }
        paired_maximum=qrt_sm121_prepared_integer_pairs::maximum_pair(paired_maximum,exponents);
    }
    int maximum=int((paired_maximum&65535u)>(paired_maximum>>16u)?paired_maximum&65535u:paired_maximum>>16u);
    maximum=qrt_sm121_lane_reduce::maximum<4u>(maximum);
    const int unit=int(int16_t(p.left_control))+int(int16_t(p.right_control));
    maximum=maximum-30+unit;
    const int carry_exponent=int((f32::bits(carry)&0x7fffffffu)>>23u)-127;
    maximum=maximum>carry_exponent?maximum:carry_exponent;
    maximum=maximum>-89?maximum:-89;
    const int power=25-maximum+unit;uint32_t modulo=0u;
    if(power>=-126){
        const float scale=f32::alignment::from_bits(uint32_t(127+power)<<23u);
#pragma unroll
        for(unsigned i=0u;i<4u;++i)modulo+=uint32_t(int32_t(products[i]*scale));
    }
    modulo=qrt_sm121_lane_reduce::sum<4u>(modulo);
    const float carry_scale=f32::alignment::from_bits(uint32_t(152-maximum)<<23u);
    modulo+=uint32_t(int32_t(carry*carry_scale));
    const auto sum=qrt_sm121_group16::decode_modulo_sum(modulo,((p.left[0]^p.right[0])&0x8000u)!=0u);
    return carry_math::normalize(sum.magnitude,sum.negative,maximum);
}

// The caller combines both complete-row flags. Unsupported rows use original
// staged arithmetic for the entire dot, never a partially accepted endpoint.
template<unsigned StagingGroups,bool Audit=false>
__device__ __forceinline__ float dot(const Row* left,const Row* right,unsigned width,
    bool admitted,uint32_t* trace=nullptr,Stats* stats=nullptr) {
    static_assert(StagingGroups==2u || StagingGroups==4u);
    const unsigned lane=threadIdx.x&3u,groups=width/16u;
    if constexpr(Audit)if(!lane&&stats)*stats={admitted?groups:0u,!admitted};
    if(!admitted){
        if constexpr(Audit){
            staged::Value original{0u,-133,false};
            for(unsigned i=0u;i<groups;++i){
                original=staged::accumulate(original,staged::load(left[i],right[i]));
                if(!lane&&trace)trace[i]=f32::bits(qrt_q1_moe_hawkeye::value_to_float(original));
            }
            return lane?0.0f:qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(original));
        }else return staged::dot<2u>(left,right,width);
    }
    float carry=0.0f;
#pragma unroll 1
    for(unsigned base=0u;base<groups;base+=StagingGroups){
        staged::LaneOperands operands[StagingGroups];
        const unsigned count=groups-base<StagingGroups?groups-base:StagingGroups;
#pragma unroll
        for(unsigned i=0u;i<StagingGroups;++i)if(i<count)operands[i]=staged::load(left[base+i],right[base+i]);
#pragma unroll
        for(unsigned i=0u;i<StagingGroups;++i)if(i<count){
            const auto& p=operands[i];const unsigned active=(p.left_control&p.right_control)>>16u;
            if(active)carry=active==65535u?accumulate<true>(carry,p,active):accumulate<false>(carry,p,active);
            if constexpr(Audit)if(!lane&&trace)trace[base+i]=f32::bits(carry);
        }
    }
    return lane?0.0f:carry;
}
} // namespace qrt_sm121_narrow_half_projection
