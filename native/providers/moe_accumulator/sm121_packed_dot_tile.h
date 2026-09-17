#pragma once
#include "sm121_float_alignment.h"
#include "sm121_canonical_normalize.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#include "sm121_wave16.h"
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_DOT_TILE_INLINE __host__ __device__ __forceinline__
#define QRT_DOT_TILE_UNROLL _Pragma("unroll")
#else
#define QRT_DOT_TILE_INLINE inline
#define QRT_DOT_TILE_UNROLL
#endif

// Component-only dataflow experiment. Several independent ordered dots share
// each packed operand read. There is no change to an individual K16 group,
// its carry, classification, original exceptional replay or final conversion.
namespace qrt_sm121_packed_dot_tile {
namespace original=qrt_q1_moe_hawkeye;
namespace aligned=qrt_sm121_float_alignment;
template<unsigned Rows,unsigned Columns> struct Result {float values[Rows][Columns];};

template<unsigned Width,unsigned LeftPitch,unsigned RightPitch,
    unsigned Rows,unsigned Columns,bool Audit=false>
QRT_DOT_TILE_INLINE Result<Rows,Columns> dot(const uint32_t* left,
    const uint32_t* right,const unsigned* left_flags,const unsigned* right_flags,
    original::Value* trace=nullptr) {
    static_assert(Width && Width%16u==0u && LeftPitch>=Width/2u && RightPitch>=Columns);
    static_assert((Rows==1u || Rows==2u) && (Columns==1u || Columns==2u));
    original::Value carry[Rows][Columns];
    QRT_DOT_TILE_UNROLL
    for(unsigned r=0u;r<Rows;++r) QRT_DOT_TILE_UNROLL for(unsigned c=0u;c<Columns;++c)
        carry[r][c]={0u,-133,false};
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
    for(unsigned base=0u;base<Width;base+=16u){
        aligned::Group groups[Rows][Columns];
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for(unsigned i=0u;i<16u;i+=2u){
            uint32_t a[Rows],b[Columns];
            QRT_DOT_TILE_UNROLL
            for(unsigned r=0u;r<Rows;++r)a[r]=left[r*LeftPitch+(base+i)/2u];
            QRT_DOT_TILE_UNROLL
            for(unsigned c=0u;c<Columns;++c)b[c]=right[((base+i)/2u)*RightPitch+c];
            QRT_DOT_TILE_UNROLL
            for(unsigned r=0u;r<Rows;++r) QRT_DOT_TILE_UNROLL for(unsigned c=0u;c<Columns;++c)
                if(left_flags[r] && right_flags[c]){
                    groups[r][c].set(i,uint16_t(a[r]),uint16_t(b[c]));
                    groups[r][c].set(i+1u,uint16_t(a[r]>>16u),uint16_t(b[c]>>16u));
                }
        }
        QRT_DOT_TILE_UNROLL
        for(unsigned r=0u;r<Rows;++r) QRT_DOT_TILE_UNROLL for(unsigned c=0u;c<Columns;++c){
            qrt_sm121_group16::AlignedSum sum;
            const bool accepted=left_flags[r] && right_flags[c] && aligned::sum(carry[r][c],groups[r][c],&sum);
            if(!accepted){
                uint32_t products[16];
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
                for(unsigned i=0u;i<16u;i+=2u){
                    const uint32_t a=left[r*LeftPitch+(base+i)/2u],b=right[((base+i)/2u)*RightPitch+c];
                    products[i]=qrt_sm121_group16::pack_product(original::multiply_bf16(uint16_t(a),uint16_t(b),-133));
                    products[i+1u]=qrt_sm121_group16::pack_product(original::multiply_bf16(uint16_t(a>>16u),uint16_t(b>>16u),-133));
                }
                sum=qrt_sm121_group16::sum_packed(carry[r][c],products);
            }
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
            carry[r][c]=qrt_sm121_wave16::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
#else
            carry[r][c]=qrt_sm121_canonical::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
#endif
            if constexpr(Audit)if(trace)trace[(base/16u*Rows+r)*Columns+c]=carry[r][c];
        }
    }
    Result<Rows,Columns> result;
    QRT_DOT_TILE_UNROLL
    for(unsigned r=0u;r<Rows;++r) QRT_DOT_TILE_UNROLL for(unsigned c=0u;c<Columns;++c)
        result.values[r][c]=original::value_to_float(qrt_sm121_group16::finish_accumulator(carry[r][c]));
    return result;
}
} // namespace qrt_sm121_packed_dot_tile
#undef QRT_DOT_TILE_INLINE
#undef QRT_DOT_TILE_UNROLL
