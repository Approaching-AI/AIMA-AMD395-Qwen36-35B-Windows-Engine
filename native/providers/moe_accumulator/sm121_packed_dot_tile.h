#pragma once
#include "sm121_f32_carry.h"
#include "sm121_canonical_normalize.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#include "sm121_wave16.h"
#define QRT_DOT_TILE_INLINE __host__ __device__ __forceinline__
#define QRT_DOT_TILE_UNROLL _Pragma("unroll")
#else
#define QRT_DOT_TILE_INLINE inline
#define QRT_DOT_TILE_UNROLL
#endif

// Component-only dataflow experiment. Each original K16 endpoint remains
// exact. Keep only one row of product groups alive, and carry normal endpoints
// in FP32. Any failed classification restarts the complete original tile;
// original extended/exceptional carries never enter the fast recurrence.
namespace qrt_sm121_packed_dot_tile {
namespace original=qrt_q1_moe_hawkeye;
namespace aligned=qrt_sm121_float_alignment;
namespace compact=qrt_sm121_f32_carry;
template<unsigned Rows,unsigned Columns> struct Result {float values[Rows][Columns];};

template<unsigned Width,unsigned LeftPitch,unsigned RightPitch,
    unsigned Rows,unsigned Columns,bool Audit>
QRT_DOT_TILE_INLINE bool try_tile(const uint32_t* left,const uint32_t* right,
    const unsigned* left_flags,const unsigned* right_flags,
    Result<Rows,Columns>* result,original::Value* trace){
    QRT_DOT_TILE_UNROLL
    for(unsigned r=0u;r<Rows;++r)if(!left_flags[r])return false;
    QRT_DOT_TILE_UNROLL
    for(unsigned c=0u;c<Columns;++c)if(!right_flags[c])return false;
    float carry[Rows][Columns]{};
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
    for(unsigned base=0u;base<Width;base+=16u){
        QRT_DOT_TILE_UNROLL
        for(unsigned r=0u;r<Rows;++r){
            aligned::Group groups[Columns];
            QRT_DOT_TILE_UNROLL
            for(unsigned i=0u;i<16u;i+=2u){
                const uint32_t a=left[r*LeftPitch+(base+i)/2u];
                QRT_DOT_TILE_UNROLL
                for(unsigned c=0u;c<Columns;++c){
                    const uint32_t b=right[((base+i)/2u)*RightPitch+c];
                    groups[c].set(i,uint16_t(a),uint16_t(b));
                    groups[c].set(i+1u,uint16_t(a>>16u),uint16_t(b>>16u));
                }
            }
            QRT_DOT_TILE_UNROLL
            for(unsigned c=0u;c<Columns;++c){
                float next;
                if(!compact::accumulate<0u>(carry[r][c],groups[c],&next))return false;
                carry[r][c]=next;
                if constexpr(Audit)if(trace)
                    trace[(base/16u*Rows+r)*Columns+c]=original::value_from_float(next,-133);
            }
        }
    }
    QRT_DOT_TILE_UNROLL
    for(unsigned r=0u;r<Rows;++r) QRT_DOT_TILE_UNROLL for(unsigned c=0u;c<Columns;++c)
        result->values[r][c]=carry[r][c];
    return true;
}

template<unsigned Width,unsigned LeftPitch,unsigned RightPitch,
    unsigned Rows,unsigned Columns,bool Audit=false>
QRT_DOT_TILE_INLINE Result<Rows,Columns> dot(const uint32_t* left,
    const uint32_t* right,const unsigned* left_flags,const unsigned* right_flags,
    original::Value* trace=nullptr){
    static_assert(Width && Width%16u==0u && LeftPitch>=Width/2u && RightPitch>=Columns);
    static_assert((Rows==1u || Rows==2u) && (Columns==1u || Columns==2u));
    Result<Rows,Columns> result;
    if(try_tile<Width,LeftPitch,RightPitch,Rows,Columns,Audit>(left,right,left_flags,right_flags,&result,trace))return result;
    // Retire every fast product before entering exceptional full-dot replay.
    // Process one original dot at a time, including all early groups. Audit
    // writes overwrite every speculative trace, even after a late rejection.
    QRT_DOT_TILE_UNROLL
    for(unsigned r=0u;r<Rows;++r) QRT_DOT_TILE_UNROLL for(unsigned c=0u;c<Columns;++c){
        original::Value carry{0u,-133,false};
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
        for(unsigned base=0u;base<Width;base+=16u){
            uint32_t products[16];
            QRT_DOT_TILE_UNROLL
            for(unsigned i=0u;i<16u;i+=2u){
                const uint32_t a=left[r*LeftPitch+(base+i)/2u],b=right[((base+i)/2u)*RightPitch+c];
                products[i]=qrt_sm121_group16::pack_product(original::multiply_bf16(uint16_t(a),uint16_t(b),-133));
                products[i+1u]=qrt_sm121_group16::pack_product(original::multiply_bf16(uint16_t(a>>16u),uint16_t(b>>16u),-133));
            }
            const auto sum=qrt_sm121_group16::sum_packed(carry,products);
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
            carry=qrt_sm121_wave16::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
#else
            carry=qrt_sm121_canonical::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
#endif
            if constexpr(Audit)if(trace)trace[(base/16u*Rows+r)*Columns+c]=carry;
        }
        result.values[r][c]=original::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
    }
    return result;
}
} // namespace qrt_sm121_packed_dot_tile
#undef QRT_DOT_TILE_INLINE
#undef QRT_DOT_TILE_UNROLL
