#pragma once
#include "sm121_coarse_projection_matrix.h"
#include "sm121_observed_prefix_bound.h"

// Isolated producer using actual native partial magnitudes to bound every
// canonical interior carry. The final center arithmetic is unchanged. Parts1
// writes no checkpoints; Parts2/4 support the existing exact prefix replay.
// Launch contracts: width divisible by64*Parts, nonempty dimensions and
// complete valid operand/flag/output spans; snapshot capacity(Parts-1)*cells.
namespace qrt_sm121_observed_prefix_projection {
namespace bound=qrt_sm121_coarse_projection_bound;
namespace observed=qrt_sm121_observed_prefix_bound;
using B16=qrt_sm121_coarse_projection_matrix::B16;
using F8=qrt_sm121_coarse_projection_matrix::F8;
constexpr unsigned threads=256u,row_tile=128u;
template<unsigned Parts>
__global__ __launch_bounds__(threads) void produce(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* weight_ok,const unsigned* input_ok,float* centers,float* errors,
    unsigned rows,unsigned tokens,unsigned width,bound::State* snapshots) {
    constexpr unsigned Chunk=64u,Fragments=1u;
    static_assert(Parts==1u || Parts==2u || Parts==4u);
    const unsigned split=width/Parts;
    const size_t cells=size_t(rows)*tokens;
    static_assert(Chunk==64u || Chunk==128u || Chunk==256u);
    static_assert(Fragments==1u || Fragments==2u);
    const unsigned lane=threadIdx.x%32u,wave=threadIdx.x/32u,source=lane%16u;
    const unsigned row=blockIdx.x*row_tile+wave*16u+source;
    const unsigned first_token=blockIdx.y*(16u*Fragments);
    const bool valid_weight=row<rows && weight_ok[row];
    F8 centers_local[Fragments]{},errors_local[Fragments]{};
    for(unsigned coarse=0u;coarse<width;coarse+=Chunk) {
        F8 partial[Fragments]{},positive[Fragments]{},prefix_max[Fragments]{},group_max[Fragments]{};
#pragma unroll
        for(unsigned fragment=0u;fragment<Fragments;++fragment)
#pragma unroll
            for(unsigned item=0u;item<8u;++item)prefix_max[fragment][item]=bound::scalar::absolute(centers_local[fragment][item]);
#pragma unroll 1
        for(unsigned base=coarse;base<coarse+Chunk && base<width;base+=16u) {
            B16 w{},wa{};
#pragma unroll
            for(unsigned k=0u;k<16u;++k){w[k]=valid_weight?weights[size_t(row)*width+base+k]:0u;wa[k]=w[k]&0x7fffu;}
#pragma unroll
            for(unsigned fragment=0u;fragment<Fragments;++fragment) {
                const unsigned token=first_token+fragment*16u+source;
                const bool valid_input=token<tokens && input_ok[token];B16 x{},xa{};
#pragma unroll
                for(unsigned k=0u;k<16u;++k){x[k]=valid_input?inputs[size_t(token)*width+base+k]:0u;xa[k]=x[k]&0x7fffu;}
                const F8 zero{};
                // Explicit zero-C products followed by FP32 additions. Native
                // C accumulation is not substituted for this error model.
                const F8 signed_product=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x,w,zero);
                const F8 absolute_product=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(xa,wa,zero);
                partial[fragment]+=signed_product;positive[fragment]+=absolute_product;
#pragma unroll
                for(unsigned item=0u;item<8u;++item){
                    const float value=bound::scalar::absolute(centers_local[fragment][item]+partial[fragment][item]);
                    prefix_max[fragment][item]=value>prefix_max[fragment][item]?value:prefix_max[fragment][item];
                    group_max[fragment][item]=absolute_product[item]>group_max[fragment][item]?absolute_product[item]:group_max[fragment][item];
                }
            }
        }
#pragma unroll
        for(unsigned fragment=0u;fragment<Fragments;++fragment) {
#pragma unroll
            for(unsigned item=0u;item<8u;++item) {
                const auto next=observed::advance<Chunk/16u>({centers_local[fragment][item],errors_local[fragment][item]},partial[fragment][item],positive[fragment][item],prefix_max[fragment][item],group_max[fragment][item]);
                centers_local[fragment][item]=next.center;errors_local[fragment][item]=next.error;
                if(Parts>1u && (coarse+Chunk)%split==0u && coarse+Chunk<width && row<rows) {
                    const unsigned token=first_token+fragment*16u+2u*item+lane/16u;
                    if(token<tokens) snapshots[size_t((coarse+Chunk)/split-1u)*cells+size_t(token)*rows+row]=
                        {next.center,valid_weight && input_ok[token]?next.error:bound::scalar::infinity()};
                }
            }
        }
    }
    if(row>=rows)return;
#pragma unroll
    for(unsigned fragment=0u;fragment<Fragments;++fragment) {
#pragma unroll
        for(unsigned item=0u;item<8u;++item) {
            const unsigned token=first_token+fragment*16u+2u*item+lane/16u;
            if(token<tokens){const size_t cell=size_t(token)*rows+row;centers[cell]=centers_local[fragment][item];
                errors[cell]=valid_weight && input_ok[token]?errors_local[fragment][item]:bound::scalar::infinity();}
        }
    }
}

} // namespace qrt_sm121_observed_prefix_projection
