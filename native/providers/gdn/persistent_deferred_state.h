#pragma once
#include "deferred_state_matrices.h"
#include "blackwell_lifetime_matrices.h"

// Isolated persistent scheduling of the established deferred-state proof.
// A CTA owns eight complete state columns and every residual needed to replay
// their history. No other CTA consumes or modifies those columns during this
// kernel. The original interval arithmetic and replay cache are unchanged.
namespace qrt_fla_persistent_deferred {
namespace prior=qrt_fla_deferred_state;
namespace scalar=qrt_fla_blackwell_scalar;
namespace matrix=qrt_fla_interval;
namespace consumer=qrt_fla_consumer_interval;
constexpr unsigned columns=8u,threads=256u,cells=524288u;

__device__ __forceinline__ void report(const prior::Resolution& r,unsigned* statistics,
    unsigned step,unsigned slot){
    __shared__ unsigned waves[8][7];
    unsigned values[7]={1u,unsigned(r.depth!=0u),r.original_dots,r.depth,r.depth,
        unsigned(r.complete_replay),unsigned(!r.ready)};
#pragma unroll
    for(unsigned i=0u;i<7u;++i){
        unsigned value=values[i];
        for(unsigned shift=16u;shift;shift>>=1u){
            const unsigned next=__shfl_down(value,shift,32u);
            value=i==4u?max(value,next):value+next;
        }
        if(!(threadIdx.x%32u))waves[threadIdx.x/32u][i]=value;
    }
    __syncthreads();
    if(threadIdx.x<7u){
        const unsigned i=threadIdx.x;unsigned value=0u;
        for(unsigned wave=0u;wave<8u;++wave)
            value=i==4u?max(value,waves[wave][i]):value+waves[wave][i];
        const unsigned record=step*(cells/threads)+blockIdx.y*64u+blockIdx.x*4u+slot;
        statistics[size_t(record)*7u+i]=value;
    }
    __syncthreads();
}

template<bool CoarseResidual>
__global__ __launch_bounds__(256) void state(
    const uint16_t* keys,const uint16_t* u,const uint16_t* weights,const float* gates,
    uint16_t* checkpoints,uint16_t* updated,float* final_state,unsigned count,
    const unsigned char* table,float* lower,float* upper,float* cache,unsigned char* flags,
    uint16_t* archive,const float* coefficients,unsigned* statistics,
    float* audit_lower,float* audit_upper){
    union Left {uint32_t weights[64][65];uint32_t keys[128][33];};
    union Right {uint32_t packed[64][16];uint16_t words[64][16][2];uint32_t increments[32][16];};
    __shared__ Left left;
    __shared__ Right right;
    __shared__ uint16_t residual[64][columns];
    __shared__ float decay[64];
    __shared__ unsigned weight_ok[64],state_ok[columns];
    const unsigned tid=threadIdx.x,head=blockIdx.y,first_column=blockIdx.x*columns;
    const unsigned steps=(count+63u)/64u;
    // The second half of the native matrix is immutable zero padding. Each
    // active halfword has a unique writer before the whole-CTA barrier.
    for(unsigned cell=tid;cell<64u*16u;cell+=threads)right.packed[cell/16u][cell%16u]=0u;
    __syncthreads();
    for(unsigned step=0u;step<=steps;++step){
        for(unsigned slot=0u;slot<4u;++slot){
            const unsigned local=slot*threads+tid,feature=local%128u,column=local/128u;
            const unsigned cell=(head*128u+first_column+column)*128u+feature;
            prior::DeviceHistory history{lower,upper,cache,flags,coefficients,cell,head};
            prior::DeviceReplay replay{keys,archive,count,head,first_column+column,feature};
            const auto result=prior::resolve(history,replay,step,
                step==steps?prior::Boundary::fp32_state:prior::Boundary::bf16_checkpoint);
            report(result,statistics,step,slot);
            if(result.ready){
                if(step==steps)final_state[cell]=result.state.lower;
                else{
                    const uint16_t word=scalar::to_bf16(result.state.lower);
                    checkpoints[size_t(step)*cells+cell]=word;
                    right.words[feature/2u][column][feature%2u]=word;
                }
            }
        }
        if(step==steps)break;
        const unsigned offset=step*64u,valid=min(64u,count-offset);
        if(tid<64u)weight_ok[tid]=1u;
        if(tid<columns)state_ok[tid]=1u;
        if(tid<valid)decay[tid]=scalar::exponential(
            gates[size_t(offset+valid-1u)*32u+head]-gates[size_t(offset+tid)*32u+head],table);
        __syncthreads();
        for(unsigned cell=tid;cell<64u*columns;cell+=threads){
            const unsigned pair=cell/columns,column=cell%columns;
            const uint32_t words=right.packed[pair][column];
            if(!scalar::eligible(uint16_t(words),uint16_t(words>>16u)))atomicAnd(state_ok+column,0u);
        }
        for(unsigned cell=tid;cell<64u*64u;cell+=threads){
            const unsigned row=cell/64u,pair=cell%64u;
            const size_t at=(size_t(offset+row)*32u+head)*128u+pair*2u;
            const uint16_t a=row<valid?weights[at]:0u,b=row<valid?weights[at+1u]:0u;
            left.weights[row][pair]=scalar::pack(a,b);
            if(!scalar::eligible(a,b))atomicAnd(weight_ok+row,0u);
        }
        __syncthreads();
        if constexpr(CoarseResidual){
            const unsigned wave=tid/32u,lane=tid%32u;
            if(wave<4u){
                const auto ranges=matrix::matrix<128u,true>(left.weights,right.packed,wave*16u,0u);
                for(unsigned item=0u;item<8u;++item){
                    const unsigned row=wave*16u+item*2u+lane/16u,column=lane%16u;
                    if(column<columns){
                        uint16_t vn=0u,scaled=0u;
                        if(row<valid){
                            const size_t at=(size_t(offset+row)*32u+head)*128u+first_column+column;
                            if(!consumer::residual({ranges.lower[item],ranges.upper[item]},
                                scalar::from_bf16(u[at]),decay[row],&vn,&scaled)){
                                const float dot=scalar::dot<128u,16u>(left.weights[row],right.packed,
                                    column,weight_ok[row]&&state_ok[column]);
                                const float difference=scalar::from_bf16(u[at])-dot;
                                vn=scalar::to_bf16(difference);scaled=scalar::to_bf16(difference*decay[row]);
                            }
                            updated[at]=vn;archive[at]=scaled;
                        }
                        residual[row][column]=scaled;
                    }
                }
            }
        }else{
            for(unsigned cell=tid;cell<64u*columns;cell+=threads){
                const unsigned row=cell/columns,column=cell%columns;uint16_t scaled=0u;
                if(row<valid){
                    const size_t at=(size_t(offset+row)*32u+head)*128u+first_column+column;
                    const float dot=scalar::dot<128u,16u>(left.weights[row],right.packed,
                        column,weight_ok[row]&&state_ok[column]);
                    const float difference=scalar::from_bf16(u[at])-dot;
                    updated[at]=scalar::to_bf16(difference);scaled=scalar::to_bf16(difference*decay[row]);
                    archive[at]=scaled;
                }
                residual[row][column]=scaled;
            }
        }
        __syncthreads();
        // W/H is dead. Reuse both operand arenas only after all of its
        // consumers have published their exact scaled residuals.
        for(unsigned cell=tid;cell<128u*32u;cell+=threads){
            const unsigned feature=cell/32u,pair=cell%32u;
            const uint16_t a=pair*2u<valid?keys[(size_t(offset+pair*2u)*16u+head/2u)*128u+feature]:0u;
            const uint16_t b=pair*2u+1u<valid?keys[(size_t(offset+pair*2u+1u)*16u+head/2u)*128u+feature]:0u;
            left.keys[feature][pair]=scalar::pack(a,b);
        }
        for(unsigned cell=tid;cell<32u*columns;cell+=threads){
            const unsigned pair=cell/columns,column=cell%columns;
            right.packed[pair][column]=scalar::pack(residual[pair*2u][column],residual[pair*2u+1u][column]);
        }
        __syncthreads();
        const unsigned wave=tid/32u,lane=tid%32u;
        // Eight waves own all 128 features. Padding permits the original
        // C64 matrix enclosure without sharing a recurrent column across CTAs.
        const auto increments=matrix::matrix<64u,true>(left.keys,right.increments,wave*16u,0u);
        for(unsigned item=0u;item<8u;++item){
            const unsigned feature=wave*16u+item*2u+lane/16u,column=lane%16u;
            if(column<columns){
                const unsigned cell=(head*128u+first_column+column)*128u+feature;
                const size_t before=size_t(step)*cells+cell,after=before+cells;
                const auto next=prior::advance({lower[before],upper[before]},coefficients[step*32u+head],
                    {increments.lower[item],increments.upper[item]});
                lower[after]=next.lower;upper[after]=next.upper;flags[before]=0u;
                if(audit_lower){audit_lower[after]=next.lower;audit_upper[after]=next.upper;}
            }
        }
        __syncthreads();
    }
}
} // namespace qrt_fla_persistent_deferred
