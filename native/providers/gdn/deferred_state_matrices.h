#pragma once
#include "interval_matrices.h"
#include "deferred_state_interval.h"
#include "../moe_accumulator/sm121_f32_carry.h"

// Isolated bounded-segment dataflow. Every published checkpoint is canonical
// BF16, and finalization returns canonical FP32 state. Deferred dot operands
// are original keys and candidate-produced canonical scaled residuals.
namespace qrt_fla_deferred_state {
namespace scalar=qrt_fla_blackwell_scalar;
namespace matrix=qrt_fla_interval;
constexpr unsigned state_cells=524288u,threads=256u;
struct DeviceHistory {
    float *lower,*upper,*values;
    unsigned char* flags;
    const float* coefficients;
    unsigned cell,head;
    __device__ Interval state(unsigned step)const{
        const size_t at=size_t(step)*state_cells+cell;return {lower[at],upper[at]};
    }
    __device__ bool exact(unsigned step)const{return flags[size_t(step)*state_cells+cell]!=0u;}
    __device__ float increment(unsigned step)const{return values[size_t(step)*state_cells+cell];}
    __device__ float decay(unsigned step)const{return coefficients[step*32u+head];}
    __device__ void cache(unsigned step,float value){
        const size_t at=size_t(step)*state_cells+cell;values[at]=value;flags[at]=1u;
    }
    __device__ void replace(unsigned step,Interval value){
        const size_t at=size_t(step)*state_cells+cell;lower[at]=value.lower;upper[at]=value.upper;
    }
};

__device__ __attribute__((noinline)) float original_delta(const uint16_t* keys,const uint16_t* residual,
    unsigned count,unsigned step,unsigned head,unsigned column,unsigned feature){
    const unsigned offset=step*64u,valid=min(64u,count-offset);
    float carry=0.0f;bool fallback=false;
    for(unsigned base=0u;base<64u;base+=16u){
        qrt_sm121_float_alignment::Group group;
#pragma unroll
        for(unsigned i=0u;i<16u;++i){
            const unsigned row=base+i;
            const uint16_t a=row<valid?keys[(size_t(offset+row)*16u+head/2u)*128u+feature]:0u;
            const uint16_t b=row<valid?residual[(size_t(offset+row)*32u+head)*128u+column]:0u;
            fallback|=!scalar::eligible(a,b);group.set(i,a,b);
        }
        float next;
        if(fallback || !qrt_sm121_f32_carry::accumulate<0u>(carry,group,&next)){fallback=true;break;}
        carry=next;
    }
    if(!fallback)return carry;
    qrt_q1_moe_hawkeye::Value exact{0u,-133,false};
    for(unsigned base=0u;base<64u;base+=16u){
        uint32_t products[16];
#pragma unroll
        for(unsigned i=0u;i<16u;++i){
            const unsigned row=base+i;
            const uint16_t a=row<valid?keys[(size_t(offset+row)*16u+head/2u)*128u+feature]:0u;
            const uint16_t b=row<valid?residual[(size_t(offset+row)*32u+head)*128u+column]:0u;
            products[i]=qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(a,b,-133));
        }
        const auto sum=qrt_sm121_group16::sum_packed(exact,products);
        exact=qrt_sm121_wave16::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
    }
    return qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(exact));
}
struct DeviceReplay {
    const uint16_t *keys,*residual;
    unsigned count,head,column,feature;
    __device__ float operator()(unsigned step)const{return original_delta(keys,residual,count,step,head,column,feature);}
};
__device__ void count_resolution(const Resolution& r,unsigned* statistics,unsigned step){
    __shared__ unsigned waves[8][7];
    unsigned values[7]={1u,unsigned(r.depth!=0u),r.original_dots,r.depth,r.depth,
        unsigned(r.complete_replay),unsigned(!r.ready)};
#pragma unroll
    for(unsigned i=0u;i<7u;++i){
        unsigned value=values[i];
        for(unsigned shift=16u;shift;shift>>=1u){
            const unsigned next=__shfl_down(value,shift,32u);value=i==4u?max(value,next):value+next;
        }
        if(!(threadIdx.x%32u))waves[threadIdx.x/32u][i]=value;
    }
    __syncthreads();
    if(threadIdx.x<7u){
        const unsigned i=threadIdx.x;unsigned value=0u;
        for(unsigned wave=0u;wave<8u;++wave)value=i==4u?max(value,waves[wave][i]):value+waves[wave][i];
        statistics[(size_t(step)*gridDim.x+blockIdx.x)*7u+i]=value;
    }
}
__global__ void initialize_history(const float* state,const float* gates,float* lower,float* upper,
    float* coefficients,unsigned count,const unsigned char* table,float* audit_lower,float* audit_upper){
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x,steps=(count+63u)/64u;
    if(cell<state_cells){lower[cell]=state[cell];upper[cell]=state[cell];
        if(audit_lower){audit_lower[cell]=state[cell];audit_upper[cell]=state[cell];}}
    if(cell<steps*32u){
        const unsigned step=cell/32u,head=cell%32u,last=min((step+1u)*64u,count)-1u;
        coefficients[cell]=scalar::exponential(gates[size_t(last)*32u+head],table);
    }
}
template<bool Final>
__global__ void checkpoint(const uint16_t* keys,const uint16_t* residual,unsigned count,unsigned steps,
    float* lower,float* upper,float* cache,unsigned char* flags,const float* coefficients,
    uint16_t* checkpoints,float* state,unsigned* statistics){
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=state_cells)return;
    const unsigned feature=cell%128u,column=cell/128u%128u,head=cell/(128u*128u);
    DeviceHistory history{lower,upper,cache,flags,coefficients,cell,head};
    DeviceReplay replay{keys,residual,count,head,column,feature};
    const auto result=resolve(history,replay,steps,Final?Boundary::fp32_state:Boundary::bf16_checkpoint);
    count_resolution(result,statistics,steps);
    if(result.ready){
        if constexpr(Final)state[cell]=result.state.lower;
        else checkpoints[size_t(steps)*state_cells+cell]=scalar::to_bf16(result.state.lower);
    }
}

// Standalone original WH consumer. The coarse variant has the existing native
// envelope and exact scalar fallback; both publish identical Vnew and scaled
// residuals. These archives are computation-owned, not observer data.
template<bool Coarse>
__global__ void residuals(const uint16_t* u,const uint16_t* w,const uint16_t* checkpoints,
    const float* gates,uint16_t* updated,uint16_t* archive,unsigned count,unsigned step,
    const unsigned char* table){
    constexpr unsigned columns=16u;
    __shared__ uint32_t weights[64][65],state[64][columns];
    __shared__ unsigned weight_ok[64],state_ok[columns];
    __shared__ float decay[64];
    const unsigned tid=threadIdx.x,head=blockIdx.y,first_column=blockIdx.x*columns;
    const unsigned offset=step*64u,valid=min(64u,count-offset);
    if(tid<64u)weight_ok[tid]=1u;
    if(tid<columns)state_ok[tid]=1u;
    if(tid<valid)decay[tid]=scalar::exponential(gates[size_t(offset+valid-1u)*32u+head]-gates[size_t(offset+tid)*32u+head],table);
    __syncthreads();
    for(unsigned cell=tid;cell<64u*columns;cell+=threads){
        const unsigned pair=cell/columns,column=cell%columns;
        const size_t at=size_t(step)*state_cells+(head*128u+first_column+column)*128u+pair*2u;
        const uint16_t a=checkpoints[at],b=checkpoints[at+1u];state[pair][column]=scalar::pack(a,b);
        if(!scalar::eligible(a,b))atomicAnd(state_ok+column,0u);
    }
    for(unsigned cell=tid;cell<64u*64u;cell+=threads){
        const unsigned row=cell/64u,pair=cell%64u;
        const size_t at=(size_t(offset+row)*32u+head)*128u+pair*2u;
        const uint16_t a=row<valid?w[at]:0u,b=row<valid?w[at+1u]:0u;
        weights[row][pair]=scalar::pack(a,b);
        if(!scalar::eligible(a,b))atomicAnd(weight_ok+row,0u);
    }
    __syncthreads();
    if constexpr(Coarse){
        const unsigned wave=tid/32u,lane=tid%32u;
        if(wave<4u){
            const auto ranges=matrix::matrix<128u,true>(weights,state,wave*16u,0u);
            for(unsigned i=0u;i<8u;++i){
                const unsigned row=wave*16u+i*2u+lane/16u,column=lane%16u;
                if(row<valid){
                    const size_t at=(size_t(offset+row)*32u+head)*128u+first_column+column;
                    uint16_t vn,scaled;
                    if(!consumer::residual({ranges.lower[i],ranges.upper[i]},scalar::from_bf16(u[at]),decay[row],&vn,&scaled)){
                        const float dot=scalar::dot<128u,columns>(weights[row],state,column,weight_ok[row]&&state_ok[column]);
                        const float difference=scalar::from_bf16(u[at])-dot;
                        vn=scalar::to_bf16(difference);scaled=scalar::to_bf16(difference*decay[row]);
                    }
                    updated[at]=vn;archive[at]=scaled;
                }
            }
        }
    }else{
        for(unsigned cell=tid;cell<valid*columns;cell+=threads){
            const unsigned row=cell/columns,column=cell%columns;
            const size_t at=(size_t(offset+row)*32u+head)*128u+first_column+column;
            const float dot=scalar::dot<128u,columns>(weights[row],state,column,weight_ok[row]&&state_ok[column]);
            const float difference=scalar::from_bf16(u[at])-dot;
            updated[at]=scalar::to_bf16(difference);archive[at]=scalar::to_bf16(difference*decay[row]);
        }
    }
}

// One wave owns16x16 state cells; eight waves cover64featuresx32columns.
// No sequential state carry, WH dot or checkpoint array occupies shared
// storage while the native update matrices execute.
__global__ void update(const uint16_t* keys,const uint16_t* archive,unsigned count,unsigned step,
    float* lower,float* upper,unsigned char* flags,const float* coefficients,
    float* audit_lower,float* audit_upper){
    __shared__ uint32_t left[64][33],right[32][32];
    const unsigned tid=threadIdx.x,head=blockIdx.y,first_column=blockIdx.x*32u,first_feature=blockIdx.z*64u;
    const unsigned offset=step*64u,valid=min(64u,count-offset);
    for(unsigned cell=tid;cell<64u*32u;cell+=threads){
        const unsigned feature=cell/32u,pair=cell%32u;
        const uint16_t a=pair*2u<valid?keys[(size_t(offset+pair*2u)*16u+head/2u)*128u+first_feature+feature]:0u;
        const uint16_t b=pair*2u+1u<valid?keys[(size_t(offset+pair*2u+1u)*16u+head/2u)*128u+first_feature+feature]:0u;
        left[feature][pair]=scalar::pack(a,b);
    }
    for(unsigned cell=tid;cell<32u*32u;cell+=threads){
        const unsigned pair=cell/32u,column=cell%32u;
        const uint16_t a=pair*2u<valid?archive[(size_t(offset+pair*2u)*32u+head)*128u+first_column+column]:0u;
        const uint16_t b=pair*2u+1u<valid?archive[(size_t(offset+pair*2u+1u)*32u+head)*128u+first_column+column]:0u;
        right[pair][column]=scalar::pack(a,b);
    }
    __syncthreads();
    const unsigned wave=tid/32u,lane=tid%32u,row_base=(wave/2u)*16u,column_base=(wave%2u)*16u;
    const auto increments=matrix::matrix<64u,true>(left,right,row_base,column_base);
    for(unsigned i=0u;i<8u;++i){
        const unsigned feature=first_feature+row_base+i*2u+lane/16u,column=first_column+column_base+lane%16u;
        const unsigned cell=(head*128u+column)*128u+feature;
        const size_t before=size_t(step)*state_cells+cell,after=before+state_cells;
        const auto next=advance({lower[before],upper[before]},coefficients[step*32u+head],
            {increments.lower[i],increments.upper[i]});
        lower[after]=next.lower;upper[after]=next.upper;flags[before]=0u;
        if(audit_lower){audit_lower[after]=next.lower;audit_upper[after]=next.upper;}
    }
}
} // namespace qrt_fla_deferred_state
