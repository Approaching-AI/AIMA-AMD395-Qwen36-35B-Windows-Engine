#pragma once
#include "consumer_interval.h"

#if defined(__HIPCC__)
#define QRT_DEFERRED_STATE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_DEFERRED_STATE_INLINE inline
#endif

// Isolated recurrence scheduling primitive. Increment intervals must enclose
// the original ordered K64 dot. A History stores candidate-owned intervals,
// original replay caches and exact nonnegative decay coefficients. References
// used by numerical observers must never implement History or choose replay.
namespace qrt_fla_deferred_state {
namespace consumer=qrt_fla_consumer_interval;
namespace interval=qrt_sm121_projection_interval;
namespace bound=qrt_sm121_pv_bound;
using interval::Interval;
enum class Boundary { bf16_checkpoint, fp32_state };

QRT_DEFERRED_STATE_INLINE Interval advance(Interval state,float decay,Interval increment){
    if(!interval::valid(state) || !interval::valid(increment) || !consumer::nonnegative(decay))
        return interval::invalid();
    // FMA at each original checkpoint is monotone in both operands for this
    // exact nonnegative coefficient. Do not reassociate a sequence of updates.
    const Interval result{consumer::fused(state.lower,decay,increment.lower),
                          consumer::fused(state.upper,decay,increment.upper)};
    return interval::valid(result)?result:interval::invalid();
}
QRT_DEFERRED_STATE_INLINE bool accepted(Interval state,Boundary boundary){
    if(!interval::valid(state))return false;
    return boundary==Boundary::fp32_state
        ? bound::bits(state.lower)==bound::bits(state.upper)
        : bound::bf16(state.lower)==bound::bf16(state.upper);
}
struct Resolution {
    Interval state;
    unsigned original_dots,depth;
    bool ready,complete_replay;
};

// The caller publishes every earlier BF16 checkpoint before constructing the
// corresponding increment. Thus its replay operands remain original even if
// the unrounded FP32 state is still an interval. Extend a cached exact tail
// backwards until this boundary is fixed. Each original dot is computed at
// most once per cell/segment; each attempt reapplies all tail FMAs in order.
// A segment finishes only at fp32_state, preserving the original state ABI.
template<class History,class Replay>
QRT_DEFERRED_STATE_INLINE Resolution resolve(History& history,Replay& replay,
    unsigned steps,Boundary boundary){
    Interval state=history.state(steps);
    if(accepted(state,boundary))return {state,0u,0u,true,false};
    unsigned dots=0u;
    for(unsigned next=steps;next;--next){
        const unsigned first=next-1u;
        if(!history.exact(first)){
            history.cache(first,replay(first));++dots;
        }
        state=history.state(first);
        for(unsigned step=first;step<steps;++step){
            const float increment=history.increment(step);
            state=advance(state,history.decay(step),{increment,increment});
        }
        if(accepted(state,boundary)){
            history.replace(steps,state);
            return {state,dots,steps-first,true,first==0u};
        }
    }
    // Full original arithmetic also retains exceptional/signed-zero behavior
    // for which finite interval propagation deliberately refuses a proof.
    const auto seed=history.state(0u);
    if(bound::bits(seed.lower)!=bound::bits(seed.upper))return {state,dots,steps,false,false};
    float value=seed.lower;
    for(unsigned step=0u;step<steps;++step)
        value=consumer::fused(value,history.decay(step),history.increment(step));
    state={value,value};history.replace(steps,state);
    return {state,dots,steps,true,true};
}
} // namespace qrt_fla_deferred_state
#undef QRT_DEFERRED_STATE_INLINE
