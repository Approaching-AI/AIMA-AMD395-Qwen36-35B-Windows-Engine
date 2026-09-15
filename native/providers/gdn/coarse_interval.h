#ifndef QRT_FLA_COARSE_INTERVAL_H
#define QRT_FLA_COARSE_INTERVAL_H
#include "consumer_interval.h"
#include "../moe_accumulator/sm121_coarse_projection_bound.h"
#if defined(__HIPCC__)
#define QRT_GDN_COARSE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_GDN_COARSE_INLINE inline
#endif
namespace qrt_fla_coarse_interval {
namespace coarse=qrt_sm121_coarse_projection_bound;
namespace interval=qrt_sm121_projection_interval;
namespace bound=qrt_sm121_pv_bound;
// The C64 producer retains the existing conditional native WMMA allowance
// and every original K16 loss. Unsupported operands invalidate the whole dot.
QRT_GDN_COARSE_INLINE interval::Interval range(coarse::State value,bool supported) {
    if(!supported || !bound::finite(value.center) || !bound::finite(value.error) || value.error<0.0f)
        return interval::invalid();
    const interval::Interval result{bound::next(value.center-value.error,false),
        bound::next(value.center+value.error,true)};
    return interval::valid(result)?result:interval::invalid();
}
}
#undef QRT_GDN_COARSE_INLINE
#endif
