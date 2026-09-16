#pragma once
#include "sm121_coarse_projection_bound.h"

#if defined(__HIPCC__)
#define QRT_PREFIX_REPLAY_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PREFIX_REPLAY_INLINE inline
#endif

namespace qrt_sm121_prefix_replay_bound {
namespace coarse = qrt_sm121_coarse_projection_bound;
namespace scalar = coarse::scalar;
using State = coarse::State;

// Preconditions: prefix and final are checkpoints of the SAME coarse producer
// and exact_prefix is the original canonical carry at that prefix boundary.
// The native zero-C WMMA error assumption is unchanged from coarse::advance.
//
// Each coarse step increases its error by at least the native-product error,
// canonical K16 loss and center-addition error for that step. Consequently
// final.error-prefix.error bounds their sum over the suffix. Telescoping the
// center additions gives:
//   canonical_final = exact_prefix + final.center-prefix.center + suffix_error.
// This does not assume a Lipschitz property of rounded canonical transitions,
// nor does it apply a new approximate carry to those transitions. The original
// path supplied every canonical-loss bound and exact_prefix stays on that path.
QRT_PREFIX_REPLAY_INLINE State suffix(State prefix, State final, float exact_prefix) {
    State result{0.0f, scalar::infinity()};
    if (!scalar::finite(prefix.center) || !scalar::finite(prefix.error) || prefix.error < 0.0f ||
        !scalar::finite(final.center) || !scalar::finite(final.error) || final.error < prefix.error ||
        !scalar::finite(exact_prefix)) return result;
    const float delta = final.center - prefix.center;
    result.center = exact_prefix + delta;
    const float subtract_error = coarse::unit(scalar::upper(
        scalar::absolute(final.center) + scalar::absolute(prefix.center)), 23u);
    const float add_error = coarse::unit(scalar::upper(
        scalar::absolute(exact_prefix) + scalar::absolute(delta)), 23u);
    const float remaining = scalar::upper(final.error - prefix.error);
    result.error = scalar::upper(remaining + scalar::upper(subtract_error + add_error));
    return result;
}
QRT_PREFIX_REPLAY_INLINE bool certificate(State prefix, State final,
    float exact_prefix, float* representative) {
    const auto value = suffix(prefix, final, exact_prefix);
    if (!coarse::certified(value)) return false;
    *representative = value.center;
    return true;
}
} // namespace qrt_sm121_prefix_replay_bound
#undef QRT_PREFIX_REPLAY_INLINE
