// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../providers/gdn/sm121_q2_gated_math.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define AIMA_GATED_INLINE __host__ __device__ __forceinline__
#else
#define AIMA_GATED_INLINE inline
#endif

namespace aima_port {
// The original q8192 Triton prefill layout reduces eight adjacent BF16 values
// in each of sixteen lanes, then XOR 8, 4, 2, 1. Its first product is item 1,
// followed by FMA items 0, 2..7. Short decode uses the separate 32-lane layout.
AIMA_GATED_INLINE float gated_prefill_lane_sum(const uint16_t* core, unsigned lane) {
  using namespace qrt_sm121_q1;
  const auto* x = core + std::size_t(lane) * 8u;
  float sum = multiply(widen(x[1]), widen(x[1]));
  sum = fmaf(widen(x[0]), widen(x[0]), sum);
  for (unsigned item = 2; item < 8; ++item)
    sum = fmaf(widen(x[item]), widen(x[item]), sum);
  return sum;
}
}  // namespace aima_port

#undef AIMA_GATED_INLINE
