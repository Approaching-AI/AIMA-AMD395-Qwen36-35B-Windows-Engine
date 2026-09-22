// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../providers/gdn/sm121_mtp_residual_math.h"

namespace aima_port {
#if defined(__HIPCC__) || defined(__CUDACC__)
__host__ __device__ __forceinline__
#else
inline
#endif
float moe_carrier_lane_sumsq(const float* row, unsigned lane) {
  // Same eight adjacent values and FP32/FMA order as residual_lane_sumsq;
  // the MoE provider has already formed the unrounded sum of BF16 operands.
  using namespace qrt_sm121_q1;
  const float* v = row + lane * 8u;
  float sum = multiply(v[1], v[1]);
  sum = fmaf(v[0], v[0], sum);
  sum = fmaf(v[2], v[2], sum);
  sum = add(sum, multiply(v[3], v[3]));
  sum = fmaf(v[4], v[4], sum);
  sum = add(sum, multiply(v[5], v[5]));
  sum = fmaf(v[6], v[6], sum);
  return add(sum, multiply(v[7], v[7]));
}
}  // namespace aima_port
