// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../providers/gdn/sm121_silu_table.h"
#if defined(__HIPCC__)
#define AIMA_CONV_HD __host__ __device__
#else
#define AIMA_CONV_HD
#endif
namespace aima_port {
AIMA_CONV_HD inline float conv_widen(std::uint16_t value) {
  return qrt_sm121_exp2::value(std::uint32_t(value) << 16);
}
AIMA_CONV_HD inline float conv_product(std::uint16_t x, std::uint16_t weight) {
  // AMD's v_dot2_bf16_bf16 truncates. The reference rounds each product
  // to nearest, ties to even, before the sequential FP32 sum.
  const float a = conv_widen(x), b = conv_widen(weight);
#if defined(__HIP_DEVICE_COMPILE__)
  float product;
  asm("v_mul_f32 %0, %1, %2" : "=v"(product) : "v"(a), "v"(b));
#else
  volatile float product = a * b;
#endif
  const auto bits = qrt_sm121_exp2::bits(product);
  return conv_widen(static_cast<std::uint16_t>(
      (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16));
}
AIMA_CONV_HD inline float conv_add(float left, float right) {
#if defined(__HIP_DEVICE_COMPILE__)
  float sum;
  asm("v_add_f32 %0, %1, %2" : "=v"(sum) : "v"(left), "v"(right));
  return sum;
#else
  volatile float sum = left + right;
  return sum;
#endif
}
AIMA_CONV_HD inline std::uint16_t conv_value(
    std::uint16_t x0, std::uint16_t x1, std::uint16_t x2, std::uint16_t x3,
    const std::uint16_t* weight, const unsigned char* silu) {
  float sum = conv_add(0.0f, conv_product(x0, weight[0]));
  sum = conv_add(sum, conv_product(x1, weight[1]));
  sum = conv_add(sum, conv_product(x2, weight[2]));
  sum = conv_add(sum, conv_product(x3, weight[3]));
  return qrt_sm121_silu::evaluate(silu, sum);
}
}  // namespace aima_port
#undef AIMA_CONV_HD
