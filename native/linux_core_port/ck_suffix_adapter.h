// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace aima_port {

// Existing Windows DLL ABI. All inputs are token-major BF16; output is
// query-local F32. This adapter does not read or copy device storage.
using CkSuffixLaunch = int (*)(
    const std::uint16_t*, const std::uint16_t*, const std::uint16_t*,
    const std::uint16_t*, const std::uint16_t*, float*, void*, unsigned, unsigned);

struct CkSuffixViews {
  const std::uint16_t* q;
  const std::uint16_t* prefix_k;
  const std::uint16_t* prefix_v;
  const std::uint16_t* suffix_k;
  const std::uint16_t* suffix_v;
  float* output;
  unsigned prefix_tokens;
  unsigned query_tokens;
};

inline CkSuffixViews ck_suffix_views(
    const void* q, const void* k, const void* v, void* output,
    std::size_t query_tokens, std::size_t kv_tokens) {
  // Preserve the imported engine's current 262144 total-KV bound. The
  // Windows suffix provider accepts at most one 8192-query chunk per call.
  if (query_tokens == 0 || query_tokens > 8192 ||
      kv_tokens <= query_tokens || kv_tokens > 262144) {
    throw std::invalid_argument("Windows CK suffix geometry is unsupported");
  }
  constexpr std::size_t query_row_bytes = 4096 * sizeof(std::uint16_t);
  constexpr std::size_t kv_row_bytes = 512 * sizeof(std::uint16_t);
  constexpr std::size_t output_row_bytes = 4096 * sizeof(float);
  const std::size_t output_bytes = query_tokens * output_row_bytes;
  const auto destination = reinterpret_cast<std::uintptr_t>(output);
  const auto maximum = std::numeric_limits<std::uintptr_t>::max();
  if (destination == 0 || destination % alignof(float) != 0 ||
      output_bytes > maximum - destination) {
    throw std::invalid_argument("Windows CK suffix output range is invalid");
  }
  const void* inputs[] = {q, k, v};
  const std::size_t sizes[] = {
      query_tokens * query_row_bytes, kv_tokens * kv_row_bytes,
      kv_tokens * kv_row_bytes};
  for (unsigned i = 0; i < 3; ++i) {
    const auto address = reinterpret_cast<std::uintptr_t>(inputs[i]);
    if (address == 0 || address % alignof(std::uint16_t) != 0 ||
        sizes[i] > maximum - address ||
        (destination < address + sizes[i] &&
         address < destination + output_bytes)) {
      throw std::invalid_argument("Windows CK suffix input range is invalid");
    }
  }
  const std::size_t prefix_tokens = kv_tokens - query_tokens;
  const std::size_t suffix_offset = prefix_tokens * kv_row_bytes;
  // Integer offsets avoid dereferencing GPU addresses on the host. Their
  // complete consumed extents and wraparound were checked above.
  return {
      static_cast<const std::uint16_t*>(q),
      static_cast<const std::uint16_t*>(k),
      static_cast<const std::uint16_t*>(v),
      reinterpret_cast<const std::uint16_t*>(
          reinterpret_cast<std::uintptr_t>(k) + suffix_offset),
      reinterpret_cast<const std::uint16_t*>(
          reinterpret_cast<std::uintptr_t>(v) + suffix_offset),
      static_cast<float*>(output), static_cast<unsigned>(prefix_tokens),
      static_cast<unsigned>(query_tokens)};
}

}  // namespace aima_port
