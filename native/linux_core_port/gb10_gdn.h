// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include <memory>

namespace aima_port {
// One batch-one owner. Construct before the model so table/provider preparation
// is included in command-to-ready time, and destroy after the engine drains.
class Gb10GdnOwner {
 public:
  Gb10GdnOwner();
  ~Gb10GdnOwner();
  Gb10GdnOwner(const Gb10GdnOwner&) = delete;
  Gb10GdnOwner& operator=(const Gb10GdnOwner&) = delete;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void gb10_prefill_gdn(std::size_t layer, const void* convolution,
                     const void* a, const void* b, void* output, void* state,
                     std::size_t tokens, bool has_initial_state);
void gb10_decode_gdn(std::size_t layer, const void* convolution,
                    const void* a, const void* b, void* output, void* state,
                    hipStream_t stream);
}  // namespace aima_port
