// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
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
// Borrowed immutable table; valid only within the live GDN owner's lifetime.
const unsigned char* gb10_rsqrt_table();
const unsigned char* gb10_exp2_table();
const uint16_t* gb10_sigmoid_table();
using GdnPrefillObserver = void (*)(const char*, const void*, std::size_t, void*);
void set_gdn_prefill_observer(std::size_t layer, GdnPrefillObserver callback,
                              void* context, bool first64 = false);
// Fixed output-only sampling: the last row of every 64-token chunk at q8192.
void observe_gdn_prefill(std::size_t layer, const char* name, const void* values,
                         std::size_t columns, std::size_t tokens);
}  // namespace aima_port
