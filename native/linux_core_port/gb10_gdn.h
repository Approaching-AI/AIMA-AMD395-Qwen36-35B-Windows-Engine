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
// Optional ordered integer-accumulator chunk-64 prefill core.
// Cold q8192 only; the resident state binding and decode arithmetic stay intact.
bool gb10_native_gdn_prefill_enabled(std::size_t tokens, bool has_initial_state);
// Reports the selected owner's optional fused recurrence for runtime evidence.
bool gb10_persistent_gdn_prefill_enabled();
struct NativeGdnMatrices { void* matrix_f32; void* inverse_bf16; };
// Prepare original Q/K normalization, BF16 beta and FP32 decay values into
// the engine's live token-major tensors. Returned matrices belong to this owner.
NativeGdnMatrices gb10_prepare_native_gdn(std::size_t layer, const void* convolution,
    const void* a, const void* b, void* q, void* k, void* v, void* g, void* beta,
    std::size_t tokens);
// G is the original chunk-local FP32 cumsum. All live spans must be distinct.
// Uses owned scratch for matrix inversion and recurrent state, then commits
// the last FP32 state directly to the engine. Returns the AOT launch count.
std::size_t gb10_native_gdn_pipeline(const void* q, const void* k, const void* v,
    const void* g, const void* beta, void* w, void* u, void* output, void* state,
    std::size_t tokens);
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
