// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aima_port {
class Gb10DecodeMoeOwner {
 public:
  Gb10DecodeMoeOwner();
  ~Gb10DecodeMoeOwner();
  Gb10DecodeMoeOwner(const Gb10DecodeMoeOwner&) = delete;
  Gb10DecodeMoeOwner& operator=(const Gb10DecodeMoeOwner&) = delete;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
struct Gb10DecodeMoeWeights {
  const void *router, *shared_gate, *shared_gate_projection, *shared_up_projection;
  const void *shared_down, *routed_gate_up, *routed_down;
};
// All buffers are the engine's real, disjoint BF16 stage outputs, except the
// eight FP32 routing weights and eight int32 expert indices. Shared input is
// the engine's packed [scalar gate,512 gates,512 ups] row.
struct Gb10DecodeMoeBuffers {
  void *shared_input, *shared_activation, *shared_down, *shared_output;
  void *router, *router_indices, *router_weights;
  void *routed_gate_up, *routed_activation, *routed_weighted, *routed_output, *combined;
};
bool gb10_decode_moe_enabled();
// Borrow immutable, identity-verified tables during cold prefill. The decode
// owner stays alive through the complete request and remains idle here.
const uint16_t* gb10_moe_silu_table();
const uint32_t* gb10_moe_router_exp_table();
// Ordered layers0..39 per token, default stream only. The device error flag
// accumulates across the complete token and is checked at layer39 before any
// token can be published. Launch/flag failures poison this request owner.
void gb10_decode_moe(std::size_t layer, const void* input,
    const Gb10DecodeMoeWeights& weights, const Gb10DecodeMoeBuffers& buffers,
    void* stream);
// Preserve the last live MoE/residual operands before their rounded carrier
// overwrites the input. Final RMSNorm must use the unrounded FP32 sum variance.
void gb10_decode_moe_save_terminal(const void* combined, const void* residual,
                                  void* carrier, void* stream);
bool gb10_decode_moe_terminal_norm(const void* carrier, const void* weight,
                                  void* output, void* stream);
}  // namespace aima_port
