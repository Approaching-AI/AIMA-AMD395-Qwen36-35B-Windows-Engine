// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aima_port {
class Gb10MoeOwner {
 public:
  Gb10MoeOwner();
  ~Gb10MoeOwner();
  Gb10MoeOwner(const Gb10MoeOwner&) = delete;
  Gb10MoeOwner& operator=(const Gb10MoeOwner&) = delete;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
// Register immutable model weights before READY, and drain/unregister before
// the model owner frees them. The provider owns only their row metadata.
void gb10_moe_register_weights(const uint16_t* const* gate_up,
                              const uint16_t* const* down, std::size_t count);
void gb10_moe_release_weights() noexcept;
void gb10_prefill_moe(std::size_t layer, const void* input, const void* residual,
    const void* router, const void* gate_up, const void* down,
    const void* shared_gate, const void* shared_gate_projection,
    const void* shared_up_projection, const void* shared_down,
    void* output, std::size_t tokens, bool terminal_only = false);
// Experimental FP32 expert GEMMs and SM121 replay for cold q8192 layers0..38. The final
// layer keeps the qualified terminal provider. Incomplete native work poisons
// the owner, so a partial carrier can never enter the next normalization.
bool gb10_native_moe_prefill_enabled(std::size_t layer, std::size_t tokens);
class Gb10NativeMoeScope {
 public:
  Gb10NativeMoeScope(std::size_t layer, const void* input, const void* residual,
      const void* gate_up, const void* down, void* output, std::size_t tokens);
  ~Gb10NativeMoeScope();
  Gb10NativeMoeScope(const Gb10NativeMoeScope&) = delete;
  Gb10NativeMoeScope& operator=(const Gb10NativeMoeScope&) = delete;
  void* router_weights() const;
  void project_experts(bool down, const void* input, const void* ids,
      const void* sorted_routes, const void* block_experts, const void* padded_count,
      void* output);
  void finish(const void* weighted, const void* shared, void* routed, void* combined);
 private:
  void* owner_;
  const void* residual_;
  void* output_;
  std::size_t layer_;
  bool completed_ = false;
};
void gb10_native_moe_shared_gate(const void* input, const void* weight, void* output);
void gb10_native_moe_shared_activation(const void* gate, const void* up, void* output);
void gb10_native_moe_shared_scale(const void* gate, const void* down, void* output);
void gb10_native_moe_router(const void* logits, void* ids);
void gb10_native_moe_expert_activation(const void* gate_up, void* output);
// One live q8192 layer boundary. Only layer zero starts a new sequence; every
// subsequent norm must consume the preceding MoE's unrounded FP32 carrier.
bool gb10_moe_input_norm(std::size_t layer, const void* carrier,
    const void* weight, void* output, std::size_t tokens);
bool gb10_moe_terminal_norm(const void* carrier, const void* weight,
                           void* output, void* stream = nullptr);
}  // namespace aima_port
