// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <memory>
namespace aima_port {
class Gb10NormalizationOwner {
 public:
  Gb10NormalizationOwner();
  ~Gb10NormalizationOwner();
  Gb10NormalizationOwner(const Gb10NormalizationOwner&) = delete;
  Gb10NormalizationOwner& operator=(const Gb10NormalizationOwner&) = delete;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
void gb10_gated_norm(const void* core, const void* z, const void* weight,
                     void* output, std::size_t tokens, void* stream = nullptr);
void gb10_residual_norm(const void* input, const void* residual, const void* weight,
    void* residual_output, void* norm_output, std::size_t tokens, void* stream = nullptr);
// Snapshot one live residual row before the MoE tail can overwrite an alias.
// The borrowed copy is consumed on the default stream before the next call.
const void* gb10_preserve_decode_residual(const void* residual, void* stream = nullptr);
// Optional model-config rotary cache and GB10 head normalization. Ordinary
// text positions are explicit; this does not accept multimodal position plans.
bool gb10_full_head_norm_rope_enabled();
void gb10_full_head_norm_rope(const void* q_gate, const void* k_raw, const void* v_raw,
    const void* q_weight, const void* k_weight, void* q_output, void* k_output, void* v_output,
    std::size_t tokens, std::size_t q_stride, std::size_t k_stride, std::size_t v_stride,
    std::size_t first_position, void* stream = nullptr);
}
