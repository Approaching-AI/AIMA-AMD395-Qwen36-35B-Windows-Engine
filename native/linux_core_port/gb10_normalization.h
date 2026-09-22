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
}
