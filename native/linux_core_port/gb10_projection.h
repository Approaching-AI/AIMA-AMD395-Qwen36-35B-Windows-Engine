// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "aima/bf16_wvsplitk.h"
#include <cstdint>
#include <memory>
namespace aima_port {
// The fixed model's complete embedding inverse-RMS table is loaded before
// command-ready. Token selection always comes from the live request.
class Gb10ProjectionOwner {
 public:
  explicit Gb10ProjectionOwner(const std::vector<uint32_t>& prompt);
  ~Gb10ProjectionOwner();
  Gb10ProjectionOwner(const Gb10ProjectionOwner&) = delete;
  Gb10ProjectionOwner& operator=(const Gb10ProjectionOwner&) = delete;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
void set_gb10_decode_token(uint32_t token);
void gb10_embedding_norm(const void* input, const void* weight, void* output,
                         std::size_t tokens, void* stream = nullptr);
void gb10_projection(const void* weight, const void* input, const void* bias,
                     void* output, std::size_t rows, std::size_t reduction, void* stream);
void gb10_projection_group(const aima::Bf16WvSplitKProjection* projections,
                          std::size_t count, const void* input,
                          std::size_t reduction, void* stream);
}
