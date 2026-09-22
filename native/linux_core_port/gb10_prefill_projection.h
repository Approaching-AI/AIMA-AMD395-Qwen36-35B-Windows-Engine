// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <memory>
namespace aima_port {
class Gb10PrefillProjectionOwner {
 public:
  Gb10PrefillProjectionOwner();
  ~Gb10PrefillProjectionOwner();
  Gb10PrefillProjectionOwner(const Gb10PrefillProjectionOwner&) = delete;
  Gb10PrefillProjectionOwner& operator=(const Gb10PrefillProjectionOwner&) = delete;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
bool gb10_prefill_projection_shape(std::size_t tokens, std::size_t rows,
                                   std::size_t reduction, bool bias);
void* gb10_prefill_projection_buffer(std::size_t tokens, std::size_t rows,
                                    std::size_t reduction, void* stream);
void gb10_prefill_projection_fallback(const void* input, const void* weights,
    std::size_t tokens, std::size_t rows, std::size_t reduction,
    bool weight_rows_contiguous, void* stream);
void gb10_prefill_projection_finish(const void* input, const void* weights,
    void* output, std::size_t tokens, std::size_t rows, std::size_t reduction,
    bool weight_rows_contiguous, void* stream);
}
