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
// Arm optional completed-GPU diagnostics after model loading/warmup. Storage
// is allocated by the owner only when AIMA_PORT_PREFILL_PROJECTION_PROFILE=1.
void gb10_prefill_projection_profile_begin();
bool gb10_prefill_projection_wmma_enabled(std::size_t reduction);
bool gb10_prefill_projection_coarse_enabled();
bool gb10_prefill_projection_tuned_gemm_enabled(std::size_t rows, std::size_t reduction);
bool gb10_prefill_gemm_algorithm_matches(const void* algorithm, std::size_t bytes, int library_version);
// Bind the original linear-OUT selector at its actual call site, rather than
// inferring layer kind from a K4096 shape shared with full attention.
class Gb10PrefillLinearOutputScope {
 public:
  Gb10PrefillLinearOutputScope(std::size_t tokens, unsigned layer);
  ~Gb10PrefillLinearOutputScope();
  Gb10PrefillLinearOutputScope(const Gb10PrefillLinearOutputScope&) = delete;
  Gb10PrefillLinearOutputScope& operator=(const Gb10PrefillLinearOutputScope&) = delete;
 private:
  void* state_ = nullptr;
};
class Gb10PrefillFullOutputScope {
 public:
  Gb10PrefillFullOutputScope(std::size_t tokens, unsigned layer);
  ~Gb10PrefillFullOutputScope();
  Gb10PrefillFullOutputScope(const Gb10PrefillFullOutputScope&) = delete;
  Gb10PrefillFullOutputScope& operator=(const Gb10PrefillFullOutputScope&) = delete;
 private:
  void* state_ = nullptr;
};
void* gb10_prefill_projection_buffer(std::size_t tokens, std::size_t rows,
                                    std::size_t reduction, void* stream);
void gb10_prefill_projection_fallback(const void* input, const void* weights,
    std::size_t tokens, std::size_t rows, std::size_t reduction,
    bool weight_rows_contiguous, void* stream);
void gb10_prefill_projection_coarse(const void* input, const void* weights,
    std::size_t tokens, std::size_t rows, std::size_t reduction,
    bool weight_rows_contiguous, void* stream);
void gb10_prefill_projection_finish(const void* input, const void* weights,
    void* output, std::size_t tokens, std::size_t rows, std::size_t reduction,
    bool weight_rows_contiguous, void* stream);
}
