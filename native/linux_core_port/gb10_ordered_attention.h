// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <memory>

namespace aima_port {
// Owns cold q8192 scratch and the four qualified gfx1151 code objects.
// Table pointers must come from the existing SHA-verified exp2 and reciprocal
// owners and must outlive this object. They are borrowed and never modified.
class Gb10OrderedAttentionOwner {
 public:
  Gb10OrderedAttentionOwner(const unsigned char* exp2, const unsigned char* reciprocal);
  ~Gb10OrderedAttentionOwner();
  Gb10OrderedAttentionOwner(const Gb10OrderedAttentionOwner&) = delete;
  Gb10OrderedAttentionOwner& operator=(const Gb10OrderedAttentionOwner&) = delete;
  // Q/output: BF16 [8192,16,256]; K/V: BF16 [8192,2,256]. No layout conversion
  // is applied to Q/K. V is packed into private scratch once per call.
  // Enqueues 193 AOT launches on the default stream and returns that count.
  std::size_t launch(const void* query, const void* key, const void* value,
                    void* output, std::size_t tokens, std::size_t cache_end,
                    std::size_t cache_start, void* stream = nullptr);
  static constexpr std::size_t scratch_bytes = 113254404;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
bool gb10_ordered_attention_setting(const char* value);
bool gb10_ordered_attention_enabled();
std::size_t gb10_ordered_attention_prefill(const void* query, const void* key,
    const void* value, void* output, std::size_t tokens, std::size_t cache_end,
    std::size_t cache_start, void* stream = nullptr);
}  // namespace aima_port
