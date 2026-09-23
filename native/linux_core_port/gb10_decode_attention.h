// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <memory>

namespace aima_port {
// Optional batch-one, default-stream owner. Construct after Gb10GdnOwner and
// before READY; its immutable exp2 table is borrowed for the complete lifetime.
class Gb10DecodeAttentionOwner {
 public:
  explicit Gb10DecodeAttentionOwner(std::size_t cache_capacity);
  ~Gb10DecodeAttentionOwner();
  Gb10DecodeAttentionOwner(const Gb10DecodeAttentionOwner&) = delete;
  Gb10DecodeAttentionOwner& operator=(const Gb10DecodeAttentionOwner&) = delete;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
bool gb10_decode_attention_enabled();
// Borrowed from the live SHA-verified decode-attention owner.
const unsigned char* gb10_attention_reciprocal_table();
bool gb10_prefill_terminal_only_enabled();
// Cold q8192 terminal query, with the original prefill denominator arithmetic.
// Q is one BF16 row; output is one F32 row. All 8192 K/V rows remain resident.
void gb10_prefill_terminal_attention(const void* query, const void* key,
    const void* value, void* output, std::size_t cache_end);
// K/V are the existing token-major [cache_end,2,256] planes, including the
// current row. Q/output are one [16,256] BF16 row. No cache writes or copies.
void gb10_decode_attention(const void* query, const void* key, const void* value,
                           void* output, std::size_t cache_end, void* stream);
}  // namespace aima_port
