// SPDX-License-Identifier: Apache-2.0
#include "gb10_ordered_attention.h"
#include "aima/aot_kernel.h"
#include "aima/sha256.h"
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima_port {
namespace {
Gb10OrderedAttentionOwner* active_ordered = nullptr;
constexpr std::size_t tokens = 8192, slab = 128, heads = 16, width = 256;
constexpr std::size_t query_bytes = tokens * heads * width * 2;
constexpr std::size_t kv_bytes = tokens * 2 * width * 2;
constexpr std::size_t score_bytes = slab * heads * tokens * 4;
constexpr std::size_t probability_bytes = slab * heads * tokens * 2;
constexpr std::size_t scale_bytes = slab * heads * (tokens / 32 + 1) * 4;
constexpr std::size_t queue_bytes = slab * heads * width * 4;
constexpr std::size_t exp2_bytes = 183174448, reciprocal_bytes = 8388640;
static_assert(score_bytes + probability_bytes + scale_bytes + kv_bytes + queue_bytes + 4 ==
              Gb10OrderedAttentionOwner::scratch_bytes);
struct OrderedAttentionImage {
  const unsigned char* data;
  std::size_t bytes;
  const char* symbol;
  const char* sha256;
  unsigned warps, shared;
};
#include "gb10_ordered_attention_images.inc"
void check(hipError_t value, const char* operation) {
  if (value != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(value));
}
struct Device {
  void* pointer = nullptr;
  ~Device() { if (pointer) hipFree(pointer); }
  Device() = default;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void allocate(std::size_t bytes) { check(hipMalloc(&pointer, bytes), "Ordered attention scratch allocation"); }
};
struct Span { const void* pointer; std::size_t bytes; unsigned alignment; };
void validate_spans(const std::vector<Span>& spans) {
  for (std::size_t i = 0; i < spans.size(); ++i) {
    const auto& s = spans[i];
    const auto first = reinterpret_cast<std::uintptr_t>(s.pointer);
    if (!first || first % s.alignment || !s.bytes || first > UINTPTR_MAX - s.bytes)
      throw std::invalid_argument("Ordered attention buffer is absent, unaligned or wraps address space");
    for (std::size_t j = 0; j < i; ++j) {
      const auto previous = reinterpret_cast<std::uintptr_t>(spans[j].pointer);
      if (first < previous + spans[j].bytes && previous < first + s.bytes)
        throw std::invalid_argument("Ordered attention live buffer spans overlap");
    }
  }
}
}  // namespace

struct Gb10OrderedAttentionOwner::Impl {
  const unsigned char* exp2;
  const unsigned char* reciprocal;
  std::array<std::unique_ptr<aima::AotKernel>, 4> kernels;
  Device scores, probability, scales, values, indices, count;
  Impl(const unsigned char* exp2_table, const unsigned char* reciprocal_table)
      : exp2(exp2_table), reciprocal(reciprocal_table) {
    if (active_ordered) throw std::logic_error("An ordered attention owner is already active");
    validate_spans({{exp2,exp2_bytes,4},{reciprocal,reciprocal_bytes,4}});
    for (unsigned i = 0; i < kernels.size(); ++i) {
      const auto& image = ordered_attention_images[i];
      if (aima::sha256_bytes(image.data, image.bytes) != image.sha256)
        throw std::runtime_error("Ordered attention embedded image identity differs");
      kernels[i] = std::make_unique<aima::AotKernel>(
          std::vector<unsigned char>(image.data, image.data + image.bytes), image.symbol);
    }
    scores.allocate(score_bytes);
    probability.allocate(probability_bytes);
    scales.allocate(scale_bytes);
    values.allocate(kv_bytes);
    indices.allocate(queue_bytes);
    count.allocate(sizeof(std::int32_t));
    std::vector<std::int32_t> queue(slab * heads * width);
    for (std::size_t i = 0; i < queue.size(); ++i) queue[i] = static_cast<std::int32_t>(i);
    const std::int32_t cells = static_cast<std::int32_t>(queue.size());
    check(hipMemcpy(indices.pointer, queue.data(), queue_bytes, hipMemcpyHostToDevice), "Ordered attention queue upload");
    check(hipMemcpy(count.pointer, &cells, sizeof(cells), hipMemcpyHostToDevice), "Ordered attention count upload");
  }
  void run(unsigned index, unsigned x, unsigned y, unsigned z, const std::vector<void*>& parameters) {
    const auto& image = ordered_attention_images[index];
    kernels[index]->launch(aima::AotLaunchConfig{x,y,z,image.warps,32,image.shared}, parameters);
  }
};

Gb10OrderedAttentionOwner::Gb10OrderedAttentionOwner(const unsigned char* exp2,
    const unsigned char* reciprocal) : impl_(std::make_unique<Impl>(exp2, reciprocal)) {
  active_ordered = this;
}
Gb10OrderedAttentionOwner::~Gb10OrderedAttentionOwner() {
  if (active_ordered == this) { hipDeviceSynchronize(); active_ordered = nullptr; }
}
bool gb10_ordered_attention_setting(const char* value) {
  if (!value || std::string(value) == "0") return false;
  if (std::string(value) == "1") return true;
  throw std::invalid_argument("Ordered attention setting must be 0 or 1");
}
bool gb10_ordered_attention_enabled() { return active_ordered != nullptr; }
std::size_t gb10_ordered_attention_prefill(const void* query, const void* key,
    const void* value, void* output, std::size_t active_tokens, std::size_t cache_end,
    std::size_t cache_start, void* stream) {
  if (!active_ordered) throw std::logic_error("Ordered attention owner is absent");
  return active_ordered->launch(query, key, value, output, active_tokens, cache_end, cache_start, stream);
}

std::size_t Gb10OrderedAttentionOwner::launch(const void* query, const void* key,
    const void* value, void* output, std::size_t active_tokens, std::size_t cache_end,
    std::size_t cache_start, void* stream) {
  if (active_ordered != this || active_tokens != tokens || cache_end != tokens || cache_start || stream)
    throw std::invalid_argument("Ordered attention requires cold q8192 on the default stream");
  auto& s = *impl_;
  validate_spans({{query,query_bytes,2},{key,kv_bytes,2},{value,kv_bytes,2},{output,query_bytes,2},
      {s.scores.pointer,score_bytes,4},{s.probability.pointer,probability_bytes,2},
      {s.scales.pointer,scale_bytes,4},{s.values.pointer,kv_bytes,2},
      {s.indices.pointer,queue_bytes,4},{s.count.pointer,4,4},
      {s.exp2,exp2_bytes,4},{s.reciprocal,reciprocal_bytes,4}});
  std::int32_t total = tokens, count = slab;
  // Stable image order is pack, QK, online probability, selected PV.
  s.run(0,16,256,1,{&value,&s.values.pointer,&total});
  std::size_t launches = 1;
  void* unused_debug = nullptr;
  for (std::int32_t start = 0; start < total; start += count) {
    s.run(1,16,1024,16,{&query,&key,&s.scores.pointer,&total,&start,&count});
    s.run(2,32,16,1,{&s.scores.pointer,&s.probability.pointer,&s.scales.pointer,
                   &s.exp2,&total,&start,&count});
    void* destination = static_cast<unsigned char*>(output) + std::size_t(start) * heads * width * 2;
    s.run(3,1024,1,1,{&s.probability.pointer,&s.values.pointer,&s.scales.pointer,&s.reciprocal,
        &s.count.pointer,&s.indices.pointer,&destination,&unused_debug,&total,&start,&count});
    launches += 3;
  }
  return launches;
}
}  // namespace aima_port
