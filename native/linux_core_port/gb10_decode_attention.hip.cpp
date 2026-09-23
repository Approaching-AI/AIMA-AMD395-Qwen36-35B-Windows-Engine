// SPDX-License-Identifier: Apache-2.0
#include "gb10_decode_attention.h"
#include "gb10_gdn.h"
#include "aima/sha256.h"
#include "../providers/ck_fmha/blackwell_attention.h"
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima_port {
namespace {
constexpr std::size_t kMaximumTokens = 262144;
constexpr std::size_t kReciprocalBytes = 8388640;
void check(hipError_t status, const char* operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}
struct Device {
  void* data = nullptr;
  ~Device() { if (data) hipFree(data); }
  void allocate(std::size_t bytes) { check(hipMalloc(&data, bytes), "Decode attention allocation"); }
  template<class T> T* as() const { return static_cast<T*>(data); }
};
struct State {
  Device reciprocal, scratch;
  std::size_t capacity = 0, score_capacity = 0;
  bool terminal_only = false;
};
State* active = nullptr;
std::vector<unsigned char> read_reciprocal(const std::filesystem::path& path) {
  if (std::filesystem::file_size(path) != kReciprocalBytes)
    throw std::runtime_error("Decode attention reciprocal table size differs");
  std::vector<unsigned char> bytes(kReciprocalBytes);
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!file || file.peek() != std::char_traits<char>::eof() ||
      aima::sha256_bytes(bytes.data(), bytes.size()) !=
        "d2e557543f6bc51f5141ba6414000cd8ed892e2e915eda19245c3cae22c16b39")
    throw std::runtime_error("Decode attention reciprocal table identity differs");
  return bytes;
}
struct Span { const void* pointer; std::size_t bytes; };
bool valid(Span span) {
  const auto first = reinterpret_cast<std::uintptr_t>(span.pointer);
  return first && !(first & 1u) && span.bytes &&
      first <= std::numeric_limits<std::uintptr_t>::max() - span.bytes;
}
bool disjoint(Span a, Span b) {
  const auto first = reinterpret_cast<std::uintptr_t>(a.pointer);
  const auto second = reinterpret_cast<std::uintptr_t>(b.pointer);
  return first + a.bytes <= second || second + b.bytes <= first;
}
// MtpCache selects the original Q2 denominator FMA. A typed view preserves
// that arithmetic while addressing the engine's separate, resident V plane.
struct Values {
  const uint16_t* plane;
  __device__ uint16_t value(unsigned token, unsigned head, unsigned column) const {
    return plane[std::size_t(token) * 512u + head * 256u + column];
  }
};
static __global__ void publish_context(const float* source, uint16_t* output) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 4096u) output[i] = qrt_blackwell_attention::f32_to_bf16(source[i]);
}
}  // namespace
struct Gb10DecodeAttentionOwner::Impl { State state; };
Gb10DecodeAttentionOwner::Gb10DecodeAttentionOwner(std::size_t capacity)
    : impl_(std::make_unique<Impl>()) {
  if (!capacity || capacity > kMaximumTokens)
    throw std::invalid_argument("Invalid decode attention cache capacity");
  if (active) throw std::runtime_error("A decode attention owner is already active");
  const char* terminal = std::getenv("AIMA_PORT_PREFILL_TERMINAL_ONLY");
  if (terminal && *terminal && std::string(terminal) != "0" && std::string(terminal) != "1")
    throw std::invalid_argument("Unsupported terminal prefill mode");
  const bool terminal_only = terminal && std::string(terminal) == "1";
  const char* setting = std::getenv("AIMA_PORT_DECODE_ATTENTION");
  if (!setting || !*setting) {
    if (terminal_only) throw std::invalid_argument("Terminal prefill requires the attention owner");
    return;
  }
  if (std::string(setting) != "1") throw std::invalid_argument("Unsupported decode attention mode");
  if (terminal_only && capacity < 8192)
    throw std::invalid_argument("Terminal prefill requires a complete q8192 cache");
  (void)gb10_exp2_table();
  const char* path = std::getenv("AIMA_PORT_ATTENTION_RCP_TABLE");
  if (!path || !*path) throw std::runtime_error("Missing decode attention reciprocal table");
  const auto bytes = read_reciprocal(std::filesystem::u8path(path));
  auto& s = impl_->state;
  s.terminal_only = terminal_only;
  s.capacity = capacity;
  s.score_capacity = (capacity + 31u) & ~std::size_t(31u);
  s.reciprocal.allocate(bytes.size());
  check(hipMemcpy(s.reciprocal.data, bytes.data(), bytes.size(), hipMemcpyHostToDevice),
        "Decode attention reciprocal upload");
  s.scratch.allocate((16u * s.score_capacity + 4096u) * sizeof(float));
  active = &s;
}
Gb10DecodeAttentionOwner::~Gb10DecodeAttentionOwner() {
  if (active == &impl_->state) { hipDeviceSynchronize(); active = nullptr; }
}
bool gb10_decode_attention_enabled() { return active != nullptr; }
const unsigned char* gb10_attention_reciprocal_table() {
  if (!active || !active->reciprocal.data)
    throw std::logic_error("Attention reciprocal owner is absent");
  return active->reciprocal.as<unsigned char>();
}
bool gb10_prefill_terminal_only_enabled() { return active && active->terminal_only; }
void gb10_prefill_terminal_attention(const void* query, const void* key,
    const void* value, void* output, std::size_t cache_end) {
  if (!gb10_prefill_terminal_only_enabled() || cache_end != 8192 || cache_end > active->capacity)
    throw std::invalid_argument("Invalid terminal prefill owner or extent");
  const std::array<Span,4> spans{{{query,8192u}, {key,cache_end*1024u},
                                {value,cache_end*1024u}, {output,16384u}}};
  if (reinterpret_cast<std::uintptr_t>(output) & 3u)
    throw std::invalid_argument("Unaligned terminal prefill F32 output");
  for (std::size_t i = 0; i < spans.size(); ++i) {
    if (!valid(spans[i])) throw std::invalid_argument("Invalid terminal prefill buffer");
    for (std::size_t j = 0; j < i; ++j)
      if (!disjoint(spans[i], spans[j])) throw std::invalid_argument("Overlapping terminal prefill buffers");
  }
  const auto* exp2 = gb10_exp2_table();
  float* scores = active->scratch.as<float>();
  hipLaunchKernelGGL(qrt_blackwell_attention::blackwell_compact_query_scores_kernel,
      dim3(8192), dim3(256), 0, nullptr, static_cast<const uint16_t*>(query),
      static_cast<const uint16_t*>(key), scores, 8191u, 1u, 8192u, 8191u);
  check(hipGetLastError(), "Terminal prefill QK");
  // MtpCache=false is intentional: the qualified CK terminal replacement uses
  // separate multiply/add for the denominator, unlike the Q2 decode FMA.
  hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_blackwell_attention::blackwell_exact_attention_kernel<
      true, true, false, false, false, false, false, false>),
      dim3(16), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(nullptr), static_cast<const uint16_t*>(nullptr),
      static_cast<const uint16_t*>(value), static_cast<float*>(output), 8191u, 0u, exp2,
      static_cast<float*>(nullptr), static_cast<float*>(nullptr), true,
      active->reciprocal.as<unsigned char>(), scores, 8192u,
      static_cast<const uint16_t*>(nullptr), 0u);
  check(hipGetLastError(), "Terminal prefill softmax/PV");
}
void gb10_decode_attention(const void* query, const void* key, const void* value,
    void* output, std::size_t cache_end, void* stream) {
  if (!active || !cache_end || cache_end > active->capacity || stream)
    throw std::invalid_argument("Invalid decode attention owner, extent or stream");
  const std::array<Span,4> spans{{{query,8192u}, {key,cache_end*1024u},
                                {value,cache_end*1024u}, {output,8192u}}};
  for (std::size_t i = 0; i < spans.size(); ++i) {
    if (!valid(spans[i])) throw std::invalid_argument("Invalid decode attention buffer");
    for (std::size_t j = 0; j < i; ++j)
      if (!disjoint(spans[i], spans[j])) throw std::invalid_argument("Overlapping decode attention buffers");
  }
  const unsigned position = static_cast<unsigned>(cache_end - 1u);
  const unsigned stride = static_cast<unsigned>((cache_end + 31u) & ~std::size_t(31u));
  const auto* exp2 = gb10_exp2_table();
  float* scores = active->scratch.as<float>();
  float* context = scores + 16u * active->score_capacity;
  hipLaunchKernelGGL(qrt_blackwell_attention::blackwell_compact_query_scores_kernel,
      dim3(stride), dim3(256), 0, nullptr, static_cast<const uint16_t*>(query),
      static_cast<const uint16_t*>(key), scores, position, 1u, stride, position);
  check(hipGetLastError(), "Decode attention QK");
  hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_blackwell_attention::blackwell_exact_attention_kernel<
      true, true, false, false, false, false, false, true, Values>),
      dim3(16), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(nullptr), static_cast<const uint16_t*>(nullptr),
      static_cast<const uint16_t*>(nullptr), context, position, 0u, exp2,
      static_cast<float*>(nullptr), static_cast<float*>(nullptr), true,
      active->reciprocal.as<unsigned char>(), scores, stride,
      Values{static_cast<const uint16_t*>(value)}, position);
  check(hipGetLastError(), "Decode attention softmax/PV");
  hipLaunchKernelGGL(publish_context, dim3(16), dim3(256), 0, nullptr,
      context, static_cast<uint16_t*>(output));
  check(hipGetLastError(), "Decode attention BF16 context");
}
}  // namespace aima_port
