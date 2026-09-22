// SPDX-License-Identifier: Apache-2.0
#include "gb10_normalization.h"
#include "gb10_gdn.h"
#include "aima/sha256.h"
#include "gb10_normalization_math.h"
#include "../providers/gdn/sm121_mtp_residual.h"
#include "../providers/gdn/sm121_mtp_kv_math.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace aima_port {
namespace {
struct State {
  float* silu = nullptr;
  void* residual = nullptr;
  uint16_t* rope = nullptr;
  ~State() { if (silu) hipFree(silu); if (residual) hipFree(residual); if (rope) hipFree(rope); }
};
State* active = nullptr;
std::vector<float> read_silu(const std::filesystem::path& path) {
  constexpr std::size_t bytes = 65536 * sizeof(float);
  if (std::filesystem::file_size(path) != bytes)
    throw std::runtime_error("Gated SiLU table size differs");
  std::vector<float> values(65536);
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(values.data()), bytes);
  if (!file || file.peek() != std::char_traits<char>::eof() ||
      aima::sha256_bytes(values.data(), bytes) !=
        "f8b4983266a2d26f64a154c0c53c6acd6616e3298e7eb2e128c7431be586c97c")
    throw std::runtime_error("Gated SiLU table identity differs");
  return values;
}
std::vector<uint16_t> read_rope(const std::filesystem::path& path) {
  constexpr std::size_t bytes = 262144 * 64 * sizeof(uint16_t);
  if (std::filesystem::file_size(path) != bytes)
    throw std::runtime_error("Full-attention rotary table size differs");
  std::vector<uint16_t> values(bytes / sizeof(uint16_t));
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(values.data()), bytes);
  if (!file || file.peek() != std::char_traits<char>::eof() ||
      aima::sha256_bytes(values.data(), bytes) !=
        "ba12ce218327d4cf23aac7dfacd8e9efbc99fd207611a8466227089838ef0e80")
    throw std::runtime_error("Full-attention rotary table identity differs");
  return values;
}
static __global__ void full_head_norm_rope_kernel(
    const uint16_t* q_gate, const uint16_t* k_raw, const uint16_t* v_raw,
    const uint16_t* q_weight, const uint16_t* k_weight, uint16_t* q_output,
    uint16_t* k_output, uint16_t* v_output, unsigned q_stride, unsigned k_stride,
    unsigned v_stride, unsigned first_position, const unsigned char* rsqrt,
    const uint16_t* rope) {
  using namespace qrt_sm121_q1;
  __shared__ float values[256], warps[4], inverse;
  __shared__ uint16_t normalized[256];
  const unsigned token = blockIdx.x, head = blockIdx.y, channel = threadIdx.x;
  const bool key = head >= 16;
  const unsigned local_head = key ? head - 16 : head;
  const uint16_t* source = key ? k_raw + std::size_t(token) * k_stride + local_head * 256u
                              : q_gate + std::size_t(token) * q_stride + head * 512u;
  const uint16_t* weight = key ? k_weight : q_weight;
  values[channel] = widen(source[channel]);
  __syncthreads();
  if (channel < head_norm_warps(key) * 32u) {
    float sum = head_norm_lane_sumsq(values, channel, key);
    for (unsigned mask = 16; mask; mask >>= 1)
      sum = add(sum, __shfl_xor(sum, mask, 32));
    if ((channel & 31u) == 0) warps[channel / 32u] = sum;
  }
  __syncthreads();
  if (channel == 0)
    inverse = qrt_sm121_rsqrt::evaluate(rsqrt,
        add(multiply(head_norm_warp_sum(warps, key), 1.f / 256.f), 1.e-6f));
  __syncthreads();
  normalized[channel] = qrt_sm121_mtp::normalized(values[channel], inverse, weight[channel]);
  __syncthreads();
  const auto* coefficients = rope + std::size_t(first_position + token) * 64u;
  const uint16_t result = qrt_sm121_mtp::key_rotated(normalized, channel, coefficients);
  if (key) {
    const std::size_t destination = std::size_t(token) * 512u + local_head * 256u + channel;
    k_output[destination] = result;
    if (v_output) v_output[destination] = v_raw[std::size_t(token) * v_stride + local_head * 256u + channel];
  } else {
    q_output[std::size_t(token) * 4096u + head * 256u + channel] = result;
  }
}
// Eight independent 32-lane heads share a CTA on the short decode route.
static __global__ void gated_kernel(const uint16_t* core, const uint16_t* z,
    const uint16_t* weight, uint16_t* output, const unsigned char* rsqrt,
    const float* silu, unsigned heads) {
  const unsigned head = blockIdx.x * 8u + threadIdx.x / 32u, lane = threadIdx.x & 31u;
  if (head >= heads) return;
  const std::size_t first = std::size_t(head) * 128u;
  float sum = qrt_sm121_q2::gated_lane_sum(core + first, lane);
  for (unsigned offset = 16; offset; offset >>= 1)
    sum = qrt_sm121_q1::add(sum, __shfl_down(sum, offset, 32));
  const float inverse = qrt_sm121_q2::gated_inverse(__shfl(sum, 0, 32), rsqrt);
  for (unsigned item = 0; item < 4; ++item) {
    const unsigned column = lane * 4u + item;
    output[first + column] = qrt_sm121_q2::gated_value(core[first + column],
        z[first + column], weight[column], inverse, silu);
  }
}
static __global__ void gated_prefill_kernel(const uint16_t* core, const uint16_t* z,
    const uint16_t* weight, uint16_t* output, const unsigned char* rsqrt,
    const float* silu, unsigned heads) {
  const unsigned head = blockIdx.x * 16u + threadIdx.x / 16u, lane = threadIdx.x & 15u;
  if (head >= heads) return;
  const std::size_t first = std::size_t(head) * 128u;
  float sum = gated_prefill_lane_sum(core + first, lane);
  for (unsigned offset = 8; offset; offset >>= 1)
    sum = qrt_sm121_q1::add(sum, __shfl_xor(sum, offset, 16));
  const float inverse = qrt_sm121_q2::gated_inverse(sum, rsqrt);
  for (unsigned item = 0; item < 8; ++item) {
    const unsigned column = lane * 8u + item;
    output[first + column] = qrt_sm121_q2::gated_value(core[first + column],
        z[first + column], weight[column], inverse, silu);
  }
}
}  // namespace
struct Gb10NormalizationOwner::Impl { State state; };
Gb10NormalizationOwner::Gb10NormalizationOwner() : impl_(std::make_unique<Impl>()) {
  if (active) throw std::runtime_error("A GB10 normalization owner is already active");
  (void)gb10_rsqrt_table();
  const char* path = std::getenv("AIMA_PORT_GATED_SILU_TABLE");
  if (!path || !*path) throw std::runtime_error("Missing gated SiLU table");
  const auto values = read_silu(std::filesystem::u8path(path));
  auto& s = impl_->state;
  if (hipMalloc(reinterpret_cast<void**>(&s.silu), values.size() * sizeof(float)) != hipSuccess ||
      hipMemcpy(s.silu, values.data(), values.size() * sizeof(float), hipMemcpyHostToDevice) != hipSuccess)
    throw std::runtime_error("Gated SiLU table upload failed");
  if (hipMalloc(&s.residual, 2048 * sizeof(uint16_t)) != hipSuccess)
    throw std::runtime_error("Decode residual snapshot allocation failed");
  const char* rope_path = std::getenv("AIMA_PORT_FULL_ATTENTION_ROPE_TABLE");
  if (rope_path && *rope_path) {
    const auto rotary = read_rope(std::filesystem::u8path(rope_path));
    if (hipMalloc(reinterpret_cast<void**>(&s.rope), rotary.size() * sizeof(uint16_t)) != hipSuccess ||
        hipMemcpy(s.rope, rotary.data(), rotary.size() * sizeof(uint16_t), hipMemcpyHostToDevice) != hipSuccess)
      throw std::runtime_error("Full-attention rotary table upload failed");
  }
  active = &s;
}
Gb10NormalizationOwner::~Gb10NormalizationOwner() {
  if (active == &impl_->state) { hipDeviceSynchronize(); active = nullptr; }
}
void gb10_gated_norm(const void* core, const void* z, const void* weight,
    void* output, std::size_t tokens, void* stream) {
  if (!active || !active->silu || !core || !z || !weight || !output || !tokens || tokens > 8192)
    throw std::invalid_argument("Invalid GB10 gated normalization binding");
  const auto* table = gb10_rsqrt_table();
  if (tokens == 8192) {
    hipLaunchKernelGGL(gated_prefill_kernel, dim3(tokens * 2u), dim3(256), 0,
        static_cast<hipStream_t>(stream), static_cast<const uint16_t*>(core),
        static_cast<const uint16_t*>(z), static_cast<const uint16_t*>(weight),
        static_cast<uint16_t*>(output), table, active->silu, static_cast<unsigned>(tokens * 32u));
  } else {
    hipLaunchKernelGGL(gated_kernel, dim3(tokens * 4u), dim3(256), 0,
        static_cast<hipStream_t>(stream), static_cast<const uint16_t*>(core),
        static_cast<const uint16_t*>(z), static_cast<const uint16_t*>(weight),
        static_cast<uint16_t*>(output), table, active->silu, static_cast<unsigned>(tokens * 32u));
  }
  if (hipGetLastError() != hipSuccess) throw std::runtime_error("GB10 gated normalization launch failed");
}
void gb10_residual_norm(const void* input, const void* residual, const void* weight,
    void* residual_output, void* norm_output, std::size_t tokens, void* stream) {
  if (!active || !tokens || tokens > 8192)
    throw std::invalid_argument("Invalid GB10 residual normalization owner/extent");
  const auto status = qrt_sm121_mtp::launch_residual_normalize(
      static_cast<const uint16_t*>(input), static_cast<const uint16_t*>(residual),
      static_cast<const uint16_t*>(weight), gb10_rsqrt_table(), static_cast<unsigned>(tokens),
      static_cast<uint16_t*>(norm_output), static_cast<uint16_t*>(residual_output),
      static_cast<hipStream_t>(stream));
  if (status != hipSuccess) throw std::runtime_error("GB10 residual normalization launch failed");
}
const void* gb10_preserve_decode_residual(const void* residual, void* stream) {
  if (!active || !active->residual || !residual || stream)
    throw std::invalid_argument("Decode residual snapshot requires a live owner and default stream");
  if (hipMemcpyAsync(active->residual, residual, 2048 * sizeof(uint16_t),
                     hipMemcpyDeviceToDevice, nullptr) != hipSuccess)
    throw std::runtime_error("Decode residual snapshot failed");
  return active->residual;
}
bool gb10_full_head_norm_rope_enabled() { return active && active->rope; }
void gb10_full_head_norm_rope(const void* q_gate, const void* k_raw, const void* v_raw,
    const void* q_weight, const void* k_weight, void* q_output, void* k_output, void* v_output,
    std::size_t tokens, std::size_t q_stride, std::size_t k_stride, std::size_t v_stride,
    std::size_t first_position, void* stream) {
  if (!gb10_full_head_norm_rope_enabled() || !q_gate || !k_raw || !q_weight || !k_weight ||
      !q_output || !k_output || q_output == k_output ||
      (tokens != 1 && tokens != 8192) || first_position >= 262144 || tokens > 262144 - first_position ||
      (q_stride != 8192 && q_stride != 9216) || (k_stride != 512 && k_stride != 9216) ||
      ((v_raw == nullptr) != (v_output == nullptr)) ||
      (v_output && (v_output == q_output || v_output == k_output || (v_stride != 512 && v_stride != 9216))) ||
      (!v_output && v_stride != 0))
    throw std::invalid_argument("Invalid GB10 full-attention head normalization binding");
  hipLaunchKernelGGL(full_head_norm_rope_kernel, dim3(tokens, 18), dim3(256), 0,
      static_cast<hipStream_t>(stream), static_cast<const uint16_t*>(q_gate),
      static_cast<const uint16_t*>(k_raw), static_cast<const uint16_t*>(v_raw),
      static_cast<const uint16_t*>(q_weight), static_cast<const uint16_t*>(k_weight),
      static_cast<uint16_t*>(q_output), static_cast<uint16_t*>(k_output), static_cast<uint16_t*>(v_output),
      static_cast<unsigned>(q_stride), static_cast<unsigned>(k_stride), static_cast<unsigned>(v_stride),
      static_cast<unsigned>(first_position), gb10_rsqrt_table(), active->rope);
  if (hipGetLastError() != hipSuccess) throw std::runtime_error("GB10 full-attention head normalization launch failed");
}
}  // namespace aima_port
