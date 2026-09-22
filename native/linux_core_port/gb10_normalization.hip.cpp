// SPDX-License-Identifier: Apache-2.0
#include "gb10_normalization.h"
#include "gb10_gdn.h"
#include "aima/sha256.h"
#include "../providers/gdn/sm121_q2_gated_math.h"
#include "../providers/gdn/sm121_mtp_residual.h"
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
  ~State() { if (silu) hipFree(silu); if (residual) hipFree(residual); }
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
// Eight independent 32-lane heads share a CTA. The arithmetic is the existing
// original short-row norm: four adjacent BF16 values per lane and ordered
// FP32 reduction. Prefill's captured host invstd stage is no longer consumed.
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
  hipLaunchKernelGGL(gated_kernel, dim3(tokens * 4u), dim3(256), 0,
      static_cast<hipStream_t>(stream), static_cast<const uint16_t*>(core),
      static_cast<const uint16_t*>(z), static_cast<const uint16_t*>(weight),
      static_cast<uint16_t*>(output), table, active->silu, static_cast<unsigned>(tokens * 32u));
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
}  // namespace aima_port
