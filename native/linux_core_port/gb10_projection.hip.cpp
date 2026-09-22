// SPDX-License-Identifier: Apache-2.0
#include "gb10_projection.h"
#include "aima/sha256.h"
#include "../providers/moe_accumulator/sm121_wave16.h"
#include "../providers/gdn/sm121_q1_math.h"
#include <limits>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace aima_port {
namespace {
constexpr unsigned vocabulary = 248320, prompt_tokens = 8192;
struct Device {
  void* data = nullptr;
  ~Device() { if (data) hipFree(data); }
  void upload(const void* source, std::size_t bytes) {
    if (hipMalloc(&data, bytes) != hipSuccess ||
        hipMemcpy(data, source, bytes, hipMemcpyHostToDevice) != hipSuccess)
      throw std::runtime_error("GB10 embedding table upload failed");
  }
};
struct State {
  Device inverse, prompt;
  uint32_t decode_token = vocabulary;
};
State* active = nullptr;
std::vector<float> read_inverse(const std::filesystem::path& path) {
  if (std::filesystem::file_size(path) != vocabulary * sizeof(float))
    throw std::runtime_error("GB10 embedding inverse table size differs");
  std::vector<float> values(vocabulary);
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
  if (!file || file.peek() != std::char_traits<char>::eof() ||
      aima::sha256_bytes(values.data(), values.size() * sizeof(float)) !=
        "f4e37f759c586bfc8fcc4d74cefdd89235f0f0c0c90cd286147e331e87509e67")
    throw std::runtime_error("GB10 embedding inverse table identity differs");
  return values;
}
static __global__ void embedding_norm_kernel(const uint16_t* input,
    const uint16_t* weight, uint16_t* output, unsigned tokens,
    const float* inverse, const uint32_t* prompt, uint32_t decode_token) {
  const unsigned cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= tokens * 2048u) return;
  const unsigned token = tokens == 1 ? decode_token : prompt[cell / 2048u];
  output[cell] = qrt_sm121_q1::bf16(qrt_sm121_q1::embedding_norm_value(
      qrt_sm121_q1::widen(input[cell]), inverse[token], weight[cell % 2048u]));
}
struct ProjectionGroup {
  const uint16_t* weights[4]{};
  uint16_t* outputs[4]{};
  const uint16_t* bias[4]{};
  unsigned rows[4]{}, count = 0, total = 0;
};
static_assert(std::is_standard_layout<ProjectionGroup>::value &&
              std::is_trivially_copyable<ProjectionGroup>::value);

// Reuse the already-qualified Windows K16 / width-26 accumulator. Each
// subgroup owns one output; groups share neither state nor temporary storage.
static __global__ void projection_kernel(ProjectionGroup group,
    const uint16_t* input, unsigned reduction) {
  const unsigned cell = blockIdx.x * 16u + threadIdx.x / 16u;
  if (cell >= group.total) return;
  unsigned local = cell, selected = 0;
  while (selected + 1u < group.count && local >= group.rows[selected])
    local -= group.rows[selected++];
  const auto* weights = group.weights[selected] + std::size_t(local) * reduction;
  const unsigned lane = threadIdx.x & 15u;
  qrt_q1_moe_hawkeye::Value carry{0u, -133, false};
  for (unsigned base = 0; base < reduction; base += 16u)
    carry = qrt_sm121_wave16::accumulate(carry, input[base + lane], weights[base + lane], lane);
  if (lane == 0) {
    float value = qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
    if (group.bias[selected]) value = qrt_sm121_q1::add(value, qrt_sm121_q1::widen(group.bias[selected][local]));
    group.outputs[selected][local] = qrt_sm121_q1::bf16(value);
  }
}
void append(ProjectionGroup& group, const void* weight, void* output,
            const void* bias, std::size_t rows) {
  if (group.count >= 4 || !weight || !output || rows <= 8 || rows % 2 ||
      rows > std::size_t(std::numeric_limits<int>::max()) - group.total)
    throw std::invalid_argument("Invalid GB10 projection binding");
  const unsigned i = group.count++;
  group.weights[i] = static_cast<const uint16_t*>(weight);
  group.outputs[i] = static_cast<uint16_t*>(output);
  group.bias[i] = static_cast<const uint16_t*>(bias);
  group.rows[i] = static_cast<unsigned>(rows);
  group.total += static_cast<unsigned>(rows);
}
void launch(const ProjectionGroup& group, const void* input, std::size_t reduction, void* stream) {
  if (!input || !group.count || (reduction != 512 && reduction != 2048 && reduction != 4096))
    throw std::invalid_argument("Unsupported GB10 projection reduction");
  hipLaunchKernelGGL(projection_kernel, dim3((group.total + 15u) / 16u), dim3(256), 0,
      static_cast<hipStream_t>(stream), group, static_cast<const uint16_t*>(input),
      static_cast<unsigned>(reduction));
  if (hipGetLastError() != hipSuccess) throw std::runtime_error("GB10 projection launch failed");
}
}  // namespace
struct Gb10ProjectionOwner::Impl { State state; };
Gb10ProjectionOwner::Gb10ProjectionOwner(const std::vector<uint32_t>& prompt)
    : impl_(std::make_unique<Impl>()) {
  if (active || prompt.size() != prompt_tokens)
    throw std::invalid_argument("GB10 projections require one q8192 request owner");
  for (uint32_t token : prompt)
    if (token >= vocabulary) throw std::invalid_argument("Embedding token exceeds vocabulary");
  const char* path = std::getenv("QRT_QWEN36_GB10_LAYER0_RMSNORM_SCALE_LUT_PATH");
  if (!path || !*path) throw std::runtime_error("Missing GB10 embedding inverse table");
  const auto inverse = read_inverse(std::filesystem::u8path(path));
  impl_->state.inverse.upload(inverse.data(), inverse.size() * sizeof(float));
  impl_->state.prompt.upload(prompt.data(), prompt.size() * sizeof(uint32_t));
  active = &impl_->state;
}
Gb10ProjectionOwner::~Gb10ProjectionOwner() {
  if (active == &impl_->state) { hipDeviceSynchronize(); active = nullptr; }
}
void set_gb10_decode_token(uint32_t token) {
  if (!active || token >= vocabulary)
    throw std::invalid_argument("Invalid live GB10 decode token");
  active->decode_token = token;
}
void gb10_embedding_norm(const void* input, const void* weight, void* output,
    std::size_t tokens, void* stream) {
  if (!active || !input || !weight || !output ||
      (tokens != prompt_tokens && tokens != 1) ||
      (tokens == 1 && active->decode_token >= vocabulary))
    throw std::invalid_argument("Invalid GB10 embedding normalization binding");
  hipLaunchKernelGGL(embedding_norm_kernel, dim3(tokens * 8u), dim3(256), 0,
      static_cast<hipStream_t>(stream), static_cast<const uint16_t*>(input),
      static_cast<const uint16_t*>(weight), static_cast<uint16_t*>(output),
      static_cast<unsigned>(tokens), static_cast<const float*>(active->inverse.data),
      static_cast<const uint32_t*>(active->prompt.data), active->decode_token);
  if (hipGetLastError() != hipSuccess) throw std::runtime_error("GB10 embedding norm launch failed");
}
void gb10_projection(const void* weight, const void* input, const void* bias,
    void* output, std::size_t rows, std::size_t reduction, void* stream) {
  ProjectionGroup group;
  append(group, weight, output, bias, rows);
  launch(group, input, reduction, stream);
}
void gb10_projection_group(const aima::Bf16WvSplitKProjection* projections,
    std::size_t count, const void* input, std::size_t reduction, void* stream) {
  if (!projections || count < 2 || count > 4)
    throw std::invalid_argument("Invalid GB10 projection group");
  ProjectionGroup group;
  for (std::size_t i = 0; i < count; ++i)
    append(group, projections[i].weight_mk, projections[i].output_1m, nullptr, projections[i].m);
  launch(group, input, reduction, stream);
}
}  // namespace aima_port
