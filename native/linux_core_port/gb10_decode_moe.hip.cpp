// SPDX-License-Identifier: Apache-2.0
#include "gb10_decode_moe.h"
#include "gb10_gdn.h"
#include "gb10_normalization.h"
#include "aima/sha256.h"
#include "../providers/gdn/sm121_mtp_moe.h"
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
void check(hipError_t status, const char* operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}
struct Device {
  void* data = nullptr;
  ~Device() { if (data) hipFree(data); }
  void allocate(std::size_t bytes) { check(hipMalloc(&data, bytes), "Decode MoE allocation"); }
  template<class T> T* as() const { return static_cast<T*>(data); }
};
struct State {
  Device silu, router_exp, invalid, terminal;
  unsigned next_layer = 0;
  bool poisoned = false, needs_terminal_save = false;
  void* terminal_carrier = nullptr;
};
State* active = nullptr;
void upload(Device& device, const char* environment, std::size_t size, const char* hash) {
  const char* path = std::getenv(environment);
  if (!path || !*path) throw std::runtime_error(std::string("Missing decode MoE table: ") + environment);
  const auto file_path = std::filesystem::u8path(path);
  if (std::filesystem::file_size(file_path) != size) throw std::runtime_error("Decode MoE table size differs");
  std::vector<unsigned char> bytes(size);
  std::ifstream file(file_path, std::ios::binary);
  file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!file || file.peek() != std::char_traits<char>::eof() ||
      aima::sha256_bytes(bytes.data(), bytes.size()) != hash)
    throw std::runtime_error("Decode MoE table identity differs");
  device.allocate(size);
  check(hipMemcpy(device.data, bytes.data(), bytes.size(), hipMemcpyHostToDevice), "Decode MoE table upload");
}
struct Span { const void* pointer; std::size_t bytes, alignment; };
bool valid(Span s) {
  const auto first = reinterpret_cast<std::uintptr_t>(s.pointer);
  return first && first % s.alignment == 0 &&
      first <= std::numeric_limits<std::uintptr_t>::max() - s.bytes;
}
bool disjoint(Span a, Span b) {
  const auto first = reinterpret_cast<std::uintptr_t>(a.pointer);
  const auto second = reinterpret_cast<std::uintptr_t>(b.pointer);
  return first + a.bytes <= second || second + b.bytes <= first;
}
void validate(const void* input, const Gb10DecodeMoeWeights& w, const Gb10DecodeMoeBuffers& o) {
  const Span reads[] = {{input,4096,2}, {w.router,1048576,2}, {w.shared_gate,4096,2},
      {w.shared_gate_projection,2097152,2}, {w.shared_up_projection,2097152,2},
      {w.shared_down,2097152,2}, {w.routed_gate_up,1073741824,2}, {w.routed_down,536870912,2}};
  const Span writes[] = {{o.shared_input,2050,2}, {o.shared_activation,1024,2}, {o.shared_down,4096,2},
      {o.shared_output,4096,2}, {o.router,512,2}, {o.router_indices,32,4}, {o.router_weights,32,4},
      {o.routed_gate_up,16384,2}, {o.routed_activation,8192,2}, {o.routed_weighted,32768,2},
      {o.routed_output,4096,2}, {o.combined,4096,2}};
  for (auto read : reads) if (!valid(read)) throw std::invalid_argument("Invalid decode MoE input");
  for (std::size_t i = 0; i < std::size(writes); ++i) {
    if (!valid(writes[i])) throw std::invalid_argument("Invalid decode MoE output");
    for (const auto read : reads)
      if (!disjoint(writes[i], read)) throw std::invalid_argument("Decode MoE input/output overlap");
    for (std::size_t j = 0; j < i; ++j)
      if (!disjoint(writes[i], writes[j])) throw std::invalid_argument("Decode MoE output overlap");
  }
}
}  // namespace
struct Gb10DecodeMoeOwner::Impl { State state; };
Gb10DecodeMoeOwner::Gb10DecodeMoeOwner() : impl_(std::make_unique<Impl>()) {
  if (active) throw std::runtime_error("A decode MoE owner is already active");
  const char* setting = std::getenv("AIMA_PORT_DECODE_MOE");
  if (!setting || !*setting) return;
  if (std::string(setting) != "1") throw std::invalid_argument("Unsupported decode MoE mode");
  (void)gb10_sigmoid_table();
  auto& s = impl_->state;
  upload(s.silu, "QRT_QWEN36_CUDA_VLLM_SILU_BF16_DOMAIN_LUT_PATH", 131096,
      "97a2a729266bb0681983aa5b2c6ddffafeaab1bab99c7658a0524b09331a11ac");
  upload(s.router_exp, "QRT_QWEN36_CUDA_ROUTER_EX2_FRACTION_LUT_PATH", 33554432,
      "b2a42c4a626469c986e33e43f16f41bde9d84de94347cd9ceaa1bd68d36bcdf0");
  s.invalid.allocate(sizeof(uint32_t));
  s.terminal.allocate(8192);
  active = &s;
}
Gb10DecodeMoeOwner::~Gb10DecodeMoeOwner() {
  if (active == &impl_->state) { hipDeviceSynchronize(); active = nullptr; }
}
bool gb10_decode_moe_enabled() { return active != nullptr; }
const uint16_t* gb10_moe_silu_table() {
  if (!active || active->poisoned || !active->silu.data)
    throw std::runtime_error("MoE SiLU table owner is unavailable");
  return active->silu.as<uint16_t>() + 12;
}
const uint32_t* gb10_moe_router_exp_table() {
  if (!active || active->poisoned || !active->router_exp.data)
    throw std::runtime_error("MoE router table owner is unavailable");
  return active->router_exp.as<uint32_t>();
}
void gb10_decode_moe(std::size_t layer, const void* input, const Gb10DecodeMoeWeights& w,
    const Gb10DecodeMoeBuffers& o, void* stream) {
  if (!active || active->poisoned || active->needs_terminal_save || active->terminal_carrier ||
      layer >= 40 || layer != active->next_layer || stream)
    throw std::invalid_argument("Invalid decode MoE owner, layer order or stream");
  validate(input, w, o);
  const auto* sigmoid = gb10_sigmoid_table();
  auto& s = *active;
  using namespace qrt_sm121_mtp;
  MoeBuffers b;
  b.router = static_cast<uint16_t*>(o.router);
  b.shared_gate = static_cast<uint16_t*>(o.shared_input);
  b.shared_gate_up = b.shared_gate + 1;
  b.shared_activated = static_cast<uint16_t*>(o.shared_activation);
  b.shared_down = static_cast<uint16_t*>(o.shared_down);
  b.shared = static_cast<uint16_t*>(o.shared_output);
  b.routed_gate_up = static_cast<uint16_t*>(o.routed_gate_up);
  b.routed_activated = static_cast<uint16_t*>(o.routed_activation);
  b.routed_weighted = static_cast<uint16_t*>(o.routed_weighted);
  b.routed = static_cast<uint16_t*>(o.routed_output);
  b.output = static_cast<uint16_t*>(o.combined);
  b.topk_ids = static_cast<uint32_t*>(o.router_indices);
  b.topk_weights = static_cast<float*>(o.router_weights);
  b.invalid = s.invalid.as<uint32_t>();
  const auto* x = static_cast<const uint16_t*>(input);
  const auto dense = [&](const void* weight, const uint16_t* values, uint16_t* output,
                         unsigned rows, unsigned columns) {
    check(mtp_moe_detail::dense(static_cast<const uint16_t*>(weight), values, output,
        rows, columns, 1, 1024, nullptr), "Decode MoE dense projection");
  };
  try {
    if (layer == 0) check(hipMemsetAsync(b.invalid, 0, sizeof(uint32_t), nullptr), "Decode MoE error reset");
    dense(w.router, x, b.router, 256, 2048);
    hipLaunchKernelGGL(mtp_moe_detail::router, dim3(1), dim3(32), 0, nullptr,
        b.router, s.router_exp.as<uint32_t>(), b.topk_ids, b.topk_weights, b.invalid);
    check(hipGetLastError(), "Decode MoE router");
    hipLaunchKernelGGL(mtp_moe_detail::shared_gate, dim3(1), dim3(32), 0, nullptr,
        x, static_cast<const uint16_t*>(w.shared_gate), b.shared_gate, b.invalid);
    check(hipGetLastError(), "Decode MoE scalar shared gate");
    // The imported loader retains separate gate/up weights. Feed those views
    // directly into the existing exact projection without concatenating them.
    dense(w.shared_gate_projection, x, b.shared_gate_up, 512, 2048);
    dense(w.shared_up_projection, x, b.shared_gate_up + 512, 512, 2048);
    check(mtp_moe_detail::routed<false>(x, static_cast<const uint16_t*>(w.routed_gate_up), b, 1, 1024, nullptr),
          "Decode MoE routed gate/up");
    hipLaunchKernelGGL(mtp_moe_detail::activate, dim3(18), dim3(256), 0, nullptr,
        b, s.silu.as<uint16_t>() + 12, 1u);
    check(hipGetLastError(), "Decode MoE activation");
    dense(w.shared_down, b.shared_activated, b.shared_down, 2048, 512);
    check(mtp_moe_detail::routed<true>(b.routed_activated, static_cast<const uint16_t*>(w.routed_down), b, 1, 1024, nullptr),
          "Decode MoE routed down");
    hipLaunchKernelGGL(mtp_moe_detail::finish, dim3(8), dim3(256), 0, nullptr, b, sigmoid, 1u);
    check(hipGetLastError(), "Decode MoE finish");
    if (layer == 39) {
      uint32_t invalid = 0;
      check(hipMemcpy(&invalid, b.invalid, sizeof(invalid), hipMemcpyDeviceToHost), "Decode MoE flag read");
      if (invalid) throw std::runtime_error("Decode MoE rejected nonfinite input or invalid expert state");
      s.needs_terminal_save = true;
    }
    s.next_layer = static_cast<unsigned>((layer + 1) % 40);
  } catch (...) {
    s.poisoned = true;
    throw;
  }
}
void gb10_decode_moe_save_terminal(const void* combined, const void* residual,
    void* carrier, void* stream) {
  if (!active || active->poisoned || !active->needs_terminal_save || active->terminal_carrier || stream ||
      !valid({combined,4096,2}) || !valid({residual,4096,2}) || !valid({carrier,4096,2}))
    throw std::invalid_argument("Invalid decode MoE terminal snapshot binding");
  auto& s = *active;
  try {
    check(hipMemcpyAsync(s.terminal.data, combined, 4096, hipMemcpyDeviceToDevice, nullptr), "Decode MoE terminal input save");
    check(hipMemcpyAsync(s.terminal.as<uint16_t>() + 2048, residual, 4096, hipMemcpyDeviceToDevice, nullptr),
          "Decode MoE terminal residual save");
    s.needs_terminal_save = false;
    s.terminal_carrier = carrier;
  } catch (...) { s.poisoned = true; throw; }
}
bool gb10_decode_moe_terminal_norm(const void* carrier, const void* weight, void* output, void* stream) {
  if (!active) return false;
  if (active->poisoned || active->needs_terminal_save)
    throw std::runtime_error("Decode MoE terminal is incomplete");
  if (!active->terminal_carrier) return false;
  if (stream || carrier != active->terminal_carrier || !valid({weight,4096,2}) ||
      !valid({output,4096,2}) || !disjoint({output,4096,2},{carrier,4096,2}) ||
      !disjoint({output,4096,2},{weight,4096,2}))
    throw std::invalid_argument("Invalid decode MoE terminal normalization binding");
  auto& s = *active;
  try {
    gb10_residual_norm(s.terminal.data, s.terminal.as<uint16_t>() + 2048,
        weight, s.terminal_carrier, output, 1, nullptr);
    s.terminal_carrier = nullptr;
  } catch (...) { s.poisoned = true; throw; }
  return true;
}
}  // namespace aima_port
