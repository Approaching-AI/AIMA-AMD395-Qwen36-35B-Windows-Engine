// SPDX-License-Identifier: Apache-2.0
#include "gb10_moe.h"
#include "gb10_moe_math.h"
#include "gb10_gdn.h"
#include "aima/sha256.h"
#include "dlfcn.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima_port {
namespace {
constexpr unsigned tokens = 8192, hidden = 2048;
constexpr std::size_t elements = std::size_t(tokens) * hidden;
struct Asset { const char* name; std::size_t bytes; const char* sha256; };
struct Setting { const char* name; const char* value; };
#include "gb10_moe_assets.inc"
using Prepare = int (*)(const char*);
using Release = void (*)();
using Error = const char* (*)();
using Register = int (*)(const uint16_t* const*, const uint16_t* const*, uint32_t);
using Launch = int (*)(const float*, const float*, const uint16_t*, const uint16_t*,
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*,
    const uint16_t*, float*, void*);
void check(hipError_t status, const char* action) {
  if (status != hipSuccess) throw std::runtime_error(std::string(action) + ": " + hipGetErrorString(status));
}
std::filesystem::path selected(const char* name) {
  const char* value = std::getenv(name);
  if (!value || !*value) throw std::runtime_error(std::string("Missing MoE input: ") + name);
  return std::filesystem::u8path(value);
}
void verify(const std::filesystem::path& path, const Asset& asset) {
  if (std::filesystem::file_size(path) != asset.bytes)
    throw std::runtime_error("MoE artifact size differs: " + path.u8string());
  std::vector<unsigned char> bytes(asset.bytes);
  std::ifstream f(path, std::ios::binary);
  f.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!f || f.peek() != std::char_traits<char>::eof() ||
      aima::sha256_bytes(bytes.data(), bytes.size()) != asset.sha256)
    throw std::runtime_error("MoE artifact identity differs: " + path.u8string());
}
struct State {
  void* library = nullptr;
  Release release = nullptr;
  Error error = nullptr;
  Register registration = nullptr;
  Launch launch = nullptr;
  float* storage = nullptr;
  std::array<const uint16_t*, 40> gate_up{}, down{};
  bool registered = false, pending = false, poisoned = false;
  std::size_t next_layer = 0;
  void* pending_carrier = nullptr;
  float* input() const { return storage; }
  float* residual() const { return storage + elements; }
  float* output() const { return storage + 2u * elements; }
  ~State() {
    if (library) {
      hipDeviceSynchronize();
      if (release) release();
      dlclose(library);
    }
    if (storage) hipFree(storage);
  }
};
State* active = nullptr;
State& bound() {
  if (!active || !active->registered || active->poisoned || !active->storage)
    throw std::runtime_error("MoE owner and model registration are incomplete");
  return *active;
}
static __global__ void widen_moe_inputs(const uint16_t* x, const uint16_t* r,
    float* xf, float* rf, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) { xf[i] = qrt_sm121_q1::widen(x[i]); rf[i] = qrt_sm121_q1::widen(r[i]); }
}
static __global__ void round_moe_carrier(const float* source, uint16_t* output, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = qrt_sm121_q1::bf16(source[i]);
}
static __global__ void normalize_moe_carrier(const float* source, const uint16_t* weight,
    uint16_t* output, const unsigned char* table) {
  __shared__ float sums[8], inverse;
  const unsigned lane = threadIdx.x;
  const std::size_t offset = std::size_t(blockIdx.x) * hidden;
  float sum = moe_carrier_lane_sumsq(source + offset, lane);
  for (unsigned mask = 16; mask; mask >>= 1)
    sum = qrt_sm121_q1::add(sum, __shfl_xor(sum, mask, 32));
  if ((lane & 31u) == 0) sums[lane / 32] = sum;
  __syncthreads();
  if (!lane) inverse = qrt_sm121_mtp::inverse(qrt_sm121_mtp::residual_sum_warps(sums), table);
  __syncthreads();
  for (unsigned i = 0; i < 8; ++i) {
    const unsigned column = lane * 8 + i;
    output[offset + column] = qrt_sm121_mtp::normalized(
        qrt_sm121_q1::widen(qrt_sm121_q1::bf16(source[offset + column])), inverse, weight[column]);
  }
}
void norm(State& s, const void* weight, void* output, unsigned rows, unsigned first) {
  if (!weight || !output || output == s.pending_carrier)
    throw std::invalid_argument("Invalid MoE normalization binding");
  hipLaunchKernelGGL(normalize_moe_carrier, dim3(rows), dim3(256), 0, nullptr,
      s.output() + std::size_t(first) * hidden, static_cast<const uint16_t*>(weight),
      static_cast<uint16_t*>(output), gb10_rsqrt_table());
  check(hipGetLastError(), "MoE carrier normalization");
}
}  // namespace

struct Gb10MoeOwner::Impl { State state; };
Gb10MoeOwner::Gb10MoeOwner() : impl_(std::make_unique<Impl>()) {
  if (active) throw std::runtime_error("A MoE owner is already active");
  (void)gb10_rsqrt_table();
  for (const auto& setting : moe_settings) {
    const char* value = std::getenv(setting.name);
    if (!value || std::string(value) != setting.value)
      throw std::runtime_error(std::string("Unsupported MoE setting: ") + setting.name);
  }
  const auto library = selected("AIMA_PORT_MOE_PROVIDER_DLL");
  const auto directory = selected("AIMA_PORT_MOE_PROVIDER_DIR");
  verify(library, moe_library);
  for (const auto& asset : moe_kernels) verify(directory / asset.name, asset);
  verify(selected("QRT_QWEN36_CUDA_ROUTER_EX2_FRACTION_LUT_PATH"), moe_router_table);
  verify(selected("QRT_QWEN36_CUDA_VLLM_SILU_BF16_DOMAIN_LUT_PATH"), moe_silu_table);
  auto& s = impl_->state;
  s.library = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!s.library) throw std::runtime_error("Cannot load the pinned MoE provider");
  auto prepare = reinterpret_cast<Prepare>(dlsym(s.library, "qrt_triton_moe_q8192_prepare"));
  s.release = reinterpret_cast<Release>(dlsym(s.library, "qrt_triton_moe_q8192_release"));
  s.error = reinterpret_cast<Error>(dlsym(s.library, "qrt_triton_moe_q8192_last_error"));
  s.registration = reinterpret_cast<Register>(dlsym(s.library, "qrt_triton_moe_q8192_register_weight_metadata"));
  s.launch = reinterpret_cast<Launch>(dlsym(s.library, "qrt_triton_moe_q8192_launch_full_v3_async"));
  if (!prepare || !s.release || !s.error || !s.registration || !s.launch)
    throw std::runtime_error("MoE provider ABI is incomplete");
  if (prepare(directory.u8string().c_str()) != 1)
    throw std::runtime_error(std::string("MoE prepare: ") + s.error());
  check(hipMalloc(reinterpret_cast<void**>(&s.storage), 3u * elements * sizeof(float)), "MoE carrier allocation");
  active = &s;
}
Gb10MoeOwner::~Gb10MoeOwner() {
  if (active == &impl_->state) { gb10_moe_release_weights(); active = nullptr; }
}
void gb10_moe_register_weights(const uint16_t* const* gate_up,
    const uint16_t* const* down, std::size_t count) {
  if (!active || active->registered || active->poisoned || !gate_up || !down || count != 40)
    throw std::invalid_argument("Invalid MoE model registration");
  for (std::size_t i = 0; i < count; ++i) {
    if (!gate_up[i] || !down[i] || gate_up[i] == down[i])
      throw std::invalid_argument("Invalid MoE model weight pair");
    for (std::size_t j = 0; j < i; ++j)
      if (gate_up[i] == gate_up[j] || down[i] == down[j] || gate_up[i] == down[j] || down[i] == gate_up[j])
        throw std::invalid_argument("Duplicate MoE model weight identity");
  }
  if (active->registration(gate_up, down, 40) != 1) {
    active->poisoned = true;
    throw std::runtime_error(std::string("MoE registration: ") + active->error());
  }
  std::copy_n(gate_up, 40, active->gate_up.begin());
  std::copy_n(down, 40, active->down.begin());
  active->registered = true;
}
void gb10_moe_release_weights() noexcept {
  if (!active || !active->registered) return;
  // Fail closed if outstanding provider work cannot drain before model free.
  if (hipDeviceSynchronize() != hipSuccess || active->registration(nullptr, nullptr, 0) != 1)
    std::abort();
  active->registered = false; active->pending = false; active->pending_carrier = nullptr;
  active->gate_up.fill(nullptr); active->down.fill(nullptr);
}
void gb10_prefill_moe(std::size_t layer, const void* input, const void* residual,
    const void* router, const void* gate_up, const void* down,
    const void* shared_gate, const void* shared_gate_projection,
    const void* shared_up_projection, const void* shared_down,
    void* output, std::size_t count) {
  auto& s = bound();
  if (layer >= 40 || count != tokens || s.pending || s.next_layer != layer ||
      !input || !residual || !router || !shared_gate || !shared_gate_projection ||
      !shared_up_projection || !shared_down || !output || gate_up != s.gate_up[layer] || down != s.down[layer])
    throw std::invalid_argument("Invalid complete q8192 MoE binding or layer order");
  observe_gdn_prefill(layer, "prefill-post-attention-norm-sampled", input, hidden, count);
  observe_gdn_prefill(layer, "prefill-post-attention-residual-sampled", residual, hidden, count);
  hipLaunchKernelGGL(widen_moe_inputs, dim3((elements + 255u) / 256u), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(input), static_cast<const uint16_t*>(residual), s.input(), s.residual(), unsigned(elements));
  check(hipGetLastError(), "MoE input conversion");
  if (s.launch(s.input(), s.residual(), static_cast<const uint16_t*>(router),
      static_cast<const uint16_t*>(gate_up), static_cast<const uint16_t*>(down),
      static_cast<const uint16_t*>(shared_gate), static_cast<const uint16_t*>(shared_gate_projection),
      static_cast<const uint16_t*>(shared_up_projection), static_cast<const uint16_t*>(shared_down), s.output(), nullptr) != 1) {
    s.poisoned = true;
    throw std::runtime_error(std::string("MoE launch: ") + s.error());
  }
  hipLaunchKernelGGL(round_moe_carrier, dim3((elements + 255u) / 256u), dim3(256), 0, nullptr,
      s.output(), static_cast<uint16_t*>(output), unsigned(elements));
  check(hipGetLastError(), "MoE output conversion");
  s.pending = true; s.next_layer = layer + 1; s.pending_carrier = output;
  observe_gdn_prefill(layer, "prefill-layer-output-sampled", output, hidden, count);
}
bool gb10_moe_input_norm(std::size_t layer, const void* carrier,
    const void* weight, void* output, std::size_t count) {
  auto& s = bound();
  if (layer >= 40 || count != tokens || !carrier || !weight || !output)
    throw std::invalid_argument("Invalid q8192 MoE norm extent");
  if (!layer) {
    if (s.pending) throw std::runtime_error("An earlier MoE carrier was not consumed");
    s.next_layer = 0; return false;
  }
  if (!s.pending || s.next_layer != layer || s.pending_carrier != carrier)
    throw std::runtime_error("MoE norm does not own the preceding live carrier");
  norm(s, weight, output, tokens, 0);
  s.pending = false;
  observe_gdn_prefill(layer - 1, "prefill-next-input-norm-sampled", output, hidden, count);
  return true;
}
bool gb10_moe_terminal_norm(const void* carrier, const void* weight, void* output, void* stream) {
  auto& s = bound();
  if (!s.pending) return false;
  const auto last = reinterpret_cast<std::uintptr_t>(s.pending_carrier) + (elements - hidden) * sizeof(uint16_t);
  if (s.next_layer != 40 || stream || reinterpret_cast<std::uintptr_t>(carrier) != last)
    throw std::invalid_argument("MoE terminal norm does not own the final prefill row");
  norm(s, weight, output, 1, tokens - 1);
  s.pending = false;
  return true;
}
}  // namespace aima_port
