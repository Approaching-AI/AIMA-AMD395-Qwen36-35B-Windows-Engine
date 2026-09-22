// SPDX-License-Identifier: Apache-2.0
#include "gb10_moe.h"
#include <hip/hip_runtime.h>
#include "gb10_moe_math.h"
#include "gb10_gdn.h"
#include "gb10_decode_moe.h"
#include "../providers/gdn/sm121_mtp_moe_math.h"
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
using DynamicLaunch = int (*)(const float*, const float*, const uint16_t*, const uint16_t*,
    const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*,
    const uint16_t*, float*, uint32_t, void*);
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
  DynamicLaunch dynamic_launch = nullptr;
  float* storage = nullptr;
  std::array<const uint16_t*, 40> gate_up{}, down{};
  bool registered = false, pending = false, poisoned = false;
  bool native_prefill = false, native_inflight = false;
  unsigned native_stage = 0;
  std::size_t next_layer = 0;
  void* pending_carrier = nullptr;
  float* input() const { return storage; }
  float* residual() const { return storage + elements; }
  float* output() const { return storage + 2u * elements; }
  // These two provider conversion slabs are idle during a native layer.
  uint32_t* native_invalid() const { return reinterpret_cast<uint32_t*>(input()); }
  float* native_weights() const { return residual(); }
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
bool native_setting() {
  const char* value = std::getenv("AIMA_PORT_NATIVE_MOE_PREFILL");
  if (!value || !*value || std::string(value) == "0") return false;
  if (std::string(value) != "1") throw std::invalid_argument("Unsupported native MoE mode");
  return true;
}
State& native_stage(unsigned stage) {
  auto& s = bound();
  if (!s.native_prefill || !s.native_inflight || s.native_stage != stage)
    throw std::logic_error("Native MoE stage order differs");
  return s;
}
void pointers(const void* a, const void* b, const void* c) {
  if (!a || !b || !c || reinterpret_cast<std::uintptr_t>(a) % 2u ||
      reinterpret_cast<std::uintptr_t>(b) % 2u || reinterpret_cast<std::uintptr_t>(c) % 2u ||
      a == c || b == c) throw std::invalid_argument("Invalid native MoE stage binding");
}
static __global__ void native_shared_gate(const uint16_t* input, const uint16_t* weight, uint16_t* output) {
  constexpr unsigned lanes = qrt_sm121_shared_gate::lanes;
  const unsigned token = blockIdx.x * 4u + threadIdx.x / lanes, lane = threadIdx.x % lanes;
  float sum = qrt_sm121_shared_gate::lane_dot(input + std::size_t(token) * hidden, weight, lane);
  for (unsigned mask = lanes / 2; mask; mask >>= 1)
    sum = qrt_sm121_q1::add(sum, __shfl_down(sum, mask, lanes));
  if (!lane) output[token] = qrt_sm121_q1::bf16(sum);
}
static __global__ void native_shared_activation(const uint16_t* gate, const uint16_t* up,
    uint16_t* output, const uint16_t* silu, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = qrt_sm121_mtp::moe_activate(gate[i], up[i], silu);
}
static __global__ void native_shared_scale(const uint16_t* gate, const uint16_t* down,
    uint16_t* output, const uint16_t* sigmoid, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = qrt_sm121_mtp::moe_shared_product(gate[i / hidden], down[i], sigmoid);
}
static __global__ void native_router(const uint16_t* logits, const uint32_t* exponent,
    uint32_t* ids, float* weights, uint32_t* invalid) {
  if (threadIdx.x) return;
  const unsigned row = blockIdx.x;
  if (!qrt_sm121_mtp::moe_route(logits + row * 256u, exponent, ids + row * 8u, weights + row * 8u)) {
    // Keep dispatch in bounds even on a rejected nonfinite row. The persistent
    // error flag is checked before the carrier can be consumed or published.
    for (unsigned i = 0; i < 8; ++i) { ids[row * 8u + i] = 0; weights[row * 8u + i] = 0; }
    atomicOr(invalid, 1u);
  }
}
static __global__ void native_expert_activation(const uint16_t* gate_up, uint16_t* output,
    const uint16_t* silu, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const unsigned base = (i / 512u) * 1024u + i % 512u;
  output[i] = qrt_sm121_mtp::moe_activate(gate_up[base], gate_up[base + 512u], silu);
}
static __global__ void native_combine(const uint16_t* weighted, const uint16_t* shared,
    const uint16_t* residual, uint16_t* routed, uint16_t* combined, float* carrier,
    uint16_t* output, uint32_t* invalid, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint16_t r = qrt_sm121_mtp::moe_routed_sum(weighted + std::size_t(i / hidden) * 8u * hidden, i % hidden);
  const uint16_t c = qrt_sm121_mtp::moe_output(shared[i], r);
  const float value = qrt_sm121_q1::add(qrt_sm121_q1::widen(residual[i]), qrt_sm121_q1::widen(c));
  routed[i] = r; combined[i] = c; carrier[i] = value; output[i] = qrt_sm121_q1::bf16(value);
  if (!qrt_sm121_mtp::moe_finite(value)) atomicOr(invalid, 2u);
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
  s.native_prefill = native_setting();
  s.library = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!s.library) throw std::runtime_error("Cannot load the pinned MoE provider");
  auto prepare = reinterpret_cast<Prepare>(dlsym(s.library, "qrt_triton_moe_q8192_prepare"));
  s.release = reinterpret_cast<Release>(dlsym(s.library, "qrt_triton_moe_q8192_release"));
  s.error = reinterpret_cast<Error>(dlsym(s.library, "qrt_triton_moe_q8192_last_error"));
  s.registration = reinterpret_cast<Register>(dlsym(s.library, "qrt_triton_moe_q8192_register_weight_metadata"));
  s.launch = reinterpret_cast<Launch>(dlsym(s.library, "qrt_triton_moe_q8192_launch_full_v3_async"));
  s.dynamic_launch = reinterpret_cast<DynamicLaunch>(dlsym(s.library, "qrt_triton_moe_q8192_launch_full_v4_dynamic_async"));
  if (!prepare || !s.release || !s.error || !s.registration || !s.launch)
    throw std::runtime_error("MoE provider ABI is incomplete");
  const char* terminal = std::getenv("AIMA_PORT_PREFILL_TERMINAL_ONLY");
  if (terminal && std::string(terminal) == "1" && !s.dynamic_launch)
    throw std::runtime_error("Terminal MoE provider ABI is incomplete");
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
  active->native_inflight = false;
  active->gate_up.fill(nullptr); active->down.fill(nullptr);
}
void gb10_prefill_moe(std::size_t layer, const void* input, const void* residual,
    const void* router, const void* gate_up, const void* down,
    const void* shared_gate, const void* shared_gate_projection,
    const void* shared_up_projection, const void* shared_down,
    void* output, std::size_t count, bool terminal_only) {
  auto& s = bound();
  if (layer >= 40 || count != tokens || s.pending || s.native_inflight || s.next_layer != layer ||
      !input || !residual || !router || !shared_gate || !shared_gate_projection ||
      !shared_up_projection || !shared_down || !output || gate_up != s.gate_up[layer] || down != s.down[layer])
    throw std::invalid_argument("Invalid complete q8192 MoE binding or layer order");
  if (s.native_prefill && layer < 39)
    throw std::invalid_argument("Native MoE layer reached the external provider");
  if (terminal_only && (layer != 39 || !s.dynamic_launch))
    throw std::invalid_argument("Invalid terminal MoE binding");
  const std::size_t offset = terminal_only ? elements - hidden : 0;
  const unsigned live_elements = terminal_only ? hidden : unsigned(elements);
  if (!terminal_only) {
    observe_gdn_prefill(layer, "prefill-post-attention-norm-sampled", input, hidden, count);
    observe_gdn_prefill(layer, "prefill-post-attention-residual-sampled", residual, hidden, count);
  }
  hipLaunchKernelGGL(widen_moe_inputs, dim3((live_elements + 255u) / 256u), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(input) + offset, static_cast<const uint16_t*>(residual) + offset,
      s.input() + offset, s.residual() + offset, live_elements);
  check(hipGetLastError(), "MoE input conversion");
  const int status = terminal_only
      ? s.dynamic_launch(s.input() + offset, s.residual() + offset, static_cast<const uint16_t*>(router),
      static_cast<const uint16_t*>(gate_up), static_cast<const uint16_t*>(down),
      static_cast<const uint16_t*>(shared_gate), static_cast<const uint16_t*>(shared_gate_projection),
      static_cast<const uint16_t*>(shared_up_projection), static_cast<const uint16_t*>(shared_down),
      s.output() + offset, 1u, nullptr)
      : s.launch(s.input(), s.residual(), static_cast<const uint16_t*>(router),
      static_cast<const uint16_t*>(gate_up), static_cast<const uint16_t*>(down),
      static_cast<const uint16_t*>(shared_gate), static_cast<const uint16_t*>(shared_gate_projection),
      static_cast<const uint16_t*>(shared_up_projection), static_cast<const uint16_t*>(shared_down), s.output(), nullptr);
  if (status != 1) {
    s.poisoned = true;
    throw std::runtime_error(std::string("MoE launch: ") + s.error());
  }
  hipLaunchKernelGGL(round_moe_carrier, dim3((live_elements + 255u) / 256u), dim3(256), 0, nullptr,
      s.output() + offset, static_cast<uint16_t*>(output) + offset, live_elements);
  check(hipGetLastError(), "MoE output conversion");
  s.pending = true; s.next_layer = layer + 1; s.pending_carrier = output;
  if (terminal_only)
    std::fprintf(stderr, "{\"event\":\"terminal_prefill_moe\",\"layer\":39,\"rows\":1,\"carrier_row\":8191}\n");
  else observe_gdn_prefill(layer, "prefill-layer-output-sampled", output, hidden, count);
}
bool gb10_native_moe_prefill_enabled(std::size_t layer, std::size_t count) {
  auto& s = bound();
  if (!s.native_prefill) return false;
  if (layer >= 40 || count != tokens) throw std::invalid_argument("Native MoE requires cold q8192");
  return layer < 39;
}
Gb10NativeMoeScope::Gb10NativeMoeScope(std::size_t layer, const void* input, const void* residual,
    const void* gate_up, const void* down, void* output, std::size_t count)
    : owner_(nullptr), residual_(residual), output_(output), layer_(layer) {
  auto& s = bound();
  if (!gb10_native_moe_prefill_enabled(layer, count) || s.pending || s.native_inflight || s.next_layer != layer ||
      !input || !residual || !output || gate_up != s.gate_up[layer] || down != s.down[layer])
    throw std::invalid_argument("Invalid native MoE layer binding");
  (void)gb10_moe_silu_table(); (void)gb10_moe_router_exp_table(); (void)gb10_sigmoid_table();
  check(hipMemsetAsync(s.native_invalid(), 0, sizeof(uint32_t), nullptr), "Native MoE error reset");
  owner_ = &s; s.native_inflight = true; s.native_stage = 0;
}
Gb10NativeMoeScope::~Gb10NativeMoeScope() {
  if (!completed_ && owner_ && active == owner_) {
    active->poisoned = true; active->native_inflight = false;
  }
}
void* Gb10NativeMoeScope::router_weights() const {
  if (completed_ || owner_ != active) throw std::logic_error("Native MoE scope expired");
  auto& s = bound();
  if (!s.native_inflight) throw std::logic_error("Native MoE scope is inactive");
  return s.native_weights();
}
void gb10_native_moe_shared_gate(const void* input, const void* weight, void* output) {
  auto& s = native_stage(0); pointers(input, weight, output);
  hipLaunchKernelGGL(native_shared_gate, dim3(tokens / 4u), dim3(64), 0, nullptr,
      static_cast<const uint16_t*>(input), static_cast<const uint16_t*>(weight), static_cast<uint16_t*>(output));
  check(hipGetLastError(), "Native MoE shared gate"); ++s.native_stage;
}
void gb10_native_moe_shared_activation(const void* gate, const void* up, void* output) {
  auto& s = native_stage(1); pointers(gate, up, output);
  hipLaunchKernelGGL(native_shared_activation, dim3(tokens * 512u / 256u), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(gate), static_cast<const uint16_t*>(up), static_cast<uint16_t*>(output),
      gb10_moe_silu_table(), tokens * 512u);
  check(hipGetLastError(), "Native MoE shared activation"); ++s.native_stage;
}
void gb10_native_moe_shared_scale(const void* gate, const void* down, void* output) {
  auto& s = native_stage(2); pointers(gate, down, output);
  hipLaunchKernelGGL(native_shared_scale, dim3(elements / 256u), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(gate), static_cast<const uint16_t*>(down), static_cast<uint16_t*>(output),
      gb10_sigmoid_table(), unsigned(elements));
  check(hipGetLastError(), "Native MoE shared scale"); ++s.native_stage;
}
void gb10_native_moe_router(const void* logits, void* ids) {
  auto& s = native_stage(3); pointers(logits, s.native_weights(), ids);
  if (reinterpret_cast<std::uintptr_t>(ids) % alignof(uint32_t))
    throw std::invalid_argument("Native MoE router indices are misaligned");
  hipLaunchKernelGGL(native_router, dim3(tokens), dim3(32), 0, nullptr,
      static_cast<const uint16_t*>(logits), gb10_moe_router_exp_table(), static_cast<uint32_t*>(ids),
      s.native_weights(), s.native_invalid());
  check(hipGetLastError(), "Native MoE router"); ++s.native_stage;
}
void gb10_native_moe_expert_activation(const void* gate_up, void* output) {
  auto& s = native_stage(4); pointers(gate_up, gb10_moe_silu_table(), output);
  hipLaunchKernelGGL(native_expert_activation, dim3(tokens * 8u * 512u / 256u), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(gate_up), static_cast<uint16_t*>(output), gb10_moe_silu_table(), tokens * 8u * 512u);
  check(hipGetLastError(), "Native MoE expert activation"); ++s.native_stage;
}
void Gb10NativeMoeScope::finish(const void* weighted, const void* shared, void* routed, void* combined) {
  if (completed_ || owner_ != active) throw std::logic_error("Native MoE scope expired");
  auto& s = native_stage(5); pointers(weighted, shared, routed); pointers(weighted, shared, combined);
  pointers(residual_, shared, routed); pointers(residual_, shared, combined);
  if (routed == combined || output_ == routed || output_ == combined || output_ == weighted || output_ == shared)
    throw std::invalid_argument("Native MoE output aliases live intermediates");
  hipLaunchKernelGGL(native_combine, dim3(elements / 256u), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(weighted), static_cast<const uint16_t*>(shared),
      static_cast<const uint16_t*>(residual_), static_cast<uint16_t*>(routed), static_cast<uint16_t*>(combined),
      s.output(), static_cast<uint16_t*>(output_), s.native_invalid(), unsigned(elements));
  check(hipGetLastError(), "Native MoE combine");
  uint32_t invalid = 0;
  check(hipMemcpy(&invalid, s.native_invalid(), sizeof(invalid), hipMemcpyDeviceToHost), "Native MoE flag read");
  if (invalid) throw std::runtime_error("Native MoE rejected nonfinite operands");
  s.pending = true; s.pending_carrier = output_; s.next_layer = layer_ + 1;
  s.native_inflight = false; completed_ = true;
  std::fprintf(stderr, "{\"event\":\"native_moe_prefill\",\"layer\":%zu,\"tokens\":8192,\"expert_kernels\":2,"
      "\"router_weights\":\"float32\",\"silu\":\"bf16_table\",\"carrier\":\"unrounded_float32\"}\n", layer_);
}
bool gb10_moe_input_norm(std::size_t layer, const void* carrier,
    const void* weight, void* output, std::size_t count) {
  auto& s = bound();
  if (s.native_inflight) throw std::logic_error("Native MoE carrier is incomplete");
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
  if (s.native_inflight) throw std::logic_error("Native MoE carrier is incomplete");
  if (!s.pending) return false;
  const auto last = reinterpret_cast<std::uintptr_t>(s.pending_carrier) + (elements - hidden) * sizeof(uint16_t);
  if (s.next_layer != 40 || stream || reinterpret_cast<std::uintptr_t>(carrier) != last)
    throw std::invalid_argument("MoE terminal norm does not own the final prefill row");
  norm(s, weight, output, 1, tokens - 1);
  s.pending = false;
  return true;
}
}  // namespace aima_port
