// SPDX-License-Identifier: Apache-2.0
#include "gb10_gdn.h"
#include "aima/sha256.h"
#include "dlfcn.h"
#include "../providers/gdn/fla_checkpoint.h"
#include "../providers/gdn/sm121_q1_gdn.h"
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace aima_port {
namespace {
struct Asset { const char* name; std::size_t bytes; const char* sha256; };
#include "gb10_gdn_assets.inc"
using Launch = qrt_fla_checkpoint::SeededLaunch;
using Prepare = int (*)(const char*);
using Release = void (*)();
using Error = const char* (*)();

void check(hipError_t status, const char* action) {
  if (status != hipSuccess) throw std::runtime_error(std::string(action) + ": " + hipGetErrorString(status));
}
std::filesystem::path selected(const char* name) {
  const char* value = std::getenv(name);
  if (!value || !*value) throw std::runtime_error(std::string("Missing GDN input: ") + name);
  return std::filesystem::u8path(value);
}
void require_setting(const char* name, const char* expected) {
  const char* value = std::getenv(name);
  if (!value || std::string(value) != expected)
    throw std::runtime_error(std::string("Unsupported GDN setting: ") + name);
}
std::vector<unsigned char> read(const std::filesystem::path& path, const Asset& asset) {
  if (std::filesystem::file_size(path) != asset.bytes)
    throw std::runtime_error("GDN artifact size differs: " + path.u8string());
  std::vector<unsigned char> data(asset.bytes);
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(data.data()), data.size());
  if (!file || file.peek() != std::char_traits<char>::eof() ||
      aima::sha256_bytes(data.data(), data.size()) != asset.sha256)
    throw std::runtime_error("GDN artifact identity differs: " + path.u8string());
  return data;
}
struct Device {
  void* data = nullptr;
  Device() = default;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device() { if (data) hipFree(data); }
  void allocate(std::size_t bytes) { check(hipMalloc(&data, bytes), "GDN allocation"); }
  void upload(const std::filesystem::path& path, const Asset& asset) {
    const auto bytes = read(path, asset);
    allocate(bytes.size());
    check(hipMemcpy(data, bytes.data(), bytes.size(), hipMemcpyHostToDevice), "GDN upload");
  }
  template<class T> T* as() const { return static_cast<T*>(data); }
};
struct State {
  void* library = nullptr;
  Release release = nullptr;
  Error error = nullptr;
  Launch cold = nullptr, seeded = nullptr;
  std::array<Device, 40> gate;
  Device beta, prefill_beta, exp2, rsqrt, raw, gates, output, decode_ab;
  GdnPrefillObserver observer = nullptr;
  void* observer_context = nullptr;
  std::size_t observer_layer = 0;
  bool observer_first64 = false;
  ~State() {
    if (library) {
      hipDeviceSynchronize();
      if (release) release();
      dlclose(library);
    }
  }
};
State* active = nullptr;

// Token-major BF16 projections and raw convolution become the existing FLA
// ABI's FP32 carriers. Prefill keeps its qualified BF16 sigmoid table; decode
// uses the distinct FP32 sigmoid endpoint from the original Q2 recurrence.
static __global__ void prepare_prefill(
    const uint16_t* conv, const uint16_t* a, const uint16_t* b,
    float* raw, float* gates, const float* g_table, const uint16_t* beta,
    unsigned tokens) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < tokens * 8192u) raw[i] = qrt_sm121_q1::widen(conv[i]);
  if (i < tokens * 32u) {
    const unsigned row = i / 32u, head = i % 32u;
    gates[row * 64u + head] = g_table[head * 65536u + a[i]];
    gates[row * 64u + 32u + head] = qrt_sm121_q1::widen(beta[b[i]]);
  }
}
static __global__ void prepare_decode(const uint16_t* conv, const uint16_t* a,
    const uint16_t* b, float* raw, float* ab) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 8192u) raw[i] = qrt_sm121_q1::widen(conv[i]);
  if (i < 32u) {
    ab[i] = qrt_sm121_q1::widen(a[i]);
    ab[32u + i] = qrt_sm121_q1::widen(b[i]);
  }
}
static __global__ void copy_core(const float* source, uint16_t* output, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = qrt_sm121_q1::bf16(source[i]);
}
static __global__ void sample_prefill(const uint16_t* source, uint16_t* samples,
                                     unsigned columns) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 128u * columns)
    samples[i] = source[((i / columns + 1u) * 64u - 1u) * columns + i % columns];
}
State& bound(std::size_t layer, const void* conv, const void* a, const void* b,
             void* output, void* state) {
  if (!active || layer >= 40 || layer % 4 == 3 || !active->gate[layer].data ||
      !conv || !a || !b || !output || !state)
    throw std::runtime_error("GDN owner or model-layer binding is incomplete");
  return *active;
}
}  // namespace

struct Gb10GdnOwner::Impl { State state; };
Gb10GdnOwner::Gb10GdnOwner() : impl_(std::make_unique<Impl>()) {
  if (active) throw std::runtime_error("A GDN owner is already active");
  for (const char* name : {"QRT_FLA_GDN_STATE_BLACKWELL", "QRT_FLA_GDN_KKT_BLACKWELL",
       "QRT_FLA_GDN_WU_BLACKWELL", "QRT_FLA_GDN_OUTPUT_BLACKWELL", "QRT_FLA_GDN_NORM_BLACKWELL",
       "QRT_FLA_GDN_INVERSE_BLACKWELL", "QRT_FLA_GDN_COOPERATIVE_EXACT", "QRT_FLA_GDN_BATCHED_EXACT",
       "QRT_FLA_GDN_SCALAR_FLOAT_MATRICES", "QRT_FLA_GDN_PAIRED_SCORE_ARENAS"}) require_setting(name, "1");
  require_setting("QRT_FLA_GDN_TILED_KKT", "2");
  require_setting("QRT_FLA_GDN_SCALAR_FLOAT_STATE", "8");
  check(hipSetDevice(0), "GDN device");
  auto& s = impl_->state;
  const auto directory = selected("AIMA_PORT_GDN_PROVIDER_DIR");
  for (const auto& asset : provider_assets) read(directory / asset.name, asset);
  s.library = dlopen((directory / "qrt_fla_chunk_gdn_provider.dll").c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!s.library) throw std::runtime_error("Cannot load the pinned GDN provider");
  const auto prepare = reinterpret_cast<Prepare>(dlsym(s.library, "qrt_aiter_fused_gdn_q8192_prepare"));
  s.release = reinterpret_cast<Release>(dlsym(s.library, "qrt_aiter_fused_gdn_q8192_release"));
  s.error = reinterpret_cast<Error>(dlsym(s.library, "qrt_aiter_fused_gdn_q8192_last_error"));
  s.cold = reinterpret_cast<Launch>(dlsym(s.library, "qrt_aiter_fused_gdn_launch_async_dynamic"));
  s.seeded = reinterpret_cast<Launch>(dlsym(s.library, "qrt_fla_gdn_launch_async_seeded_f32_v1"));
  if (!prepare || !s.release || !s.error || !s.cold || !s.seeded)
    throw std::runtime_error("GDN provider ABI is incomplete");
  // These same two files are SHA-checked by the provider before its own upload.
  s.exp2.upload(selected("QRT_FLA_GDN_SM121_EXP2_TABLE"), exp2_asset);
  s.rsqrt.upload(selected("QRT_FLA_GDN_SM121_RSQRT_TABLE"), rsqrt_asset);
  s.beta.upload(selected("AIMA_PORT_GDN_BETA_TABLE"), beta_asset);
  const auto gate_directory = selected("AIMA_PORT_GDN_GATE_DIR");
  s.prefill_beta.upload(gate_directory / prefill_beta_asset.name, prefill_beta_asset);
  for (unsigned layer = 0; layer < 40; ++layer)
    if (layer % 4 != 3) s.gate[layer].upload(gate_directory / gate_assets[layer].name, gate_assets[layer]);
  if (prepare(directory.u8string().c_str()) != 1)
    throw std::runtime_error(std::string("GDN preparation failed: ") + s.error());
  s.raw.allocate(8192ull * 8192 * sizeof(float));
  s.gates.allocate(8192ull * 64 * sizeof(float));
  s.output.allocate(8192ull * 4096 * sizeof(float));
  s.decode_ab.allocate(64 * sizeof(float));
  active = &s;
}
Gb10GdnOwner::~Gb10GdnOwner() { if (active == &impl_->state) active = nullptr; }
const unsigned char* gb10_rsqrt_table() {
  if (!active || !active->rsqrt.data) throw std::runtime_error("GDN rsqrt owner is absent");
  return active->rsqrt.as<unsigned char>();
}

void set_gdn_prefill_observer(std::size_t layer, GdnPrefillObserver callback,
                              void* context, bool first64) {
  if (!active || layer >= 40 || layer % 4 == 3 || !callback || !context || active->observer)
    throw std::runtime_error("GDN prefill observation owner is invalid");
  active->observer = callback;
  active->observer_context = context;
  active->observer_layer = layer;
  active->observer_first64 = first64;
}
void observe_gdn_prefill(std::size_t layer, const char* name, const void* values,
                         std::size_t columns, std::size_t tokens) {
  if (!active || !active->observer || layer != active->observer_layer) return;
  if (!values || values == active->output.data || !name || tokens != 8192 ||
      (columns != 1 && columns != 32 && columns != 256 && columns != 512 &&
       columns != 2048 && columns != 4096 && columns != 8192))
    throw std::runtime_error("GDN prefill observation geometry is invalid");
  // This buffer is dead at each caller: before the FLA call, or after its
  // output has been converted into the engine's distinct BF16 destination.
  hipLaunchKernelGGL(sample_prefill, dim3((128u * columns + 255u) / 256u), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(values), active->output.as<uint16_t>(), static_cast<unsigned>(columns));
  check(hipGetLastError(), "GDN prefill observation gather");
  active->observer(name, active->output.data, 128u * columns * sizeof(uint16_t), active->observer_context);
  if (active->observer_first64) {
    const std::string label(name), suffix("-sampled");
    if (label.size() <= suffix.size() ||
        label.compare(label.size() - suffix.size(), suffix.size(), suffix) != 0)
      throw std::runtime_error("First64 observation requires a sampled surface name");
    const auto first = label.substr(0, label.size() - suffix.size()) + "-first64";
    // Read the first real 64 rows directly. No gather or arithmetic writes to
    // the input; the same output-only collector owns transfer and byte guards.
    active->observer(first.c_str(), values, 64u * columns * sizeof(uint16_t), active->observer_context);
  }
}

void gb10_prefill_gdn(std::size_t layer, const void* conv, const void* a,
    const void* b, void* output, void* state, std::size_t tokens, bool has_initial) {
  auto& s = bound(layer, conv, a, b, output, state);
  if (tokens != 8192) throw std::runtime_error("GDN experiment requires exact q8192 prefill");
  observe_gdn_prefill(layer, "prefill-conv-sampled", conv, 8192, tokens);
  hipLaunchKernelGGL(prepare_prefill, dim3(tokens * 8192u / 256u), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(conv), static_cast<const uint16_t*>(a), static_cast<const uint16_t*>(b),
      s.raw.as<float>(), s.gates.as<float>(), s.gate[layer].as<float>(), s.prefill_beta.as<uint16_t>(),
      static_cast<unsigned>(tokens));
  check(hipGetLastError(), "GDN prefill input conversion");
  const auto launch = has_initial ? s.seeded : s.cold;
  if (launch(s.raw.as<float>(), s.gates.as<float>(), s.output.as<float>(),
             static_cast<float*>(state), 0, nullptr, static_cast<int32_t>(tokens)) != 1)
    throw std::runtime_error(std::string("GDN prefill failed: ") + s.error());
  hipLaunchKernelGGL(copy_core, dim3(tokens * 4096u / 256u), dim3(256), 0, nullptr,
      s.output.as<float>(), static_cast<uint16_t*>(output), static_cast<unsigned>(tokens * 4096u));
  check(hipGetLastError(), "GDN prefill output conversion");
  observe_gdn_prefill(layer, "prefill-core-sampled", output, 4096, tokens);
}
void gb10_decode_gdn(std::size_t layer, const void* conv, const void* a,
    const void* b, void* output, void* state, hipStream_t stream) {
  auto& s = bound(layer, conv, a, b, output, state);
  hipLaunchKernelGGL(prepare_decode, dim3(32), dim3(256), 0, stream,
      static_cast<const uint16_t*>(conv), static_cast<const uint16_t*>(a), static_cast<const uint16_t*>(b),
      s.raw.as<float>(), s.decode_ab.as<float>());
  check(hipGetLastError(), "GDN decode input conversion");
  hipLaunchKernelGGL(qrt_sm121_q1::recurrent, dim3(32), dim3(128), 0, stream,
      s.raw.as<float>(), s.decode_ab.as<float>(), s.decode_ab.as<float>() + 32,
      static_cast<float*>(state), false, s.output.as<float>(), nullptr, nullptr, nullptr,
      s.gate[layer].as<float>(), s.beta.as<float>(), s.exp2.as<unsigned char>(), s.rsqrt.as<unsigned char>(),
      false, nullptr, nullptr);
  check(hipGetLastError(), "GDN original Q2 arithmetic update");
  hipLaunchKernelGGL(copy_core, dim3(16), dim3(256), 0, stream,
      s.output.as<float>(), static_cast<uint16_t*>(output), 4096u);
  check(hipGetLastError(), "GDN decode output conversion");
}
}  // namespace aima_port
