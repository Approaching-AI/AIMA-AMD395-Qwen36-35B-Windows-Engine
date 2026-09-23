// SPDX-License-Identifier: Apache-2.0
#include "gb10_gdn.h"
#include "aima/aot_kernel.h"
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
#include "gb10_gdn_ordered_images.inc"
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
bool native_prefill_setting(const char* value) {
  if (!value || std::string(value) == "0") return false;
  if (std::string(value) == "1") return true;
  throw std::invalid_argument("Native GDN prefill setting must be 0 or 1");
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
  Device native_matrix, native_inverse;
  std::array<std::unique_ptr<aima::AotKernel>, static_cast<unsigned>(OrderedStage::Count)> ordered;
  GdnPrefillObserver observer = nullptr;
  void* observer_context = nullptr;
  std::size_t observer_layer = 0;
  bool observer_first64 = false;
  bool native_prefill = false;
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
// Same contiguous eight-values-per-lane and XOR16 reduction as the qualified
// FLA normalizer. Read BF16 directly instead of materializing an FP32 carrier.
static __global__ void prepare_native_qk(const uint16_t* conv, uint16_t* q,
    uint16_t* k, unsigned tokens, const unsigned char* rsqrt) {
  const unsigned row = blockIdx.x * 16u + threadIdx.x / 16u, lane = threadIdx.x % 16u;
  if (row >= tokens * 16u) return;
  const unsigned token = row / 16u, head = row % 16u;
  float qv[8], kv[8];
  for (unsigned i = 0; i < 8; ++i) {
    const unsigned offset = token * 8192u + head * 128u + lane * 8u + i;
    qv[i] = qrt_sm121_q1::widen(conv[offset]);
    kv[i] = qrt_sm121_q1::widen(conv[offset + 2048u]);
  }
  float qs = qrt_sm121_q1::embedding_lane_sumsq(qv);
  float ks = qrt_sm121_q1::embedding_lane_sumsq(kv);
  for (unsigned delta = 8; delta; delta >>= 1) {
    qs = qrt_sm121_q1::add(qs, __shfl_xor(qs, delta, 16));
    ks = qrt_sm121_q1::add(ks, __shfl_xor(ks, delta, 16));
  }
  const float qr = qrt_sm121_rsqrt::evaluate(rsqrt, qrt_sm121_q1::add(qs, 1.e-6f));
  const float kr = qrt_sm121_rsqrt::evaluate(rsqrt, qrt_sm121_q1::add(ks, 1.e-6f));
  for (unsigned i = 0; i < 8; ++i) {
    const unsigned offset = row * 128u + lane * 8u + i;
    q[offset] = qrt_sm121_q1::bf16(qrt_sm121_q1::multiply(qv[i], qr));
    k[offset] = qrt_sm121_q1::bf16(qrt_sm121_q1::multiply(kv[i], kr));
  }
}
static __global__ void prepare_native_v_gate(const uint16_t* conv, const uint16_t* a,
    const uint16_t* b, uint16_t* v, float* g, float* beta, const float* g_table,
    const uint16_t* beta_table, unsigned tokens) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < tokens * 4096u) v[i] = conv[(i / 4096u) * 8192u + 4096u + i % 4096u];
  if (i < tokens * 32u) {
    g[i] = g_table[(i % 32u) * 65536u + a[i]];
    // The imported ABI carries FP32 beta; the original prefill boundary is BF16.
    beta[i] = qrt_sm121_q1::widen(beta_table[b[i]]);
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
  s.native_prefill = native_prefill_setting(std::getenv("AIMA_PORT_NATIVE_GDN_PREFILL"));
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
  if (s.native_prefill) {
    s.native_matrix.allocate(8192ull * 32 * 64 * sizeof(float));
    s.native_inverse.allocate(8192ull * 32 * 64 * sizeof(uint16_t));
    for (unsigned i = 0; i < s.ordered.size(); ++i) {
      const auto& image = ordered_images[i];
      if (aima::sha256_bytes(image.data, image.bytes) != image.sha256)
        throw std::runtime_error("Native GDN ordered embedded image identity differs");
      s.ordered[i] = std::make_unique<aima::AotKernel>(
          std::vector<unsigned char>(image.data, image.data + image.bytes), image.name);
    }
  }
  active = &s;
}
Gb10GdnOwner::~Gb10GdnOwner() { if (active == &impl_->state) active = nullptr; }
const unsigned char* gb10_rsqrt_table() {
  if (!active || !active->rsqrt.data) throw std::runtime_error("GDN rsqrt owner is absent");
  return active->rsqrt.as<unsigned char>();
}
const unsigned char* gb10_exp2_table() {
  if (!active || !active->exp2.data) throw std::runtime_error("GDN exp2 owner is absent");
  return active->exp2.as<unsigned char>();
}
const uint16_t* gb10_sigmoid_table() {
  if (!active || !active->prefill_beta.data) throw std::runtime_error("GDN sigmoid owner is absent");
  return active->prefill_beta.as<uint16_t>();
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
       columns != 2048 && columns != 4096 && columns != 8192 && columns != 16384))
    throw std::runtime_error("GDN prefill observation geometry is invalid");
  // This buffer is dead at each caller: before the FLA call, or after its
  // output has been converted into the engine's distinct BF16 destination.
  // The widest routed-MoE surface gathers 4MiB into the existing 128MiB
  // allocation. The source keeps its full [8192,8,2048] row stride.
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
  if (s.native_prefill) throw std::runtime_error("Native GDN prefill cannot dispatch the FLA adapter");
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
bool gb10_native_gdn_prefill_enabled(std::size_t tokens, bool has_initial) {
  if (!active) throw std::runtime_error("Native GDN prefill requires its owner");
  if (!active->native_prefill) return false;
  if (tokens != 8192 || has_initial)
    throw std::invalid_argument("Native GDN prefill comparison requires cold q8192");
  return true;
}
NativeGdnMatrices gb10_prepare_native_gdn(std::size_t layer, const void* conv,
    const void* a, const void* b, void* q, void* k, void* v, void* g, void* beta,
    std::size_t tokens) {
  if (!active || !active->native_prefill || tokens != 8192 || layer >= 40 || layer % 4 == 3 ||
      !active->gate[layer].data || !active->rsqrt.data || !active->prefill_beta.data ||
      !active->native_matrix.data || !active->native_inverse.data)
    throw std::invalid_argument("Native GDN preparation owner or geometry is invalid");
  const void* pointers[] = {conv, a, b, q, k, v, g, beta};
  const std::size_t bytes[] = {tokens * 8192 * 2, tokens * 32 * 2, tokens * 32 * 2,
      tokens * 2048 * 2, tokens * 2048 * 2, tokens * 4096 * 2, tokens * 32 * 4, tokens * 32 * 4};
  for (unsigned i = 0; i < 8; ++i) {
    const auto first = reinterpret_cast<std::uintptr_t>(pointers[i]);
    if (!first || first % (i >= 6 ? 4 : 2) || first > UINTPTR_MAX - bytes[i])
      throw std::invalid_argument("Native GDN preparation pointer is invalid");
    for (unsigned j = 0; j < i; ++j) {
      const auto other = reinterpret_cast<std::uintptr_t>(pointers[j]);
      if (first < other + bytes[j] && other < first + bytes[i])
        throw std::invalid_argument("Native GDN preparation spans overlap");
    }
  }
  auto& s = *active;
  hipLaunchKernelGGL(prepare_native_qk, dim3(tokens), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(conv), static_cast<uint16_t*>(q), static_cast<uint16_t*>(k),
      static_cast<unsigned>(tokens), s.rsqrt.as<unsigned char>());
  check(hipGetLastError(), "Native GDN original Q/K normalization");
  hipLaunchKernelGGL(prepare_native_v_gate, dim3(tokens * 4096u / 256u), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(conv), static_cast<const uint16_t*>(a), static_cast<const uint16_t*>(b),
      static_cast<uint16_t*>(v), static_cast<float*>(g), static_cast<float*>(beta),
      s.gate[layer].as<float>(), s.prefill_beta.as<uint16_t>(), static_cast<unsigned>(tokens));
  check(hipGetLastError(), "Native GDN original V and gate preparation");
  return {s.native_matrix.data, s.native_inverse.data};
}
std::size_t gb10_native_gdn_pipeline(const void* q, const void* k, const void* v,
    const void* g, const void* beta, void* w, void* u, void* output, void* state,
    std::size_t tokens) {
  if (!active || !active->native_prefill || tokens != 8192)
    throw std::invalid_argument("Native GDN pipeline owner or geometry is invalid");
  auto& s = *active;
  for (const auto& kernel : s.ordered)
    if (!kernel) throw std::runtime_error("Native GDN ordered module is absent");
  constexpr std::size_t state_bytes = 32ull * 128 * 128 * sizeof(float);
  constexpr std::size_t chunk_bytes = 64ull * 32 * 128 * sizeof(uint16_t);
  const void* pointers[] = {q, k, v, g, beta, w, u, output, state,
      s.raw.data, s.native_matrix.data, s.native_inverse.data, s.exp2.data};
  const std::size_t bytes[] = {tokens * 2048 * 2, tokens * 2048 * 2,
      tokens * 4096 * 2, tokens * 32 * 4, tokens * 32 * 4,
      tokens * 4096 * 2, tokens * 4096 * 2, tokens * 4096 * 2, state_bytes,
      8192ull * 8192 * sizeof(float), tokens * 32 * 64 * 4,
      tokens * 32 * 64 * 2, exp2_asset.bytes};
  for (unsigned i = 0; i < sizeof(pointers) / sizeof(pointers[0]); ++i) {
    const auto first = reinterpret_cast<std::uintptr_t>(pointers[i]);
    const unsigned alignment = (i == 3 || i == 4 || i == 8 || i == 9 || i == 10 || i == 12) ? 4 : 2;
    if (!first || first % alignment || first > UINTPTR_MAX - bytes[i])
      throw std::invalid_argument("Native GDN pipeline pointer is invalid");
    for (unsigned j = 0; j < i; ++j) {
      const auto other = reinterpret_cast<std::uintptr_t>(pointers[j]);
      if (first < other + bytes[j] && other < first + bytes[i])
        throw std::invalid_argument("Native GDN pipeline live spans overlap");
    }
  }
  std::size_t launches = 0;
  auto launch = [&](OrderedStage stage, unsigned x, unsigned y, unsigned z,
                    const std::vector<void*>& parameters) {
    const auto index = static_cast<unsigned>(stage);
    const auto& image = ordered_images[index];
    s.ordered[index]->launch(aima::AotLaunchConfig{x, y, z, image.warps, 32, image.shared}, parameters);
    ++launches;
  };
  void* matrix = s.native_matrix.data;
  void* inverse = s.native_inverse.data;
  const void* table = s.exp2.data;
  void* unused_debug = nullptr;
  std::int32_t count = 8192;
  launch(OrderedStage::Kkt, 1024, 8, 32, {&k, &k, &beta, &g, &table, &matrix, &count});
  // The inverse writes its lower block triangle; upper blocks must be +0.
  check(hipMemset(inverse, 0, tokens * 32 * 64 * 2), "Native GDN inverse clear");
  launch(OrderedStage::Inverse, 128, 32, 1, {&matrix, &inverse, &count});
  launch(OrderedStage::W, 1024, 16, 32, {&inverse, &k, &beta, &g, &table, &w, &count});
  launch(OrderedStage::U, 1024, 16, 32, {&inverse, &v, &beta, &u, &unused_debug, &count});
  // KKT is dead after inversion. Its first half now holds BF16 scores.
  void* scores = matrix;
  launch(OrderedStage::Scores, 1024, 8, 32, {&q, &k, &beta, &g, &table, &scores, &count});
  // The FLA conversion carrier is idle throughout native prefill. Its first
  // 5MiB hold two FP32 states and two single-chunk BF16 residual carriers.
  auto* scratch = s.raw.as<unsigned char>();
  void* states[] = {scratch, scratch + state_bytes};
  void* residual = scratch + 2 * state_bytes;
  void* v_new = scratch + 2 * state_bytes + chunk_bytes;
  check(hipMemset(states[0], 0, state_bytes), "Native GDN cold state clear");
  const void* incoming = states[0];
  std::int32_t chunk_tokens = 64;
  for (std::size_t chunk = 0; chunk < tokens / 64; ++chunk) {
    const auto first = chunk * 64;
    const void* chunk_q = static_cast<const uint16_t*>(q) + first * 2048;
    const void* chunk_k = static_cast<const uint16_t*>(k) + first * 2048;
    const void* chunk_w = static_cast<const uint16_t*>(w) + first * 4096;
    const void* chunk_u = static_cast<const uint16_t*>(u) + first * 4096;
    const void* chunk_g = static_cast<const float*>(g) + first * 32;
    const void* chunk_scores = static_cast<const uint16_t*>(scores) + first * 32 * 64;
    void* chunk_output = static_cast<uint16_t*>(output) + first * 4096;
    void* next = chunk + 1 == tokens / 64 ? state : states[(chunk + 1) % 2];
    launch(OrderedStage::Residual, 8, 16, 32,
        {&chunk_w, &chunk_u, &incoming, &chunk_g, &table, &v_new, &residual, &chunk_tokens});
    launch(OrderedStage::State, 16, 16, 32,
        {&chunk_k, &residual, &incoming, &chunk_g, &table, &next, &chunk_tokens});
    launch(OrderedStage::Output, 8, 16, 32,
        {&chunk_q, &v_new, &incoming, &chunk_g, &chunk_scores, &table, &chunk_output, &chunk_tokens});
    incoming = next;
  }
  return launches;
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
