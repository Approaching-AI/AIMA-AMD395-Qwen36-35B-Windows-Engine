// SPDX-License-Identifier: Apache-2.0
#include "gb10_prefill_projection.h"
#include "../providers/moe_accumulator/sm121_staged_half_projection.h"
#include "../providers/moe_accumulator/bf16_midpoint_selector.h"
#include "../providers/gdn/sm121_q1_math.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace aima_port {
namespace {
namespace staged = qrt_sm121_staged_half_projection;
namespace half = qrt_sm121_scaled_half_products;
constexpr unsigned max_tokens = 8192, max_rows = 12352, max_k = 4096;
constexpr unsigned window_capacity = 1u << 20;
constexpr unsigned max_windows = (max_tokens * max_rows + window_capacity - 1) / window_capacity;
void check(hipError_t error) {
  if (error != hipSuccess) throw std::runtime_error("Prefill projection GPU operation failed");
}
struct Device {
  void* data = nullptr;
  ~Device() { if (data) hipFree(data); }
  void allocate(std::size_t bytes) {
    if (hipMalloc(&data, bytes) != hipSuccess) throw std::runtime_error("Prefill projection scratch allocation failed");
  }
  template<class T> T* as() const { return static_cast<T*>(data); }
};
struct Event {
  hipEvent_t value = nullptr;
  Event() { check(hipEventCreate(&value)); }
  ~Event() { if (value) hipEventDestroy(value); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};
struct Profile {
  // Four common boundaries and three boundaries per bounded replay window.
  // No per-projection allocation or host read of a live selection counter.
  std::array<Event, 4 + 3 * max_windows> events;
  Device counts;
  std::array<unsigned, max_windows> host_counts{};
  bool armed = false, inflight = false, fallback = false;
  unsigned ordinal = 0, rows = 0, width = 0;
  Profile() { counts.allocate(max_windows * sizeof(unsigned)); }
  void mark(unsigned index) { check(hipEventRecord(events.at(index).value, nullptr)); }
  float ms(unsigned first, unsigned last) const {
    float elapsed = 0;
    check(hipEventElapsedTime(&elapsed, events.at(first).value, events.at(last).value));
    return elapsed;
  }
  void begin(unsigned n, unsigned k) {
    if (inflight) throw std::logic_error("Prefill projection profile already in flight");
    rows = n; width = k; fallback = false; inflight = true;
    mark(0);
  }
  void finish(unsigned windows, bool contiguous, unsigned ppb, bool linear_output) {
    check(hipEventSynchronize(events.at(3 + 3 * windows).value));
    check(hipMemcpy(host_counts.data(), counts.data, windows * sizeof(unsigned), hipMemcpyDeviceToHost));
    double selection_ms = 0, replay_ms = 0;
    unsigned long long candidates = 0;
    for (unsigned i = 0; i < windows; ++i) {
      const unsigned capacity = std::min(window_capacity, max_tokens * rows - i * window_capacity);
      if (host_counts[i] > capacity) throw std::runtime_error("Prefill projection profile count exceeds window");
      candidates += host_counts[i];
      selection_ms += ms(4 + 3 * i, 5 + 3 * i);
      replay_ms += ms(5 + 3 * i, 6 + 3 * i);
    }
    std::fprintf(stderr, "{\"event\":\"prefill_projection_profile\",\"ordinal\":%u,"
        "\"tokens\":%u,\"rows\":%u,\"reduction\":%u,\"weight_rows_contiguous\":%s,"
        "\"producer\":\"%s\",\"windows\":%u,\"cells\":%llu,\"candidates\":%llu,"
        "\"bound_ppb\":%u,\"linear_output\":%s,"
        "\"producer_ms\":%.6f,\"operands_ms\":%.6f,\"norm_bound_ms\":%.6f,"
        "\"selection_ms\":%.6f,\"replay_ms\":%.6f,\"total_gpu_ms\":%.6f,"
        "\"completed_gpu_events\":true,\"diagnostic_only\":true}\n",
        ordinal++, max_tokens, rows, width, contiguous ? "true" : "false",
        fallback ? "wmma-fallback" : "hipblaslt", windows,
        static_cast<unsigned long long>(max_tokens) * rows, candidates,
        ppb, linear_output ? "true" : "false",
        ms(0, 1), ms(1, 2), ms(2, 3), selection_ms, replay_ms, ms(0, 3 + 3 * windows));
    inflight = false;
  }
};
struct State {
  Device raw, inputs, weights, input_l2, weight_l2, indices, count;
  std::unique_ptr<Profile> profile;
  bool wmma = false, linear_bound = false, linear_output = false;
};
State* active = nullptr;
bool enabled(const char* name) {
  const char* value = std::getenv(name);
  return value && std::strcmp(value, "1") == 0;
}

using WmmaBf16 = unsigned short __attribute__((ext_vector_type(16)));
using WmmaF32 = float __attribute__((ext_vector_type(8)));
// Same ascending-K16 WMMA producer and M64/N128 LDS geometry as the existing
// Windows provider, generalized to the imported engine's two weight views.
// It covers shapes for which the installed hipBLASLt has no FP32 destination
// solution; the same selector and exact replay follow either producer.
static __global__ void fallback_matmul(const uint16_t* input, const uint16_t* weights,
    float* output, unsigned tokens, unsigned rows, unsigned width, bool contiguous) {
  constexpr unsigned stride = 66;
  __shared__ uint16_t weight_tile[128][stride];
  __shared__ uint16_t input_tile[64][stride];
  const unsigned thread = threadIdx.x, wave = thread / 32u, lane = thread % 32u;
  const unsigned source = lane % 16u, segment = lane / 16u;
  const unsigned row_base = blockIdx.x * 128u, token_base = blockIdx.y * 64u;
  WmmaF32 accumulators[4] = {};
  for (unsigned base = 0; base < width; base += 64u) {
    for (unsigned cell = thread; cell < 128u * 64u; cell += 256u) {
      const unsigned row = contiguous ? cell / 64u : cell % 128u;
      const unsigned k = contiguous ? cell % 64u : cell / 128u;
      weight_tile[row][k] = row_base + row < rows ? weights[contiguous
          ? std::size_t(row_base + row) * width + base + k
          : std::size_t(base + k) * rows + row_base + row] : uint16_t(0);
    }
    for (unsigned cell = thread; cell < 64u * 64u; cell += 256u) {
      const unsigned token = cell / 64u, k = cell % 64u;
      input_tile[token][k] = token_base + token < tokens
          ? input[std::size_t(token_base + token) * width + base + k] : uint16_t(0);
    }
    __syncthreads();
    for (unsigned offset = 0; offset < 64u; offset += 16u) {
      WmmaBf16 weight_fragment;
      for (unsigned e = 0; e < 16; ++e) weight_fragment[e] = weight_tile[wave * 16u + source][offset + e];
      for (unsigned fragment = 0; fragment < 4; ++fragment) {
        WmmaBf16 input_fragment;
        for (unsigned e = 0; e < 16; ++e) input_fragment[e] = input_tile[fragment * 16u + source][offset + e];
        accumulators[fragment] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            input_fragment, weight_fragment, accumulators[fragment]);
      }
    }
    __syncthreads();
  }
  const unsigned row = row_base + wave * 16u + source;
  if (row >= rows) return;
  for (unsigned fragment = 0; fragment < 4; ++fragment)
    for (unsigned e = 0; e < 8; ++e) {
      const unsigned token = token_base + fragment * 16u + 2u * e + segment;
      if (token < tokens) output[std::size_t(token) * rows + row] = accumulators[fragment][e];
    }
}

// Reuse the qualified lossless BF16 -> scaled-FP16 K16 representation. A
// fused [K,N] weight view is gathered by output row without altering K order.
static __global__ void prepare_operands(const uint16_t* source, half::Row* output,
    unsigned rows, unsigned width, bool contiguous) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned groups = width / 16u;
  if (index >= rows * groups) return;
  const unsigned row = contiguous ? index / groups : index % rows;
  const unsigned group = contiguous ? index % groups : index / rows;
  uint16_t original[16];
  for (unsigned i = 0; i < 16; ++i) {
    const unsigned k = group * 16u + i;
    original[i] = source[contiguous ? std::size_t(row) * width + k : std::size_t(k) * rows + row];
  }
  output[std::size_t(row) * groups + group] = half::prepare(original);
}

// Same F64 Cauchy-Schwarz norm bound and outward inflation as the existing
// Windows projection selector. This is an admission estimate, not a claim
// that an approximate GEMM is reference-exact.
static __global__ void row_l2(const uint16_t* source, float* output,
    unsigned rows, unsigned width, bool contiguous) {
  __shared__ double sums[256];
  __shared__ unsigned nonzero[256];
  const unsigned row = blockIdx.x;
  if (row >= rows) return;
  double sum = 0.;
  unsigned any = 0;
  for (unsigned k = threadIdx.x; k < width; k += 256u) {
    const uint16_t word = source[contiguous ?
        std::size_t(row) * width + k : std::size_t(k) * rows + row];
    any |= word & 0x7fffu;
    const double x = qrt_sm121_q1::widen(word);
    sum += x * x;
  }
  sums[threadIdx.x] = sum;
  nonzero[threadIdx.x] = any;
  __syncthreads();
  for (unsigned step = 128; step; step >>= 1) {
    if (threadIdx.x < step) {
      sums[threadIdx.x] += sums[threadIdx.x + step];
      nonzero[threadIdx.x] |= nonzero[threadIdx.x + step];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float bound = static_cast<float>(sqrt(sums[0])) * 1.00002f;
    // A stored zero denotes an actually zero row, including during warmup.
    // Preserve a conservative positive bound if a tiny norm flushes to zero.
    if (nonzero[0] && bound < 0x1p-126f) bound = 0x1p-126f;
    output[row] = bound;
  }
}

static __global__ void select_and_round(const float* raw, uint16_t* output,
    const float* input_l2, const float* weight_l2, unsigned rows,
    unsigned start, unsigned size, unsigned bound_ppb,
    unsigned* count, unsigned* indices) {
  const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local >= size) return;
  const unsigned index = start + local, row = index % rows, token = index / rows;
  const float left = input_l2[token], right = weight_l2[row];
  if (left == 0.f || right == 0.f) { output[index] = 0; return; }
  const float value = raw[index];
  const unsigned bits = qrt_sm121_exp2::bits(value), low = bits & 65535u;
  const unsigned distance = low >= 32768u ? low - 32768u : 32768u - low;
  const float bound = left * right * (float(bound_ppb) * 1.e-9f);
  const bool selected = ((bits >> 23u) & 255u) < 32u || distance <= 512u ||
                        qrt_bf16_midpoint::within_error(value, bound);
  if (selected) indices[atomicAdd(count, 1u)] = index;
  output[index] = qrt_sm121_q1::bf16(value);
}

static __global__ void replay_selected(const half::Row* inputs,
    const half::Row* weights, uint16_t* output, unsigned rows, unsigned width,
    const unsigned* count, const unsigned* indices) {
  const unsigned lane = threadIdx.x & 3u, groups = width / 16u;
  const unsigned size = *count;
  for (unsigned candidate = blockIdx.x * 64u + threadIdx.x / 4u;
       candidate < size; candidate += gridDim.x * 64u) {
    const unsigned index = indices[candidate], token = index / rows, row = index % rows;
    const float corrected = staged::dot<2>(inputs + std::size_t(token) * groups,
        weights + std::size_t(row) * groups, width);
    if (!lane) output[index] = qrt_sm121_q1::bf16(corrected);
  }
}
State& bound(std::size_t tokens, std::size_t rows, std::size_t reduction, void* stream) {
  if (!active || stream || !gb10_prefill_projection_shape(tokens, rows, reduction, false))
    throw std::invalid_argument("Prefill projection requires its q8192 default-stream owner");
  if (active->linear_output && (rows != 2048 || reduction != 4096))
    throw std::invalid_argument("Linear OUT selector requires N2048/K4096");
  return *active;
}
}  // namespace

struct Gb10PrefillProjectionOwner::Impl { State state; };
Gb10PrefillProjectionOwner::Gb10PrefillProjectionOwner() : impl_(std::make_unique<Impl>()) {
  if (active) throw std::runtime_error("A prefill projection owner is already active");
  auto& s = impl_->state;
  s.raw.allocate(std::size_t(max_tokens) * max_rows * sizeof(float));
  s.inputs.allocate(std::size_t(max_tokens) * (max_k / 16u) * sizeof(half::Row));
  s.weights.allocate(std::size_t(max_rows) * (max_k / 16u) * sizeof(half::Row));
  s.input_l2.allocate(max_tokens * sizeof(float));
  s.weight_l2.allocate(max_rows * sizeof(float));
  s.indices.allocate(window_capacity * sizeof(unsigned));
  s.count.allocate(sizeof(unsigned));
  if (enabled("AIMA_PORT_PREFILL_PROJECTION_PROFILE")) s.profile = std::make_unique<Profile>();
  s.wmma = enabled("AIMA_PORT_PREFILL_WMMA");
  s.linear_bound = enabled("AIMA_PORT_PREFILL_LINEAR_BOUND");
  active = &s;
}
Gb10PrefillProjectionOwner::~Gb10PrefillProjectionOwner() {
  if (active == &impl_->state) { hipDeviceSynchronize(); active = nullptr; }
}
bool gb10_prefill_projection_shape(std::size_t tokens, std::size_t rows,
    std::size_t reduction, bool bias) {
  if (tokens != max_tokens) return false;
  if (!rows || rows > max_rows || bias ||
      (reduction != 512 && reduction != 2048 && reduction != 4096))
    throw std::invalid_argument("Unsupported GB10 prefill projection shape");
  return true;
}
void gb10_prefill_projection_profile_begin() {
  if (!active) throw std::invalid_argument("Prefill projection profiling requires its owner");
  if (auto* profile = active->profile.get()) {
    if (profile->inflight) throw std::logic_error("Cannot reset an active prefill projection profile");
    profile->armed = true;
    profile->ordinal = 0;
  }
}
bool gb10_prefill_projection_wmma_enabled() { return active && active->wmma; }
Gb10PrefillLinearOutputScope::Gb10PrefillLinearOutputScope(std::size_t tokens, unsigned layer) {
  if (tokens != max_tokens) return;
  if (!active || layer >= 40 || layer % 4 == 3)
    throw std::invalid_argument("Linear OUT scope requires its owner and a linear layer");
  if (!active->linear_bound) return;
  if (active->linear_output || (active->profile && active->profile->inflight))
    throw std::logic_error("Linear OUT scope cannot nest or interrupt a projection");
  state_ = active;
  active->linear_output = true;
}
Gb10PrefillLinearOutputScope::~Gb10PrefillLinearOutputScope() {
  if (state_ && active == state_) active->linear_output = false;
}
void* gb10_prefill_projection_buffer(std::size_t tokens, std::size_t rows,
    std::size_t reduction, void* stream) {
  auto& s = bound(tokens, rows, reduction, stream);
  if (s.profile && s.profile->armed) s.profile->begin(static_cast<unsigned>(rows), static_cast<unsigned>(reduction));
  return s.raw.data;
}
void gb10_prefill_projection_fallback(const void* input, const void* weights,
    std::size_t tokens, std::size_t rows, std::size_t reduction,
    bool contiguous, void* stream) {
  auto& s = bound(tokens, rows, reduction, stream);
  if (!input || !weights) throw std::invalid_argument("Invalid prefill WMMA binding");
  if (s.profile && s.profile->armed) s.profile->fallback = true;
  hipLaunchKernelGGL(fallback_matmul, dim3((rows + 127u) / 128u, (tokens + 63u) / 64u),
      dim3(256), 0, nullptr, static_cast<const uint16_t*>(input),
      static_cast<const uint16_t*>(weights), s.raw.as<float>(),
      static_cast<unsigned>(tokens), static_cast<unsigned>(rows),
      static_cast<unsigned>(reduction), contiguous);
  check(hipGetLastError());
}
void gb10_prefill_projection_finish(const void* input, const void* weights,
    void* output, std::size_t tokens, std::size_t rows, std::size_t reduction,
    bool contiguous, void* stream) {
  auto& s = bound(tokens, rows, reduction, stream);
  if (!input || !weights || !output || input == output || weights == output || output == s.raw.data)
    throw std::invalid_argument("Invalid prefill projection input/output binding");
  const unsigned t = static_cast<unsigned>(tokens), n = static_cast<unsigned>(rows),
                 k = static_cast<unsigned>(reduction);
  Profile* profile = s.profile && s.profile->armed ? s.profile.get() : nullptr;
  if (profile) {
    if (!profile->inflight || profile->rows != n || profile->width != k)
      throw std::logic_error("Prefill projection profile producer/finish geometry differs");
    profile->mark(1);
  }
  const auto* x = static_cast<const uint16_t*>(input);
  const auto* w = static_cast<const uint16_t*>(weights);
  auto* y = static_cast<uint16_t*>(output);
  hipLaunchKernelGGL(prepare_operands, dim3((t * (k / 16u) + 255u) / 256u), dim3(256), 0,
      nullptr, x, s.inputs.as<half::Row>(), t, k, true);
  check(hipGetLastError());
  hipLaunchKernelGGL(prepare_operands, dim3((n * (k / 16u) + 255u) / 256u), dim3(256), 0,
      nullptr, w, s.weights.as<half::Row>(), n, k, contiguous);
  check(hipGetLastError());
  if (profile) profile->mark(2);
  hipLaunchKernelGGL(row_l2, dim3(t), dim3(256), 0, nullptr, x, s.input_l2.as<float>(), t, k, true);
  check(hipGetLastError());
  hipLaunchKernelGGL(row_l2, dim3(n), dim3(256), 0, nullptr, w, s.weight_l2.as<float>(), n, k, contiguous);
  check(hipGetLastError());
  if (profile) profile->mark(3);
  // The original linear and full-attention OUT selectors use different
  // admission bounds despite having the same N2048/K4096 geometry. An opt-in
  // scope identifies the actual linear call. Unscoped K4096 keeps 10000 ppb.
  // Complete GB10 continuation still qualifies every producer/selector pair.
  const unsigned ppb = k == 4096 && !s.linear_output ? 10000u : 1000u;
  unsigned window = 0;
  for (unsigned start = 0; start < t * n; start += window_capacity) {
    const unsigned size = std::min(window_capacity, t * n - start);
    if (profile) profile->mark(4 + 3 * window);
    check(hipMemsetAsync(s.count.data, 0, sizeof(unsigned), nullptr));
    hipLaunchKernelGGL(select_and_round, dim3((size + 255u) / 256u), dim3(256), 0,
        nullptr, s.raw.as<float>(), y, s.input_l2.as<float>(), s.weight_l2.as<float>(),
        n, start, size, ppb, s.count.as<unsigned>(), s.indices.as<unsigned>());
    check(hipGetLastError());
    if (profile) profile->mark(5 + 3 * window);
    hipLaunchKernelGGL(replay_selected, dim3(256), dim3(256), 0,
        nullptr, s.inputs.as<half::Row>(), s.weights.as<half::Row>(), y, n, k,
        s.count.as<unsigned>(), s.indices.as<unsigned>());
    check(hipGetLastError());
    if (profile) {
      profile->mark(6 + 3 * window);
      check(hipMemcpyAsync(profile->counts.as<unsigned>() + window, s.count.data,
          sizeof(unsigned), hipMemcpyDeviceToDevice, nullptr));
    }
    ++window;
  }
  if (profile) profile->finish(window, contiguous, ppb, s.linear_output);
}
}  // namespace aima_port
