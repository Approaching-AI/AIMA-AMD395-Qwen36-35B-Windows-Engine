// SPDX-License-Identifier: Apache-2.0
#include "gb10_prefill_projection.h"
#include "../providers/moe_accumulator/sm121_staged_half_projection.h"
#include "../providers/moe_accumulator/bf16_midpoint_selector.h"
#include "../providers/moe_accumulator/sm121_coarse_projection_matrix.h"
#include "../providers/gdn/sm121_q1_math.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace aima_port {
namespace {
namespace staged = qrt_sm121_staged_half_projection;
namespace half = qrt_sm121_scaled_half_products;
namespace coarse = qrt_sm121_coarse_projection_matrix;
constexpr unsigned max_tokens = 8192, max_rows = 12352, max_k = 4096;
constexpr unsigned routed_rows = 8192 * 8, routed_experts = 256, sorted_capacity = 73472;
constexpr unsigned window_capacity = 1u << 20;
constexpr unsigned routed_window_capacity = 4u << 20;
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
  bool armed = false, inflight = false, fallback = false, coarse_output = false;
  bool routed = false, weighted = false;
  unsigned ordinal = 0, rows = 0, width = 0, invalid_intervals = 0;
  unsigned route_ordinal = 0, row_count = max_tokens;
  unsigned selection_cells = 0, window_size = window_capacity;
  Profile() { counts.allocate(max_windows * sizeof(unsigned)); }
  void mark(unsigned index) { check(hipEventRecord(events.at(index).value, nullptr)); }
  float ms(unsigned first, unsigned last) {
    float elapsed = 0;
    check(hipEventElapsedTime(&elapsed, events.at(first).value, events.at(last).value));
    if (!std::isfinite(elapsed)) throw std::runtime_error("Nonfinite HIP projection elapsed time");
    if (elapsed < 0) ++invalid_intervals;
    return elapsed;
  }
  void begin(unsigned n, unsigned k, bool experts = false, bool down = false) {
    if (inflight) throw std::logic_error("Prefill projection profile already in flight");
    rows = n; width = k; fallback = false; coarse_output = false;
    routed = experts; weighted = down; row_count = experts ? routed_rows : max_tokens;
    selection_cells = (experts ? sorted_capacity : max_tokens) * n;
    window_size = experts ? routed_window_capacity : window_capacity;
    invalid_intervals = 0; inflight = true;
    mark(0);
  }
  void finish(unsigned windows, bool contiguous, unsigned ppb, bool linear_output,
      bool batched = false) {
    const unsigned final_event = batched ? 6u : 3u + 3u * windows;
    check(hipEventSynchronize(events.at(final_event).value));
    check(hipMemcpy(host_counts.data(), counts.data, windows * sizeof(unsigned), hipMemcpyDeviceToHost));
    double selection_ms = 0, replay_ms = 0;
    unsigned long long candidates = 0;
    for (unsigned i = 0; i < windows; ++i) {
      const unsigned capacity = std::min(window_size, selection_cells - i * window_size);
      if (host_counts[i] > capacity) throw std::runtime_error("Prefill projection profile count exceeds window");
      candidates += host_counts[i];
      if (!batched) {
        selection_ms += ms(4 + 3 * i, 5 + 3 * i);
        replay_ms += ms(5 + 3 * i, 6 + 3 * i);
      }
    }
    if (batched) { selection_ms = ms(4, 5); replay_ms = ms(5, 6); }
    const float producer_ms = ms(0, 1), operands_ms = ms(1, 2), norm_ms = ms(2, 3);
    const float total_ms = ms(0, final_event);
    std::fprintf(stderr, "{\"event\":\"%s\",\"ordinal\":%u,"
        "\"tokens\":%u,\"rows\":%u,\"reduction\":%u,\"weight_rows_contiguous\":%s,"
        "\"producer\":\"%s\",\"windows\":%u,\"cells\":%llu,\"candidates\":%llu,"
        "\"selection_cells_capacity\":%u,\"window_capacity\":%u,\"batched_replay\":%s,"
        "\"bound_ppb\":%u,\"linear_output\":%s,"
        "\"coarse_output\":%s,\"routed\":%s,\"weighted_output\":%s,\"invalid_elapsed_intervals\":%u,\"timing_valid\":%s,"
        "\"producer_ms\":%.6f,\"operands_ms\":%.6f,\"norm_bound_ms\":%.6f,"
        "\"selection_ms\":%.6f,\"replay_ms\":%.6f,\"total_gpu_ms\":%.6f,"
        "\"completed_gpu_events\":true,\"diagnostic_only\":true}\n",
        routed ? "routed_projection_profile" : "prefill_projection_profile",
        routed ? route_ordinal++ : ordinal++, row_count, rows, width, contiguous ? "true" : "false",
        routed ? "wmma-routed-k16" : (coarse_output ? "coarse-c64" : (fallback ? "wmma-fallback" :
            (contiguous && gb10_prefill_projection_tuned_gemm_enabled(rows, width)
                ? "hipblaslt-tuned-5651" : "hipblaslt"))), windows,
        static_cast<unsigned long long>(row_count) * rows, candidates,
        selection_cells, window_size, batched ? "true" : "false",
        ppb, linear_output ? "true" : "false",
        coarse_output ? "true" : "false", routed ? "true" : "false", weighted ? "true" : "false",
        invalid_intervals, invalid_intervals ? "false" : "true",
        producer_ms, operands_ms, norm_ms, selection_ms, replay_ms, total_ms);
    inflight = false;
  }
};
struct State {
  Device raw, inputs, weights, input_l2, weight_l2, indices, count, coarse_errors;
  std::unique_ptr<Profile> profile;
  bool wmma = false, wmma_output_only = false, linear_bound = false, linear_output = false;
  bool coarse_full = false, full_output = false, coarse_produced = false;
  bool tuned_gemm = false, tuned_gemm_input_only = false;
  bool routed = false, batch_replay = false, batch_armed = false;
  unsigned batched_projections = 0, batched_windows = 0;
};
State* active = nullptr;
bool enabled(const char* name) {
  const char* value = std::getenv(name);
  return value && std::strcmp(value, "1") == 0;
}
void validate_producers(const State& s) {
  if (s.tuned_gemm_input_only && !s.tuned_gemm)
    throw std::invalid_argument("Input-only tuned GEMM requires tuned GEMM selection");
  if (s.tuned_gemm && (s.wmma || s.wmma_output_only) &&
      !(s.tuned_gemm_input_only && s.wmma && s.wmma_output_only))
    throw std::invalid_argument("Tuned GEMM and WMMA require disjoint input/output scopes");
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
    unsigned* count, unsigned* indices, const float* errors = nullptr) {
  const unsigned offset = blockIdx.y * window_capacity;
  if (offset >= size) return;
  const unsigned remaining = size - offset;
  const unsigned capacity = remaining < window_capacity ? remaining : window_capacity;
  const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local >= capacity) return;
  count += blockIdx.y;
  indices += offset;
  const unsigned index = start + offset + local, row = index % rows, token = index / rows;
  if (errors) {
    const float value = raw[index];
    if (!coarse::bound::certified({value, errors[index]})) indices[atomicAdd(count, 1u)] = index;
    output[index] = qrt_sm121_q1::bf16(value);
    return;
  }
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
  count += blockIdx.y;
  indices += std::size_t(blockIdx.y) * window_capacity;
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

// Keep the expert accumulator in FP32 until admission and original SM121
// replay finish. The BF16-only imported AOT endpoint loses the midpoint
// information needed to distinguish the AMD and reference dot reductions.
template<bool Down>
static __global__ void routed_matmul(const uint16_t* input, const uint16_t* weights,
    const int32_t* sorted, const int32_t* experts, const int32_t* padded,
    float* output, uint32_t* invalid) {
  constexpr unsigned n = Down ? 2048u : 1024u, k = Down ? 512u : 2048u, stride = 66;
  const int32_t count = *padded;
  if (count < int32_t(routed_rows) || count > int32_t(sorted_capacity) || count % 32) {
    if (!threadIdx.x) atomicOr(invalid, 4u);
    return;
  }
  const unsigned route_base = blockIdx.y * 32u, row_base = blockIdx.x * 128u;
  if (route_base >= unsigned(count)) return;
  const int32_t expert = experts[blockIdx.y];
  if (expert < 0 || expert >= int32_t(routed_experts)) {
    if (!threadIdx.x) atomicOr(invalid, 8u);
    return;
  }
  __shared__ uint16_t weight_tile[128][stride], input_tile[32][stride];
  __shared__ unsigned routes[32];
  const unsigned thread = threadIdx.x, wave = thread / 32u, lane = thread % 32u;
  const unsigned source = lane % 16u, segment = lane / 16u;
  if (thread < 32u) {
    const int32_t route = sorted[route_base + thread];
    if (route < 0 || route > int32_t(routed_rows)) atomicOr(invalid, 16u);
    routes[thread] = route >= 0 && route < int32_t(routed_rows) ? unsigned(route) : routed_rows;
  }
  __syncthreads();
  WmmaF32 accumulators[2] = {};
  for (unsigned base = 0; base < k; base += 64u) {
    for (unsigned cell = thread; cell < 128u * 64u; cell += 256u)
      weight_tile[cell / 64u][cell % 64u] = weights[
          (std::size_t(expert) * n + row_base + cell / 64u) * k + base + cell % 64u];
    for (unsigned cell = thread; cell < 32u * 64u; cell += 256u) {
      const unsigned route = routes[cell / 64u];
      input_tile[cell / 64u][cell % 64u] = route < routed_rows
          ? input[std::size_t(Down ? route : route / 8u) * k + base + cell % 64u] : uint16_t(0);
    }
    __syncthreads();
    for (unsigned offset = 0; offset < 64u; offset += 16u) {
      WmmaBf16 weight_fragment;
      for (unsigned e = 0; e < 16; ++e) weight_fragment[e] = weight_tile[wave * 16u + source][offset + e];
      for (unsigned fragment = 0; fragment < 2; ++fragment) {
        WmmaBf16 input_fragment;
        for (unsigned e = 0; e < 16; ++e) input_fragment[e] = input_tile[fragment * 16u + source][offset + e];
        accumulators[fragment] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            input_fragment, weight_fragment, accumulators[fragment]);
      }
    }
    __syncthreads();
  }
  const unsigned row = row_base + wave * 16u + source;
  for (unsigned fragment = 0; fragment < 2; ++fragment)
    for (unsigned e = 0; e < 8; ++e) {
      const unsigned route = routes[fragment * 16u + 2u * e + segment];
      if (route < routed_rows) output[std::size_t(route) * n + row] = accumulators[fragment][e];
    }
}

// Scan in the dispatcher's existing expert order. Candidate replay therefore
// reuses adjacent expert weight rows without another expert sorting buffer.
// The queue is bounded by the number of scanned cells, including padding.
template<bool Down>
static __global__ void select_routed(const float* raw, uint16_t* output,
    const float* input_l2, const float* weight_l2, const int32_t* ids,
    const float* route_weights, const int32_t* sorted, const int32_t* experts,
    const int32_t* padded, unsigned start, unsigned size, unsigned* count,
    unsigned* indices, uint32_t* invalid) {
  constexpr unsigned n = Down ? 2048u : 1024u;
  const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local >= size) return;
  const unsigned sorted_row = (start + local) / n, column = (start + local) % n;
  const int32_t padded_rows = *padded;
  if (padded_rows < int32_t(routed_rows) || padded_rows > int32_t(sorted_capacity) || padded_rows % 32) {
    atomicOr(invalid, 4u); return;
  }
  if (sorted_row >= unsigned(padded_rows)) return;
  const int32_t route = sorted[sorted_row];
  if (route == int32_t(routed_rows)) return;
  if (route < 0 || route >= int32_t(routed_rows)) { atomicOr(invalid, 16u); return; }
  const int32_t expert = ids[route];
  const unsigned index = unsigned(route) * n + column;
  if (expert < 0 || expert >= int32_t(routed_experts) || expert != experts[sorted_row / 32u]) {
    output[index] = 0; atomicOr(invalid, 8u); return;
  }
  const float scale = Down ? route_weights[route] : 1.f;
  const float value = qrt_sm121_q1::multiply(raw[index], scale);
  if ((qrt_sm121_exp2::bits(value) & 0x7f800000u) == 0x7f800000u ||
      (qrt_sm121_exp2::bits(scale) & 0x7f800000u) == 0x7f800000u || scale < 0.f || scale > 1.f) {
    output[index] = 0; atomicOr(invalid, 32u); return;
  }
  const unsigned bits = qrt_sm121_exp2::bits(value), low = bits & 65535u;
  const unsigned distance = low >= 32768u ? low - 32768u : 32768u - low;
  const float error = input_l2[Down ? unsigned(route) : unsigned(route) / 8u] *
      weight_l2[unsigned(expert) * n + column] * 1.e-6f * scale;
  if (distance <= 512u || qrt_bf16_midpoint::within_error(value, error))
    indices[atomicAdd(count, 1u)] = index;
  output[index] = qrt_sm121_q1::bf16(value);
}

template<bool Down>
static __global__ void replay_routed(const half::Row* inputs, const half::Row* weights,
    const int32_t* ids, const float* route_weights, uint16_t* output,
    const unsigned* count, const unsigned* indices, uint32_t* invalid) {
  constexpr unsigned n = Down ? 2048u : 1024u, k = Down ? 512u : 2048u, groups = k / 16u;
  const unsigned lane = threadIdx.x & 3u, size = *count;
  if (size > routed_window_capacity) { if (!threadIdx.x) atomicOr(invalid, 64u); return; }
  for (unsigned candidate = blockIdx.x * 64u + threadIdx.x / 4u;
       candidate < size; candidate += gridDim.x * 64u) {
    const unsigned index = indices[candidate], route = index / n, column = index % n;
    if (route >= routed_rows) { if (!lane) atomicOr(invalid, 64u); continue; }
    const int32_t expert = ids[route];
    if (expert < 0 || expert >= int32_t(routed_experts)) { if (!lane) atomicOr(invalid, 8u); continue; }
    float value = staged::dot<2>(inputs + std::size_t(Down ? route : route / 8u) * groups,
        weights + (std::size_t(expert) * n + column) * groups, k);
    if (!lane) {
      if constexpr (Down) value = qrt_sm121_q1::multiply(value, route_weights[route]);
      if ((qrt_sm121_exp2::bits(value) & 0x7f800000u) == 0x7f800000u) atomicOr(invalid, 32u);
      output[index] = qrt_sm121_q1::bf16(value);
    }
  }
}

template<bool Down>
void launch_routed(State& s, const uint16_t* input, const uint16_t* weights,
    const int32_t* ids, const float* route_weights, const int32_t* sorted,
    const int32_t* experts, const int32_t* padded, uint16_t* output, uint32_t* invalid) {
  constexpr unsigned n = Down ? 2048u : 1024u, k = Down ? 512u : 2048u;
  constexpr unsigned input_rows = Down ? routed_rows : max_tokens, weight_rows = routed_experts * n;
  Profile* profile = s.profile && s.profile->armed ? s.profile.get() : nullptr;
  if (profile) profile->begin(n, k, true, Down);
  hipLaunchKernelGGL((routed_matmul<Down>), dim3(n / 128u, sorted_capacity / 32u), dim3(256), 0,
      nullptr, input, weights, sorted, experts, padded, s.raw.as<float>(), invalid);
  check(hipGetLastError());
  if (profile) profile->mark(1);
  hipLaunchKernelGGL(prepare_operands, dim3((input_rows * (k / 16u) + 255u) / 256u), dim3(256), 0,
      nullptr, input, s.inputs.as<half::Row>(), input_rows, k, true);
  check(hipGetLastError());
  hipLaunchKernelGGL(prepare_operands, dim3((weight_rows * (k / 16u) + 255u) / 256u), dim3(256), 0,
      nullptr, weights, s.weights.as<half::Row>(), weight_rows, k, true);
  check(hipGetLastError());
  if (profile) profile->mark(2);
  hipLaunchKernelGGL(row_l2, dim3(input_rows), dim3(256), 0, nullptr,
      input, s.input_l2.as<float>(), input_rows, k, true);
  check(hipGetLastError());
  hipLaunchKernelGGL(row_l2, dim3(weight_rows), dim3(256), 0, nullptr,
      weights, s.weight_l2.as<float>(), weight_rows, k, true);
  check(hipGetLastError());
  if (profile) profile->mark(3);
  unsigned window = 0;
  for (unsigned start = 0; start < sorted_capacity * n; start += routed_window_capacity) {
    const unsigned size = std::min(routed_window_capacity, sorted_capacity * n - start);
    if (profile) profile->mark(4 + 3 * window);
    check(hipMemsetAsync(s.count.data, 0, sizeof(unsigned), nullptr));
    hipLaunchKernelGGL((select_routed<Down>), dim3((size + 255u) / 256u), dim3(256), 0,
        nullptr, s.raw.as<float>(), output, s.input_l2.as<float>(), s.weight_l2.as<float>(),
        ids, route_weights, sorted, experts, padded, start, size, s.count.as<unsigned>(),
        s.indices.as<unsigned>(), invalid);
    check(hipGetLastError());
    if (profile) profile->mark(5 + 3 * window);
    hipLaunchKernelGGL((replay_routed<Down>), dim3(256), dim3(256), 0,
        nullptr, s.inputs.as<half::Row>(), s.weights.as<half::Row>(), ids, route_weights,
        output, s.count.as<unsigned>(), s.indices.as<unsigned>(), invalid);
    check(hipGetLastError());
    if (profile) {
      profile->mark(6 + 3 * window);
      check(hipMemcpyAsync(profile->counts.as<unsigned>() + window, s.count.data,
          sizeof(unsigned), hipMemcpyDeviceToDevice, nullptr));
    }
    ++window;
  }
  if (profile) profile->finish(window, true, 1000u, false);
}

State& bound(std::size_t tokens, std::size_t rows, std::size_t reduction, void* stream) {
  if (!active || stream || !gb10_prefill_projection_shape(tokens, rows, reduction, false))
    throw std::invalid_argument("Prefill projection requires its q8192 default-stream owner");
  if ((active->linear_output || active->full_output) && (rows != 2048 || reduction != 4096))
    throw std::invalid_argument("Scoped OUT selector requires N2048/K4096");
  return *active;
}
}  // namespace

struct Gb10PrefillProjectionOwner::Impl { State state; };
Gb10PrefillProjectionOwner::Gb10PrefillProjectionOwner() : impl_(std::make_unique<Impl>()) {
  if (active) throw std::runtime_error("A prefill projection owner is already active");
  auto& s = impl_->state;
  s.routed = enabled("AIMA_PORT_NATIVE_MOE_PREFILL");
  s.batch_replay = enabled("AIMA_PORT_PREFILL_BATCH_REPLAY");
  s.raw.allocate((s.routed ? std::size_t(routed_rows) * 2048u : std::size_t(max_tokens) * max_rows) * sizeof(float));
  s.inputs.allocate(std::size_t(max_tokens) * (max_k / 16u) * sizeof(half::Row));
  s.weights.allocate((s.routed ? std::size_t(routed_experts) * 1024u * (2048u / 16u) :
      std::size_t(max_rows) * (max_k / 16u)) * sizeof(half::Row));
  s.input_l2.allocate((s.routed ? routed_rows : max_tokens) * sizeof(float));
  s.weight_l2.allocate((s.routed ? routed_experts * 2048u : max_rows) * sizeof(float));
  const std::size_t queue_cells = s.batch_replay ? std::size_t(max_tokens) * max_rows :
      (s.routed ? routed_window_capacity : window_capacity);
  s.indices.allocate(queue_cells * sizeof(unsigned));
  s.count.allocate((s.batch_replay ? max_windows : 1u) * sizeof(unsigned));
  if (enabled("AIMA_PORT_PREFILL_PROJECTION_PROFILE")) s.profile = std::make_unique<Profile>();
  s.wmma = enabled("AIMA_PORT_PREFILL_WMMA");
  s.wmma_output_only = enabled("AIMA_PORT_PREFILL_WMMA_OUTPUT_ONLY");
  s.linear_bound = enabled("AIMA_PORT_PREFILL_LINEAR_BOUND");
  s.coarse_full = enabled("AIMA_PORT_PREFILL_FULL_COARSE");
  s.tuned_gemm = enabled("AIMA_PORT_PREFILL_GEMM_TUNED");
  s.tuned_gemm_input_only = enabled("AIMA_PORT_PREFILL_GEMM_TUNED_INPUT_ONLY");
  validate_producers(s);
  if (s.coarse_full) s.coarse_errors.allocate(std::size_t(max_tokens) * 2048u * sizeof(float));
  active = &s;
}
Gb10PrefillProjectionOwner::~Gb10PrefillProjectionOwner() {
  if (active == &impl_->state) {
    const auto status = hipDeviceSynchronize();
    const auto& s = impl_->state;
    if (s.batch_replay)
      std::fprintf(stderr, "{\"event\":\"prefill_batch_replay_summary\","
          "\"submitted_projections\":%u,\"windows\":%u,\"selector_grids\":%u,\"replay_grids\":%u,"
          "\"device_synchronized\":%s,\"warmup_excluded\":%s,\"diagnostic_only\":true}\n",
          s.batched_projections, s.batched_windows, s.batched_projections,
          s.batched_projections, status == hipSuccess ? "true" : "false",
          s.batch_armed ? "true" : "false");
    active = nullptr;
  }
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
    profile->route_ordinal = 0;
  }
  active->batch_armed = true;
  active->batched_projections = active->batched_windows = 0;
}
bool gb10_prefill_projection_wmma_enabled(std::size_t reduction) {
  return active && active->wmma && (!active->wmma_output_only || reduction == 4096);
}
bool gb10_prefill_projection_coarse_enabled() {
  return active && active->coarse_full && active->full_output;
}
bool gb10_prefill_projection_tuned_gemm_enabled(std::size_t rows, std::size_t reduction) {
  return active && active->tuned_gemm &&
      ((reduction == 2048 && (rows == 8192 || rows == 4096)) ||
       (!active->tuned_gemm_input_only && reduction == 4096 && rows == 2048));
}
bool gb10_prefill_gemm_algorithm_matches(const void* algorithm, std::size_t bytes, int version) {
  // Pinned installed 1.0.1 distribution, solution5651 and128MiB preference.
  // Reject a changed heuristic surface instead of silently choosing another.
  constexpr unsigned char expected[24] = {19,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0};
  return algorithm && version == 100100 && bytes == sizeof(expected) &&
      std::memcmp(algorithm, expected, sizeof(expected)) == 0;
}
Gb10PrefillLinearOutputScope::Gb10PrefillLinearOutputScope(std::size_t tokens, unsigned layer) {
  if (tokens != max_tokens) return;
  if (!active || layer >= 40 || layer % 4 == 3)
    throw std::invalid_argument("Linear OUT scope requires its owner and a linear layer");
  if (!active->linear_bound) return;
  if (active->linear_output || active->full_output || (active->profile && active->profile->inflight))
    throw std::logic_error("Linear OUT scope cannot nest or interrupt a projection");
  state_ = active;
  active->linear_output = true;
}
Gb10PrefillLinearOutputScope::~Gb10PrefillLinearOutputScope() {
  if (state_ && active == state_) active->linear_output = false;
}
Gb10PrefillFullOutputScope::Gb10PrefillFullOutputScope(std::size_t tokens, unsigned layer) {
  if (tokens != max_tokens) return;
  if (!active || layer >= 40 || layer % 4 != 3)
    throw std::invalid_argument("Full OUT scope requires its owner and a full-attention layer");
  if (!active->coarse_full) return;
  if (active->linear_output || active->full_output || (active->profile && active->profile->inflight))
    throw std::logic_error("Full OUT scope cannot nest or interrupt a projection");
  state_ = active;
  active->full_output = true;
}
Gb10PrefillFullOutputScope::~Gb10PrefillFullOutputScope() {
  if (state_ && active == state_) active->full_output = false;
}
void* gb10_prefill_projection_buffer(std::size_t tokens, std::size_t rows,
    std::size_t reduction, void* stream) {
  auto& s = bound(tokens, rows, reduction, stream);
  s.coarse_produced = false;
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
void gb10_prefill_projection_coarse(const void* input, const void* weights,
    std::size_t tokens, std::size_t rows, std::size_t reduction,
    bool contiguous, void* stream) {
  auto& s = bound(tokens, rows, reduction, stream);
  if (!input || !weights || !contiguous || !gb10_prefill_projection_coarse_enabled() ||
      !s.coarse_errors.data || s.coarse_produced)
    throw std::invalid_argument("Coarse OUT producer requires its live contiguous full-attention scope");
  if (s.profile && s.profile->armed) s.profile->coarse_output = true;
  // These two norm buffers are dead on the coarse route; reuse them as row
  // eligibility flags. Unsupported rows retain infinite error and full replay.
  hipLaunchKernelGGL(coarse::eligibility, dim3(2048), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(weights), s.weight_l2.as<unsigned>(), 2048u, 4096u);
  check(hipGetLastError());
  hipLaunchKernelGGL(coarse::eligibility, dim3(max_tokens), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(input), s.input_l2.as<unsigned>(), max_tokens, 4096u);
  check(hipGetLastError());
  hipLaunchKernelGGL((coarse::produce<64u, 1u, true, 19u, true>), dim3(16, 512), dim3(256), 0, nullptr,
      static_cast<const uint16_t*>(weights), static_cast<const uint16_t*>(input),
      s.weight_l2.as<unsigned>(), s.input_l2.as<unsigned>(), s.raw.as<float>(),
      s.coarse_errors.as<float>(), 2048u, max_tokens, 4096u);
  check(hipGetLastError());
  s.coarse_produced = true;
}
void gb10_prefill_projection_finish(const void* input, const void* weights,
    void* output, std::size_t tokens, std::size_t rows, std::size_t reduction,
    bool contiguous, void* stream) {
  auto& s = bound(tokens, rows, reduction, stream);
  if (!input || !weights || !output || input == output || weights == output || output == s.raw.data)
    throw std::invalid_argument("Invalid prefill projection input/output binding");
  const unsigned t = static_cast<unsigned>(tokens), n = static_cast<unsigned>(rows),
                 k = static_cast<unsigned>(reduction);
  const bool coarse_output = gb10_prefill_projection_coarse_enabled();
  if (coarse_output && !s.coarse_produced)
    throw std::logic_error("Coarse OUT finish has no completed producer dispatch");
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
  if (!coarse_output) {
    hipLaunchKernelGGL(row_l2, dim3(t), dim3(256), 0, nullptr, x, s.input_l2.as<float>(), t, k, true);
    check(hipGetLastError());
    hipLaunchKernelGGL(row_l2, dim3(n), dim3(256), 0, nullptr, w, s.weight_l2.as<float>(), n, k, contiguous);
    check(hipGetLastError());
  }
  if (profile) profile->mark(3);
  // The original linear and full-attention OUT selectors use different
  // admission bounds despite having the same N2048/K4096 geometry. An opt-in
  // scope identifies the actual linear call. Unscoped K4096 keeps 10000 ppb.
  // Complete GB10 continuation still qualifies every producer/selector pair.
  const unsigned ppb = coarse_output ? 0u : (k == 4096 && !s.linear_output ? 10000u : 1000u);
  if (s.batch_replay) {
    // Every window owns its counter and its full worst-case queue extent.
    // One grid selects all windows, then one grid replays independent queues.
    // Predicate, original K16 carry order and BF16 endpoints stay unchanged.
    const unsigned cells = t * n;
    const unsigned windows = (cells + window_capacity - 1u) / window_capacity;
    if (profile) profile->mark(4);
    check(hipMemsetAsync(s.count.data, 0, windows * sizeof(unsigned), nullptr));
    hipLaunchKernelGGL(select_and_round, dim3((std::min(cells, window_capacity) + 255u) / 256u, windows),
        dim3(256), 0, nullptr, s.raw.as<float>(), y, s.input_l2.as<float>(), s.weight_l2.as<float>(),
        n, 0u, cells, ppb, s.count.as<unsigned>(), s.indices.as<unsigned>(),
        coarse_output ? s.coarse_errors.as<float>() : nullptr);
    check(hipGetLastError());
    if (profile) profile->mark(5);
    hipLaunchKernelGGL(replay_selected, dim3(256, windows), dim3(256), 0,
        nullptr, s.inputs.as<half::Row>(), s.weights.as<half::Row>(), y, n, k,
        s.count.as<unsigned>(), s.indices.as<unsigned>());
    check(hipGetLastError());
    if (profile) {
      profile->mark(6);
      check(hipMemcpyAsync(profile->counts.data, s.count.data,
          windows * sizeof(unsigned), hipMemcpyDeviceToDevice, nullptr));
      profile->finish(windows, contiguous, ppb, s.linear_output, true);
    }
    if (s.batch_armed) {
      ++s.batched_projections;
      s.batched_windows += windows;
    }
    s.coarse_produced = false;
    return;
  }
  unsigned window = 0;
  for (unsigned start = 0; start < t * n; start += window_capacity) {
    const unsigned size = std::min(window_capacity, t * n - start);
    if (profile) profile->mark(4 + 3 * window);
    check(hipMemsetAsync(s.count.data, 0, sizeof(unsigned), nullptr));
    hipLaunchKernelGGL(select_and_round, dim3((size + 255u) / 256u), dim3(256), 0,
        nullptr, s.raw.as<float>(), y, s.input_l2.as<float>(), s.weight_l2.as<float>(),
        n, start, size, ppb, s.count.as<unsigned>(), s.indices.as<unsigned>(),
        coarse_output ? s.coarse_errors.as<float>() : nullptr);
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
  s.coarse_produced = false;
}
void gb10_prefill_routed_projection(const void* input, const void* weights,
    const void* ids, const void* route_weights, const void* sorted_routes,
    const void* block_experts, const void* padded_count, void* output,
    uint32_t* invalid, bool down) {
  if (!active || !active->routed || active->linear_output || active->full_output ||
      active->coarse_produced || (active->profile && active->profile->inflight))
    throw std::logic_error("Routed projection requires an idle native MoE projection owner");
  const void* pointers[] = {input, weights, ids, route_weights, sorted_routes,
                           block_experts, padded_count, output, invalid};
  for (unsigned i = 0; i < 9; ++i) {
    const unsigned alignment = i < 2 || i == 7 ? 2u : 4u;
    if (!pointers[i] || reinterpret_cast<std::uintptr_t>(pointers[i]) % alignment)
      throw std::invalid_argument("Routed projection operand is null or misaligned");
    for (unsigned j = 0; j < i; ++j)
      if (pointers[i] == pointers[j]) throw std::invalid_argument("Routed projection operands alias");
    for (Device* d : {&active->raw, &active->inputs, &active->weights,
                     &active->input_l2, &active->weight_l2, &active->indices, &active->count})
      if (pointers[i] == d->data) throw std::invalid_argument("Routed projection operand aliases scratch");
  }
  const auto call = down ? launch_routed<true> : launch_routed<false>;
  call(*active, static_cast<const uint16_t*>(input), static_cast<const uint16_t*>(weights),
       static_cast<const int32_t*>(ids), static_cast<const float*>(route_weights),
       static_cast<const int32_t*>(sorted_routes), static_cast<const int32_t*>(block_experts),
       static_cast<const int32_t*>(padded_count), static_cast<uint16_t*>(output), invalid);
}
}  // namespace aima_port
