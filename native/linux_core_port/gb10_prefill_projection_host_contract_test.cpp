// SPDX-License-Identifier: Apache-2.0
#include "gb10_prefill_projection.hip.cpp"
#include <cassert>
#include <iostream>
using namespace aima_port;
template<class F> void reject(F fn) {
  bool rejected = false;
  try { fn(); } catch (const std::exception&) { rejected = true; }
  assert(rejected);
}
int main() {
  constexpr unsigned rows = 32, width = 64, groups = rows * width / 16;
  std::vector<uint16_t> original(rows * width), transposed(rows * width);
  std::vector<half::Row> contiguous(groups + 1), gathered(groups + 1);
  for (unsigned i = 0; i < original.size(); ++i) original[i] = uint16_t(0x3f00u + i % 32u);
  original[3] = 1; original[48] = 0x8000; original[101] = 0;
  for (unsigned row = 0; row < rows; ++row)
    for (unsigned k = 0; k < width; ++k) transposed[k * rows + row] = original[row * width + k];
  contiguous.back().control = gathered.back().control = 0x12345678;
  blockDim = dim3(256);
  for (unsigned i = 0; i < groups + 256; ++i) {
    blockIdx = dim3(i / 256); threadIdx = dim3(i % 256);
    prepare_operands(original.data(), contiguous.data(), rows, width, true);
    prepare_operands(transposed.data(), gathered.data(), rows, width, false);
  }
  for (unsigned i = 0; i < groups; ++i) {
    assert(memcmp(&contiguous[i], &gathered[i], sizeof(half::Row)) == 0);
    for (unsigned k = 0; k < 16; ++k) assert(half::original(contiguous[i], k) == original[i * 16 + k]);
  }
  assert(half::unit(contiguous[0]) == -32768);
  assert(contiguous.back().control == 0x12345678 && gathered.back().control == 0x12345678);

  // Exercise actual classification/compaction, including a full bounded
  // window. The atomics are sequential host operations; replay is recorded.
  std::vector<float> raw(window_capacity, qrt_sm121_exp2::value(0x3f808000));
  std::vector<uint16_t> output(window_capacity + 256, 0x1234);
  std::vector<unsigned> indices(window_capacity + 256, 0xdeadbeef);
  std::vector<float> left(8192, 1.f), right(128, 1.f);
  unsigned count = 0;
  for (unsigned i = 0; i < window_capacity + 256; ++i) {
    blockIdx = dim3(i / 256); threadIdx = dim3(i % 256);
    select_and_round(raw.data(), output.data(), left.data(), right.data(), 128,
                     0, window_capacity, 1000, &count, indices.data());
  }
  assert(count == window_capacity);
  for (unsigned i = 0; i < window_capacity; ++i) {
    assert(indices[i] == i && output[i] == 0x3f80);
    assert(qrt_sm121_exp2::bits(raw[i]) == 0x3f808000);
  }
  for (unsigned i = window_capacity; i < indices.size(); ++i) {
    assert(indices[i] == 0xdeadbeef && output[i] == 0x1234);
  }
  const uint32_t cases[] = {0x3f800000, 0x3f808000, 0x3f8081ff, 0x3f808201,
                           0xbf808000, 0x00800000, 0x3f807fff, 0x3f808000};
  for (unsigned i = 0; i < 8; ++i) raw[5 + i] = qrt_sm121_exp2::value(cases[i]);
  std::fill(output.begin(), output.begin() + 20, 0x1234);
  std::fill(indices.begin(), indices.begin() + 20, 0xdeadbeef);
  std::fill(left.begin(), left.end(), 1.f);
  std::fill(right.begin(), right.end(), 1.f);
  right[12] = 0.f; count = 0;
  // With an explicitly zero admission bound, only the fixed radius and tiny
  // endpoint predicate select; the zero row is exact without replay.
  for (unsigned i = 0; i < 16; ++i) {
    blockIdx = dim3(0); threadIdx = dim3(i);
    select_and_round(raw.data(), output.data(), left.data(), right.data(), 128,
                     5, 8, 0, &count, indices.data());
  }
  assert(count == 5);
  const unsigned expected[] = {6, 7, 9, 10, 11};
  for (unsigned i = 0; i < count; ++i) assert(indices[i] == expected[i]);
  assert(output[12] == 0 && output[4] == 0x1234 && output[13] == 0x1234);
  // A value beyond the fixed radius is admitted by the independent bound.
  count = 0; blockIdx = dim3(0); threadIdx = dim3(0);
  select_and_round(raw.data(), output.data(), left.data(), right.data(), 128,
                   8, 1, 100000, &count, indices.data());
  assert(count == 1 && indices[0] == 8);

  assert(!gb10_prefill_projection_shape(4096, 8192, 2048, false));
  assert(gb10_prefill_projection_shape(8192, 1, 2048, false));
  assert(gb10_prefill_projection_shape(8192, 12352, 4096, false));
  reject([&]{gb10_prefill_projection_shape(8192, 12353, 2048, false);});
  reject([&]{gb10_prefill_projection_shape(8192, 32, 1024, false);});
  reject([&]{gb10_prefill_projection_shape(8192, 32, 2048, true);});
  reject([&]{gb10_prefill_projection_buffer(8192, 32, 2048, nullptr);});
  State state;
  for (Device* d : {&state.raw, &state.inputs, &state.weights, &state.input_l2,
                   &state.weight_l2, &state.indices, &state.count}) d->allocate(64);
  active = &state;
  assert(gb10_prefill_projection_buffer(8192, 32, 2048, nullptr) == state.raw.data);
  reject([&]{gb10_prefill_projection_buffer(8192, 32, 2048, reinterpret_cast<void*>(1));});
  reject([&]{gb10_prefill_projection_buffer(4096, 32, 2048, nullptr);});
  reject([&]{Gb10PrefillProjectionOwner duplicate;});
  reject([&]{gb10_prefill_projection_finish(nullptr, original.data(), output.data(), 8192, 32, 2048, true, nullptr);});
  reject([&]{gb10_prefill_projection_finish(original.data(), original.data(), state.raw.data, 8192, 32, 2048, true, nullptr);});
  fake_events.clear();
  gb10_prefill_projection_fallback(original.data(), transposed.data(), 8192, 32, 2048, false, nullptr);
  gb10_prefill_projection_finish(original.data(), transposed.data(), output.data(), 8192, 32, 2048, false, nullptr);
  assert(fake_events == std::vector<std::string>({"fallback_matmul", "prepare_operands", "prepare_operands", "row_l2", "row_l2", "memset", "select_and_round", "replay_selected"}));

  // Optional event/count storage is bounded independently of model tensors.
  // Warmup is excluded, stale or nested measurement bindings are rejected,
  // and the maximum 97-window shape cannot overrun its event/count arrays.
  state.profile = std::make_unique<Profile>();
  assert(max_windows == 97 && fake_live_profile_events == 295);
  fake_events.clear();
  gb10_prefill_projection_buffer(8192, 32, 2048, nullptr);
  gb10_prefill_projection_finish(original.data(), transposed.data(), output.data(), 8192, 32, 2048, false, nullptr);
  assert(fake_profile_serial == 0 && fake_count_copies == 0);
  gb10_prefill_projection_profile_begin();
  reject([&]{gb10_prefill_projection_finish(original.data(), transposed.data(), output.data(), 8192, 32, 2048, false, nullptr);});
  gb10_prefill_projection_buffer(8192, 32, 2048, nullptr);
  reject([&]{gb10_prefill_projection_buffer(8192, 32, 2048, nullptr);});
  reject([&]{gb10_prefill_projection_profile_begin();});
  reject([&]{gb10_prefill_projection_finish(original.data(), transposed.data(), output.data(), 8192, 33, 2048, false, nullptr);});
  gb10_prefill_projection_fallback(original.data(), transposed.data(), 8192, 32, 2048, false, nullptr);
  gb10_prefill_projection_finish(original.data(), transposed.data(), output.data(), 8192, 32, 2048, false, nullptr);
  assert(fake_count_copies == 1 && fake_count_host_bytes == sizeof(unsigned));
  assert(fake_profile_serial == 7 && !state.profile->inflight && state.profile->ordinal == 1);
  gb10_prefill_projection_buffer(8192, 12352, 4096, nullptr);
  gb10_prefill_projection_finish(original.data(), transposed.data(), output.data(), 8192, 12352, 4096, true, nullptr);
  assert(fake_count_copies == 98 && fake_count_host_bytes == 97 * sizeof(unsigned));
  assert(fake_profile_serial == 302 && !state.profile->inflight && state.profile->ordinal == 2);
  state.profile.reset();
  assert(fake_live_profile_events == 0);
  active = nullptr;
  reject([&]{gb10_prefill_projection_profile_begin();});
  std::cout << "{\"lossless_operand_values\":2048,\"both_weight_layouts_match\":true,"
               "\"full_window_candidates\":1048576,\"window_guards_pass\":true,"
               "\"input_immutable\":true,\"selector_edges_pass\":true,"
               "\"queued_kernel_order_pass\":true,\"invalid_bindings_rejected\":14,"
               "\"profile_max_windows\":97,\"profile_lifetime_pass\":true,"
               "\"profile_warmup_excluded\":true,\"profile_completed_events_only\":true,"
               "\"gpu_replay_executed\":false}\n";
}
