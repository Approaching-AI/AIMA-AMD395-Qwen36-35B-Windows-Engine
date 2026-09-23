// SPDX-License-Identifier: Apache-2.0
#include "gb10_prefill_projection.hip.cpp"
#include <cassert>
#include <iostream>
using namespace aima_port;
unsigned rejected_bindings = 0;
template<class F> void reject(F fn) {
  bool rejected = false;
  try { fn(); } catch (const std::exception&) { rejected = true; }
  assert(rejected);
  ++rejected_bindings;
}
int main() {
  constexpr unsigned rows = 32, width = 64, groups = rows * width / 16;
  std::vector<uint16_t> original(rows * width), transposed(rows * width);
  std::vector<half::Row> contiguous(groups + 1), gathered(groups + 1);
  std::vector<half::Row> group_major(groups + 1), group_major_gathered(groups + 1);
  for (unsigned i = 0; i < original.size(); ++i) original[i] = uint16_t(0x3f00u + i % 32u + (i / width) * 3u);
  original[3] = 1; original[48] = 0x8000; original[101] = 0;
  for (unsigned row = 0; row < rows; ++row)
    for (unsigned k = 0; k < width; ++k) transposed[k * rows + row] = original[row * width + k];
  contiguous.back().control = gathered.back().control = 0x12345678;
  group_major.back().control = group_major_gathered.back().control = 0x12345678;
  blockDim = dim3(256);
  for (unsigned i = 0; i < groups + 256; ++i) {
    blockIdx = dim3(i / 256, 0, 0); threadIdx = dim3(i % 256);
    prepare_operands(original.data(), contiguous.data(), rows, width, true);
    prepare_operands(transposed.data(), gathered.data(), rows, width, false);
    prepare_operands(original.data(), group_major.data(), rows, width, true, true);
    prepare_operands(transposed.data(), group_major_gathered.data(), rows, width, false, true);
  }
  for (unsigned i = 0; i < groups; ++i) {
    assert(memcmp(&contiguous[i], &gathered[i], sizeof(half::Row)) == 0);
    const auto slot = (i % (width / 16u)) * rows + i / (width / 16u);
    assert(memcmp(&contiguous[i], &group_major[slot], sizeof(half::Row)) == 0);
    assert(memcmp(&contiguous[i], &group_major_gathered[slot], sizeof(half::Row)) == 0);
    for (unsigned k = 0; k < 16; ++k) assert(half::original(contiguous[i], k) == original[i * 16 + k]);
  }
  assert(half::unit(contiguous[0]) == -32768);
  assert(contiguous.back().control == 0x12345678 && gathered.back().control == 0x12345678);
  assert(group_major.back().control == 0x12345678 && group_major_gathered.back().control == 0x12345678);
  for (unsigned row = 0; row < rows; ++row)
    for (unsigned group = 0; group < width / 16u; ++group)
      for (unsigned lane = 0; lane < 4; ++lane) {
        threadIdx = dim3(lane);
        const auto first = staged::load(contiguous[group], contiguous[row * (width / 16u) + group]);
        const auto second = staged::load(contiguous[group], group_major[group * rows + row]);
        assert(memcmp(&first, &second, sizeof(first)) == 0);
      }

  // Exercise actual classification/compaction, including a full bounded
  // window. The atomics are sequential host operations; replay is recorded.
  std::vector<float> raw(window_capacity, qrt_sm121_exp2::value(0x3f808000));
  std::vector<uint16_t> output(window_capacity + 256, 0x1234);
  std::vector<unsigned> indices(window_capacity + 256, 0xdeadbeef);
  std::vector<float> left(8192, 1.f), right(128, 1.f);
  unsigned count = 0;
  for (unsigned i = 0; i < window_capacity + 256; ++i) {
    blockIdx = dim3(i / 256, 0, 0); threadIdx = dim3(i % 256);
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
    blockIdx = dim3(0, 0, 0); threadIdx = dim3(i);
    select_and_round(raw.data(), output.data(), left.data(), right.data(), 128,
                     5, 8, 0, &count, indices.data());
  }
  assert(count == 5);
  const unsigned expected[] = {6, 7, 9, 10, 11};
  for (unsigned i = 0; i < count; ++i) assert(indices[i] == expected[i]);
  assert(output[12] == 0 && output[4] == 0x1234 && output[13] == 0x1234);
  // A value beyond the fixed radius is admitted by the independent bound.
  count = 0; blockIdx = dim3(0, 0, 0); threadIdx = dim3(0);
  select_and_round(raw.data(), output.data(), left.data(), right.data(), 128,
                   8, 1, 100000, &count, indices.data());
  assert(count == 1 && indices[0] == 8);

  // Coarse selection uses its own interval, including exceptional envelopes;
  // dead L2 buffers must not influence admission or overwrite guard cells.
  std::vector<float> errors(20, 0.f);
  for (unsigned i = 5; i <= 11; ++i) raw[i] = 1.f;
  raw[6] = qrt_sm121_exp2::value(0x3f808000);
  errors[8] = INFINITY; errors[9] = NAN; errors[10] = -1.f; errors[11] = 1.e-6f;
  std::fill(left.begin(), left.end(), NAN); std::fill(right.begin(), right.end(), NAN);
  output[4] = output[12] = 0x1234; count = 0;
  for (unsigned i = 0; i < 16; ++i) {
    blockIdx = dim3(0, 0, 0); threadIdx = dim3(i);
    select_and_round(raw.data(), output.data(), left.data(), right.data(), 128,
        5, 7, 0, &count, indices.data(), errors.data());
  }
  assert(count == 4 && indices[0] == 6 && indices[1] == 8 && indices[2] == 9 && indices[3] == 10);
  assert(output[4] == 0x1234 && output[12] == 0x1234 && output[11] == 0x3f80);

  // Execute the real 2D selector with an empty window, a worst-case full
  // window, and a partial tail. Each queue owns its own count and extent.
  {
    constexpr unsigned cells = 2u * window_capacity + 17u;
    std::vector<float> values(cells, 1.f), il2((cells + 127u) / 128u, 1.f), wl2(128, 1.f);
    std::vector<uint16_t> rounded(cells + 64u, 0x1234);
    std::vector<unsigned> selected(cells + 64u, 0xdeadbeef);
    std::array<unsigned, 4> counts{0, 0, 0, 0x12345678};
    for (unsigned i = window_capacity; i < 2u * window_capacity; ++i)
      values[i] = qrt_sm121_exp2::value(0x3f808000);
    for (unsigned i = 2u * window_capacity; i < cells; i += 2u)
      values[i] = qrt_sm121_exp2::value(0x3f808000);
    for (unsigned window = 0; window < 4; ++window)
      for (unsigned i = 0; i < window_capacity + 256u; ++i) {
        blockIdx = dim3(i / 256u, window, 0); threadIdx = dim3(i % 256u);
        select_and_round(values.data(), rounded.data(), il2.data(), wl2.data(), 128,
            0, cells, 0, counts.data(), selected.data());
      }
    assert((counts == std::array<unsigned, 4>{0, window_capacity, 9, 0x12345678}));
    for (unsigned i = 0; i < window_capacity; ++i) {
      assert(selected[i] == 0xdeadbeef);
      assert(selected[window_capacity + i] == window_capacity + i);
    }
    for (unsigned i = 0; i < 9; ++i)
      assert(selected[2u * window_capacity + i] == 2u * window_capacity + 2u * i);
    for (unsigned i = 2u * window_capacity + 9; i < selected.size(); ++i)
      assert(selected[i] == 0xdeadbeef);
    for (unsigned i = 0; i < cells; ++i) assert(rounded[i] == 0x3f80);
    for (unsigned i = cells; i < rounded.size(); ++i) assert(rounded[i] == 0x1234);

    // Only address routing is tested here: host shuffle stubs cannot establish
    // GPU dot correctness. A nonzero window must read its own queue/counter.
    counts = {0, 1, 0, 0x12345678}; selected[window_capacity] = 7;
    std::array<uint16_t, 33> batch_output, shifted_output;
    batch_output.fill(0x1234); shifted_output.fill(0x1234);
    gridDim = dim3(1); blockIdx = dim3(0, 1, 0); threadIdx = dim3(0);
    replay_selected(contiguous.data(), contiguous.data(), batch_output.data(), 32, 64,
        counts.data(), selected.data());
    blockIdx = dim3(0, 0, 0);
    replay_selected(contiguous.data(), contiguous.data(), shifted_output.data(), 32, 64,
        counts.data() + 1, selected.data() + window_capacity);
    assert(batch_output == shifted_output && batch_output[7] != 0x1234);
    for (unsigned i = 0; i < batch_output.size(); ++i)
      if (i != 7) assert(batch_output[i] == 0x1234);
    replay_selected(contiguous.data(), contiguous.data(), batch_output.data(), 32, 64,
        counts.data(), selected.data());
    assert(batch_output == shifted_output);
    std::array<uint16_t, 33> major_output; major_output.fill(0x1234);
    blockIdx = dim3(0, 1, 0);
    replay_selected_group_major(contiguous.data(), group_major.data(), major_output.data(), 32, 64,
        counts.data(), selected.data());
    assert(major_output == shifted_output);
  }

  // Real selector execution over a complete expert-ordered queue. Reverse
  // route order makes confusing a sorted row with a logical route observable.
  unsigned routed_candidates = 0;
  {
    constexpr unsigned cells = routed_window_capacity, route_count = cells / 1024u;
    std::vector<float> values(cells, qrt_sm121_exp2::value(0x3f808000));
    std::vector<uint16_t> rounded(cells + 256u, 0x1234);
    std::vector<unsigned> selected(cells + 256u, 0xdeadbeef);
    std::vector<int32_t> sorted(sorted_capacity, int32_t(routed_rows));
    std::vector<int32_t> ids(routed_rows, 0), experts(sorted_capacity / 32u, 0);
    std::vector<float> route_weights(routed_rows, 1.f), il2(routed_rows, 1.f), wl2(256u * 2048u, 1.f);
    for (unsigned i = 0; i < route_count; ++i) {
      sorted[i] = int32_t(route_count - 1u - i);
      ids[sorted[i]] = experts[i / 32u] = int32_t((i / 32u) % 256u);
    }
    int32_t padded = routed_rows;
    uint32_t invalid = 0;
    for (unsigned i = 0; i < cells + 256u; ++i) {
      blockIdx = dim3(i / 256u, 0, 0); threadIdx = dim3(i % 256u);
      select_routed<false>(values.data(), rounded.data(), il2.data(), wl2.data(), ids.data(),
          route_weights.data(), sorted.data(), experts.data(), &padded, 0, cells,
          &routed_candidates, selected.data(), &invalid);
    }
    assert(routed_candidates == cells && !invalid);
    for (unsigned i = 0; i < cells; ++i) {
      assert(selected[i] == unsigned(sorted[i / 1024u]) * 1024u + i % 1024u);
      assert(rounded[i] == 0x3f80 && qrt_sm121_exp2::bits(values[i]) == 0x3f808000);
    }
    for (unsigned i = cells; i < cells + 256u; ++i)
      assert(selected[i] == 0xdeadbeef && rounded[i] == 0x1234);
    // Down admission is applied after the FP32 routing multiply, before BF16.
    sorted[0] = 0; ids[0] = experts[0] = 17; route_weights[0] = .5f;
    values[0] = qrt_sm121_exp2::value(0x40008000);
    values[1] = 2.f; values[2] = qrt_sm121_exp2::value(0xc0008000);
    count = 0;
    for (unsigned i = 0; i < 3; ++i) {
      blockIdx = dim3(0, 0, 0); threadIdx = dim3(i);
      select_routed<true>(values.data(), rounded.data(), il2.data(), wl2.data(), ids.data(),
          route_weights.data(), sorted.data(), experts.data(), &padded, 0, 3,
          &count, selected.data(), &invalid);
    }
    assert(count == 2 && selected[0] == 0 && selected[1] == 2 && !invalid);
    assert(rounded[0] == 0x3f80 && rounded[1] == 0x3f80 && rounded[2] == 0xbf80);
    auto one = [&] {
      count = 0; invalid = 0; rounded[0] = 0x1234;
      blockIdx = dim3(0, 0, 0); threadIdx = dim3(0);
      select_routed<true>(values.data(), rounded.data(), il2.data(), wl2.data(), ids.data(),
          route_weights.data(), sorted.data(), experts.data(), &padded, 0, 1,
          &count, selected.data(), &invalid);
    };
    for (int32_t value : {-1, 256}) { ids[0] = value; one(); assert(invalid == 8 && !count); }
    ids[0] = 18; one(); assert(invalid == 8 && !count); ids[0] = 17;
    for (float value : {-1.f, 2.f, INFINITY, NAN}) {
      route_weights[0] = value; one(); assert(invalid == 32 && !count);
    }
    route_weights[0] = .5f;
    for (int32_t value : {-1, int32_t(routed_rows + 1)}) {
      sorted[0] = value; one(); assert(invalid == 16 && !count && rounded[0] == 0x1234);
    }
    sorted[0] = routed_rows; one(); assert(!invalid && !count && rounded[0] == 0x1234);
    sorted[0] = 0;
    for (int32_t value : {65535, 65537, 73473}) {
      padded = value; one(); assert(invalid == 4 && !count && rounded[0] == 0x1234);
    }
  }

  assert(!gb10_prefill_projection_shape(4096, 8192, 2048, false));
  assert(gb10_prefill_projection_shape(8192, 1, 2048, false));
  assert(gb10_prefill_projection_shape(8192, 12352, 4096, false));
  reject([&]{gb10_prefill_projection_shape(8192, 12353, 2048, false);});
  reject([&]{gb10_prefill_projection_shape(8192, 32, 1024, false);});
  reject([&]{gb10_prefill_projection_shape(8192, 32, 2048, true);});
  reject([&]{gb10_prefill_projection_buffer(8192, 32, 2048, nullptr);});
  State state;
  for (Device* d : {&state.raw, &state.inputs, &state.weights, &state.input_l2,
                   &state.weight_l2, &state.indices}) d->allocate(64);
  state.count.allocate(max_windows * sizeof(unsigned));
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
  fake_negative_elapsed = true;
  assert(state.profile->ms(0, 1) < 0 && state.profile->invalid_intervals == 1);
  fake_negative_elapsed = false;
  state.batch_replay = true;
  fake_events.clear(); fake_projection_launches.clear();
  gb10_prefill_projection_buffer(8192, 12352, 4096, nullptr);
  gb10_prefill_projection_finish(original.data(), transposed.data(), output.data(), 8192, 12352, 4096, true, nullptr);
  assert(fake_count_copies == 99 && fake_count_host_bytes == 97 * sizeof(unsigned));
  assert(fake_profile_serial == 309 && !state.profile->inflight && state.profile->ordinal == 3);
  assert(std::count(fake_events.begin(), fake_events.end(), "select_and_round") == 1);
  assert(std::count(fake_events.begin(), fake_events.end(), "replay_selected") == 1);
  assert(fake_projection_launches.size() == 6);
  assert(fake_projection_launches[4].name == "select_and_round");
  assert(fake_projection_launches[4].grid.x == 4096 && fake_projection_launches[4].grid.y == 97);
  assert(fake_projection_launches[5].name == "replay_selected");
  assert(fake_projection_launches[5].grid.x == 256 && fake_projection_launches[5].grid.y == 97);
  state.profile.reset();
  assert(fake_live_profile_events == 0);
  fake_events.clear();
  gb10_prefill_projection_finish(original.data(), transposed.data(), output.data(), 8192, 1, 2048, true, nullptr);
  assert(fake_events == std::vector<std::string>({"prepare_operands", "prepare_operands", "row_l2", "row_l2", "memset", "select_and_round", "replay_selected"}));
  assert(state.batched_projections == 2 && state.batched_windows == 98);
  state.group_major_weights = true;
  for (bool batch : {false, true}) {
    state.batch_replay = batch;
    fake_events.clear(); fake_projection_launches.clear();
    gb10_prefill_projection_finish(original.data(), transposed.data(), output.data(), 8192, 12352, 4096, true, nullptr);
    assert(std::count(fake_events.begin(), fake_events.end(), "replay_selected_group_major") == (batch ? 1 : 97));
    assert(std::count(fake_events.begin(), fake_events.end(), "replay_selected") == 0);
    for (const auto& launch : fake_projection_launches) if (launch.name == "replay_selected_group_major")
      assert(launch.grid.x == 256 && launch.grid.y == (batch ? 97 : 1));
  }
  state.group_major_weights = false;
  state.batch_replay = false;
  assert(!gb10_prefill_projection_wmma_enabled(4096));
  assert(!gb10_prefill_projection_tuned_gemm_enabled(8192,2048));
  state.tuned_gemm = true;
  assert(gb10_prefill_projection_tuned_gemm_enabled(8192,2048));
  assert(gb10_prefill_projection_tuned_gemm_enabled(4096,2048));
  assert(gb10_prefill_projection_tuned_gemm_enabled(2048,4096));
  assert(!gb10_prefill_projection_tuned_gemm_enabled(512,2048));
  assert(!gb10_prefill_projection_tuned_gemm_enabled(32,2048));
  assert(!gb10_prefill_projection_tuned_gemm_enabled(2048,512));
  unsigned char algorithm[24] = {19,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0};
  assert(gb10_prefill_gemm_algorithm_matches(algorithm,24,100100));
  assert(!gb10_prefill_gemm_algorithm_matches(nullptr,24,100100));
  assert(!gb10_prefill_gemm_algorithm_matches(algorithm,23,100100));
  assert(!gb10_prefill_gemm_algorithm_matches(algorithm,24,100101));
  for(unsigned i=0;i<24;++i) {
    algorithm[i]^=1;assert(!gb10_prefill_gemm_algorithm_matches(algorithm,24,100100));algorithm[i]^=1;
  }
  unsigned producer_controls = 0;
  for (unsigned flags = 0; flags < 16; ++flags) {
    state.tuned_gemm = flags & 1;
    state.tuned_gemm_input_only = flags & 2;
    state.wmma = flags & 4;
    state.wmma_output_only = flags & 8;
    // Admission is specified independently by the complete configuration set.
    const bool admitted = flags == 0 || flags == 1 || flags == 3 ||
        flags == 4 || flags == 8 || flags == 12 || flags == 15;
    if (!admitted) reject([&]{validate_producers(state);});
    else {
      validate_producers(state);
      if (state.tuned_gemm) {
        assert(gb10_prefill_projection_tuned_gemm_enabled(8192,2048));
        assert(gb10_prefill_projection_tuned_gemm_enabled(4096,2048));
        assert(gb10_prefill_projection_tuned_gemm_enabled(2048,4096) == (flags == 1));
        assert(!gb10_prefill_projection_tuned_gemm_enabled(512,2048));
        assert(!gb10_prefill_projection_tuned_gemm_enabled(32,2048));
        assert(!gb10_prefill_projection_wmma_enabled(2048));
        assert(gb10_prefill_projection_wmma_enabled(4096) == (flags == 15));
      }
    }
    ++producer_controls;
  }
  assert(producer_controls == 16);
  state.tuned_gemm = false;
  state.tuned_gemm_input_only = false;
  state.wmma = true;
  state.wmma_output_only = false;
  assert(gb10_prefill_projection_wmma_enabled(2048) && gb10_prefill_projection_wmma_enabled(4096));
  state.wmma_output_only = true;
  assert(!gb10_prefill_projection_wmma_enabled(2048) && gb10_prefill_projection_wmma_enabled(4096));
  { Gb10PrefillLinearOutputScope disabled(8192, 0); assert(!state.linear_output); }
  state.linear_bound = true;
  { Gb10PrefillLinearOutputScope short_shape(1024, 0); assert(!state.linear_output); }
  reject([&]{Gb10PrefillLinearOutputScope full_layer(8192, 3);});
  reject([&]{Gb10PrefillLinearOutputScope bad_layer(8192, 40);});
  unsigned scoped_layers = 0;
  for (unsigned layer = 0; layer < 40; ++layer) if (layer % 4 != 3) {
    { Gb10PrefillLinearOutputScope scope(8192, layer);
      assert(state.linear_output);
      assert(gb10_prefill_projection_buffer(8192, 2048, 4096, nullptr) == state.raw.data);
      reject([&]{Gb10PrefillLinearOutputScope nested(8192, layer);});
      reject([&]{gb10_prefill_projection_buffer(8192, 2048, 2048, nullptr);});
      reject([&]{gb10_prefill_projection_buffer(8192, 4096, 4096, nullptr);});
      ++scoped_layers;
    }
    assert(!state.linear_output);
  }
  try { Gb10PrefillLinearOutputScope scope(8192, 0); throw std::runtime_error("scope-unwind"); }
  catch (const std::runtime_error&) {}
  assert(!state.linear_output && scoped_layers == 30);
  { Gb10PrefillFullOutputScope disabled(8192, 3); assert(!state.full_output); }
  state.coarse_full = true;
  state.coarse_errors.allocate(64);
  assert(!gb10_prefill_projection_coarse_enabled());
  reject([&]{gb10_prefill_projection_coarse(original.data(), transposed.data(), 8192, 2048, 4096, true, nullptr);});
  reject([&]{Gb10PrefillFullOutputScope linear_layer(8192, 0);});
  reject([&]{Gb10PrefillFullOutputScope invalid_layer(8192, 43);});
  unsigned full_scopes = 0;
  for (unsigned layer = 3; layer < 40; layer += 4) {
    { Gb10PrefillFullOutputScope scope(8192, layer);
      assert(state.full_output && gb10_prefill_projection_coarse_enabled());
      reject([&]{Gb10PrefillFullOutputScope nested(8192, layer);});
      reject([&]{Gb10PrefillLinearOutputScope overlapping(8192, 0);});
      reject([&]{gb10_prefill_projection_buffer(8192, 512, 4096, nullptr);});
      gb10_prefill_projection_buffer(8192, 2048, 4096, nullptr);
      reject([&]{gb10_prefill_projection_finish(original.data(), transposed.data(), output.data(), 8192, 2048, 4096, true, nullptr);});
      reject([&]{gb10_prefill_projection_coarse(original.data(), transposed.data(), 8192, 2048, 4096, false, nullptr);});
      fake_events.clear();
      gb10_prefill_projection_coarse(original.data(), transposed.data(), 8192, 2048, 4096, true, nullptr);
      assert(fake_events.size() == 3 && fake_events[0] == "coarse::eligibility" && fake_events[1] == "coarse::eligibility");
      assert(fake_events[2].find("coarse::produce<64u, 1u, true, 19u, true>") != std::string::npos);
      reject([&]{gb10_prefill_projection_coarse(original.data(), transposed.data(), 8192, 2048, 4096, true, nullptr);});
      gb10_prefill_projection_finish(original.data(), transposed.data(), output.data(), 8192, 2048, 4096, true, nullptr);
      assert(!state.coarse_produced && std::find(fake_events.begin(), fake_events.end(), "row_l2") == fake_events.end());
      assert(std::count(fake_events.begin(), fake_events.end(), "replay_selected") == 16);
      ++full_scopes;
    }
    assert(!state.full_output && !gb10_prefill_projection_coarse_enabled());
  }
  try { Gb10PrefillFullOutputScope scope(8192, 3); throw std::runtime_error("scope-unwind"); }
  catch (const std::runtime_error&) {}
  assert(!state.full_output && full_scopes == 10);
  // Both complete routed shapes reuse the bounded queue and maintain their
  // own profiling ordinal, without contaminating dense projection counts.
  std::array<uint64_t, 36> operand_storage{};
  std::array<void*, 9> operands{};
  for (unsigned i = 0; i < operands.size(); ++i) operands[i] = &operand_storage[i * 4];
  auto routed_call = [&](bool down) {
    gb10_prefill_routed_projection(operands[0], operands[1], operands[2], operands[3],
        operands[4], operands[5], operands[6], operands[7], static_cast<uint32_t*>(operands[8]), down);
  };
  reject([&]{routed_call(false);});
  state.routed = true;
  for (unsigned i = 0; i < operands.size(); ++i) {
    auto saved = operands[i];
    operands[i] = nullptr; reject([&]{routed_call(false);});
    operands[i] = static_cast<char*>(saved) + 1; reject([&]{routed_call(false);});
    operands[i] = state.raw.data; reject([&]{routed_call(false);});
    operands[i] = operands[(i + 1) % operands.size()]; reject([&]{routed_call(false);});
    operands[i] = saved;
  }
  state.linear_output = true; reject([&]{routed_call(false);}); state.linear_output = false;
  state.full_output = true; reject([&]{routed_call(false);}); state.full_output = false;
  state.coarse_produced = true; reject([&]{routed_call(false);}); state.coarse_produced = false;
  state.profile = std::make_unique<Profile>();
  gb10_prefill_projection_profile_begin();
  state.profile->inflight = true; reject([&]{routed_call(false);}); state.profile->inflight = false;
  const auto previous_copies = fake_count_copies;
  for (bool down : {false, true}) {
    fake_events.clear(); routed_call(down);
    const auto windows = down ? 36 : 18;
    assert(std::count(fake_events.begin(), fake_events.end(), "(select_routed<Down>)") == windows);
    assert(std::count(fake_events.begin(), fake_events.end(), "(replay_routed<Down>)") == windows);
    assert(fake_count_host_bytes == windows * sizeof(unsigned));
    assert(!state.profile->inflight && state.profile->ordinal == 0 && state.profile->routed);
  }
  assert(state.profile->route_ordinal == 2 && fake_count_copies == previous_copies + 54);
  state.profile.reset(); assert(fake_live_profile_events == 0);


  unsigned routed_ordered_cases=0,routed_ordered_launches=0,routed_ordered_rejections=0;
  const auto before_routed_ordered=rejected_bindings;
  for(unsigned tile:{32u,64u,128u}){
    state.routed_ordered_batch=tile;load_routed_ordered(state);
    reject([&]{load_routed_ordered(state);});
    for(bool profiled:{false,true})for(bool down:{false,true}){
      if(profiled)state.profile=std::make_unique<Profile>();
      gb10_prefill_projection_profile_begin();
      fake_events.clear();fake_module_launches.clear();
      const auto windows=down?36u:18u;const auto& image=routed_ordered_image(tile,down);
      routed_call(down);
      assert(fake_module_launches.size()==windows);
      assert(std::count(fake_events.begin(),fake_events.end(),"prepare_operands")==0);
      assert(std::count(fake_events.begin(),fake_events.end(),"row_l2")==2);
      assert(std::count(fake_events.begin(),fake_events.end(),"(select_routed<Down>)")==windows);
      assert(std::count(fake_events.begin(),fake_events.end(),"(replay_routed<Down>)")==0);
      for(const auto& launch:fake_module_launches){
        auto address=[](const void* q){return reinterpret_cast<std::uintptr_t>(q);};
        assert(launch.name=="routed_replay_kernel"&&launch.shared==image.shared);
        assert(launch.pointers==std::vector<std::uintptr_t>({address(operands[0]),address(operands[1]),
          address(operands[2]),address(operands[3]),address(state.count.data),address(state.indices.data),
          address(operands[7]),0,address(operands[8])}));
        assert(launch.scalars==std::vector<std::int32_t>({65536}));
      }
      assert(state.routed_ordered_projections==1&&state.routed_ordered_launches==windows);
      if(profiled){assert(!state.profile->inflight&&state.profile->route_ordinal==1);state.profile.reset();}
      assert(fake_live_profile_events==0);++routed_ordered_cases;routed_ordered_launches+=windows;
    }
    auto direct=[&]{launch_routed_ordered(state,static_cast<const uint16_t*>(operands[0]),
      static_cast<const uint16_t*>(operands[1]),static_cast<const int32_t*>(operands[2]),
      static_cast<const float*>(operands[3]),static_cast<uint16_t*>(operands[7]),
      static_cast<uint32_t*>(operands[8]),false);};
    for(unsigned i:{0u,1u,2u,3u,7u,8u}){auto saved=operands[i];operands[i]=nullptr;reject(direct);operands[i]=saved;}
    for(Device* device:{&state.count,&state.indices}){auto saved=device->data;device->data=nullptr;reject(direct);device->data=saved;}
    fake_ordered_launch_error=1;reject(direct);fake_ordered_launch_error=0;
    state.routed=false;reject(direct);reject([&]{validate_producers(state);});state.routed=true;
    for(unsigned down:{0u,1u}){
      state.routed_ordered[down].reset();fake_events.clear();reject([&]{routed_call(false);});assert(fake_events.empty());
    }
  }
  state.routed_ordered_batch=0;reject([&]{load_routed_ordered(state);});
  routed_ordered_rejections=rejected_bindings-before_routed_ordered;
  assert(routed_ordered_cases==12&&routed_ordered_launches==324&&routed_ordered_rejections==43);
  state.routed = false;
  // The actual AOT loader and kernel ABI execute against recording HIP calls.
  // Every accepted shape is checked in both queue modes and weight layouts;
  // this establishes argument/ownership behavior, not GPU dot arithmetic.
  unsigned ordered_cases=0,ordered_launches=0,ordered_rejections=0;
  const unsigned before_ordered_rejections=rejected_bindings;
  assert(ordered_replay_batch_setting(nullptr)==0&&ordered_replay_batch_setting("0")==0);
  for(const char* v:{"","1","16","33","064","256","-1","true"})
    reject([&]{ordered_replay_batch_setting(v);});
  for(unsigned tile:{32u,64u,128u}){
    assert(ordered_replay_batch_setting(std::to_string(tile).c_str())==tile);
    state.ordered_batch=tile;load_ordered_replay(state);
    reject([&]{load_ordered_replay(state);});
    const auto& image=ordered_replay_image(tile);
    for(bool batched:{false,true})for(bool packed:{false,true})
      for(unsigned reduction:{512u,2048u,4096u})for(unsigned n:{1u,32u,2048u,12352u}){
        state.batch_replay=batched;gb10_prefill_projection_profile_begin();
        fake_events.clear();fake_module_launches.clear();fake_projection_launches.clear();
        gb10_prefill_projection_finish(original.data(),transposed.data(),output.data(),8192,n,reduction,packed,nullptr);
        const unsigned windows=(8192u*n+window_capacity-1u)/window_capacity;
        const unsigned launches=batched?1u:windows;
        assert(fake_module_launches.size()==launches);
        assert(std::count(fake_events.begin(),fake_events.end(),"prepare_operands")==0);
        assert(std::count(fake_events.begin(),fake_events.end(),"row_l2")==2);
        assert(std::count(fake_events.begin(),fake_events.end(),"select_and_round")==launches);
        assert(std::count(fake_events.begin(),fake_events.end(),"replay_selected")==0);
        for(const auto& launch:fake_module_launches){
          auto address=[](const void* q){return reinterpret_cast<std::uintptr_t>(q);};
          assert(launch.pointers==std::vector<std::uintptr_t>({address(original.data()),address(transposed.data()),
            address(state.count.data),address(state.indices.data),address(output.data()),0}));
          assert(launch.scalars==std::vector<std::int32_t>({int(n),int(reduction),packed?int(reduction):1,packed?1:int(n)}));
          assert(launch.y==(batched?windows:1u)&&launch.shared==image.shared);
        }
        assert(state.ordered_projections==1&&state.ordered_launches==launches);
        ++ordered_cases;ordered_launches+=launches;
      }

    state.batch_replay=true;
    state.profile=std::make_unique<Profile>();gb10_prefill_projection_profile_begin();
    const auto serial_before=fake_profile_serial,copies_before=fake_count_copies;
    fake_events.clear();fake_module_launches.clear();
    gb10_prefill_projection_buffer(8192,12352,4096,nullptr);
    gb10_prefill_projection_finish(original.data(),transposed.data(),output.data(),8192,12352,4096,true,nullptr);
    assert(fake_profile_serial==serial_before+7&&fake_count_copies==copies_before+1);
    assert(fake_count_host_bytes==97*sizeof(unsigned)&&!state.profile->inflight);
    assert(fake_module_launches.size()==1&&fake_module_launches[0].y==97);
    assert(std::count(fake_events.begin(),fake_events.end(),"prepare_operands")==0);
    state.profile.reset();assert(fake_live_profile_events==0);
    auto call=[&](unsigned n=32,unsigned k=2048,unsigned windows=1){
      launch_dense_replay(state,original.data(),transposed.data(),output.data(),n,k,windows,false);
    };
    for(unsigned n:{0u,12353u})reject([&]{call(n);});
    for(unsigned k:{0u,16u,1024u,4097u})reject([&]{call(32,k);});
    for(unsigned windows:{0u,2u,98u})reject([&]{call(32,2048,windows);});
    state.batch_replay=false;reject([&]{call(2048,2048,2);});state.batch_replay=true;
    for(Device* d:{&state.count,&state.indices}){
      auto saved=d->data;d->data=nullptr;reject([&]{call();});d->data=saved;
    }
    state.group_major_weights=true;reject([&]{call();});reject([&]{validate_producers(state);});state.group_major_weights=false;
    fake_ordered_launch_error=1;reject([&]{call();});fake_ordered_launch_error=0;
    state.ordered.reset();reject([&]{call();});
    fake_events.clear();reject([&]{gb10_prefill_projection_finish(original.data(),transposed.data(),output.data(),8192,32,2048,true,nullptr);});
    assert(fake_events.empty());
  }
  state.ordered_batch=0;reject([&]{load_ordered_replay(state);});
  ordered_rejections=rejected_bindings-before_ordered_rejections;
  assert(ordered_cases==144&&ordered_rejections==63);
  active=nullptr;
  // Constructors select one verified image and avoid both prepared dense
  // allocations. The RAII owner remains unique and unloads after GPU sync.
  for(const char* tile:{"32","64","128"}){
    setenv("AIMA_PORT_PREFILL_ORDERED_REPLAY",tile,1);
    {Gb10PrefillProjectionOwner owner;
      assert(active&&active->ordered&&!active->inputs.data&&!active->weights.data);
      reject([&]{Gb10PrefillProjectionOwner duplicate;});}
    assert(!active);
  }
  setenv("AIMA_PORT_PREFILL_GROUP_MAJOR_WEIGHTS","1",1);
  reject([&]{Gb10PrefillProjectionOwner conflicting;});assert(!active);
  unsetenv("AIMA_PORT_PREFILL_GROUP_MAJOR_WEIGHTS");unsetenv("AIMA_PORT_PREFILL_ORDERED_REPLAY");


  setenv("AIMA_PORT_ROUTED_ORDERED_REPLAY","64",1);
  reject([&]{Gb10PrefillProjectionOwner no_native_moe;});assert(!active);
  setenv("AIMA_PORT_NATIVE_MOE_PREFILL","1",1);
  for(const char* tile:{"32","64","128"}){
    setenv("AIMA_PORT_ROUTED_ORDERED_REPLAY",tile,1);
    setenv("AIMA_PORT_PREFILL_ORDERED_REPLAY","64",1);
    {Gb10PrefillProjectionOwner owner;
      assert(active&&active->ordered&&active->routed_ordered[0]&&active->routed_ordered[1]);
      assert(!active->inputs.data&&!active->weights.data);
      reject([&]{Gb10PrefillProjectionOwner duplicate;});}
    assert(!active);
  }
  unsetenv("AIMA_PORT_PREFILL_ORDERED_REPLAY");
  {Gb10PrefillProjectionOwner owner;
    assert(active&&active->inputs.data&&active->weights.data&&!active->ordered);
    assert(active->routed_ordered[0]&&active->routed_ordered[1]);}
  unsetenv("AIMA_PORT_NATIVE_MOE_PREFILL");unsetenv("AIMA_PORT_ROUTED_ORDERED_REPLAY");
  assert(!active);
  reject([&]{gb10_prefill_projection_profile_begin();});
  reject([&]{Gb10PrefillLinearOutputScope missing_owner(8192, 0);});
  reject([&]{Gb10PrefillFullOutputScope missing_owner(8192, 3);});
  assert(!gb10_prefill_projection_wmma_enabled(4096));
  std::cout << "{\"lossless_operand_values\":2048,\"both_weight_layouts_match\":true,"
               "\"full_window_candidates\":1048576,\"window_guards_pass\":true,"
               "\"batch_selector_empty_full_partial_guards_pass\":true,\"batch_queue_address_routing_pass\":true,"
               "\"batch_profile_max_windows\":97,\"batch_profile_events_per_projection\":7,"
               "\"batch_profile_counter_copies\":1,\"batch_unprofiled_dispatch_pass\":true,"
               "\"group_major_layout_values\":4096,\"group_major_guards_and_operand_lanes_pass\":true,"
               "\"group_major_replay_addressing_pass\":true,\"group_major_both_submission_modes_pass\":true,"
               "\"input_immutable\":true,\"selector_edges_pass\":true,"
               "\"queued_kernel_order_pass\":true,\"invalid_bindings_rejected\":" << rejected_bindings << ","
               "\"profile_max_windows\":97,\"profile_lifetime_pass\":true,"
               "\"profile_warmup_excluded\":true,\"profile_completed_events_only\":true,"
               "\"linear_output_scoped_layers\":30,\"linear_output_scope_unwind_pass\":true,"
               "\"full_output_scoped_layers\":10,\"full_output_scope_unwind_pass\":true,"
               "\"coarse_interval_exceptional_edges_pass\":true,\"invalid_event_time_reported\":true,"
               "\"producer_configuration_controls\":16,\"tuned_input_output_scopes_disjoint\":true,"
               "\"routed_full_window_candidates\":" << routed_candidates << ","
               "\"routed_sorted_scatter_guards_pass\":true,\"routed_weighted_endpoint_edges_pass\":true,"
               "\"routed_invalid_dispatch_rejected\":true,\"routed_profile_windows\":54,"
               "\"routed_ordered_cases\":" << routed_ordered_cases << ","
               "\"routed_ordered_aot_launches\":" << routed_ordered_launches << ","
               "\"routed_ordered_invalid_bindings\":" << routed_ordered_rejections << ","
               "\"routed_ordered_raw_bf16_abi_and_no_preparation_pass\":true,"
               "\"routed_ordered_scratch_and_image_ownership_pass\":true,"
               "\"ordered_projection_cases\":" << ordered_cases << ","
               "\"ordered_module_launches\":" << ordered_launches << ","
               "\"ordered_invalid_bindings_rejected\":" << ordered_rejections << ","
               "\"ordered_raw_bf16_abi_and_no_preparation_pass\":true,"
               "\"ordered_completed_profile_max_window_pass\":true,"
               "\"ordered_owner_images_and_scratch_lifetime_pass\":true,"
               "\"gpu_replay_executed\":false}\n";
}
