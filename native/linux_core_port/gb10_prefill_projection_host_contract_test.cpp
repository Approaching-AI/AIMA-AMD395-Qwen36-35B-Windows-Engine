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

  // Coarse selection uses its own interval, including exceptional envelopes;
  // dead L2 buffers must not influence admission or overwrite guard cells.
  std::vector<float> errors(20, 0.f);
  for (unsigned i = 5; i <= 11; ++i) raw[i] = 1.f;
  raw[6] = qrt_sm121_exp2::value(0x3f808000);
  errors[8] = INFINITY; errors[9] = NAN; errors[10] = -1.f; errors[11] = 1.e-6f;
  std::fill(left.begin(), left.end(), NAN); std::fill(right.begin(), right.end(), NAN);
  output[4] = output[12] = 0x1234; count = 0;
  for (unsigned i = 0; i < 16; ++i) {
    blockIdx = dim3(0); threadIdx = dim3(i);
    select_and_round(raw.data(), output.data(), left.data(), right.data(), 128,
        5, 7, 0, &count, indices.data(), errors.data());
  }
  assert(count == 4 && indices[0] == 6 && indices[1] == 8 && indices[2] == 9 && indices[3] == 10);
  assert(output[4] == 0x1234 && output[12] == 0x1234 && output[11] == 0x3f80);

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
      blockIdx = dim3(i / 256u); threadIdx = dim3(i % 256u);
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
      blockIdx = dim3(0); threadIdx = dim3(i);
      select_routed<true>(values.data(), rounded.data(), il2.data(), wl2.data(), ids.data(),
          route_weights.data(), sorted.data(), experts.data(), &padded, 0, 3,
          &count, selected.data(), &invalid);
    }
    assert(count == 2 && selected[0] == 0 && selected[1] == 2 && !invalid);
    assert(rounded[0] == 0x3f80 && rounded[1] == 0x3f80 && rounded[2] == 0xbf80);
    auto one = [&] {
      count = 0; invalid = 0; rounded[0] = 0x1234;
      blockIdx = dim3(0); threadIdx = dim3(0);
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
  fake_negative_elapsed = true;
  assert(state.profile->ms(0, 1) < 0 && state.profile->invalid_intervals == 1);
  fake_negative_elapsed = false;
  state.profile.reset();
  assert(fake_live_profile_events == 0);
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
  state.routed = false;
  active = nullptr;
  reject([&]{gb10_prefill_projection_profile_begin();});
  reject([&]{Gb10PrefillLinearOutputScope missing_owner(8192, 0);});
  reject([&]{Gb10PrefillFullOutputScope missing_owner(8192, 3);});
  assert(!gb10_prefill_projection_wmma_enabled(4096));
  std::cout << "{\"lossless_operand_values\":2048,\"both_weight_layouts_match\":true,"
               "\"full_window_candidates\":1048576,\"window_guards_pass\":true,"
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
               "\"gpu_replay_executed\":false}\n";
}
