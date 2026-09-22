// SPDX-License-Identifier: Apache-2.0
#include "gb10_moe.hip.cpp"
#include <cassert>
#include <iostream>
namespace aima_port {
unsigned char roots[64];
const unsigned char* gb10_rsqrt_table() { return roots; }
void observe_gdn_prefill(std::size_t, const char*, const void*, std::size_t, std::size_t) {}
}
using namespace aima_port;
unsigned calls = 0, registrations = 0, rejected = 0;
int returned = 1;
const char* failure() { return "injected provider failure"; }
int registration(const uint16_t* const* gu, const uint16_t* const* dn, uint32_t n) {
  if (n) assert(gu && dn && n == 40);
  else assert(!gu && !dn);
  ++registrations; return 1;
}
int launch(const float* x, const float* r, const uint16_t* router, const uint16_t* gu,
    const uint16_t* dn, const uint16_t* sg, const uint16_t* gp, const uint16_t* up,
    const uint16_t* sd, float* out, void* stream) {
  assert(active && x == active->input() && r == active->residual() && out == active->output());
  assert(router && sg && gp && up && sd && !stream);
  assert(gu == active->gate_up[active->next_layer] && dn == active->down[active->next_layer]);
  fake_events.push_back("provider"); ++calls; return returned;
}
template<class F> void reject(F fn) {
  bool failed = false;
  try { fn(); } catch (const std::exception&) { failed = true; }
  assert(failed); ++rejected;
}
int main(int argc, char** argv) {
  assert(argc == 2);
  constexpr unsigned n = 3 * 2048;
  std::vector<uint16_t> x(n), residual(n), output(n + 256, 0x1234);
  std::vector<float> xf(n + 256, -111), rf(n + 256, -222), carrier(n);
  for (unsigned i = 0; i < n; ++i) {
    x[i] = uint16_t(0x3d00u + i % 0x300u);
    residual[i] = uint16_t(0xbe00u + i % 0x200u);
    carrier[i] = qrt_sm121_q1::add(qrt_sm121_q1::widen(x[i]), qrt_sm121_q1::widen(residual[i]));
  }
  const auto original_x = x, original_r = residual;
  blockDim = dim3(256);
  for (unsigned i = 0; i < n + 256; ++i) {
    blockIdx = dim3(i / 256); threadIdx = dim3(i % 256);
    widen_moe_inputs(x.data(), residual.data(), xf.data(), rf.data(), n);
    round_moe_carrier(carrier.data(), output.data(), n);
  }
  for (unsigned i = 0; i < n; ++i) {
    assert(qrt_sm121_exp2::bits(xf[i]) == uint32_t(x[i]) << 16);
    assert(qrt_sm121_exp2::bits(rf[i]) == uint32_t(residual[i]) << 16);
    assert(output[i] == qrt_sm121_q1::bf16(carrier[i]));
  }
  for (unsigned i = n; i < n + 256; ++i)
    assert(xf[i] == -111 && rf[i] == -222 && output[i] == 0x1234);
  assert(x == original_x && residual == original_r);
  for (unsigned row = 0; row < 3; ++row)
    for (unsigned lane = 0; lane < 256; ++lane)
      assert(qrt_sm121_exp2::bits(moe_carrier_lane_sumsq(carrier.data() + row * 2048, lane)) ==
          qrt_sm121_exp2::bits(qrt_sm121_mtp::residual_lane_sumsq(
              x.data() + row * 2048, residual.data() + row * 2048, lane)));
  float halfway[] = {qrt_sm121_exp2::value(0x3f808000), qrt_sm121_exp2::value(0x3f818000), qrt_sm121_exp2::value(0xbf818000)};
  uint16_t ties[4] = {0,0,0,0x1234};
  for (unsigned i = 0; i < 4; ++i) { blockIdx = dim3(0); threadIdx = dim3(i); round_moe_carrier(halfway,ties,3); }
  assert(ties[0] == 0x3f80 && ties[1] == 0x3f82 && ties[2] == 0xbf82 && ties[3] == 0x1234);

  State state;
  check(hipMalloc(reinterpret_cast<void**>(&state.storage), 3 * elements * sizeof(float)), "test allocation");
  state.launch = launch; state.registration = registration; state.error = failure;
  const uint16_t* gu[40]; const uint16_t* dn[40];
  for (unsigned i = 0; i < 40; ++i) {
    gu[i] = reinterpret_cast<const uint16_t*>(std::uintptr_t(0x100000 + i * 0x1000));
    dn[i] = reinterpret_cast<const uint16_t*>(std::uintptr_t(0x200000 + i * 0x1000));
  }
  auto run = [&](unsigned layer, unsigned count = 8192) {
    gb10_prefill_moe(layer,x.data(),residual.data(),x.data(),gu[layer],dn[layer],
        x.data(),x.data(),x.data(),x.data(),output.data(),count);
  };
  reject([&]{run(0);});
  active = &state;
  reject([&]{run(0);}); reject([&]{Gb10MoeOwner duplicate;});
  reject([&]{gb10_moe_register_weights(gu,dn,39);});
  const auto saved = gu[1]; gu[1] = gu[0];
  reject([&]{gb10_moe_register_weights(gu,dn,40);}); gu[1] = saved;
  gb10_moe_register_weights(gu,dn,40);
  reject([&]{gb10_moe_register_weights(gu,dn,40);});
  reject([&]{run(0,8191);}); reject([&]{run(1);});
  reject([&]{gb10_moe_input_norm(40,x.data(),x.data(),output.data(),8192);});
  assert(!gb10_moe_input_norm(0,x.data(),x.data(),output.data(),8192));
  for (unsigned layer = 0; layer < 40; ++layer) {
    if (layer) {
      reject([&]{gb10_moe_input_norm(layer,x.data(),x.data(),residual.data(),8192);});
      assert(gb10_moe_input_norm(layer,output.data(),x.data(),residual.data(),8192));
      reject([&]{gb10_moe_input_norm(layer,output.data(),x.data(),residual.data(),8192);});
    }
    fake_events.clear(); run(layer);
    assert(fake_events == std::vector<std::string>({"widen_moe_inputs","provider","round_moe_carrier"}));
    reject([&]{run(layer);});
  }
  const void* last = reinterpret_cast<const void*>(reinterpret_cast<std::uintptr_t>(output.data()) + (elements-hidden)*2);
  reject([&]{gb10_moe_terminal_norm(output.data(),x.data(),residual.data());});
  reject([&]{gb10_moe_terminal_norm(last,x.data(),residual.data(),reinterpret_cast<void*>(1));});
  assert(gb10_moe_terminal_norm(last,x.data(),residual.data()));
  assert(!gb10_moe_terminal_norm(last,x.data(),residual.data()));
  assert(calls == 40 && registrations == 1);
  assert(!gb10_moe_input_norm(0,x.data(),x.data(),output.data(),8192));
  returned = 0; fake_events.clear(); reject([&]{run(0);});
  assert(state.poisoned && fake_events == std::vector<std::string>({"widen_moe_inputs","provider"}));
  reject([&]{run(0);});
  gb10_moe_release_weights(); assert(!state.registered && registrations == 2);
  gb10_moe_release_weights(); assert(registrations == 2); active = nullptr;

  const std::filesystem::path file = argv[1];
  const Asset fixture{"fixture",3,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"};
  { std::ofstream f(file); f << "abc"; } verify(file,fixture);
  { std::ofstream f(file); f << "abd"; } reject([&]{verify(file,fixture);});
  { std::ofstream f(file); f << "ab"; } reject([&]{verify(file,fixture);});
  std::cout << "{\"conversion_values_checked\":18435,\"guard_elements_checked\":768,"
      "\"carrier_lane_sums_verified\":768,\"original_inputs_unchanged\":true,"
      "\"ordered_layer_calls\":40,\"terminal_consumed_once\":true,"
      "\"registered_weights_drained\":true,\"failed_provider_poisoned\":true,"
      "\"invalid_bindings_and_artifacts_rejected\":" << rejected << ",\"gpu_reduction_executed\":false}\n";
}
