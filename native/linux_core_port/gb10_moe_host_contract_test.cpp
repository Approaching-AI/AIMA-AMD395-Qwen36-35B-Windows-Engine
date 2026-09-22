// SPDX-License-Identifier: Apache-2.0
#include "gb10_moe.hip.cpp"
#include <cassert>
#include <iostream>
namespace aima_port {
unsigned char roots[64];
uint16_t silu_values[65536]{}, sigmoid_values[65536]{};
uint32_t exponent_values[4]{};
bool tables_live = true;
const uint16_t* gb10_sigmoid_table() { if (!tables_live) throw std::runtime_error("Missing table"); return sigmoid_values; }
const uint16_t* gb10_moe_silu_table() { if (!tables_live) throw std::runtime_error("Missing table"); return silu_values; }
const uint32_t* gb10_moe_router_exp_table() { if (!tables_live) throw std::runtime_error("Missing table"); return exponent_values; }
const unsigned char* gb10_rsqrt_table() { return roots; }
void observe_gdn_prefill(std::size_t, const char*, const void*, std::size_t, std::size_t) {}
unsigned routed_projection_calls = 0;
void gb10_prefill_routed_projection(const void* input, const void* weights,
    const void* ids, const void* route_weights, const void* sorted,
    const void* experts, const void* padded, void* output, uint32_t* invalid, bool down) {
  assert(active && active->native_inflight && active->native_stage == (down ? 6u : 4u));
  assert(input == (down ? active->native_activation : active->native_input));
  assert(weights == (down ? active->down[active->next_layer] : active->gate_up[active->next_layer]));
  assert(ids == active->native_ids && route_weights == active->native_weights());
  assert(sorted && experts && padded && output && invalid == active->native_invalid());
  fake_events.push_back(down ? "routed_down_projection" : "routed_gate_up_projection");
  ++routed_projection_calls;
}
}
using namespace aima_port;
unsigned calls = 0, dynamic_calls = 0, registrations = 0, rejected = 0;
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
int dynamic_launch(const float* x, const float* r, const uint16_t* router, const uint16_t* gu,
    const uint16_t* dn, const uint16_t* sg, const uint16_t* gp, const uint16_t* up,
    const uint16_t* sd, float* out, uint32_t rows, void* stream) {
  const auto offset=elements-hidden;
  assert(active && active->next_layer==39 && rows==1 && !stream);
  assert(x==active->input()+offset && r==active->residual()+offset && out==active->output()+offset);
  assert(router && sg && gp && up && sd && gu==active->gate_up[39] && dn==active->down[39]);
  fake_events.push_back("dynamic_provider");++dynamic_calls;return returned;
}
template<class F> void reject(F fn) {
  bool failed = false;
  try { fn(); } catch (const std::exception&) { failed = true; }
  assert(failed); ++rejected;
}
int main(int argc, char** argv) {
  assert(argc == 2);
  for (const char* value : {"", "0", "1", "invalid"}) {
#ifdef _WIN32
    _putenv_s("AIMA_PORT_NATIVE_MOE_PREFILL",value);
#else
    setenv("AIMA_PORT_NATIVE_MOE_PREFILL",value,1);
#endif
    if (std::string(value)=="invalid") reject([]{native_setting();});
    else assert(native_setting()==(std::string(value)=="1"));
  }
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
  auto run = [&](unsigned layer, unsigned count = 8192, bool terminal = false) {
    gb10_prefill_moe(layer,x.data(),residual.data(),x.data(),gu[layer],dn[layer],
        x.data(),x.data(),x.data(),x.data(),output.data(),count,terminal);
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
  // A second request computes only the final MoE row while preserving the
  // same carrier base identity and exactly-once terminal norm handoff.
  x.resize(elements); residual.resize(elements); output.resize(elements);
  assert(!gb10_moe_input_norm(0,x.data(),x.data(),output.data(),8192));
  reject([&]{run(0,8192,true);});
  for(unsigned layer=0;layer<40;++layer) {
    if(layer)assert(gb10_moe_input_norm(layer,output.data(),x.data(),residual.data(),8192));
    if(layer==39) {
      reject([&]{run(layer,8192,true);});
      state.dynamic_launch=dynamic_launch;
    }
    fake_events.clear();moe_launches.clear();run(layer,8192,layer==39);
    if(layer==39) {
      assert(fake_events==std::vector<std::string>({"widen_moe_inputs","dynamic_provider","round_moe_carrier"}));
      const auto off=elements-hidden;
      assert(moe_launches.size()==2 && moe_launches[0].blocks==8 && moe_launches[1].blocks==8);
      assert(moe_launches[0].args==std::vector<std::uintptr_t>({moe_address(x.data()+off),moe_address(residual.data()+off),
          moe_address(state.input()+off),moe_address(state.residual()+off),hidden}));
      assert(moe_launches[1].args==std::vector<std::uintptr_t>({moe_address(state.output()+off),moe_address(output.data()+off),hidden}));
    }
  }
  assert(calls==79 && dynamic_calls==1 && state.pending_carrier==output.data());
  last=output.data()+elements-hidden;
  assert(gb10_moe_terminal_norm(last,x.data(),residual.data()));
  assert(!gb10_moe_terminal_norm(last,x.data(),residual.data()));
  assert(!gb10_moe_input_norm(0,x.data(),x.data(),output.data(),8192));
  // Native layers reuse only the two idle provider conversion slabs. Verify
  // the FP32 routing pointer, all launch boundaries and exactly-once carrier
  // handoff, followed by the unchanged terminal provider at layer39.
  state.native_prefill = true;
  reject([&]{gb10_native_moe_prefill_enabled(0,8191);});
  reject([&]{gb10_native_moe_prefill_enabled(40,8192);});
  assert(!gb10_native_moe_prefill_enabled(39,8192));
  reject([&]{run(0);});
  auto native = [&](unsigned layer) {
    return Gb10NativeMoeScope(layer,x.data(),residual.data(),gu[layer],dn[layer],output.data(),8192);
  };
  tables_live=false; reject([&]{auto missing=native(0);}); tables_live=true;
  reject([&]{auto wrong=native(1);});
  std::vector<uint16_t> temporary(8192);
  void *sg=temporary.data(), *gate=temporary.data()+512, *up=temporary.data()+1024,
       *act=temporary.data()+1536, *sd=temporary.data()+2048, *shared=temporary.data()+2560,
       *logits=temporary.data()+3072, *ids=temporary.data()+3584, *expert=temporary.data()+4096,
       *activated=temporary.data()+4608, *weighted=temporary.data()+5120,
       *routed=temporary.data()+5632, *combined=temporary.data()+6144;
  void *sorted=temporary.data()+6656, *experts=temporary.data()+7168, *padded=temporary.data()+7680;
  unsigned native_layers=0;
  for (unsigned layer=0;layer<39;++layer) {
    if(layer) assert(gb10_moe_input_norm(layer,output.data(),x.data(),residual.data(),8192));
    fake_events.clear(); moe_launches.clear();
    auto scope=native(layer); ++native_layers;
    assert(scope.router_weights()==state.native_weights() && state.native_inflight);
    reject([&]{auto duplicate=native(layer);});
    reject([&]{scope.finish(weighted,shared,routed,combined);});
    reject([&]{gb10_moe_input_norm(layer,output.data(),x.data(),residual.data(),8192);});
    gb10_native_moe_shared_gate(x.data(),x.data(),sg);
    gb10_native_moe_shared_activation(gate,up,act);
    gb10_native_moe_shared_scale(sg,sd,shared);
    gb10_native_moe_router(logits,ids);
    reject([&]{gb10_native_moe_expert_activation(expert,activated);});
    reject([&]{scope.project_experts(false,residual.data(),ids,sorted,experts,padded,expert);});
    reject([&]{scope.project_experts(false,x.data(),logits,sorted,experts,padded,expert);});
    scope.project_experts(false,x.data(),ids,sorted,experts,padded,expert);
    reject([&]{scope.project_experts(false,x.data(),ids,sorted,experts,padded,expert);});
    reject([&]{gb10_native_moe_expert_activation(gate,activated);});
    gb10_native_moe_expert_activation(expert,activated);
    reject([&]{scope.finish(weighted,shared,routed,combined);});
    scope.project_experts(true,activated,ids,sorted,experts,padded,weighted);
    reject([&]{scope.finish(expert,shared,routed,combined);});
    scope.finish(weighted,shared,routed,combined);
    assert(fake_events==std::vector<std::string>({"native_shared_gate","native_shared_activation",
        "native_shared_scale","native_router","routed_gate_up_projection","native_expert_activation",
        "routed_down_projection","native_combine"}));
    assert(moe_launches.size()==6 && moe_launches[3].args[3]==moe_address(state.native_weights()) &&
        moe_launches[3].args[4]==moe_address(state.native_invalid()));
    assert(state.pending && !state.native_inflight && state.pending_carrier==output.data());
    reject([&]{scope.router_weights();}); reject([&]{scope.finish(weighted,shared,routed,combined);});
    reject([&]{scope.project_experts(true,activated,ids,sorted,experts,padded,weighted);});
  }
  assert(gb10_moe_input_norm(39,output.data(),x.data(),residual.data(),8192));
  run(39,8192,true);
  assert(gb10_moe_terminal_norm(last,x.data(),residual.data()));
  assert(!gb10_moe_input_norm(0,x.data(),x.data(),output.data(),8192));
  {auto incomplete=native(0);}
  assert(state.poisoned && !state.native_inflight); reject([&]{auto poisoned=native(0);});
  // Explicit fixture reset isolates a device-error failure from abandonment.
  state.poisoned=false;
  {
    auto scope=native(0);
    gb10_native_moe_shared_gate(x.data(),x.data(),sg);
    gb10_native_moe_shared_activation(gate,up,act);
    gb10_native_moe_shared_scale(sg,sd,shared);
    gb10_native_moe_router(logits,ids);
    scope.project_experts(false,x.data(),ids,sorted,experts,padded,expert);
    gb10_native_moe_expert_activation(expert,activated);
    scope.project_experts(true,activated,ids,sorted,experts,padded,weighted);
    *state.native_invalid()=1;
    reject([&]{scope.finish(weighted,shared,routed,combined);});
    assert(!state.pending);
  }
  assert(state.poisoned && !state.native_inflight);
  state.poisoned=false;state.native_prefill=false;
  assert(!gb10_native_moe_prefill_enabled(0,8192));
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
      "\"ordered_layer_calls\":80,\"terminal_row_only_calls\":1,\"terminal_consumed_once\":true,"
      "\"registered_weights_drained\":true,\"failed_provider_poisoned\":true,"
      "\"complete_native_layers\":" << native_layers << ",\"native_abandonment_and_device_error_poisoned\":true,"
      "\"borrowed_routed_projection_calls\":" << routed_projection_calls << ","
      "\"invalid_bindings_and_artifacts_rejected\":" << rejected << ",\"gpu_reduction_executed\":false}\n";
}
