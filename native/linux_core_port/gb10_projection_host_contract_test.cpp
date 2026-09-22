// SPDX-License-Identifier: Apache-2.0
#include "gb10_projection.hip.cpp"
#include <cassert>
#include <iostream>
using namespace aima_port;
template<class F> void reject(F fn) {
  bool rejected = false;
  try { fn(); } catch (const std::exception&) { rejected = true; }
  assert(rejected);
}
int main(int argc, char** argv) {
  assert(argc == 2);
  // Execute the actual elementwise GPU kernel using CPU coordinates. Distinct
  // live tokens exercise row/feature addressing, then decode changes the token.
  std::vector<uint16_t> x(3 * 2048), weight(2048), output(x.size() + 256, 0x1234);
  std::vector<float> inverse(vocabulary, 0.f);
  const uint32_t prompt[] = {7, 11, vocabulary - 1};
  inverse[7] = 2.f; inverse[11] = .5f; inverse[vocabulary - 1] = 4.f;
  for (unsigned i = 0; i < x.size(); ++i) x[i] = i % 2 ? 0xbf80 : 0x3f80;
  for (unsigned i = 0; i < weight.size(); ++i) weight[i] = i % 3 ? 0 : 0x3f80;
  blockDim = dim3(256);
  for (unsigned i = 0; i < output.size(); ++i) {
    blockIdx = dim3(i / 256); threadIdx = dim3(i % 256);
    embedding_norm_kernel(x.data(), weight.data(), output.data(), 3,
                          inverse.data(), prompt, vocabulary);
  }
  for (unsigned i = 0; i < x.size(); ++i) {
    const float expected = (i % 2 ? -1.f : 1.f) * inverse[prompt[i / 2048]] *
                           (i % 2048 % 3 ? 1.f : 2.f);
    assert(output[i] == qrt_sm121_q1::bf16(expected));
  }
  for (unsigned i = x.size(); i < output.size(); ++i) assert(output[i] == 0x1234);
  for (unsigned i = 0; i < 2048; ++i) {
    blockIdx = dim3(i / 256); threadIdx = dim3(i % 256);
    embedding_norm_kernel(x.data(), weight.data(), output.data(), 1,
                          inverse.data(), nullptr, 11);
    assert(output[i] == qrt_sm121_q1::bf16((i % 2 ? -.5f : .5f) * (i % 3 ? 1.f : 2.f)));
  }
  aima::Bf16WvSplitKProjection group[] = {
    {x.data(), output.data(), 8192}, {weight.data(), output.data() + 3, 4096},
    {x.data() + 1, output.data() + 7, 32}, {weight.data() + 1, output.data() + 9, 32}};
  ProjectionGroup bindings;
  for (const auto& p : group) append(bindings, p.weight_mk, p.output_1m, nullptr, p.m);
  assert(bindings.count == 4 && bindings.total == 12352);
  for (unsigned i = 0; i < 4; ++i) {
    assert(bindings.weights[i] == group[i].weight_mk && bindings.outputs[i] == group[i].output_1m);
    assert(bindings.rows[i] == group[i].m && bindings.bias[i] == nullptr);
  }
  fake_stream = reinterpret_cast<void*>(0x10);
  gb10_projection_group(group, 4, x.data(), 2048, fake_stream);
  gb10_projection(x.data(), weight.data(), weight.data(), output.data(), 2048, 4096, fake_stream);
  assert(fake_events == std::vector<std::string>({"projection_kernel", "projection_kernel"}));
  reject([&]{gb10_projection_group(group, 5, x.data(), 2048, fake_stream);});
  reject([&]{gb10_projection_group(nullptr, 2, x.data(), 2048, fake_stream);});
  reject([&]{gb10_projection(x.data(), weight.data(), nullptr, output.data(), 8, 2048, fake_stream);});
  reject([&]{gb10_projection(x.data(), weight.data(), nullptr, output.data(), 33, 2048, fake_stream);});
  reject([&]{gb10_projection(x.data(), weight.data(), nullptr, output.data(), 32, 1024, fake_stream);});
  reject([&]{gb10_projection(x.data(), nullptr, nullptr, output.data(), 32, 2048, fake_stream);});
  reject([&]{append(bindings, x.data(), output.data(), nullptr, 32);});
  ProjectionGroup overflow;
  append(overflow, x.data(), output.data(), nullptr, 2147483646);
  reject([&]{append(overflow, x.data(), output.data(), nullptr, 32);});
  reject([&]{set_gb10_decode_token(144);});
  State state; active = &state;
  reject([&]{gb10_embedding_norm(x.data(), weight.data(), output.data(), 1, fake_stream);});
  set_gb10_decode_token(144);
  assert(state.decode_token == 144);
  gb10_embedding_norm(x.data(), weight.data(), output.data(), 1, fake_stream);
  gb10_embedding_norm(x.data(), weight.data(), output.data(), 8192, fake_stream);
  reject([&]{set_gb10_decode_token(vocabulary);});
  reject([&]{gb10_embedding_norm(x.data(), weight.data(), output.data(), 8191, fake_stream);});
  reject([&]{Gb10ProjectionOwner duplicate(std::vector<uint32_t>(8192, 0));});
  active = nullptr;
  reject([&]{Gb10ProjectionOwner short_prompt(std::vector<uint32_t>(8191, 0));});
  reject([&]{Gb10ProjectionOwner invalid_token(std::vector<uint32_t>(8192, vocabulary));});
  // The production reader rejects both byte truncation and same-length swaps.
  const auto path = std::filesystem::u8path(argv[1]);
  { std::ofstream f(path, std::ios::binary); f << "bad"; }
  reject([&]{read_inverse(path);});
  { std::ofstream f(path, std::ios::binary); f.write(reinterpret_cast<const char*>(inverse.data()), inverse.size() * 4); }
  reject([&]{read_inverse(path);});
  std::cout << "{\"normalization_values_checked\":8192,\"guards_pass\":true,"
               "\"grouped_binding_pass\":true,\"invalid_projection_bindings_rejected\":8,"
               "\"invalid_owner_bindings_rejected\":7,\"artifact_faults_rejected\":2,"
               "\"gpu_projection_executed\":false}\n";
}
