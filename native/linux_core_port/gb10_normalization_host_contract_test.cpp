// SPDX-License-Identifier: Apache-2.0
#include "gb10_normalization.hip.cpp"
#include <algorithm>
#include <cassert>
#include <iostream>
namespace aima_port {
unsigned char fake_root_table[64];
bool fake_gdn_alive = true;
const unsigned char* gb10_rsqrt_table() {
  if (!fake_gdn_alive) throw std::runtime_error("No GDN owner");
  return fake_root_table;
}
}
using namespace aima_port;
unsigned rejected = 0;
template<class F> void reject(F fn) {
  bool failed = false;
  try { fn(); } catch (const std::exception&) { failed = true; }
  assert(failed); ++rejected;
}
void set_table(const char* path) {
#ifdef _WIN32
  _putenv_s("AIMA_PORT_GATED_SILU_TABLE", path);
#else
  setenv("AIMA_PORT_GATED_SILU_TABLE", path, 1);
#endif
}
int main(int argc, char** argv) {
  assert(argc == 3);
  uint16_t input[16]{}, residual[16]{}, weight[16]{}, output[16]{}, norm[16]{};
  auto gated = [&](std::size_t n) { gb10_gated_norm(input, residual, weight, output, n, fake_stream); };
  auto residual_norm = [&](std::size_t n) { gb10_residual_norm(input, residual, weight, output, norm, n, fake_stream); };
  reject([&]{gated(1);}); reject([&]{residual_norm(1);});
  reject([&]{gb10_preserve_decode_residual(input);});
  fake_gdn_alive = false;
  reject([&]{Gb10NormalizationOwner no_gdn;});
  fake_gdn_alive = true;
  set_table(""); reject([&]{Gb10NormalizationOwner missing;});
  set_table(argv[1]);
  {
    Gb10NormalizationOwner owner;
    assert(active && active->silu);
    std::vector<uint16_t> row(2048), original;
    for (unsigned i=0; i<2048; ++i) row[i] = uint16_t(i*29u);
    original = row;
    const auto* saved = static_cast<const uint16_t*>(gb10_preserve_decode_residual(row.data()));
    assert(saved != row.data() && std::equal(original.begin(),original.end(),saved));
    std::fill(row.begin(),row.end(),0);
    assert(std::equal(original.begin(),original.end(),saved) && fake_copies == 1);
    reject([&]{gb10_preserve_decode_residual(nullptr);});
    reject([&]{gb10_preserve_decode_residual(input,reinterpret_cast<void*>(1));});
    reject([&]{Gb10NormalizationOwner duplicate;});
    fake_stream = reinterpret_cast<void*>(0x100);
    gated(8192);
    const auto& g = launches.back();
    assert(g.name == "gated_kernel" && g.grid.x == 32768 && g.block.x == 256 && g.stream == fake_stream);
    assert(g.args == std::vector<std::uintptr_t>({value(input),value(residual),value(weight),
        value(output),value(fake_root_table),value(active->silu),8192u*32u}));
    for (const unsigned rows : {1u, 8192u}) {
      residual_norm(rows);
      const auto& r = launches.back();
      assert(r.name == "residual_normalize_rows" && r.grid.x == rows && r.block.x == 256 && r.stream == fake_stream);
      assert(r.args == std::vector<std::uintptr_t>({value(input),value(residual),value(weight),
          value(fake_root_table),value(norm),value(output)}));
    }
    reject([&]{gated(0);}); reject([&]{gated(8193);});
    reject([&]{residual_norm(0);}); reject([&]{residual_norm(8193);});
    reject([&]{gb10_gated_norm(nullptr,residual,weight,output,1);});
    reject([&]{gb10_gated_norm(input,nullptr,weight,output,1);});
    reject([&]{gb10_gated_norm(input,residual,nullptr,output,1);});
    reject([&]{gb10_gated_norm(input,residual,weight,nullptr,1);});
    reject([&]{gb10_residual_norm(nullptr,residual,weight,output,norm,1);});
    reject([&]{gb10_residual_norm(input,nullptr,weight,output,norm,1);});
    reject([&]{gb10_residual_norm(input,residual,nullptr,output,norm,1);});
    reject([&]{gb10_residual_norm(input,residual,weight,nullptr,norm,1);});
    reject([&]{gb10_residual_norm(input,residual,weight,output,nullptr,1);});
    reject([&]{gb10_residual_norm(input,residual,weight,output,output,1);});
    fake_gdn_alive = false;
    reject([&]{gated(1);}); reject([&]{residual_norm(1);});
    fake_gdn_alive = true;
    assert(launches.size() == 3);
  }
  assert(!active);
  reject([&]{gated(1);}); reject([&]{residual_norm(1);});
  const auto path = std::filesystem::u8path(argv[2]);
  { std::ofstream file(path, std::ios::binary); file << "bad"; }
  reject([&]{read_silu(path);});
  { std::vector<float> wrong(65536); std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(wrong.data()), wrong.size()*4); }
  reject([&]{read_silu(path);});
  std::cout << "{\"table_ownership_and_sha_verified\":true,\"borrowed_rsqrt_binding_verified\":true,"
      "\"dispatches_verified\":3,\"residual_alias_snapshot_verified\":true,\"invalid_bindings_and_artifacts_rejected\":" << rejected
      << ",\"gpu_reduction_executed\":false}\n";
}
