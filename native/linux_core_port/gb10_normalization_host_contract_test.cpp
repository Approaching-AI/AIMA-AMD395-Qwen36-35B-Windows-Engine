// SPDX-License-Identifier: Apache-2.0
#include "gb10_normalization.hip.cpp"
#include <algorithm>
#include <array>
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
void set_rope_table(const char* path) {
#ifdef _WIN32
  _putenv_s("AIMA_PORT_FULL_ATTENTION_ROPE_TABLE", path);
#else
  setenv("AIMA_PORT_FULL_ATTENTION_ROPE_TABLE", path, 1);
#endif
}
void check_original_prefill_midpoint(const float* silu, const char* rsqrt_path) {
  // Original qualified q8192 prefill, layer0/position57/head24. The complete
  // input core SHA256 is 30dbf01ee37d2f5c98e72c470e96c41892b32d01a7f048ea950e186898c12834.
  // This immutable fixture belongs only to the host test, never the runtime.
  const uint16_t core[128] = {
    0x398d,0x37e5,0xb88e,0xb871,0xb8d8,0x3a33,0xb84d,0x3a3a,
    0xb8ab,0x3906,0x345b,0xb614,0xb855,0xb8b1,0xb8cb,0x380c,
    0x39bf,0x3a1c,0x3913,0x37f7,0x3926,0x38c1,0x38a2,0xb89e,
    0x37a7,0x38f8,0xb827,0xb806,0xb84c,0xb8eb,0xb891,0x38f7,
    0xb7eb,0xb877,0xb7b6,0xb8ce,0xb839,0x3764,0x383d,0x374e,
    0xb899,0xb80b,0xb793,0x3899,0xb885,0x3846,0xb891,0x3ea5,
    0x387b,0xb985,0xb8fb,0x3960,0xba92,0xb888,0xb70a,0xb8b1,
    0xb7cc,0x3a50,0xb86a,0xb8d7,0xb83b,0xb91b,0xb87b,0xb887,
    0x3a46,0xba10,0xb853,0x379b,0xb900,0xb731,0x394d,0xb925,
    0x37e6,0x3910,0xb7b8,0xb83d,0x38ed,0xb70a,0xb8dc,0xbaff,
    0x387b,0x388b,0xb8bc,0xb706,0x3758,0xb91e,0x38ba,0x390c,
    0x38fe,0xb815,0x3804,0x3899,0x3795,0x36d8,0x37dd,0xb9c7,
    0xb8aa,0xb8a1,0x3904,0xb771,0x3818,0xb8d7,0xb7b9,0xb868,
    0xb8f1,0xb8a7,0x393e,0xb791,0x3841,0x3989,0xb928,0x38e0,
    0xb784,0x36b8,0x3893,0xb820,0x39f3,0xb813,0x36ec,0xb84c,
    0xb893,0x37f9,0x384c,0xb73d,0x37fc,0xb916,0xb78a,0xb86d};
  std::vector<unsigned char> rsqrt(qrt_sm121_rsqrt::table_bytes);
  std::ifstream f(rsqrt_path,std::ios::binary);
  f.read(reinterpret_cast<char*>(rsqrt.data()),rsqrt.size());
  assert(f && f.peek()==std::char_traits<char>::eof() && qrt_sm121_rsqrt::valid_layout(rsqrt.data(),rsqrt.size()));
  assert(aima::sha256_bytes(rsqrt.data(),rsqrt.size()) ==
      "ca0230a8bae9bd101ac368f8a7c34007cda637df6513dbe4714253c36b940850");
  std::array<float,16> prefill;
  float short_row[32];
  for(unsigned lane=0;lane<16;++lane)prefill[lane]=gated_prefill_lane_sum(core,lane);
  for(unsigned lane=0;lane<32;++lane)short_row[lane]=qrt_sm121_q2::gated_lane_sum(core,lane);
  for(unsigned mask=8;mask;mask>>=1) {
    const auto before=prefill;
    for(unsigned lane=0;lane<16;++lane)prefill[lane]=qrt_sm121_q1::add(before[lane],before[lane^mask]);
  }
  for(unsigned mask=16;mask;mask>>=1)
    for(unsigned lane=0;lane<mask;++lane)short_row[lane]=qrt_sm121_q1::add(short_row[lane],short_row[lane+mask]);
  for(float sum:prefill) {
    const auto inverse=qrt_sm121_q2::gated_inverse(sum,rsqrt.data());
    assert(qrt_sm121_q2::gated_value(core[51],0xbf4f,0x3f70,inverse,silu)==0xbae6);
  }
  const auto short_inverse=qrt_sm121_q2::gated_inverse(short_row[0],rsqrt.data());
  assert(qrt_sm121_q2::gated_value(core[51],0xbf4f,0x3f70,short_inverse,silu)==0xbae5);
}
int main(int argc, char** argv) {
  assert(argc == 5);
  set_rope_table("");
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
    assert(!gb10_full_head_norm_rope_enabled());
    reject([&]{gb10_full_head_norm_rope(input,residual,nullptr,weight,weight,output,norm,nullptr,1,9216,9216,0,8192);});
    check_original_prefill_midpoint(active->silu,argv[3]);
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
    assert(g.name == "gated_prefill_kernel" && g.grid.x == 16384 && g.block.x == 256 && g.stream == fake_stream);
    assert(g.args == std::vector<std::uintptr_t>({value(input),value(residual),value(weight),
        value(output),value(fake_root_table),value(active->silu),8192u*32u}));
    for (const unsigned rows : {1u, 2u}) {
      gated(rows);
      const auto& d = launches.back();
      assert(d.name == "gated_kernel" && d.grid.x == rows * 4u && d.block.x == 256 && d.stream == fake_stream);
      assert(d.args == std::vector<std::uintptr_t>({value(input),value(residual),value(weight),
          value(output),value(fake_root_table),value(active->silu),rows*32u}));
    }
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
    assert(launches.size() == 5);
  }
  assert(!active);
  reject([&]{gated(1);}); reject([&]{residual_norm(1);});
  const auto path = std::filesystem::u8path(argv[2]);
  { std::ofstream file(path, std::ios::binary); file << "bad"; }
  reject([&]{read_silu(path);});
  reject([&]{read_rope(path);});
  { std::vector<uint16_t> wrong(262144*64); std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char*>(wrong.data()),wrong.size()*2); }
  reject([&]{read_rope(path);});
  set_rope_table(path.u8string().c_str());
  reject([&]{Gb10NormalizationOwner wrong_rope;});
  set_rope_table(argv[4]);
  set_table(argv[1]);
  const auto previous_launches = launches.size();
  {
    Gb10NormalizationOwner owner;
    assert(gb10_full_head_norm_rope_enabled() && active->rope);
    auto call = [&](const void* q, const void* k, const void* v, const void* qw, const void* kw,
                    void* qo, void* ko, void* vo, std::size_t n, std::size_t qs,
                    std::size_t ks, std::size_t vs, std::size_t pos) {
      gb10_full_head_norm_rope(q,k,v,qw,kw,qo,ko,vo,n,qs,ks,vs,pos,fake_stream);
    };
    for (const auto rows : {1u,8192u}) {
      call(input,residual,nullptr,weight,weight,output,norm,nullptr,rows,8192,512,0,262144-rows);
      const auto& x=launches.back();
      assert(x.name=="full_head_norm_rope_kernel" && x.grid.x==rows && x.grid.y==18 && x.block.x==256 && x.stream==fake_stream);
      assert(x.args==std::vector<std::uintptr_t>({value(input),value(residual),0,value(weight),value(weight),value(output),value(norm),0,8192,512,0,262144-rows,value(fake_root_table),value(active->rope)}));
    }
    uint16_t copied[16]{};
    call(input,residual,input,weight,weight,output,norm,copied,8192,9216,9216,9216,0);
    const auto& copied_launch=launches.back();
    assert(copied_launch.args==std::vector<std::uintptr_t>({value(input),value(residual),value(input),value(weight),value(weight),value(output),value(norm),value(copied),9216,9216,9216,0,value(fake_root_table),value(active->rope)}));
    for (unsigned null_index=0;null_index<6;++null_index) {
      const void* inputs[]={input,residual,weight,weight};void* outputs[]={output,norm};
      if(null_index<4)inputs[null_index]=nullptr;else outputs[null_index-4]=nullptr;
      reject([&]{call(inputs[0],inputs[1],nullptr,inputs[2],inputs[3],outputs[0],outputs[1],nullptr,1,9216,9216,0,8192);});
    }
    for(const auto rows:{0u,2u,8191u,8193u})
      reject([&]{call(input,residual,nullptr,weight,weight,output,norm,nullptr,rows,9216,9216,0,0);});
    reject([&]{call(input,residual,nullptr,weight,weight,output,norm,nullptr,1,9216,9216,0,262144);});
    reject([&]{call(input,residual,nullptr,weight,weight,output,norm,nullptr,8192,9216,9216,0,262144-8191);});
    reject([&]{call(input,residual,nullptr,weight,weight,output,norm,nullptr,1,0,9216,0,0);});
    reject([&]{call(input,residual,nullptr,weight,weight,output,norm,nullptr,1,9216,511,0,0);});
    reject([&]{call(input,residual,nullptr,weight,weight,output,output,nullptr,1,9216,9216,0,0);});
    reject([&]{call(input,residual,input,weight,weight,output,norm,nullptr,1,9216,9216,0,0);});
    reject([&]{call(input,residual,nullptr,weight,weight,output,norm,copied,1,9216,9216,9216,0);});
    reject([&]{call(input,residual,input,weight,weight,output,norm,copied,1,9216,9216,0,0);});
    reject([&]{call(input,residual,nullptr,weight,weight,output,norm,nullptr,1,9216,9216,512,0);});
    for(auto alias:{output,norm})
      reject([&]{call(input,residual,input,weight,weight,output,norm,alias,1,9216,9216,9216,0);});
    fake_gdn_alive=false;
    reject([&]{call(input,residual,nullptr,weight,weight,output,norm,nullptr,1,9216,9216,0,0);});
    fake_gdn_alive=true;
  }
  assert(!gb10_full_head_norm_rope_enabled() && launches.size()==previous_launches+3);
  set_rope_table("");
  { std::vector<float> wrong(65536); std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(wrong.data()), wrong.size()*4); }
  reject([&]{read_silu(path);});
  std::cout << "{\"table_ownership_and_sha_verified\":true,\"borrowed_rsqrt_binding_verified\":true,"
      "\"dispatches_verified\":8,\"full_head_norm_rope_dispatches_verified\":3,\"optional_rope_owner_and_disabled_mode_verified\":true,\"prefill_and_short_gated_layouts_separated\":true,\"residual_alias_snapshot_verified\":true,\"invalid_bindings_and_artifacts_rejected\":" << rejected
      << ",\"original_prefill_midpoint_passes\":true,\"short_layout_negative_control_differs\":true,\"gpu_reduction_executed\":false}\n";
}
