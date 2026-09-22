// SPDX-License-Identifier: Apache-2.0
// Record calls into the already separately qualified attention kernels. Their
// arithmetic is deliberately not emulated by this ownership/binding test.
#include <hip/hip_runtime.h>
#define QRT_BLACKWELL_ATTENTION_H
namespace qrt_blackwell_attention {
inline uint16_t f32_to_bf16(float) { return 0; }
inline void blackwell_compact_query_scores_kernel(const uint16_t*,const uint16_t*,float*,
    unsigned,unsigned,unsigned,unsigned) {}
template<bool Serial,bool Scores,bool Split,bool Native,bool Strided,bool Warp,bool Prepared,bool Mtp,class View>
void blackwell_exact_attention_kernel(const uint16_t*,const uint16_t*,const uint16_t*,float*,
    unsigned,unsigned,const unsigned char*,float*,float*,bool,const unsigned char*,const float*,unsigned,View,unsigned) {
  static_assert(Serial && Scores && Mtp && !Split && !Native && !Strided && !Warp && !Prepared);
}
}
#include "gb10_decode_attention.hip.cpp"
#include <algorithm>
#include <iostream>
namespace aima_port {
bool fake_gdn_alive = true;
unsigned char fake_exp2[64];
const unsigned char* gb10_exp2_table() {
  if (!fake_gdn_alive) throw std::runtime_error("Missing GDN exp2 owner");
  return fake_exp2;
}
}
using namespace aima_port;
unsigned rejected = 0;
template<class F>void reject(F fn) {
  bool failed=false;try{fn();}catch(const std::exception&){failed=true;}
  assert(failed);++rejected;
}
void setting(const char* name,const char* value) {
#ifdef _WIN32
  _putenv_s(name,value);
#else
  setenv(name,value,1);
#endif
}
int main(int argc,char** argv) {
  assert(argc==3);
  setting("AIMA_PORT_DECODE_ATTENTION","");
  std::vector<uint16_t> q(4096,0x1234), k(8193u*512u,0x2345), v(k.size(),0x3456), out(4096,0x4567);
  auto call=[&](size_t n){gb10_decode_attention(q.data(),k.data(),v.data(),out.data(),n,nullptr);};
  reject([&]{call(1);});
  fake_gdn_alive=false;
  {Gb10DecodeAttentionOwner disabled(8193);assert(!active && allocations==0);}
  reject([&]{Gb10DecodeAttentionOwner bad(0);});
  reject([&]{Gb10DecodeAttentionOwner bad(262145);});
  setting("AIMA_PORT_DECODE_ATTENTION","x");reject([&]{Gb10DecodeAttentionOwner bad(8193);});
  setting("AIMA_PORT_DECODE_ATTENTION","1");reject([&]{Gb10DecodeAttentionOwner no_gdn(8193);});
  fake_gdn_alive=true;
  setting("AIMA_PORT_ATTENTION_RCP_TABLE","");reject([&]{Gb10DecodeAttentionOwner missing(8193);});
  setting("AIMA_PORT_ATTENTION_RCP_TABLE",argv[2]);
  {std::ofstream f(argv[2],std::ios::binary);f << "bad";}
  reject([&]{Gb10DecodeAttentionOwner truncated(8193);});
  auto bytes=read_reciprocal(std::filesystem::u8path(argv[1]));bytes.back()^=1u;
  {std::ofstream f(argv[2],std::ios::binary);f.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());}
  reject([&]{Gb10DecodeAttentionOwner corrupted(8193);});
  setting("AIMA_PORT_ATTENTION_RCP_TABLE",argv[1]);
  for(int fail=0;fail<2;++fail) {
    fail_allocation=allocation_calls+fail;
    reject([&]{Gb10DecodeAttentionOwner failed(8193);});
    assert(!active && allocations==0);
  }
  fail_allocation=-1;fail_copy=true;
  reject([&]{Gb10DecodeAttentionOwner failed_copy(8193);});
  assert(!active && allocations==0);fail_copy=false;
  unsigned complete_calls=0;
  {
    Gb10DecodeAttentionOwner owner(8193);
    assert(active && active->capacity==8193 && active->score_capacity==8224 && allocations==2);
    reject([&]{Gb10DecodeAttentionOwner duplicate(8193);});
    for(size_t n:{1u,32u,33u,8192u,8193u}) {
      const size_t before=launches.size();const int copies_before=copies;call(n);++complete_calls;
      assert(launches.size()==before+3 && copies==copies_before);
      const auto& scores=launches[before];const auto& attention=launches[before+1];const auto& publish=launches[before+2];
      const size_t stride=(n+31u)&~size_t(31u);auto* scratch=active->scratch.as<float>();
      auto* context=scratch+16u*8224u;
      assert(scores.grid.x==stride && attention.grid.x==16 && publish.grid.x==16);
      for(size_t i=before;i<launches.size();++i)
        assert(launches[i].block.x==256 && launches[i].grid.y==1 && !launches[i].stream);
      assert(scores.args==std::vector<std::uintptr_t>({address(q.data()),address(k.data()),address(scratch),n-1,1,stride,n-1}));
      assert(attention.args==std::vector<std::uintptr_t>({0,0,0,address(context),n-1,0,address(fake_exp2),0,0,1,
          address(active->reciprocal.data),address(scratch),stride,address(v.data()),n-1}));
      assert(publish.args==std::vector<std::uintptr_t>({address(context),address(out.data())}));
    }
    reject([&]{call(0);});reject([&]{call(8194);});
    reject([&]{gb10_decode_attention(q.data(),k.data(),v.data(),out.data(),1,reinterpret_cast<void*>(1));});
    const std::array<const void*,4> pointers{{q.data(),k.data(),v.data(),out.data()}};
    for(size_t i=0;i<4;++i)for(unsigned control=0;control<3;++control) {
      auto args=pointers;args[i]=control==0 ? nullptr : control==1 ? reinterpret_cast<void*>(address(args[i])+1u)
          : reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max()-1u);
      reject([&]{gb10_decode_attention(args[0],args[1],args[2],const_cast<void*>(args[3]),8193,nullptr);});
    }
    for(size_t i=0;i<4;++i)for(size_t j=0;j<i;++j) {
      auto args=pointers;args[i]=reinterpret_cast<void*>(address(args[j])+2u);
      reject([&]{gb10_decode_attention(args[0],args[1],args[2],const_cast<void*>(args[3]),8193,nullptr);});
    }
    // Fail before any work is submitted if the borrowed owner disappears.
    fake_gdn_alive=false;const auto before=launches.size();reject([&]{call(1);});
    assert(launches.size()==before);fake_gdn_alive=true;
    for(size_t which=1;which<=3;++which) {
      fail_launch=launches.size()+which;reject([&]{call(33);});
      assert(launches.size()==fail_launch);fail_launch=0;
    }
    // Addressing the actual resident plane reaches both KV heads and the last
    // valid token, with no token-stride conversion.
    Values view{v.data()};v[8192u*512u+511u]=0x7abc;
    assert(view.value(8192,1,255)==0x7abc);v.back()=0x3456;
  }
  assert(!active && allocations==0 && drains==1);
  reject([&]{call(1);});
  for(const auto& pair:{std::make_pair(&q,uint16_t(0x1234)),std::make_pair(&k,uint16_t(0x2345)),
      std::make_pair(&v,uint16_t(0x3456)),std::make_pair(&out,uint16_t(0x4567))})
    assert(std::all_of(pair.first->begin(),pair.first->end(),[&](uint16_t x){return x==pair.second;}));
  {Gb10DecodeAttentionOwner maximum(262144);assert(active->score_capacity==262144);}
  assert(!active && allocations==0 && drains==2);
  std::cout << "{\"complete_bindings\":" << complete_calls << ",\"rejected_controls\":" << rejected
      << ",\"buffers_unchanged\":true,\"all_allocations_released\":true,\"kernel_arithmetic_executed\":false}" << std::endl;
}
