// SPDX-License-Identifier: Apache-2.0
#include "gb10_ordered_attention.cpp"
#include <algorithm>
#include <cassert>
#include <iostream>
using namespace aima_port;
unsigned rejected=0;
template<class F>void reject(F fn){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}assert(failed);++rejected;}
std::uintptr_t address(const void* p){return reinterpret_cast<std::uintptr_t>(p);}
const unsigned char* table(std::uintptr_t p){return reinterpret_cast<const unsigned char*>(p);}
void clean(){assert(live_bytes==0);for(const auto& m:modules)assert(!m.live);}
int main(){
  const auto* exp=table(0x2100000000ull);const auto* reciprocal=table(0x2200000000ull);
  assert(!gb10_ordered_attention_enabled());
  assert(!gb10_ordered_attention_setting(nullptr)&&!gb10_ordered_attention_setting("0")&&gb10_ordered_attention_setting("1"));
  for(const char* bad:{"","true","01","2","-1"})reject([&]{gb10_ordered_attention_setting(bad);});
  reject([&]{gb10_ordered_attention_prefill(nullptr,nullptr,nullptr,nullptr,8192,8192,0);});
  const auto first_module_attempts=module_attempts;
  for(const auto* bad:{static_cast<const unsigned char*>(nullptr),table(address(exp)+1),table(UINTPTR_MAX-3),reciprocal})
    reject([&]{Gb10OrderedAttentionOwner owner(bad,reciprocal);});
  for(const auto* bad:{static_cast<const unsigned char*>(nullptr),table(address(reciprocal)+1),table(UINTPTR_MAX-3),exp,table(address(exp)+exp2_bytes-4)})
    reject([&]{Gb10OrderedAttentionOwner owner(exp,bad);});
  assert(module_attempts==first_module_attempts&&allocation_attempts==0);clean();
  for(int step=0;step<4;++step){
    fail_module=module_attempts+step;reject([&]{Gb10OrderedAttentionOwner owner(exp,reciprocal);});clean();fail_module=-1;
    fail_function=function_attempts+step;reject([&]{Gb10OrderedAttentionOwner owner(exp,reciprocal);});clean();fail_function=-1;
  }
  for(int step=0;step<6;++step){
    fail_allocation=allocation_attempts+step;reject([&]{Gb10OrderedAttentionOwner owner(exp,reciprocal);});clean();fail_allocation=-1;
  }
  for(int step=0;step<2;++step){
    fail_copy=copy_attempts+step;reject([&]{Gb10OrderedAttentionOwner owner(exp,reciprocal);});clean();fail_copy=-1;
  }
  const std::array<const void*,4> operands{{reinterpret_cast<void*>(0x1000000000ull),
      reinterpret_cast<void*>(0x1100000000ull),reinterpret_cast<void*>(0x1200000000ull),reinterpret_cast<void*>(0x1300000000ull)}};
  unsigned complete_calls=0,binding_rejections=0,failure_cases=0;
  {
    Gb10OrderedAttentionOwner owner(exp,reciprocal);
    assert(gb10_ordered_attention_enabled());
    const auto owned_module_attempts=module_attempts;
    reject([&]{Gb10OrderedAttentionOwner duplicate(exp,reciprocal);});
    assert(module_attempts==owned_module_attempts&&gb10_ordered_attention_enabled());
    assert(live_bytes==Gb10OrderedAttentionOwner::scratch_bytes);
    std::vector<Allocation> owned;for(const auto& a:allocations)if(a.live)owned.push_back(a);
    assert(owned.size()==6);
    const std::vector<std::size_t> expected_sizes={score_bytes,probability_bytes,scale_bytes,kv_bytes,queue_bytes,4};
    for(unsigned i=0;i<owned.size();++i)assert(owned[i].bytes==expected_sizes[i]);
    const auto* indices=static_cast<const std::int32_t*>(owned[4].pointer);
    const auto* count=static_cast<const std::int32_t*>(owned[5].pointer);
    assert(*count==524288);for(unsigned i=0;i<524288;++i)assert(indices[i]==int(i));
    auto call=[&](const std::array<const void*,4>& p,std::size_t n=8192,std::size_t end=8192,std::size_t start=0,void* stream=nullptr){
      return gb10_ordered_attention_prefill(p[0],p[1],p[2],const_cast<void*>(p[3]),n,end,start,stream);
    };
    for(unsigned repeat=0;repeat<2;++repeat){
      launches.clear();const int copies_before=copy_attempts,allocations_before=allocation_attempts;
      assert(call(operands)==193&&launches.size()==193);++complete_calls;
      assert(copy_attempts==copies_before&&allocation_attempts==allocations_before);
      const auto& pack=launches[0];
      assert(pack.symbol=="pack_value_kernel"&&pack.x==16&&pack.y==256&&pack.z==1&&pack.shared==2048);
      assert(pack.pointers==std::vector<std::uintptr_t>({address(operands[2]),address(owned[3].pointer)}));
      assert(pack.scalars==std::vector<std::int32_t>({8192}));
      for(unsigned step=0;step<64;++step){
        const int first=step*128;const std::vector<std::int32_t> scalars={8192,first,128};
        const auto& qk=launches[1+step*3];const auto& prob=launches[2+step*3];const auto& pv=launches[3+step*3];
        assert(qk.symbol=="ordered_qk_kernel"&&qk.x==16&&qk.y==1024&&qk.z==16&&qk.shared==0&&qk.scalars==scalars);
        assert(prob.symbol=="probability_kernel"&&prob.x==32&&prob.y==16&&prob.z==1&&prob.shared==0&&prob.scalars==scalars);
        assert(pv.symbol=="selected_pv_kernel"&&pv.x==1024&&pv.y==1&&pv.z==1&&pv.shared==16&&pv.scalars==scalars);
        assert(qk.pointers==std::vector<std::uintptr_t>({address(operands[0]),address(operands[1]),address(owned[0].pointer)}));
        assert(prob.pointers==std::vector<std::uintptr_t>({address(owned[0].pointer),address(owned[1].pointer),address(owned[2].pointer),address(exp)}));
        assert(pv.pointers==std::vector<std::uintptr_t>({address(owned[1].pointer),address(owned[3].pointer),address(owned[2].pointer),address(reciprocal),
          address(owned[5].pointer),address(owned[4].pointer),address(operands[3])+std::size_t(first)*4096*2,0}));
      }
    }
    auto no_work=[&](auto fn){
      const auto before=launch_attempts;reject(fn);++binding_rejections;assert(launch_attempts==before);
    };
    for(std::size_t n:{0u,8191u,8193u})no_work([&]{call(operands,n);});
    for(std::size_t end:{0u,8191u,8193u,262144u})no_work([&]{call(operands,8192,end);});
    no_work([&]{call(operands,8192,8192,1);});
    no_work([&]{call(operands,8192,8192,0,reinterpret_cast<void*>(1));});
    for(unsigned i=0;i<4;++i){
      auto p=operands;p[i]=nullptr;no_work([&]{call(p);});
      p=operands;p[i]=reinterpret_cast<void*>(address(p[i])+1);no_work([&]{call(p);});
      p=operands;p[i]=reinterpret_cast<void*>(UINTPTR_MAX-1);no_work([&]{call(p);});
      for(unsigned j=0;j<i;++j){
        p=operands;p[i]=p[j];no_work([&]{call(p);});
        p=operands;p[i]=reinterpret_cast<void*>(address(p[j])+(j==0?query_bytes:kv_bytes)-2);no_work([&]{call(p);});
      }
      for(const auto& a:owned){p=operands;p[i]=a.pointer;no_work([&]{call(p);});}
      for(const auto* t:{exp,reciprocal}){p=operands;p[i]=t;no_work([&]{call(p);});}
    }
    for(int step:{0,1,2,3,192}){
      const auto before=launch_attempts;fail_launch=before+step;reject([&]{call(operands);});++failure_cases;
      assert(launch_attempts==before+step+1);fail_launch=-1;
    }
    assert(*count==524288);for(unsigned i=0;i<524288;++i)assert(indices[i]==int(i));
  }
  clean();assert(drains==1&&binding_rejections==65&&rejected==102&&!gb10_ordered_attention_enabled());
  std::cout<<"{\"complete_q8192_calls\":"<<complete_calls<<",\"aot_launches_per_call\":193,\"query_slabs\":64,"
    <<"\"scratch_bytes\":"<<Gb10OrderedAttentionOwner::scratch_bytes<<",\"per_call_allocations\":0,\"per_call_host_copies\":0,"
    <<"\"all_image_and_argument_bindings_pass\":true,\"all_window_output_offsets_pass\":true,\"input_scratch_table_aliases_rejected\":true,"
    <<"\"binding_rejections\":"<<binding_rejections<<",\"construction_faults_rejected\":25,\"launch_failures_propagated\":"<<failure_cases<<","
    <<"\"configuration_and_owner_rejections\":7,\"single_owner_published_after_initialization\":true,"
    <<"\"queue_contents_unchanged\":true,\"all_resources_released\":true,\"allocation_guards_pass\":true,\"gpu_arithmetic_executed\":false}"<<std::endl;
}
