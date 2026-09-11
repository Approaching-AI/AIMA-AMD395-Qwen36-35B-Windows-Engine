"""Exercise the native checkpoint export and its bounded segment plan."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class FlaCheckpointTests(unittest.TestCase):
    def test_invalid_plan_never_submits_and_each_slot_is_written_once(self):
        provider = (ROOT / 'native/providers/gdn/qrt_fla_chunk_gdn_q8192_provider.cpp').read_text()
        export = function(provider, 'QRT_FLA_GDN_EXPORT int qrt_fla_gdn_launch_async_checkpoints_v1(')
        source = r'''
#include "native/providers/gdn/fla_checkpoint.h"
#include <algorithm>
#include <cassert>
#include <initializer_list>
#define QRT_FLA_GDN_EXPORT
constexpr unsigned kQkvRows=8192,kGateRows=64,kValueFeatures=4096;
bool state=true,batch=true,cooperative=true;
unsigned launches=0;
bool blackwell_state_enabled(){return state;}
bool blackwell_batched_enabled(){return batch;}
namespace qrt_fla_blackwell_cooperative { bool enabled(){return cooperative;} }
void set_error_text(const char*){}
int launch_pipeline_async(const float*,const float*,float*,float*,int,void*,int32_t,
                          const qrt_fla_checkpoint::Plan*){++launches;return 1;}
''' + export + r'''
using namespace qrt_fla_checkpoint;
float* address(uintptr_t value){return reinterpret_cast<float*>(value);}
Plan plan(){
 Plan p{};p.struct_size=sizeof(p);p.abi_version=kVersion;p.count=3;
 p.prefix_tokens[0]=64;p.prefix_tokens[1]=1024;p.prefix_tokens[2]=7168;
 for(unsigned i=0;i<3;++i){p.states[i]=address(0x10000000u+i*0x400000u);p.state_bytes[i]=kStateBytes;}
 return p;
}
int run(const Plan* p,int tokens=7169){return qrt_fla_gdn_launch_async_checkpoints_v1(
 address(0x20000000u),address(0x40000000u),address(0x50000000u),address(0x60000000u),0,nullptr,tokens,p);}
void reject(const Plan* p,int tokens=7169){unsigned before=launches;assert(!run(p,tokens)&&launches==before);}
int main(){
 auto good=plan();assert(run(&good)&&launches==1);reject(nullptr);
 for(int n:{-1,0,1,64,7168,65537})reject(&good,n);
 for(unsigned n:{0u,4u,UINT32_MAX}){auto p=good;p.count=n;reject(&p);}
 for(unsigned n:{0u,unsigned(sizeof(Plan)-1),unsigned(sizeof(Plan)+1)}){auto p=good;p.struct_size=n;reject(&p);}
 {auto p=good;p.abi_version=2;reject(&p);p=good;p.reserved=1;reject(&p);p=good;p.reserved_tail=1;reject(&p);}
 for(unsigned i=0;i<3;++i){
  auto p=good;p.states[i]=nullptr;reject(&p);p=good;p.state_bytes[i]=kStateBytes-1;reject(&p);
  p=good;p.states[i]=address(UINTPTR_MAX-3u);reject(&p);p=good;p.states[i]=address(0x10000001u);reject(&p);
  for(unsigned position:{0u,1u,63u,65u,7169u,UINT32_MAX}){p=good;p.prefix_tokens[i]=position;reject(&p);}
  for(uintptr_t addr:{uintptr_t(0x20000000u),uintptr_t(0x40000000u),uintptr_t(0x50000000u),uintptr_t(0x60000000u),uintptr_t(0x601ffffcu)}){
   p=good;p.states[i]=address(addr);reject(&p);
  }
 }
 {auto p=good;p.prefix_tokens[1]=64;reject(&p);p=good;p.states[1]=address(0x10000004u);reject(&p);}
 {auto p=good;p.count=1;reject(&p);for(unsigned i=1;i<3;++i){p.prefix_tokens[i]=0;p.states[i]=nullptr;p.state_bytes[i]=0;}
  assert(run(&p));p.states[2]=address(0x80000000);reject(&p);}
 state=false;reject(&good);state=true;batch=false;reject(&good);batch=true;cooperative=false;reject(&good);cooperative=true;
 {auto p=good;p.states[2]=address(0x60200000u);assert(run(&p));} // Exactly adjacent buffers are disjoint.
 // Exercise both sides of every K64 and 1024 boundary up to the supported
 // extent. The neutral logical tail never creates a duplicate checkpoint.
 for(unsigned tokens=65;tokens<=65536;++tokens){
  auto p=plan();p.count=1;p.prefix_tokens[0]=64;
  for(unsigned i=1;i<3;++i){p.prefix_tokens[i]=0;p.states[i]=nullptr;p.state_bytes[i]=0;}
  const unsigned last=(tokens-1)/64*64;
  if(last>64){p.count=2;p.prefix_tokens[1]=last;p.states[1]=good.states[1];p.state_bytes[1]=kStateBytes;}
  assert(valid(&p,tokens));unsigned seen[3]{};
  for(unsigned offset=0;offset<tokens;){unsigned count=std::min(1024u,tokens-offset);
   auto s=segment(&p,offset,count);
   for(unsigned i=0;i<s.count;++i){assert(s.prefix_tokens[i]&&s.prefix_tokens[i]<=count&&s.prefix_tokens[i]%64==0);
    unsigned j=0;while(j<p.count&&p.states[j]!=s.states[i])++j;assert(j<p.count);
    assert(offset+s.prefix_tokens[i]==p.prefix_tokens[j]);++seen[j];}
   offset+=count;
  }
  for(unsigned i=0;i<p.count;++i)assert(seen[i]==1);
 }
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / 'checkpoint')
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=undefined', '-fno-sanitize-recover=all',
                            '-I', str(ROOT), '-x', 'c++', '-', '-o', exe],
                           input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
