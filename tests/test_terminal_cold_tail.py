"""Exercise the real terminal-provider predicate at a one-token cold tail."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class TerminalColdTailTests(unittest.TestCase):
    def test_scoped_tail_and_standalone_route_selection(self):
        whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        names = (
            'bool qwen36_chunk_prefill_continuation_active() {',
            'bool qwen36_final_layer_full_prefix_requested(',
            'bool qwen36_whole_full_attention_layer_provider_layer(',
            'bool qwen36_whole_full_attention_layer_provider_enabled(',
        )
        source = r'''
#include <cassert>
#include <climits>
#include <cstring>
#include <iostream>
constexpr unsigned kRetainedPrefillTokens=8192u;
constexpr unsigned QRT_QWEN36_LAYER_COUNT=40u,kDescriptorBatchFinalLayer=39u;
bool full_prefix_flag=false,mtp_rows=false,whole_requested=true,exact_path=true;
bool g_qwen36_whole_provider_selected_moe_backend_active=true;
bool g_qwen36_mtp_tensor_namespace_active=false;
struct ScopedQwen36PrefixBatchSuffix{
 bool terminal_only=false;
 inline static ScopedQwen36PrefixBatchSuffix* active=nullptr;
};
namespace qrt_mtp_target_rows {struct Scope{
 static bool requested(unsigned){return mtp_rows;}
};}
bool env_flag_enabled(const char* name){
 assert(!std::strcmp(name,"QRT_QWEN36_FINAL_LAYER_FULL_PREFIX"));return full_prefix_flag;
}
bool maximum_context_streamed_prefill_tokens(unsigned n){return n==65536u||n==131072u||n==262144u;}
bool qwen36_exact_arbitrary_product_path_enabled(unsigned n){return exact_path&&n>0u&&n<=8192u;}
bool qwen36_whole_full_attention_layer_provider_requested(){return whole_requested;}
''' + '\n'.join(function(whole, name) for name in names) + r'''
int main(){
 const unsigned sizes[]={0,1,2,3,4,63,64,65,1023,1024,1025,7169,8191,8192,8193,262144,UINT_MAX};
 unsigned cases=0,provider_cases=0;
 ScopedQwen36PrefixBatchSuffix scope;
 for(unsigned state=0;state<3;++state){
  ScopedQwen36PrefixBatchSuffix::active=state?&scope:nullptr;scope.terminal_only=state==2;
  for(unsigned flags=0;flags<4;++flags){
   full_prefix_flag=flags&1;mtp_rows=flags&2;
   for(unsigned n:sizes){
    const bool old=n>1&&n<=8192&&(full_prefix_flag||mtp_rows||state==2);
    const bool expected=old||(n==1&&state==2);
    assert(qwen36_final_layer_full_prefix_requested(n)==expected);++cases;
    for(unsigned selectors=0;selectors<8;++selectors){
     exact_path=selectors&1;whole_requested=selectors&2;
     g_qwen36_whole_provider_selected_moe_backend_active=selectors&4;
     for(unsigned layer=0;layer<41;++layer){
      const bool extent=n>=8192||(exact_path&&n>0&&n<=8192);
      const bool member=layer<39&&layer%4==3;
      const bool terminal=layer==39&&(maximum_context_streamed_prefill_tokens(n)||expected);
      const bool wanted=extent&&whole_requested&&g_qwen36_whole_provider_selected_moe_backend_active&&(member||terminal);
      assert(qwen36_whole_full_attention_layer_provider_enabled(layer,n)==wanted);++provider_cases;
     }
    }
   }
  }
 }
 scope.terminal_only=true;ScopedQwen36PrefixBatchSuffix::active=&scope;
 exact_path=whole_requested=g_qwen36_whole_provider_selected_moe_backend_active=true;
 full_prefix_flag=mtp_rows=false;
 assert(qwen36_whole_full_attention_layer_provider_enabled(39,1));
 ScopedQwen36PrefixBatchSuffix::active=nullptr;
 assert(!qwen36_whole_full_attention_layer_provider_enabled(39,1));
 std::cout<<"terminal_cold_tail=pass predicate_cases="<<cases<<" provider_cases="<<provider_cases
          <<" cold_tail_one_admitted=1 standalone_one_preserved=1\n";
}
'''
        with tempfile.TemporaryDirectory() as temp:
            path, exe = Path(temp) / 'main.cpp', Path(temp) / 'check'
            path.write_text(source)
            subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', str(path), '-o', str(exe)], check=True, timeout=40)
            subprocess.run([str(exe)], check=True, timeout=20)
            # The actual old predicate still compiles, then fails specifically
            # at the real one-token terminal admission exercised above.
            current = function(whole, names[1])
            old_source = subprocess.check_output(['git', 'show',
                'a7e2092d4cbe68f5f8ce4b61c8dd708453ac25b3:native/providers/whole_provider.cpp'],
                cwd=ROOT, text=True, timeout=15)
            path.write_text(source.replace(current, function(old_source, names[1])))
            subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            str(path), '-o', str(exe)], check=True, timeout=40)
            rejected = subprocess.run([str(exe)], capture_output=True, text=True, timeout=20)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('qwen36_final_layer_full_prefix_requested(n)==expected', rejected.stderr)
            print('terminal_cold_tail_old_predicate=detected')
