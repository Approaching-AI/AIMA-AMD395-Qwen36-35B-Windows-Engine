"""Exercise actual plan prewarming/dispatch and failure boundaries without a GPU."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MatrixProducerSelectionTests(unittest.TestCase):
    def test_actual_prewarm_and_dispatch_agree_and_fail_before_submission(self):
        provider = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        start = provider.index('bool resident_q8192_matrix_producer_choice(')
        wrapper = provider[start:provider.index('bool resident_bf16_matrix_matmul_f32_output_with_heuristic_index(', start)]
        start = provider.index('bool prewarm_q8192_resident_bf16_matrix_plans(')
        prewarm = provider[start:provider.index('bool prewarm_q262144_resident_bf16_matrix_plans(', start)]
        source = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <tuple>
#include "q8192_matrix_producer_policy.h"
using hipStream_t = void*;
constexpr unsigned kQkvRows=8192, QRT_QWEN36_HIDDEN_SIZE=2048, kRetainedPrefillTokens=8192,
 kZRows=4096, kAbRows=64, kOutProjectionRows=2048, kValueFeatures=4096,
 QRT_QWEN36_MOE_EXPERT_INTERMEDIATE=512, kLayer3FullAttentionQRows=8192,
 kLayer3FullAttentionKRows=512, kLayer3FullAttentionQFeatures=4096, QRT_QWEN36_VOCAB_SIZE=248320;
using Key=std::tuple<unsigned,unsigned,unsigned,bool,unsigned>;
struct ResidentBf16MatrixProvider { std::set<Key> plans; } owner;
struct ResidentBf16MatrixPlan {} plan;
unsigned ensures=0, submissions=0, last_choice=99;
bool fail_plan=false, fail_submit=false;
unsigned env_u32_or_default(const char*, unsigned value) { return value; }
bool env_flag_enabled(const char*) { return false; }
bool ensure_resident_bf16_matrix_provider(ResidentBf16MatrixProvider** out, std::string*, std::string*) {
 ++ensures; *out=&owner; return true;
}
bool create_resident_bf16_matrix_plan(ResidentBf16MatrixProvider* p, unsigned rows, unsigned k,
 unsigned tokens, bool f32, unsigned choice, ResidentBf16MatrixPlan** out, std::string*, std::string*) {
 assert(p==&owner); if (fail_plan && choice==4) return false;
 p->plans.emplace(rows,k,tokens,f32,choice); *out=&plan; return true;
}
bool resident_bf16_matrix_matmul_impl(const uint16_t* w, const uint16_t* x, void* y,
 unsigned rows, unsigned k, unsigned tokens, bool f32, float alpha, float beta, unsigned choice,
 hipStream_t stream, const std::string& stage, std::string*, std::string*) {
 assert(w && x && y && rows && k && tokens && f32 && alpha==1 && beta==0 && stream==y && stage=="test");
 ++submissions; last_choice=choice; return !fail_submit;
}
''' + wrapper + prewarm + r'''
void setting(const char* value) {
#ifdef _WIN32
 _putenv_s("QRT_QWEN36_Q8192_MATRIX_PRODUCER_ALGORITHM", value ? value : "");
#else
 if(value) setenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_ALGORITHM",value,1);
 else unsetenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_ALGORITHM");
#endif
}
void scope(const char* value) {
#ifdef _WIN32
 _putenv_s("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE", value ? value : "");
#else
 if(value) setenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE",value,1);
 else unsetenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE");
#endif
}
int main() {
 scope(nullptr);
 std::string stage,error; size_t count=0; uint16_t w=0,x=0; float y=0;
 auto run=[&](unsigned rows,unsigned k,unsigned tokens) {
  return resident_bf16_matrix_matmul_f32_output(&w,&x,&y,rows,k,tokens,&y,"test",&stage,&error);
 };
 const std::array<std::pair<unsigned,unsigned>,4> shapes={{{8192,2048},{4096,2048},{9216,2048},{2048,4096}}};
 for(const char* mode : {static_cast<const char*>(nullptr),"","0","4"}) {
  setting(mode); owner.plans.clear();
  assert(prewarm_q8192_resident_bf16_matrix_plans(&count,&stage,&error) && count==owner.plans.size());
  const unsigned expected=mode && !std::strcmp(mode,"4") ? 4 : 0;
  for(auto shape:shapes) {
   if(expected) assert(owner.plans.count(Key{shape.first,shape.second,8192,true,4})==1);
   assert(run(shape.first,shape.second,8192) && last_choice==expected);
   for(unsigned tokens:{1u,4u,65u,7169u,8191u,8193u,16384u})
    assert(run(shape.first,shape.second,tokens) && last_choice==0);
   unsigned choice=99;
   assert(qrt_q8192_matrix_producer::resolve(mode,shape.first,shape.second,8192,false,&choice) && choice==0);
  }
  for(auto shape:{std::pair{8191u,2048u},std::pair{8192u,2047u},std::pair{2048u,2048u},std::pair{64u,2048u}})
   assert(run(shape.first,shape.second,8192) && last_choice==0);
 }
 setting("4");
 for(const char* selected_scope:{"all","qkv","out"}) {
  scope(selected_scope);owner.plans.clear();
  assert(prewarm_q8192_resident_bf16_matrix_plans(&count,&stage,&error));
  unsigned planned=0;
  for(auto key:owner.plans)planned+=std::get<4>(key)==4;
  assert(planned==(!std::strcmp(selected_scope,"all")?4u:(!std::strcmp(selected_scope,"qkv")?3u:1u)));
  for(auto shape:shapes) {
   unsigned expected=!std::strcmp(selected_scope,"all") || (!std::strcmp(selected_scope,"qkv")?shape.second==2048:shape.second==4096)?4:0;
   assert(run(shape.first,shape.second,8192) && last_choice==expected);
   assert(owner.plans.count(Key{shape.first,shape.second,8192,true,4})==unsigned(expected==4));
   assert(run(shape.first,shape.second,7169) && last_choice==0);
   assert(run(shape.first,shape.second,1) && last_choice==0);
  }
 }
 for(const char* bad:{"QKV","qkv,out","output","alljunk"}) {
  scope(bad);const auto before_ensures=ensures,before_submissions=submissions;
  assert(!prewarm_q8192_resident_bf16_matrix_plans(&count,&stage,&error) && ensures==before_ensures);
  assert(!run(8192,2048,8192) && submissions==before_submissions);
 }
 scope(nullptr);
 for(const char* mode:{"1","20","-4","4junk"," 4","04"}) {
  setting(mode); const auto before_ensures=ensures,before_submissions=submissions;
  assert(!prewarm_q8192_resident_bf16_matrix_plans(&count,&stage,&error));
  assert(ensures==before_ensures && stage=="hipblaslt_q8192_matrix_producer_algorithm");
  assert(!run(8192,2048,8192) && submissions==before_submissions);
 }
 setting("4"); fail_plan=true;
 assert(!prewarm_q8192_resident_bf16_matrix_plans(&count,&stage,&error));
 fail_plan=false; fail_submit=true;
 assert(!run(8192,2048,8192) && last_choice==4);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-matrix-policy-') as tmp:
            exe = str(Path(tmp) / 'policy')
            subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-Wall','-Wextra','-Werror',
                '-I',str(ROOT/'native/providers'),'-x','c++','-','-o',exe], input=source,
                text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, capture_output=True, text=True, timeout=5)

if __name__ == '__main__':
    unittest.main()
