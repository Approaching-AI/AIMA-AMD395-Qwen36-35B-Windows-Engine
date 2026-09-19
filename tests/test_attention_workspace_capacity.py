"""Validate a capacity hint independently from live token extents."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AttentionWorkspaceCapacityTests(unittest.TestCase):
    def test_every_extent_rounding_invalid_values_and_unchanged_rejection(self):
        source = r'''
#include "native/providers/ck_fmha/attention_workspace_capacity.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
int main(){
 using qrt_attention_workspace_capacity::select;
 constexpr unsigned maximum=qrt_sm121_attention_capacity::kTokens;
 uint64_t checked=0,rejected=0;
 for(unsigned needed=1;needed<=maximum;++needed)for(unsigned quantum:{1u,8192u}){
  for(unsigned reserve:{0u,1u,8192u,32768u,131072u,262144u,maximum}){
   const auto value=std::to_string(reserve);unsigned actual=77,requested=99;
   const uint64_t want=std::max(needed,reserve);
   const unsigned expected=static_cast<unsigned>(std::min<uint64_t>(maximum,
       (want+quantum-1u)/quantum*quantum));
   assert(select(value.c_str(),needed,quantum,actual,requested));
   assert(actual==expected&&actual>=needed&&actual>=reserve&&requested==reserve);
   ++checked;
  }
 }
 for(const char* option:std::initializer_list<const char*>{nullptr,"","0","0000"}){
  unsigned actual=77,requested=99;
  assert(select(option,32769,8192,actual,requested)&&actual==40960&&requested==0);
  ++checked;
 }
 const auto over=std::to_string(maximum+1u);
 for(const char* option:{"-1","+1","1 "," 1","true","1x","4294967296",
       "18446744073709551615","999999999999999999999999999999999999",over.c_str()}){
  unsigned actual=77,requested=99;
  assert(!select(option,8192,8192,actual,requested)&&actual==77&&requested==99);++rejected;
 }
 for(unsigned needed:{0u,maximum+1u,UINT32_MAX})for(unsigned quantum:{1u,8192u}){
  unsigned actual=77,requested=99;
  assert(!select("262144",needed,quantum,actual,requested)&&actual==77&&requested==99);++rejected;
 }
 for(unsigned quantum:{0u,2u,8191u,8193u,UINT32_MAX}){
  unsigned actual=77,requested=99;
  assert(!select("262144",8192,quantum,actual,requested)&&actual==77&&requested==99);++rejected;
 }
 std::printf("capacity_checks=%llu rejected_checks=%llu\n",(unsigned long long)checked,(unsigned long long)rejected);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-workspace-capacity-') as tmp:
            binary = str(Path(tmp)/'capacity')
            subprocess.run(['c++','-std=c++17','-O1','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-fno-sanitize-recover=all','-I',str(ROOT),
                '-x','c++','-','-o',binary], input=source,text=True,check=True,timeout=30)
            result = subprocess.run([binary],capture_output=True,text=True,check=True,timeout=15)
            self.assertIn('capacity_checks=3706308',result.stdout)
            self.assertIn('rejected_checks=21',result.stdout)
