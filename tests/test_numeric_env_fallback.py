"""A cached setting must not freeze defaults that depend on the request shape."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class NumericEnvironmentFallbackTests(unittest.TestCase):
    def test_actual_parser_keeps_fallback_per_call(self):
        whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        signatures = ('unsigned int env_u32_or_default(const char *name, unsigned int fallback) {',
                      'int env_i32_or_default(const char *name, int fallback) {',
                      'uint64_t parse_env_u64_or_default(const char *name, uint64_t default_value) {')
        current = '\n'.join(function(whole, signature) for signature in signatures)
        preamble = r'''
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
bool cache_enabled=true;
bool q1_decode_control_plane_cache_enabled(){return cache_enabled;}
'''
        checks = r'''
template<class T,class Parser>
void check_other_parser(const std::string& prefix,Parser parse,const std::vector<T>& defaults,
 const std::vector<const char*>& invalid,const std::vector<const char*>& configured,const std::vector<T>& values){
 unsigned fallback_checks=0,value_checks=0;
 for(size_t i=0;i<invalid.size();++i){
  const std::string name=prefix+"_FALLBACK_"+std::to_string(i);
  if(invalid[i])assert(setenv(name.c_str(),invalid[i],1)==0);else assert(unsetenv(name.c_str())==0);
  for(T wanted:defaults){assert(parse(name.c_str(),wanted)==wanted);++fallback_checks;}
  assert(setenv(name.c_str(),"17",1)==0);
  for(T wanted:defaults){assert(parse(name.c_str(),wanted)==(cache_enabled?wanted:T(17)));++fallback_checks;}
 }
 for(size_t i=0;i<configured.size();++i){
  const std::string name=prefix+"_CONFIGURED_"+std::to_string(i);
  assert(setenv(name.c_str(),configured[i],1)==0);
  for(T wanted:defaults){assert(parse(name.c_str(),wanted)==values[i]);++value_checks;}
  assert(unsetenv(name.c_str())==0);
  for(T wanted:defaults){assert(parse(name.c_str(),wanted)==(cache_enabled?values[i]:wanted));++value_checks;}
 }
 std::cout<<"numeric_env_parser="<<prefix<<" cache="<<cache_enabled
          <<" fallback_checks="<<fallback_checks<<" configured_checks="<<value_checks<<"\n";
}
int main(int argc,char**argv){
 assert(argc==2);cache_enabled=std::string(argv[1])=="cached";
 const unsigned defaults[]={8191u,0u,7168u,UINT_MAX,31u,0u};
 unsigned fallback_checks=0,value_checks=0;
 const char* invalid[]={nullptr,"","bad","17tail","4294967296","18446744073709551616"};
 for(unsigned i=0;i<6;++i){
  const std::string name="QRT_TEST_FALLBACK_"+std::to_string(i);
  if(invalid[i])assert(setenv(name.c_str(),invalid[i],1)==0);
  else assert(unsetenv(name.c_str())==0);
  for(unsigned wanted:defaults){
   assert(env_u32_or_default(name.c_str(),wanted)==wanted);++fallback_checks;
  }
  assert(setenv(name.c_str(),"17",1)==0);
  for(unsigned wanted:defaults){
   assert(env_u32_or_default(name.c_str(),wanted)==(cache_enabled?wanted:17u));++fallback_checks;
  }
 }
 const char* configured[]={"0","1","4294967295"," 15","+31"};
 const unsigned values[]={0u,1u,UINT_MAX,15u,31u};
 for(unsigned i=0;i<5;++i){
  const std::string name="QRT_TEST_CONFIGURED_"+std::to_string(i);
  assert(setenv(name.c_str(),configured[i],1)==0);
  for(unsigned wanted:defaults){assert(env_u32_or_default(name.c_str(),wanted)==values[i]);++value_checks;}
  assert(unsetenv(name.c_str())==0);
  for(unsigned wanted:defaults){
   assert(env_u32_or_default(name.c_str(),wanted)==(cache_enabled?values[i]:wanted));++value_checks;
  }
 }
 std::cout<<"numeric_env_fallback=pass cache="<<cache_enabled
          <<" fallback_checks="<<fallback_checks<<" configured_checks="<<value_checks<<"\n";
 check_other_parser<int>("QRT_TEST_I32",env_i32_or_default,{8191,0,-1,INT_MAX,INT_MIN,0},
  {nullptr,"","bad","17tail","2147483648","-2147483649"},
  {"0","1","2147483647","-2147483648","-31"," 15"},{0,1,INT_MAX,INT_MIN,-31,15});
 check_other_parser<uint64_t>("QRT_TEST_U64",parse_env_u64_or_default,{8191u,0u,UINT64_MAX,7168u,0u},
  {nullptr,"","bad","17tail"},
  {"0","1","18446744073709551615","0xff","017"," 15"},{0,1,UINT64_MAX,255,15,15});
}
'''
        preamble += '#include <vector>\n'
        with tempfile.TemporaryDirectory() as temp:
            source, exe = Path(temp) / 'main.cpp', Path(temp) / 'check'
            source.write_text(preamble + current + checks)
            subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', str(source), '-o', str(exe)], check=True, timeout=40)
            for mode in ('cached', 'dynamic'):
                subprocess.run([str(exe), mode], check=True, timeout=20)
            old = subprocess.check_output(['git', 'show',
                '99fb67f554892c921b945bfd9a802666aeadcbce:native/providers/whole_provider.cpp'],
                cwd=ROOT, text=True, timeout=15)
            for signature in signatures:
                # Restore one real old parser at a time so all three negative
                # controls must reach their own changing-default assertion.
                prior = current.replace(function(whole, signature), function(old, signature))
                source.write_text(preamble + prior + checks)
                subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                                str(source), '-o', str(exe)], check=True, timeout=40)
                failed = subprocess.run([str(exe), 'cached'], capture_output=True, text=True, timeout=20)
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn('==wanted', failed.stderr)
                print('numeric_env_old_cached_fallback=detected parser=' + signature.split('(')[0])
