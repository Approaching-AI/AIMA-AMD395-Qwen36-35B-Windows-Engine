"""Actual row observers must read the requested row and reject out-of-range positions."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class SelectedPrefillTraceTests(unittest.TestCase):
    def test_real_float_and_bf16_observers_select_and_bound_the_same_row(self):
        source = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        definitions = '\n'.join(function(source, 'bool ' + name + '(') for name in (
            'emit_qwen36_exact_arbitrary_layer_boundary_trace',
            'emit_qwen36_exact_arbitrary_linear_stage_trace',
            'emit_qwen36_exact_arbitrary_linear_stage_bf16_trace'))
        harness = r'''
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <climits>
constexpr unsigned QRT_QWEN36_HIDDEN_SIZE=2048;
using hipError_t=int;constexpr int hipSuccess=0,hipMemcpyDeviceToHost=1;
const char* hipGetErrorString(int){return "copy failure";}
unsigned position=1,copies=0;bool enabled=true;const void* copied=nullptr;size_t bytes=0;
unsigned env_u32_or_default(const char* name,unsigned fallback){
 if(std::strstr(name,"TRACE_POSITION"))return position==UINT_MAX?fallback:position;
 if(std::strstr(name,"TRACE_LAYER"))return enabled?12u:UINT_MAX;
 return fallback;
}
bool raw_env_flag_enabled(const char*){return enabled;}
bool qwen36_all_norm_capture_active(unsigned){return false;}
bool qwen36_exact_arbitrary_product_path_enabled(unsigned){return true;}
bool dump_qwen36_selected_full_stage(unsigned,unsigned,const char*,const void*,bool,size_t,std::string*,std::string*){return true;}
uint64_t qrt_fnv1a64_f32(const float*,size_t){return 0;}
std::string hex_u64(uint64_t){return "unused-diagnostic-hash";}
int hipMemcpy(void* destination,const void* source,size_t count,int){
 ++copies;copied=source;bytes=count;std::memcpy(destination,source,count);return 0;
}
''' + definitions + r'''
int main(){
 constexpr unsigned tokens=3,width=4096;
 std::vector<float> hidden(tokens*2048u,1.0f),linear(tokens*width,2.0f);
 std::vector<uint16_t> bf16(tokens*width,0x4000u);
 std::string stage,error;std::ostringstream log;auto* previous=std::cerr.rdbuf(log.rdbuf());
 for(unsigned selected:{0u,1u,2u,UINT_MAX}){
  position=selected;const unsigned row=selected==UINT_MAX?tokens-1:selected;
  assert(emit_qwen36_exact_arbitrary_layer_boundary_trace(12,tokens,"norm",hidden.data(),&stage,&error));
  assert(copied==hidden.data()+row*2048u&&bytes==2048u*4);
  assert(emit_qwen36_exact_arbitrary_linear_stage_trace(12,tokens,"qkv",linear.data(),width,&stage,&error));
  assert(copied==linear.data()+row*width&&bytes==width*4u);
  assert(emit_qwen36_exact_arbitrary_linear_stage_bf16_trace(12,tokens,"qkv",bf16.data(),width,&stage,&error));
  assert(copied==bf16.data()+row*width&&bytes==width*2u);
 }
 const unsigned before=copies;position=tokens;
 assert(!emit_qwen36_exact_arbitrary_layer_boundary_trace(12,tokens,"norm",hidden.data(),&stage,&error));
 assert(stage=="exact_arbitrary_layer_boundary_trace_position");
 assert(!emit_qwen36_exact_arbitrary_linear_stage_trace(12,tokens,"qkv",linear.data(),width,&stage,&error));
 assert(stage=="exact_arbitrary_linear_stage_trace_position");
 assert(!emit_qwen36_exact_arbitrary_linear_stage_bf16_trace(12,tokens,"qkv",bf16.data(),width,&stage,&error));
 assert(stage=="exact_arbitrary_linear_stage_bf16_trace_position");
 assert(copies==before);enabled=false;
 assert(emit_qwen36_exact_arbitrary_layer_boundary_trace(12,tokens,"norm",hidden.data(),&stage,&error));
 assert(emit_qwen36_exact_arbitrary_linear_stage_trace(12,tokens,"qkv",linear.data(),width,&stage,&error));
 assert(emit_qwen36_exact_arbitrary_linear_stage_bf16_trace(12,tokens,"qkv",bf16.data(),width,&stage,&error));
 assert(copies==before);std::cerr.rdbuf(previous);
 assert(log.str().find("trace_position=1")!=std::string::npos);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            executable = str(Path(directory) / 'trace')
            subprocess.run(['c++', '-std=c++17', '-O1', '-fsanitize=address,undefined',
                            '-fno-sanitize-recover=all', '-x', 'c++', '-', '-o', executable],
                           input=harness, text=True, check=True, capture_output=True, timeout=30)
            subprocess.run([executable], check=True, capture_output=True, timeout=10)
