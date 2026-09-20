"""Exercise the actual layer-zero prefetch dispatch that bypasses normal decode."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackedPrefetchTests(unittest.TestCase):
    def test_prefetch_selects_the_next_target_projection_arithmetic(self):
        whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        function = whole[whole.index('void launch_q1_float_projection('):
                         whole.index('__global__ void selected_q1_bf16_projection_sm121_kernel(')]
        first = whole.index('            const bool packed_prefetch =')
        calls = whole[first:whole.index('            status = hipGetLastError();', first)]
        prelude = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
using hipStream_t=void*;
struct dim3 {unsigned x,y;explicit dim3(unsigned a,unsigned b=1):x(a),y(b){}};
constexpr unsigned kThreads=256u,kQkvRows=8192u,kZRows=4096u,kAbRows=32u;
bool gdn=false,sm121=false;
bool env_flag_enabled(const char* key){
 if(!std::strcmp(key,"QRT_QWEN36_Q1_SM121_GDN"))return gdn;
 assert(!std::strcmp(key,"QRT_QWEN36_Q1_F32_PROJECTION_SM121"));return sm121;
}
namespace qrt_sm121_q1_packed_runtime {
bool applies_to_prefix(size_t prefix){return prefix>=262144u;}
}
struct Session {size_t prefix_tokens=0;}g_qwen36_resident_session;
struct Workspace {float* device_norm_f32=nullptr;};
struct Event {std::string kernel;const void* input;const void* weight;void* output;unsigned rows;};
std::vector<Event> events;
hipStream_t wanted_stream=reinterpret_cast<void*>(0x1234);
void record(const char* kernel,dim3 grid,dim3 block,hipStream_t stream,
 const float* input,const uint16_t* weight,float* output,unsigned rows,unsigned queries){
 assert(stream==wanted_stream&&queries==1u&&block.x==256u&&grid.x==(rows+15u)/16u);
 events.push_back({kernel,input,weight,output,rows});
}
void record(const char* kernel,dim3 grid,dim3 block,hipStream_t stream,
 const uint16_t* weight,const float* input,float* output,unsigned rows,unsigned queries=0u){
 assert(stream==wanted_stream&&block.x==256u);
 assert(queries?grid.x==rows:grid.x==(rows+15u)/16u);
 events.push_back({kernel,input,weight,output,rows});
}
#define hipLaunchKernelGGL(kernel,grid,block,shared,stream,...) record(#kernel,grid,block,stream,__VA_ARGS__)
'''
        main = r'''
int main(){
 uint16_t weights[4]{};float normalized[1]{},outputs[4]{};
 Workspace owner{normalized};auto* workspace=&owner;hipStream_t direct_output_stream=wanted_stream;
 const uint16_t *qkv_weights=weights,*z_weights=weights+1,*a_weights=weights+2,*b_weights=weights+3;
 float *device_qkv=outputs,*device_z=outputs+1,*device_a=outputs+2,*device_b=outputs+3;
 const std::array<unsigned,4> rows={kQkvRows,kZRows,kAbRows,kAbRows};
 for(bool enabled:{false,true})for(bool original:{false,true})for(size_t prefix:{8192u,262143u,262144u,263168u}){
  gdn=enabled;sm121=original;g_qwen36_resident_session.prefix_tokens=prefix;events.clear();
  CALLS
  assert(events.size()==4u);
  for(unsigned i=0;i<4u;++i){const auto& e=events[i];
   assert(e.input==normalized&&e.weight==weights+i&&e.output==outputs+i&&e.rows==rows[i]);
   const std::string expected=enabled&&prefix>=262144u?"qrt_sm121_packed_dense::projection<2048u,float,float>":
    original?"selected_q1_float_projection_sm121_kernel":"selected_float_projection_kernel";
   assert(e.kernel.find(expected)!=std::string::npos);
  }
 }
}
'''.replace('CALLS', calls)
        with tempfile.TemporaryDirectory(prefix='qrt-packed-prefetch-') as temp:
            source = Path(temp) / 'test.cpp'
            source.write_text(prelude + function + main)
            executable = Path(temp) / 'test'
            built = subprocess.run([os.getenv('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                    str(source), '-o', str(executable)], capture_output=True, text=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
