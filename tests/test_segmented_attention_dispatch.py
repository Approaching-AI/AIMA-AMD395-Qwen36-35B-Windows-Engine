"""Exercise the actual resident dispatch and its owned scratch arguments."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SegmentedAttentionDispatchTests(unittest.TestCase):
    def test_context_selection_arguments_and_submission_failures(self):
        whole = (ROOT/'native/providers/whole_provider.cpp').read_text()
        start = whole.index('        const bool use_segmented_attention =')
        body = whole[start:whole.index('        hipLaunchKernelGGL(qwen36_resident_full_attention_grouped_bf16_post_kernel,',start)]
        source = r'''
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <string>
#include <initializer_list>
using hipStream_t=void*;using hipError_t=int;
struct dim3 {unsigned x;explicit dim3(unsigned n):x(n){}};
struct Session {size_t prefix_tokens;}g_qwen36_resident_session{};
namespace qrt_sm121_q1_packed_runtime {bool applies_to_prefix(size_t n){return n>=262144u;}}
struct Layer {const void *device_v,*device_decode_tail_v;size_t history_tokens;}layer;
struct Workspace {float *device_full_attention_wave32_full_dimension_partial_acc,
 *device_full_attention_wave32_full_dimension_partial_max,*device_full_attention_wave32_full_dimension_partial_sum;}workspace;
struct Tables {const unsigned char* exp2;}q1_full_core_tables;
float *score_scratch,*device_context_output;
const unsigned char* rcp;
unsigned absolute_position,total_tokens,workspace_score_scratch_token_capacity,layer_index=3u;
hipStream_t stream=reinterpret_cast<void*>(0x1234);
unsigned calls=0,events=0;bool segmented=false;int failure=0;
namespace qrt_sm121_q1_segmented_attention {
int launch(const float* scores,const uint16_t* prefix,const uint16_t* tail,float* acc,float* max,float* sum,
 float* output,unsigned history,unsigned tokens,unsigned stride,const unsigned char* ex,const unsigned char* reciprocal,hipStream_t queue){
 ++calls;segmented=true;
 assert(scores==score_scratch&&prefix==layer.device_v&&tail==layer.device_decode_tail_v);
 assert(acc==workspace.device_full_attention_wave32_full_dimension_partial_acc);
 assert(max==workspace.device_full_attention_wave32_full_dimension_partial_max);
 assert(sum==workspace.device_full_attention_wave32_full_dimension_partial_sum);
 assert(acc!=max&&max!=sum&&sum!=acc&&scores!=acc&&scores!=max&&scores!=sum);
 assert(output==device_context_output&&history==layer.history_tokens&&tokens==total_tokens&&stride==workspace_score_scratch_token_capacity);
 assert(ex==q1_full_core_tables.exp2&&reciprocal==rcp&&queue==stream);return failure==1?1:0;
}}
void legacy(dim3 grid,dim3 block,hipStream_t queue,const uint16_t* q,const uint16_t* k,const uint16_t* value,float* output,
 unsigned position,unsigned output_start,const unsigned char* ex,float* acc,float* den,bool vllm,const unsigned char* reciprocal,
 const float* scores,unsigned stride,const uint16_t* tail,unsigned history){
 ++calls;segmented=false;
 assert(grid.x==16u&&block.x==256u&&queue==stream&&!q&&!k&&value==layer.device_v&&output==device_context_output);
 assert(position==absolute_position&&!output_start&&ex==q1_full_core_tables.exp2&&!acc&&!den&&vllm&&reciprocal==rcp);
 assert(scores==score_scratch&&stride==workspace_score_scratch_token_capacity&&tail==layer.device_decode_tail_v&&history==layer.history_tokens);
}
#define hipLaunchKernelGGL(kernel,grid,block,shared,queue,...) legacy(grid,block,queue,__VA_ARGS__)
int hipGetLastError(){return failure==1?1:0;}
enum class Q1LayerProfileBoundary {kSoftmaxEnd};
int record_qwen36_resident_decode_q1_layer_profile_boundary(Workspace* owner,unsigned layer_index,Q1LayerProfileBoundary,hipStream_t queue){
 assert(owner==&workspace&&layer_index==3u&&queue==stream);++events;return failure==2?1:0;
}
bool execute(){auto checked=[](int status,const char*){return status==0;};
BODY
return true;}
int main(){
 float scratch[8]{};uint16_t caches[2]{};unsigned char tables[2]{};
 score_scratch=scratch;device_context_output=scratch+4;workspace={scratch+1,scratch+2,scratch+3};
 layer.device_v=caches;layer.device_decode_tail_v=caches+1;q1_full_core_tables.exp2=tables;rcp=tables+1;
 for(size_t prefix:{8192u,262143u,262144u,263168u})for(unsigned tail:{0u,511u})for(int fail:{0,1,2}){
  g_qwen36_resident_session.prefix_tokens=prefix;layer.history_tokens=prefix;
  absolute_position=unsigned(prefix)+tail;total_tokens=absolute_position+1u;workspace_score_scratch_token_capacity=unsigned(prefix)+1537u;
  calls=events=0;failure=fail;assert(execute()==(fail==0));assert(calls==1u&&segmented==(prefix>=262144u));
  assert(events==(fail==1?0u:1u));
 }
}
'''.replace('BODY',body)
        with tempfile.TemporaryDirectory(prefix='qrt-segment-dispatch-') as temporary:
            path=Path(temporary)/'probe.cpp';path.write_text(source)
            executable=path.with_suffix('')
            built=subprocess.run([os.getenv('CXX','c++'),'-std=c++17','-Wall','-Wextra','-Werror',
                str(path),'-o',str(executable)],capture_output=True,text=True,timeout=30)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(executable)],capture_output=True,text=True,timeout=10)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)


if __name__ == '__main__':
    unittest.main()
