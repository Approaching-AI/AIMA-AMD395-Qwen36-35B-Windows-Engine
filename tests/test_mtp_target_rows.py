"""Actual target-row ownership and unchanged LM-head row consumption."""
from pathlib import Path
import json
import struct
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class MtpTargetRowsTests(unittest.TestCase):
    def compile_run(self, source, directory):
        executable = directory / 'probe'
        build = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
            '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-pthread',
            '-I', str(ROOT / 'native/providers'), str(source), '-o', str(executable)],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(build.returncode, 0, build.stderr)
        result = subprocess.run([str(executable), str(directory)],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_owner_failure_lifetime_bounds_rounding_and_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.compile_run(ROOT / 'tests/native/mtp_target_rows_host.cpp', directory)
            self.assertEqual((directory / 'actual.hidden.bf16.bin').read_bytes(),
                             struct.pack('<H', 0xc000) * (2 * 2048))
            self.assertEqual((directory / 'actual.shifted.u32.bin').read_bytes(), struct.pack('<II', 11, 701))
            metadata = json.loads((directory / 'actual.json').read_text())
            self.assertEqual(metadata['rows'], 2)
            self.assertEqual(metadata['first_position'], 0)
            self.assertEqual(metadata['sampled_token'], 701)
            self.assertFalse(metadata['discarded_prefill'])
            self.assertFalse(metadata['mtp_inference_enabled'])
            self.assertFalse(metadata['numerical_acceptance_claimed'])
            self.assertEqual((directory / 'occupied.json').read_text(), 'original')

    def test_actual_provider_selection_and_prefix_terminal_consumer(self):
        whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        functions = '\n'.join(function(whole, signature) for signature in (
            'bool select_qwen36_mtp_lm_head_input(', 'bool capture_qwen36_prefix_batch_terminal('))
        source = r'''
#include "mtp_target_rows.h"
#include <array>
#include <cassert>
#include <cmath>
#include <string>
constexpr size_t QRT_QWEN36_HIDDEN_SIZE=2048;
constexpr uint32_t QRT_QWEN36_VOCAB_SIZE=248320;
struct Layer1InputRmsnormRun {
 std::string name,stage;std::vector<unsigned> selected_token_ids;std::vector<float> gpu_output;
 size_t selected_token_count=0,output_elements=0,output_bytes=0;
 uint64_t selected_token_ids_hash=0,gpu_output_hash=0;bool correctness_pass=false;
};
struct LmHeadRun {std::vector<uint32_t> gpu_topk_ids;std::vector<float> gpu_topk_logits;
 bool correctness_pass=false;std::string failure_stage,failure;};
struct ScopedQwen36PrefixBatchSuffix {
 inline static ScopedQwen36PrefixBatchSuffix *active=nullptr;
 unsigned tokens=3;bool terminal_only=false,terminal_valid=false;
 std::array<uint32_t,2> terminal_ids{};std::array<float,2> terminal_logits{};
};
uint64_t qrt_fnv1a64_bytes(const void*,size_t){return 1;}
uint64_t qrt_fnv1a64_f32(const float*,size_t){return 2;}
unsigned projection_calls=0;
bool run_lm_head(const Layer1InputRmsnormRun& input,const std::string&,unsigned tokens,LmHeadRun* head){
 assert(tokens==3 && input.correctness_pass);
 assert(input.selected_token_count==1 && input.selected_token_ids==std::vector<unsigned>{2});
 assert(input.gpu_output.size()==2048 && input.gpu_output.front()==3 && input.gpu_output.back()==3);
 ++projection_calls;head->gpu_topk_ids={701,702};head->gpu_topk_logits={10,9};head->correctness_pass=true;return true;
}
''' + functions + r'''
int main(){
 Layer1InputRmsnormRun source;
 source.selected_token_ids={0,1,2};source.selected_token_count=3;
 source.output_elements=3*2048;source.output_bytes=source.output_elements*4;
 source.gpu_output.resize(source.output_elements);source.correctness_pass=true;
 for(size_t row=0;row<3;++row)std::fill_n(source.gpu_output.data()+row*2048,2048,float(row+1));
 const auto original=source.gpu_output;
 std::string stage,failure;
 for(bool terminal_only:{false,true}){
  ScopedQwen36PrefixBatchSuffix suffix;suffix.terminal_only=terminal_only;
  ScopedQwen36PrefixBatchSuffix::active=&suffix;
  const std::vector<unsigned> positions=terminal_only?std::vector<unsigned>{2}:std::vector<unsigned>{0,1,2};
  Layer1InputRmsnormRun head_input;
  assert(select_qwen36_mtp_lm_head_input(source,positions,&head_input,&stage,&failure));
  assert(head_input.selected_token_ids==positions && head_input.selected_token_count==positions.size());
  assert(head_input.gpu_output.size()==positions.size()*2048 && head_input.correctness_pass);
  assert(capture_qwen36_prefix_batch_terminal(head_input,"real-model",3,&stage,&failure));
  assert(suffix.terminal_valid && suffix.terminal_ids[0]==701 && suffix.terminal_logits[0]==10);
  assert(!capture_qwen36_prefix_batch_terminal(head_input,"real-model",3,&stage,&failure));
 }
 ScopedQwen36PrefixBatchSuffix::active=nullptr;
 assert(projection_calls==2 && source.gpu_output==original);
 Layer1InputRmsnormRun selected;
 assert(!select_qwen36_mtp_lm_head_input(source,{3},&selected,&stage,&failure));
 assert(!selected.correctness_pass);
 assert(select_qwen36_mtp_lm_head_input(source,{2},&selected,&stage,&failure));
 assert(selected.correctness_pass && selected.gpu_output.size()==2048);
 source.correctness_pass=false;
 assert(!select_qwen36_mtp_lm_head_input(source,{2},&selected,&stage,&failure));
 assert(!selected.correctness_pass && selected.gpu_output.empty() && selected.selected_token_ids.empty());
 source.correctness_pass=true;
 assert(!select_qwen36_mtp_lm_head_input(source,{2},&source,&stage,&failure));
 assert(source.gpu_output==original && source.correctness_pass);
 source.correctness_pass=true;source.output_bytes--;
 assert(!select_qwen36_mtp_lm_head_input(source,{2},&selected,&stage,&failure));
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'provider.cpp'
            path.write_text(source)
            self.compile_run(path, directory)

    def test_actual_last_layer_liveness_keeps_all_mtp_consumers(self):
        whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        header = (ROOT / 'native/providers/final_query_liveness.h').read_text()
        begin = whole.index('            const qrt_final_query_liveness::Scope final_query_scope(')
        statement = whole[begin:whole.index(';', begin) + 1]
        scope = function(header, 'struct Scope {') + ';'
        source = r'''
#include "mtp_target_rows.h"
#include <cassert>
#include <cstring>
namespace qrt_final_query_liveness {inline thread_local bool active=false;
''' + scope + r'''
}
bool enabled=true,checkpoints=false,g_qwen36_mtp_tensor_namespace_active=false;
size_t g_qwen36_chunked_prefill_total_tokens=0;
bool env_flag_enabled(const char* name){assert(!std::strcmp(name,"QRT_QWEN36_FINAL_QUERY_LIVENESS"));return enabled;}
bool raw_env_flag_enabled(const char* name){assert(!std::strcmp(name,"QRT_QWEN36_PREFIX_CHECKPOINTS"));return checkpoints;}
struct ScopedQwen36PrefixBatchSuffix {inline static void* active=nullptr;};
constexpr unsigned kRetainedPrefillTokens=8192,kDescriptorBatchFinalLayer=39;
bool choose(unsigned layer,unsigned prefill_tokens,bool explicit_final_targets_are_q1=true,bool final_full_prefix_attention=true){
 struct {unsigned full_attention_layer;} segment{layer};
''' + statement + r'''
 return qrt_final_query_liveness::active;
}
int main(){
 assert(choose(39,8192));assert(!qrt_final_query_liveness::active);
 assert(!choose(35,8192)&&!choose(39,7169)&&!choose(39,8192,false)&&!choose(39,8192,true,false));
 std::vector<uint32_t> prompt(8192,23);
 qrt_mtp_target_rows::PrefillRows batch(prompt.data(),prompt.size(),0,prompt.size());
 {
  qrt_mtp_target_rows::Scope mtp(&batch);
  // Every prompt row is now a real MTP consumer, including rows before 8191.
  assert(!choose(39,8192));
  {qrt_mtp_target_rows::Scope suspended(nullptr);assert(choose(39,8192));}
  assert(!choose(39,8192));
 }
 assert(choose(39,8192));
 checkpoints=true;assert(!choose(39,8192));checkpoints=false;
 g_qwen36_chunked_prefill_total_tokens=16384;assert(!choose(39,8192));g_qwen36_chunked_prefill_total_tokens=0;
 ScopedQwen36PrefixBatchSuffix::active=&batch;assert(!choose(39,8192));ScopedQwen36PrefixBatchSuffix::active=nullptr;
 g_qwen36_mtp_tensor_namespace_active=true;assert(!choose(39,8192));g_qwen36_mtp_tensor_namespace_active=false;
 enabled=false;assert(!choose(39,8192));
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'liveness.cpp'
            path.write_text(source)
            self.compile_run(path, directory)


if __name__ == '__main__':
    unittest.main()
