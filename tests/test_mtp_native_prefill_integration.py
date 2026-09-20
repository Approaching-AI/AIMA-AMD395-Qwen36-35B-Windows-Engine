"""Execute actual provider admission and publication around the native probe."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MtpNativePrefillIntegrationTests(unittest.TestCase):
    def test_actual_opt_in_admission_and_failed_publication(self):
        whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        start = whole.index('    const char *mtp_target_trace_prefix =')
        end = whole.index('\n    qrt_qwen36_prefill_descriptor_batch_timing_t preload_timing', start)
        admission = whole[start:end]
        end = whole.index('    mtp_rows_publication.complete();', end) + len('    mtp_rows_publication.complete();')
        start = whole.rfind('    if (auto *batch = qrt_mtp_target_rows::Scope::active) {', 0, end)
        publication = whole[start:end]
        source = r'''
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include "mtp_target_rows.h"
#include "mtp_target_rows_trace.h"
struct Request {const uint32_t* input_tokens;size_t input_token_count;const char* model_dir="actual-model";};
struct Result {bool completed=false;unsigned output_token_count=0;uint32_t output_tokens[1]{};std::string stage;};
struct ScopedQwen36PrefixBatchSuffix {inline static ScopedQwen36PrefixBatchSuffix* active=nullptr;size_t prefix=0;};
size_t g_qwen36_chunked_prefill_total_tokens=0;
bool direct_provider_orchestration=true,fused_layer_stack_provider=true,history=true,weights=true;
bool source_ok=true,probe_ok=true,target_ok=true;unsigned acquired=0,probes=0;
constexpr uint64_t start_ns=0;
bool env_flag_enabled(const char* name){
 if(!std::strcmp(name,"QRT_PREFILL_DESCRIPTOR_BATCH_RESIDENT_MODEL_MTP"))return weights;
 assert(!std::strcmp(name,"QRT_PREFILL_DESCRIPTOR_BATCH_ENABLE_RESIDENT_HISTORY_CARRIER"));return history;
}
void qrt_qwen36_whole_provider_set_failure(Result* out,const std::string& stage,const std::string&,uint64_t){
 out->completed=false;out->stage=stage;
}
std::shared_ptr<int> acquire_qwen36_mtp_model_weight_source(const char* model,std::string* stage,std::string* failure){
 assert(!std::strcmp(model,"actual-model"));++acquired;
 if(!source_ok){*stage="actual_source_failure";*failure="test";return {};}
 return std::make_shared<int>(21);
}
namespace qrt_sm121_mtp_runtime {
bool probe_prefill(const qrt_mtp_target_rows::PrefillRows& batch,std::shared_ptr<int> source,
 const char* prefix,std::string& stage,std::string& failure){
 assert(source&&*source==21&&!std::strcmp(prefix,"native-prefix"));++probes;
 assert(batch.published()&&!batch.first_position()&&!batch.discarded_prefill());
 assert(batch.hidden().size()==batch.rows()*2048u&&batch.shifted_tokens().back()==82u);
 assert(batch.sampled_token()==82u&&batch.shifted_tokens().front()==101u);
 if(!probe_ok){stage="actual_probe_failure";failure="test";return false;}return true;
}
}
int invoke(const Request* request,Result* out_result){
''' + admission + r'''
 if(!target_ok)return 0;
 if(auto* batch=qrt_mtp_target_rows::Scope::active){
  const std::vector<float> hidden(batch->rows()*2048u,1.25f);
  assert(batch->stage(batch->local_rows(),hidden,82u));
 }
 out_result->completed=true;out_result->output_token_count=1;out_result->output_tokens[0]=82u;
''' + publication + r'''
 return 1;
}
int main(){
 assert(!unsetenv("QRT_QWEN36_MTP_TARGET_ROWS_DUMP_PREFIX"));
 assert(!unsetenv("QRT_QWEN36_MTP_NATIVE_PREFILL_PROBE_PREFIX"));
 std::vector<uint32_t> prompt(8192);for(unsigned i=0;i<prompt.size();++i)prompt[i]=100u+i;
 Request request{prompt.data(),prompt.size()};Result result;
 assert(invoke(&request,&result)==1&&result.completed&&result.output_tokens[0]==82u);
 assert(!acquired&&!probes&&!qrt_mtp_target_rows::Scope::active);
 assert(!setenv("QRT_QWEN36_MTP_NATIVE_PREFILL_PROBE_PREFIX","native-prefix",1));
 for(unsigned count:{1u,7168u,8193u,16384u}){
  request.input_token_count=count;assert(!invoke(&request,&result));
  assert(result.stage=="mtp_native_prefill_probe_request"&&!acquired&&!probes);
 }
 request.input_token_count=7169;weights=false;assert(!invoke(&request,&result));weights=true;
 g_qwen36_chunked_prefill_total_tokens=8192;assert(!invoke(&request,&result));g_qwen36_chunked_prefill_total_tokens=0;
 ScopedQwen36PrefixBatchSuffix suffix;ScopedQwen36PrefixBatchSuffix::active=&suffix;
 assert(!invoke(&request,&result));ScopedQwen36PrefixBatchSuffix::active=nullptr;
 for(bool* gate:{&direct_provider_orchestration,&fused_layer_stack_provider,&history}){
  *gate=false;assert(!invoke(&request,&result));assert(result.stage=="mtp_target_rows_request");*gate=true;
 }
 assert(!acquired&&!probes&&!qrt_mtp_target_rows::Scope::active);
 assert(invoke(&request,&result)==1&&acquired==1&&probes==1&&result.output_tokens[0]==82u);
 assert(!qrt_mtp_target_rows::Scope::active);
 for(unsigned failure=0;failure<3;++failure){
  qrt_mtp_target_rows::PrefillRows batch(prompt.data(),7169,0,7169);
  qrt_mtp_target_rows::Scope scope(&batch);
  source_ok=failure!=0;probe_ok=failure!=1;target_ok=failure!=2;
  const auto previous_acquired=acquired,previous_probes=probes;
  assert(!invoke(&request,&result));
  assert(!batch.valid()&&!batch.published()&&batch.hidden().empty()&&batch.shifted_tokens().empty());
  if(failure==0)assert(result.stage=="actual_source_failure"&&acquired==previous_acquired+1&&probes==previous_probes);
  if(failure==1)assert(result.stage=="actual_probe_failure"&&acquired==previous_acquired+1&&probes==previous_probes+1);
  if(failure==2)assert(acquired==previous_acquired&&probes==previous_probes);
 }
 source_ok=probe_ok=target_ok=true;
 request.input_token_count=8192;
 assert(invoke(&request,&result)==1&&result.output_tokens[0]==82u&&!qrt_mtp_target_rows::Scope::active);
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'check.cpp'
            path.write_text(source)
            exe = directory / 'check'
            build = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(ROOT / 'native/providers'),
                str(path), '-o', str(exe)], capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
