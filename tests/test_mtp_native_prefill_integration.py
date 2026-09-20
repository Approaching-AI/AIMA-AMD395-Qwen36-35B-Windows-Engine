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
        start = whole.index('        g_qwen36_resident_session.committed_decode_token_count = expected_count;')
        end = whole.index('        ++workspace.guarded_token_commit_count;', start)
        ordinary_commit = whole[start:end]
        source = r'''
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include "mtp_target_rows.h"
#include "mtp_target_rows_trace.h"
struct Request {const uint32_t* input_tokens;size_t input_token_count;const char* model_dir="actual-model";
 const void* resident_engine=reinterpret_cast<void*>(0x1234);size_t output_token_capacity=32u;};
struct Result {bool completed=false;unsigned output_token_count=0;uint32_t output_tokens[1]{};std::string stage;};
namespace qrt_sm121_mtp {
struct TargetFrontier {const void* owner;uint64_t generation,model_epoch;const uint32_t* processed_inputs;
 size_t processed_count;uint32_t current_token;};
struct RequestCheckpoint {std::shared_ptr<int> retained;};
}
struct Session {
 bool valid=true,provider_completed=true,current_token_valid=true;
 const void* owner_engine=reinterpret_cast<void*>(0x1234);
 uint64_t generation=7u;std::string model_dir="actual-model";size_t prefix_tokens=8192u,committed_decode_token_count=0u;
 uint32_t current_token_id=82u;qrt_sm121_mtp::RequestCheckpoint native_mtp_checkpoint;
 std::vector<uint32_t> native_mtp_processed_inputs;
} g_qwen36_resident_session;
std::recursive_mutex g_qwen36_resident_session_mutex;
struct ScopedQwen36PrefixBatchSuffix {inline static ScopedQwen36PrefixBatchSuffix* active=nullptr;size_t prefix=0;};
size_t g_qwen36_chunked_prefill_total_tokens=0;
bool direct_provider_orchestration=true,fused_layer_stack_provider=true,history=true,weights=true;
bool source_ok=true,probe_ok=true,target_ok=true;unsigned acquired=0,probes=0;
bool request_seed=false,seed_ok=true;unsigned seeds=0;
constexpr uint64_t start_ns=0;
bool env_flag_enabled(const char* name){
 if(!std::strcmp(name,"QRT_QWEN36_MTP_NATIVE_REQUEST_SEED"))return request_seed;
 if(!std::strcmp(name,"QRT_PREFILL_DESCRIPTOR_BATCH_RESIDENT_MODEL_MTP"))return weights;
 assert(!std::strcmp(name,"QRT_PREFILL_DESCRIPTOR_BATCH_ENABLE_RESIDENT_HISTORY_CARRIER"));return history;
}
void qrt_qwen36_whole_provider_set_failure(Result* out,const std::string& stage,const std::string&,uint64_t){
 out->completed=false;out->stage=stage;
}
struct Source {int value=21;uint64_t epoch()const{return 10u;}};
std::shared_ptr<Source> acquire_qwen36_mtp_model_weight_source(const char* model,std::string* stage,std::string* failure){
 assert(!std::strcmp(model,"actual-model"));++acquired;
 if(!source_ok){*stage="actual_source_failure";*failure="test";return {};}
 return std::make_shared<Source>();
}
namespace qrt_sm121_mtp_runtime {
bool probe_prefill(const qrt_mtp_target_rows::PrefillRows& batch,std::shared_ptr<Source> source,
 const char* prefix,std::string& stage,std::string& failure){
 assert(source&&source->value==21&&!std::strcmp(prefix,"native-prefix"));++probes;
 assert(batch.published()&&!batch.first_position()&&!batch.discarded_prefill());
 assert(batch.hidden().size()==batch.rows()*2048u&&batch.shifted_tokens().back()==82u);
 assert(batch.sampled_token()==82u&&batch.shifted_tokens().front()==101u);
 if(!probe_ok){stage="actual_probe_failure";failure="test";return false;}return true;
}
bool probe_prefill_request(const qrt_mtp_target_rows::PrefillRows& batch,std::shared_ptr<Source> source,
 const qrt_sm121_mtp::TargetFrontier& actual,unsigned capacity,const char* prefix,
 qrt_sm121_mtp::RequestCheckpoint* output,std::string& stage,std::string& failure){
 ++seeds;assert(actual.owner==g_qwen36_resident_session.owner_engine&&actual.generation==7u&&actual.model_epoch==10u);
 assert(actual.current_token==82u&&actual.processed_count==batch.rows()&&capacity==batch.rows()+32u);
 assert(batch.matches_input(actual.processed_inputs,actual.processed_count));
 if(!probe_prefill(batch,source,prefix,stage,failure))return false;
 if(!seed_ok){stage="actual_seed_failure";failure="test";return false;}
 output->retained=std::make_shared<int>(99);return true;
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
 request_seed=true;assert(!invoke(&request,&result)&&result.stage=="mtp_native_request_seed_request");
 request_seed=false;
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
 request_seed=true;
 auto& session=g_qwen36_resident_session;
 for(unsigned invalid=0;invalid<10u;++invalid){
  const auto saved=session;
  switch(invalid){
   case 0:session.valid=false;break;case 1:session.provider_completed=false;break;
   case 2:session.current_token_valid=false;break;case 3:session.owner_engine=nullptr;break;
   case 4:session.generation=0u;break;case 5:session.model_dir="different";break;
   case 6:session.prefix_tokens--;break;case 7:session.committed_decode_token_count=1u;break;
   case 8:session.current_token_id++;break;case 9:request.resident_engine=nullptr;break;
  }
  const auto count=seeds;assert(!invoke(&request,&result)&&seeds==count);
  assert(result.stage==(invalid==9u?"mtp_native_request_seed_request":"mtp_native_request_seed_frontier"));
  session=saved;request.resident_engine=session.owner_engine;
 }
 seed_ok=false;assert(!invoke(&request,&result)&&result.stage=="actual_seed_failure");
 assert(!session.native_mtp_checkpoint.retained&&session.native_mtp_processed_inputs.empty());
 seed_ok=true;assert(invoke(&request,&result)==1);
 assert(session.native_mtp_checkpoint.retained&&*session.native_mtp_checkpoint.retained==99);
 assert(session.native_mtp_processed_inputs==prompt);
 // A later trace failure must not publish the newly prepared native seed.
 const auto saved_checkpoint=session.native_mtp_checkpoint.retained;
 assert(!setenv("QRT_QWEN36_MTP_TARGET_ROWS_DUMP_PREFIX","/nonexistent-native-mtp-parent/trace",1));
 assert(!invoke(&request,&result)&&result.stage=="mtp_target_rows_trace");
 assert(session.native_mtp_checkpoint.retained==saved_checkpoint&&session.native_mtp_processed_inputs==prompt);
 assert(!unsetenv("QRT_QWEN36_MTP_TARGET_ROWS_DUMP_PREFIX"));
 const size_t expected_count=1u;const uint32_t next_token_id=83u;
''' + ordinary_commit + r'''
 assert(!session.native_mtp_checkpoint.retained&&session.native_mtp_processed_inputs.empty());
 assert(session.committed_decode_token_count==1u&&session.current_token_id==83u);
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
