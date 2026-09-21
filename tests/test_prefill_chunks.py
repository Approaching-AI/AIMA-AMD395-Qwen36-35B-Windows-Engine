"""Execute actual chunk KV ownership with original bytes and failed copies."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import attention_capacity, function

ROOT = Path(__file__).resolve().parents[1]


class PrefillChunkTests(unittest.TestCase):
    def test_coordinator_positions_single_callback_and_failed_replacement(self):
        chunk = (ROOT / 'native/providers/prefill_chunks.h').read_text()
        whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        coordinator = function(chunk, 'int run_qwen36_chunked_prefill(')
        pool = '\n'.join(function(whole, name) + ';' for name in (
            'struct DescriptorDeviceAllocationPoolBlock',
            'struct DescriptorDeviceAllocationPoolStats'))
        pool += r'''
bool g_descriptor_product_reuse_device_allocations=false;
std::vector<DescriptorDeviceAllocationPoolBlock> g_descriptor_device_allocation_pool;
std::mutex g_descriptor_device_allocation_pool_mutex;
DescriptorDeviceAllocationPoolStats g_descriptor_device_allocation_pool_stats;
'''
        pool += '\n'.join(function(whole, name) for name in (
            'void reset_descriptor_device_allocation_pool_stats(',
            'uint64_t descriptor_device_allocation_pool_cached_bytes(',
            'void release_descriptor_device_allocation_pool('))
        pool += function(whole, 'struct ScopedDescriptorProductDeviceAllocationReuse') + ';'
        pool += '\n'.join(function(whole, name) for name in (
            'hipError_t qrt_descriptor_device_malloc(', 'void free_device('))
        source = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "native/src/qrt.h"
#include "native/providers/mtp_target_rows.h"
''' + attention_capacity() + r'''
enum hipError_t{hipSuccess,hipErrorInvalidValue,hipErrorOutOfMemory};
const char* hipGetErrorString(hipError_t){return "injected";}
std::set<void*> allocations;
unsigned allocation_attempts=0,allocation_count=0,free_count=0,fail_allocation=0;
hipError_t hipMalloc(void** p,size_t bytes){
 *p=nullptr;if(++allocation_attempts==fail_allocation)return hipErrorOutOfMemory;
 *p=std::malloc(bytes);assert(*p&&allocations.insert(*p).second);++allocation_count;return hipSuccess;
}
hipError_t hipFree(void* p){assert(p&&allocations.erase(p)==1);++free_count;std::free(p);return hipSuccess;}
bool q1_terminal_device_corridor_workspace_contains(const void*){return false;}
bool qwen36_resident_decode_activation_workspace_contains(const void*){return false;}
bool whole_repeated_layer_fixed_weight_contains(const void*){return false;}
constexpr size_t kQwen36ResidentDecodeTailCapacityTokens=1536;
struct Linear{bool valid=true;size_t prefix_tokens=8192,decode_qkv_token_count=0,decode_recurrent_token_count=0;};
struct Attention{
 size_t decode_tail_k_bytes=1536*1024,decode_tail_v_bytes=1536*1024;
 size_t prefill_reserved_tokens=0,history_tokens=8192,k_bytes=8192*1024,v_bytes=8192*1024;
};
struct Workspace{size_t full_attention_score_scratch_token_capacity=0;};
namespace qrt_sm121_mtp {
struct TargetFrontier{const void* owner;uint64_t generation,model_epoch;const uint32_t* processed_inputs;size_t processed_count;uint32_t current_token;};
struct RequestCheckpoint{std::shared_ptr<int> retained;};
struct PromptStep{hipError_t status;const char* stage;unsigned retained_tokens;bool completion_unknown;};
}
struct Session{
 bool valid=false,current_token_valid=false,last_decode_top2_valid=false,mtp_target_hidden_valid=false;
 size_t prefix_tokens=0,committed_decode_token_count=0,last_decode_top2_position=0;
 uint64_t prompt_token_ids_fnv1a64=0,generation=99,full_attention_decode_tail_bytes=10*1536*2048;
 uint64_t full_attention_kv_bytes=10*8192*2048;
 const void* owner_engine=nullptr;uint32_t current_token_id=0;
 qrt_sm121_mtp::RequestCheckpoint native_mtp_checkpoint;std::vector<uint32_t> native_mtp_processed_inputs;
 std::array<Linear,40> linear_layers{};std::array<Attention,40> full_attention_layers{};Workspace activation_workspace;
} g_qwen36_resident_session;
std::recursive_mutex g_qwen36_resident_session_mutex;
size_t g_qwen36_chunked_prefill_total_tokens=0;
struct ScopedQwen36PrefixBatchSuffix{inline static void* active=nullptr;};
uint64_t clock_ns=0;
uint64_t qrt_now_ns(){return ++clock_ns;}
uint64_t qrt_elapsed_ns(uint64_t a,uint64_t b){return b-a;}
''' + pool + r'''
uint64_t qrt_fnv1a64_bytes(const void* p,size_t n){
 uint64_t h=1469598103934665603ULL;auto* bytes=static_cast<const unsigned char*>(p);
 for(size_t i=0;i<n;++i){h^=bytes[i];h*=1099511628211ULL;}return h;
}
uint64_t qrt_fnv1a64_update_bytes(uint64_t h,const void* p,size_t n){return h^qrt_fnv1a64_bytes(p,n);}
bool raw_env_flag_enabled(const char*){return false;}
bool native_mtp=false,g_qwen36_resident_completion_unknown=false;
bool env_flag_enabled(const char* name){return std::strcmp(name,"QRT_QWEN36_MTP_NATIVE_DECODE")||native_mtp;}
bool qwen36_resident_decode_activation_workspace_layout_valid(const Workspace&){return true;}
unsigned seeds=0,suffixes=0,single_tails=0,callbacks=0,releases=0,fail_suffix=0,reservations=0,fail_reservation=0;
bool fail_seed=false,cancel=false,bad_counter=false,bad_handoff=false;
unsigned bad_seed_identity=0;
unsigned throw_suffix=0;
const uint32_t* actual_prompt=nullptr;size_t requested_total=0;
unsigned mtp_calls=0,fail_mtp_at=0;bool unknown_mtp=false,mtp_source_ok=true;
struct MtpSource{uint64_t epoch()const{return 5u;}};
std::shared_ptr<MtpSource> acquire_qwen36_mtp_model_weight_source(const char*,std::string* stage,std::string* failure){
 if(!mtp_source_ok){*stage="injected_mtp_source";*failure="failed";return {};}
 return std::make_shared<MtpSource>();
}
namespace qrt_sm121_mtp_runtime {
class ChunkedPrefillSeed{
 unsigned processed_=0;uint64_t epoch_=0;
 qrt_sm121_mtp::PromptStep step(){
  if(++mtp_calls==fail_mtp_at)return {hipErrorInvalidValue,"injected_mtp_seed",processed_,unknown_mtp};
  return {hipSuccess,"complete",processed_,false};
 }
public:
 uint64_t epoch()const{return epoch_;}
 qrt_sm121_mtp::PromptStep begin(const qrt_mtp_target_rows::PrefillRows& batch,std::shared_ptr<MtpSource> source,
  const qrt_sm121_mtp::TargetFrontier& actual,unsigned capacity){
  assert(!processed_&&source&&actual.model_epoch==5u&&capacity==std::min(size_t(262144u),requested_total+512u));
  epoch_=5u;return append(batch,actual);
 }
 qrt_sm121_mtp::PromptStep append(const qrt_mtp_target_rows::PrefillRows& batch,const qrt_sm121_mtp::TargetFrontier& actual){
  assert(batch.published()&&batch.first_position()==processed_&&batch.prompt_tokens()==requested_total);
  assert(actual.owner==g_qwen36_resident_session.owner_engine&&actual.generation==99u&&actual.model_epoch==epoch_);
  assert(actual.processed_count==processed_+batch.rows()&&actual.processed_count==g_qwen36_resident_session.prefix_tokens);
  assert(std::equal(actual_prompt,actual_prompt+actual.processed_count,actual.processed_inputs));
  assert(actual.current_token==batch.sampled_token()&&batch.hidden().size()==batch.rows()*2048u);
  assert(batch.shifted_tokens().back()==(batch.discarded_prefill()?actual_prompt[requested_total-1u]:42u));
  assert(!g_qwen36_resident_session.native_mtp_checkpoint.retained&&g_qwen36_resident_session.native_mtp_processed_inputs.empty());
  processed_+=batch.rows();return step();
 }
 qrt_sm121_mtp::PromptStep save(qrt_sm121_mtp::RequestCheckpoint* output,const qrt_sm121_mtp::TargetFrontier& actual){
  assert(processed_==requested_total&&actual.processed_count==requested_total&&actual.current_token==42u);
  auto result=step();if(result.status==hipSuccess)output->retained=std::make_shared<int>(42);return result;
 }
};
}
void capture_mtp_chunk(uint32_t sample){
 auto* batch=qrt_mtp_target_rows::Scope::active;
 assert(bool(batch)==native_mtp);
 if(batch){std::vector<float> hidden(batch->rows()*2048u,1.25f);
  assert(batch->stage(batch->local_rows(),hidden,sample)&&batch->publish(sample));}
}
void qrt_qwen36_whole_provider_set_failure(qrt_qwen36_whole_provider_result_t* r,const std::string& stage,const std::string&,uint64_t start){
 r->completed=0;std::strncpy(r->failure_stage,stage.c_str(),sizeof(r->failure_stage)-1);r->wall_clock_ns=qrt_elapsed_ns(start,qrt_now_ns());
}
bool release_qwen36_resident_session_locked(){++releases;g_qwen36_resident_session=Session{};return true;}
int qrt_qwen36_whole_provider_prefill_v1(const qrt_qwen36_whole_provider_request_t* r,qrt_qwen36_whole_provider_result_t* out){
 assert(!g_descriptor_product_reuse_device_allocations&&allocations.empty());
 ++seeds;assert(r->input_tokens==actual_prompt&&r->input_token_count==8192&&r->output_token_capacity==1);
 assert(!r->prefill_emit_callback&&(r->flags&QRT_QWEN36_WHOLE_PROVIDER_FLAG_PREFIX_SEED_CAPTURE));
 auto& s=g_qwen36_resident_session;s=Session{};s.valid=true;s.prefix_tokens=8192;
 s.owner_engine=r->resident_engine;s.current_token_id=999u;s.current_token_valid=true;
 if(bad_seed_identity==1u)s.owner_engine=nullptr;
 if(bad_seed_identity==2u)s.current_token_valid=false;
 if(bad_seed_identity==3u)s.current_token_id++;
 s.prompt_token_ids_fnv1a64=r->expected_prompt_token_ids_fnv1a64;
 s.activation_workspace.full_attention_score_scratch_token_capacity=g_qwen36_chunked_prefill_total_tokens+1537;
 out->completed=1;out->output_token_count=1;out->output_tokens[0]=999;
 if(!fail_seed)capture_mtp_chunk(999u);
 return !fail_seed;
}
hipError_t resize_qwen36_prefill_chunk_tail(Attention& a,size_t count){
 assert(std::none_of(g_descriptor_device_allocation_pool.begin(),g_descriptor_device_allocation_pool.end(),
  [](const auto& block){return block.in_use;}));
 if(g_descriptor_product_reuse_device_allocations)assert(count==1536u);
 a.decode_tail_k_bytes=a.decode_tail_v_bytes=count*1024;return hipSuccess;
}
hipError_t reserve_qwen36_prefill_chunk_attention(Attention& a,size_t count){
 assert(!g_descriptor_product_reuse_device_allocations&&allocations.empty());
 assert(count==requested_total&&a.history_tokens==8192&&!a.prefill_reserved_tokens);
 if(++reservations==fail_reservation)return hipErrorInvalidValue;
 a.prefill_reserved_tokens=count;return hipSuccess;
}
hipError_t promote_qwen36_prefill_chunk_attention(Attention& a,size_t prefix,size_t count){
 assert(g_descriptor_product_reuse_device_allocations);
 assert(std::none_of(g_descriptor_device_allocation_pool.begin(),g_descriptor_device_allocation_pool.end(),
  [](const auto& block){return block.in_use;}));
 assert(a.history_tokens==prefix&&prefix+count<=a.prefill_reserved_tokens);
 a.history_tokens=prefix+count;a.k_bytes=a.v_bytes=a.history_tokens*1024;
 if(a.history_tokens==a.prefill_reserved_tokens)a.prefill_reserved_tokens=0;
 return hipSuccess;
}
bool mock_completed_suffix(const qrt_qwen36_whole_provider_prefix_request_v1_t& r,uint32_t* teachers,
 std::string* stage,std::string* failure,bool terminal_only,qrt_qwen36_whole_provider_result_t* out){
 ++suffixes;auto& s=g_qwen36_resident_session;
 assert(!teachers&&terminal_only&&r.expected_prefix_token_count==s.prefix_tokens);
 assert(r.suffix_tokens==actual_prompt+s.prefix_tokens&&r.suffix_token_count==std::min(size_t(8192),requested_total-s.prefix_tokens));
 // Actual nested allocator scope must preserve the caller's chunk lease. The
 // same two temporary spans serve every complete chunk and the actual tail.
 ScopedDescriptorProductDeviceAllocationReuse inner(true);assert(!inner.owns_scope);
 struct Handoff{void* a=nullptr;void* b=nullptr;~Handoff(){free_device(b);free_device(a);}} handoff;
 if(qrt_descriptor_device_malloc(&handoff.a,size_t(r.suffix_token_count)*8u)!=hipSuccess||
    qrt_descriptor_device_malloc(&handoff.b,size_t(r.suffix_token_count)*4u)!=hipSuccess){
  *stage="injected_allocation";*failure="failed";return false;
 }
 assert(allocation_count==2&&allocations.size()==2);
 std::memset(handoff.a,int(suffixes&255u),size_t(r.suffix_token_count)*8u);
 std::memset(handoff.b,int((suffixes+1u)&255u),size_t(r.suffix_token_count)*4u);
 if(suffixes==throw_suffix)throw std::runtime_error("injected chunk exception");
 if(suffixes==fail_suffix){*stage="injected_suffix";*failure="failed";return false;}
 if(bad_handoff)handoff.b=nullptr;
 for(unsigned i=0;i<40;++i)if(i%4!=3){auto& l=s.linear_layers[i];l.decode_qkv_token_count=l.decode_recurrent_token_count=r.suffix_token_count;}
 if(bad_counter)++s.linear_layers[12].decode_qkv_token_count;
 s.committed_decode_token_count=r.suffix_token_count;s.current_token_valid=s.last_decode_top2_valid=true;
 s.current_token_id=42u;capture_mtp_chunk(42u);
 s.last_decode_top2_position=s.prefix_tokens+r.suffix_token_count-1;
 out->completed=1;out->output_token_count=1;out->output_tokens[0]=42;out->continuation.output_token_emitted=1;
 out->continuation.output_token_id=42;out->continuation.output_logit=5.5;out->output_tokens_fnv1a64=234;
 return true;
}
bool run_qwen36_resident_batch_suffix(const qrt_qwen36_whole_provider_prefix_request_v1_t& r,uint32_t* teachers,
 std::string* stage,std::string* failure,bool terminal_only,qrt_qwen36_whole_provider_result_t* out){
 assert(r.suffix_token_count>1u);return mock_completed_suffix(r,teachers,stage,failure,terminal_only,out);
}
bool run_qwen36_prefill_single_tail(const qrt_qwen36_whole_provider_prefix_request_v1_t& r,
 std::string* stage,std::string* failure,qrt_qwen36_whole_provider_result_t* out){
 assert(r.suffix_token_count==1u&&r.expected_prefix_token_count+1u==requested_total);
 for(unsigned i=3;i<40;i+=4)assert(g_qwen36_resident_session.full_attention_layers[i].decode_tail_k_bytes==1536u*1024u);
 ++single_tails;return mock_completed_suffix(r,nullptr,stage,failure,true,out);
}
int QRT_CDECL emit(void*,uint32_t token,uint64_t elapsed){
 assert(!g_descriptor_product_reuse_device_allocations&&allocations.empty());
 ++callbacks;assert(token==42&&elapsed>1&&g_qwen36_resident_session.prefix_tokens==requested_total);
 if(native_mtp){assert(g_qwen36_resident_session.native_mtp_checkpoint.retained);
  assert(g_qwen36_resident_session.native_mtp_processed_inputs.size()==requested_total&&
   std::equal(actual_prompt,actual_prompt+requested_total,g_qwen36_resident_session.native_mtp_processed_inputs.begin()));}
 assert(!g_qwen36_resident_session.committed_decode_token_count);return !cancel;
}
''' + coordinator + r'''
int main(){
 std::vector<uint32_t> prompt(qrt_sm121_attention_capacity::kTokens+1024u);
 for(size_t i=0;i<prompt.size();++i)prompt[i]=uint32_t(i%245000);
 actual_prompt=prompt.data();auto result=std::make_unique<qrt_qwen36_whole_provider_result_t>();
 auto run=[&](size_t total){
 requested_total=total;seeds=suffixes=single_tails=callbacks=releases=reservations=0;*result={};result->preload_wall_clock_ns=789;
 mtp_calls=0;g_qwen36_resident_completion_unknown=false;
  assert(!g_descriptor_product_reuse_device_allocations&&allocations.empty()&&g_descriptor_device_allocation_pool.empty());
  allocation_attempts=allocation_count=free_count=0;
  qrt_qwen36_whole_provider_request_t r{};r.resident_engine=reinterpret_cast<qrt_engine_t*>(uintptr_t(16));
  r.input_tokens=prompt.data();r.input_token_count=total;r.output_token_capacity=512;r.prefill_emit_callback=emit;
  r.model_dir="actual-model";
  r.expected_prompt_token_ids_fnv1a64=qrt_fnv1a64_bytes(prompt.data(),total*4);
  const int ok=run_qwen36_chunked_prefill(r,result.get(),qrt_now_ns());
  assert(!g_qwen36_chunked_prefill_total_tokens&&!g_descriptor_product_reuse_device_allocations);
  assert(allocations.empty()&&g_descriptor_device_allocation_pool.empty()&&allocation_count==free_count);return ok;
 };
 for(size_t total:{8193u,8194u,8195u,8196u,8255u,8256u,8257u,9215u,9216u,9217u,
                  16383u,16384u,16385u,17408u,32768u,65536u,66560u,122880u,123904u,
                  131072u,132096u,262140u,262142u,262143u,262144u,263168u}){
  assert(run(total)==1&&callbacks==1&&seeds==1&&suffixes==(total+8191)/8192-1&&!releases);
  assert(reservations==10);
  assert(single_tails==(total%8192u==1u?1u:0u));
  assert(allocation_count==2&&g_descriptor_device_allocation_pool_stats.request_count==suffixes*2u);
  assert(g_descriptor_device_allocation_pool_stats.new_allocation_count==2);
  assert(g_descriptor_device_allocation_pool_stats.reuse_count==(suffixes-1u)*2u);
  assert(g_descriptor_device_allocation_pool_stats.release_count==suffixes*2u);
  assert(result->resident_session_valid&&result->resident_session_prefix_token_count==total);
  assert(result->resident_session_generation==99&&result->output_tokens[0]==42&&result->output_token_capacity==512);
  assert(result->preload_wall_clock_ns==789&&result->prefill_emit_completed&&!result->prefill_emit_rejected);
  assert(result->prompt_token_ids_fnv1a64==qrt_fnv1a64_bytes(prompt.data(),total*4));
 }
 fail_seed=true;assert(!run(16384)&&!callbacks&&releases==1&&!g_qwen36_resident_session.valid);fail_seed=false;
 for(unsigned failure:{1u,5u,10u}){
  fail_reservation=failure;
  assert(!run(32768)&&reservations==failure&&!suffixes&&!callbacks&&releases==1&&!g_qwen36_resident_session.valid);
 }
 fail_reservation=0;
 for(unsigned failure:{1u,2u}){fail_suffix=failure;assert(!run(32768)&&!callbacks&&releases==1&&!g_qwen36_resident_session.valid);}
 fail_suffix=0;bad_counter=true;assert(!run(16384)&&!callbacks&&releases==1);bad_counter=false;
 // A failure after crossing the old 64k limit must discard every partially
 // advanced owner and must not publish an intermediate callback.
 fail_suffix=8u;assert(!run(66560)&&!callbacks&&releases==1&&!g_qwen36_resident_session.valid);
 fail_suffix=0;
 fail_suffix=16u;assert(!run(132096)&&!callbacks&&releases==1&&!g_qwen36_resident_session.valid);
 fail_suffix=32u;assert(!run(263168)&&!callbacks&&releases==1&&!g_qwen36_resident_session.valid);
 fail_suffix=0;
 for(unsigned failure:{1u,2u}){
  fail_allocation=failure;assert(!run(32768)&&!callbacks&&releases==1&&!g_qwen36_resident_session.valid);
  assert(allocation_count==failure-1u&&!std::strcmp(result->failure_stage,"injected_allocation"));
 }
 fail_allocation=0;
 for(unsigned failure:{1u,8u,16u,32u}){
  throw_suffix=failure;assert(!run(263168)&&!callbacks&&releases==1&&!g_qwen36_resident_session.valid);
  assert(!std::strcmp(result->failure_stage,"qwen36_chunked_prefill_exception"));
 }
 throw_suffix=0;bad_handoff=true;
 assert(!run(32768)&&suffixes==1&&!callbacks&&releases==1&&!g_qwen36_resident_session.valid);
 assert(!std::strcmp(result->failure_stage,"qwen36_chunked_prefill_scratch_handoff"));bad_handoff=false;
 cancel=true;assert(!run(17408)&&callbacks==1&&releases==1&&result->prefill_emit_rejected&&!result->resident_session_valid);cancel=false;
 assert(!run(8192)&&!seeds&&!callbacks&&!releases);
 assert(!run(8191)&&!seeds&&!callbacks&&!releases);
 assert(!run(qrt_sm121_attention_capacity::kTokens+1024u)&&!seeds&&!callbacks&&!releases);
 assert(run(16384)&&callbacks==1&&result->completed);
 native_mtp=true;
 for(size_t total:{8193u,8194u,8195u,8257u,16383u,16384u,16385u,17408u,
                  65536u,131072u,262140u,262142u,262143u}){
  assert(run(total)&&callbacks==1&&mtp_calls==(total+8191u)/8192u+1u);
  assert(single_tails==(total%8192u==1u?1u:0u));
  assert(g_qwen36_resident_session.native_mtp_checkpoint.retained&&result->resident_session_valid);
 }
 for(unsigned at:{1u,2u,3u,4u})for(bool unknown:{false,true}){
  fail_mtp_at=at;unknown_mtp=unknown;
  assert(!run(17408u)&&mtp_calls==at&&!callbacks&&releases==1&&!result->resident_session_valid);
  assert(g_qwen36_resident_completion_unknown==unknown);
 }
 fail_mtp_at=0;unknown_mtp=false;mtp_source_ok=false;
 assert(!run(16384u)&&!mtp_calls&&!callbacks&&releases==1);mtp_source_ok=true;
 for(unsigned invalid:{1u,2u,3u}){
  bad_seed_identity=invalid;assert(!run(16384u)&&!mtp_calls&&!callbacks&&releases==1);
  assert(!std::strcmp(result->failure_stage,"mtp_chunked_prefill_seed_frontier"));
 }
 bad_seed_identity=0;
 // The last short chunk can fail before or after private MTP production.
 for(unsigned at:{2u,3u})for(bool unknown:{false,true}){
  fail_mtp_at=at;unknown_mtp=unknown;
  assert(!run(8193u)&&mtp_calls==at&&!callbacks&&releases==1&&!result->resident_session_valid);
  assert(g_qwen36_resident_completion_unknown==unknown);
 }
 fail_mtp_at=0;unknown_mtp=false;
 cancel=true;assert(!run(17408u)&&callbacks==1&&releases==1);cancel=false;
 assert(!run(262144u)&&!seeds&&!mtp_calls&&!callbacks&&!releases);
 assert(run(16384u)&&callbacks==1&&result->completed);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / 'coordinator')
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            '-I', str(ROOT), '-x', 'c++', '-', '-o', exe],
                           input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=20, capture_output=True)

    def test_kv_promotion_tail_resize_and_failure_atomicity(self):
        whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        chunk = (ROOT / 'native/providers/prefill_chunks.h').read_text()
        declarations = '\n'.join(function(whole, name) + ';' for name in (
            'enum class Qwen36ResidentSessionElementKind',
            'struct Qwen36ResidentSessionFullAttentionLayer'))
        operations = '\n'.join(function(chunk, name) for name in (
            'hipError_t reserve_qwen36_prefill_chunk_attention(',
            'hipError_t resize_qwen36_prefill_chunk_tail(',
            'hipError_t promote_qwen36_prefill_chunk_attention('))
        source = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorOutOfMemory,hipErrorUnknown};
constexpr int hipMemcpyDeviceToDevice=1;
std::set<void*> live;
bool fail_allocate=false;
void* fail_free=nullptr;
unsigned copies=0,fail_copy=0;
hipError_t hipMalloc(void** out,size_t bytes) {
 if(fail_allocate){*out=nullptr;return hipErrorOutOfMemory;}
 *out=std::malloc(bytes);assert(*out&&live.insert(*out).second);return hipSuccess;
}
''' + function(whole, 'hipError_t qrt_unpooled_device_malloc(') + r'''
// Match the production inclusion environment: plain hipMalloc after the macro
// belongs to transient scratch. Persistent owner operations must bypass it.
hipError_t forbidden_pooled_malloc(void**,size_t){assert(false&&"persistent KV entered scratch pool");return hipErrorUnknown;}
hipError_t hipFree(void* p){if(p==fail_free)return hipErrorUnknown;assert(p&&live.erase(p)==1);std::free(p);return hipSuccess;}
hipError_t hipMemcpy(void* out,const void* in,size_t bytes,int kind){
 assert(kind==hipMemcpyDeviceToDevice);
 if(++copies==fail_copy)return hipErrorUnknown;
 std::memcpy(out,in,bytes);return hipSuccess;
}
''' + attention_capacity() + declarations + '\n#define hipMalloc forbidden_pooled_malloc\n' + operations + r'''
#undef hipMalloc
int main(){
 Qwen36ResidentSessionFullAttentionLayer layer{};
 constexpr size_t prefix=8192,tokens=8192,initial_bytes=prefix*1024;
 layer.valid=true;layer.history_tokens=prefix;
 layer.element_kind=Qwen36ResidentSessionElementKind::kBf16;
 layer.k_bytes=layer.v_bytes=initial_bytes;
 assert(hipMalloc(&layer.device_allocation,initial_bytes*2)==hipSuccess);
 layer.device_k=layer.device_allocation;
 layer.device_v=static_cast<unsigned char*>(layer.device_k)+initial_bytes;
 std::memset(layer.device_k,3,initial_bytes);std::memset(layer.device_v,5,initial_bytes);
 layer.decode_tail_capacity_tokens=1536;layer.decode_tail_k_bytes=layer.decode_tail_v_bytes=1536*1024;
 assert(hipMalloc(&layer.device_decode_tail_allocation,1536*2048)==hipSuccess);
 layer.device_decode_tail_k=layer.device_decode_tail_allocation;
 layer.device_decode_tail_v=static_cast<unsigned char*>(layer.device_decode_tail_k)+1536*1024;
 const auto original_tail=layer.device_decode_tail_allocation;
 fail_allocate=true;assert(resize_qwen36_prefill_chunk_tail(layer,8192)==hipErrorOutOfMemory);
 assert(layer.device_decode_tail_allocation==original_tail&&layer.decode_tail_capacity_tokens==1536&&live.size()==2);
 fail_allocate=false;layer.decode_tail_token_count=1;
 assert(resize_qwen36_prefill_chunk_tail(layer,8192)==hipErrorInvalidValue);
 layer.decode_tail_token_count=0;
 assert(resize_qwen36_prefill_chunk_tail(layer,8192)==hipSuccess&&live.size()==2);
 assert(layer.decode_tail_capacity_tokens==8192&&layer.decode_tail_k_bytes==tokens*1024);
 std::memset(layer.device_decode_tail_k,7,tokens*1024);std::memset(layer.device_decode_tail_v,11,tokens*1024);
 layer.decode_tail_token_count=tokens;
 const auto original=layer.device_allocation;
 for(unsigned failure=0;failure<=4;++failure){
  copies=0;fail_copy=failure;fail_allocate=failure==0;
  assert(promote_qwen36_prefill_chunk_attention(layer,prefix,tokens)==(failure?hipErrorUnknown:hipErrorOutOfMemory));
  assert(layer.device_allocation==original&&layer.history_tokens==prefix&&layer.decode_tail_token_count==tokens);
  assert(layer.k_bytes==initial_bytes&&layer.v_bytes==initial_bytes&&live.size()==2);
  for(auto pair:{std::pair<void*,unsigned char>{layer.device_k,3},{layer.device_v,5},
                {layer.device_decode_tail_k,7},{layer.device_decode_tail_v,11}}){
   auto* bytes=static_cast<unsigned char*>(pair.first);
   assert(std::all_of(bytes,bytes+initial_bytes,[&](unsigned char value){return value==pair.second;}));
  }
 }
 copies=fail_copy=0;fail_allocate=false;
 assert(promote_qwen36_prefill_chunk_attention(layer,prefix,tokens)==hipSuccess&&copies==4&&live.size()==2);
 assert(layer.history_tokens==16384&&!layer.decode_tail_token_count&&layer.k_bytes==16384u*1024);
 for(auto pair:{std::pair<void*,unsigned char>{layer.device_k,3},{layer.device_v,5}}){
  auto* bytes=static_cast<unsigned char*>(pair.first);
  assert(std::all_of(bytes,bytes+initial_bytes,[&](unsigned char v){return v==pair.second;}));
  assert(std::all_of(bytes+initial_bytes,bytes+initial_bytes*2,[&](unsigned char v){return v==(pair.second==3?7:11);}));
 }
 // A genuine1024-row tail after two full chunks uses the new V offset and
 // preserves both prior chunks; no padding rows become part of the history.
 std::memset(layer.device_decode_tail_k,13,1024u*1024);std::memset(layer.device_decode_tail_v,17,1024u*1024);
 layer.decode_tail_token_count=1024;
 assert(promote_qwen36_prefill_chunk_attention(layer,8192,1024)==hipErrorInvalidValue);
 assert(promote_qwen36_prefill_chunk_attention(layer,16384,1024)==hipSuccess&&live.size()==2);
 assert(layer.history_tokens==17408&&!layer.decode_tail_token_count);
 for(auto pair:{std::pair<void*,unsigned char>{layer.device_k,13},{layer.device_v,17}}){
  auto* bytes=static_cast<unsigned char*>(pair.first);
  assert(bytes[0]==(pair.second==13?3:5)&&bytes[initial_bytes]==(pair.second==13?7:11));
  assert(std::all_of(bytes+initial_bytes*2,bytes+17408u*1024,[&](unsigned char v){return v==pair.second;}));
 }
 assert(resize_qwen36_prefill_chunk_tail(layer,1536)==hipSuccess&&live.size()==2);
 assert(layer.decode_tail_capacity_tokens==1536&&layer.decode_tail_k_bytes==1536u*1024);
 hipFree(layer.device_allocation);hipFree(layer.device_decode_tail_allocation);assert(live.empty());

 // A reserved owner appends the same original bytes without allocating or
 // recopying any committed history. Its larger V stride is private until the
 // last real tail fills it, restoring the ordinary compact published layout.
 layer={};layer.valid=true;layer.history_tokens=prefix;
 layer.element_kind=Qwen36ResidentSessionElementKind::kBf16;
 layer.k_bytes=layer.v_bytes=initial_bytes;
 assert(hipMalloc(&layer.device_allocation,initial_bytes*2)==hipSuccess);
 layer.device_k=layer.device_allocation;
 layer.device_v=static_cast<unsigned char*>(layer.device_k)+initial_bytes;
 std::memset(layer.device_k,3,initial_bytes);std::memset(layer.device_v,5,initial_bytes);
 layer.decode_tail_capacity_tokens=tokens;layer.decode_tail_k_bytes=layer.decode_tail_v_bytes=tokens*1024;
 assert(hipMalloc(&layer.device_decode_tail_allocation,tokens*2048)==hipSuccess);
 layer.device_decode_tail_k=layer.device_decode_tail_allocation;
 layer.device_decode_tail_v=static_cast<unsigned char*>(layer.device_decode_tail_k)+tokens*1024;
 std::memset(layer.device_decode_tail_k,7,tokens*1024);std::memset(layer.device_decode_tail_v,11,tokens*1024);
 const auto seed_owner=layer.device_allocation;
 constexpr size_t capacity=17408;
 for(size_t invalid:{size_t(0),prefix,qrt_sm121_attention_capacity::kTokens+size_t(1)})
  assert(reserve_qwen36_prefill_chunk_attention(layer,invalid)==hipErrorInvalidValue&&live.size()==2);
 for(unsigned failure=0;failure<4;++failure){
  copies=0;fail_copy=failure<3?failure:0;fail_allocate=failure==0;
  fail_free=failure==3?seed_owner:nullptr;
  assert(reserve_qwen36_prefill_chunk_attention(layer,capacity)==(failure?hipErrorUnknown:hipErrorOutOfMemory));
  assert(layer.device_allocation==seed_owner&&!layer.prefill_reserved_tokens&&layer.history_tokens==prefix&&live.size()==2);
  assert(layer.device_v==static_cast<unsigned char*>(seed_owner)+initial_bytes);
  for(auto pair:{std::pair<void*,unsigned char>{layer.device_k,3},{layer.device_v,5}}){
   auto* bytes=static_cast<unsigned char*>(pair.first);
   assert(std::all_of(bytes,bytes+initial_bytes,[&](unsigned char v){return v==pair.second;}));
  }
 }
 fail_free=nullptr;fail_allocate=false;copies=fail_copy=0;
 assert(reserve_qwen36_prefill_chunk_attention(layer,capacity)==hipSuccess&&copies==2&&live.size()==2);
 const auto reserved_owner=layer.device_allocation;
 assert(layer.prefill_reserved_tokens==capacity&&layer.device_v==static_cast<unsigned char*>(reserved_owner)+capacity*1024);
 assert(reserve_qwen36_prefill_chunk_attention(layer,capacity)==hipErrorInvalidValue);
 layer.decode_tail_token_count=tokens;
 fail_allocate=true; // Every subsequent append must succeed with allocation disabled.
 for(unsigned failure:{1u,2u}){
  copies=0;fail_copy=failure;
  assert(promote_qwen36_prefill_chunk_attention(layer,prefix,tokens)==hipErrorUnknown);
  assert(layer.device_allocation==reserved_owner&&layer.history_tokens==prefix&&layer.k_bytes==initial_bytes);
  assert(layer.decode_tail_token_count==tokens&&layer.prefill_reserved_tokens==capacity&&live.size()==2);
  for(auto pair:{std::pair<void*,unsigned char>{layer.device_k,3},{layer.device_v,5}}){
   auto* bytes=static_cast<unsigned char*>(pair.first);
   assert(std::all_of(bytes,bytes+initial_bytes,[&](unsigned char v){return v==pair.second;}));
  }
 }
 copies=fail_copy=0;
 assert(promote_qwen36_prefill_chunk_attention(layer,prefix,tokens)==hipSuccess&&copies==2);
 assert(layer.device_allocation==reserved_owner&&layer.prefill_reserved_tokens==capacity&&layer.history_tokens==16384);
 assert(layer.device_v==static_cast<unsigned char*>(reserved_owner)+capacity*1024);
 // Reject corrupt capacity before writing either tensor or publishing counters.
 layer.decode_tail_token_count=1024;layer.prefill_reserved_tokens=16384;
 copies=0;assert(promote_qwen36_prefill_chunk_attention(layer,16384,1024)==hipErrorInvalidValue&&!copies);
 layer.prefill_reserved_tokens=capacity;
 auto* reserved_v=layer.device_v;layer.device_v=static_cast<unsigned char*>(reserved_v)+1;
 assert(promote_qwen36_prefill_chunk_attention(layer,16384,1024)==hipErrorInvalidValue&&!copies);
 layer.device_v=reserved_v;
 std::memset(layer.device_decode_tail_k,13,1024u*1024);std::memset(layer.device_decode_tail_v,17,1024u*1024);
 assert(promote_qwen36_prefill_chunk_attention(layer,16384,1024)==hipSuccess&&copies==2);
 assert(layer.device_allocation==reserved_owner&&!layer.prefill_reserved_tokens&&!layer.decode_tail_token_count);
 assert(layer.history_tokens==capacity&&layer.k_bytes==capacity*1024&&layer.v_bytes==layer.k_bytes);
 assert(layer.device_v==static_cast<unsigned char*>(layer.device_k)+layer.k_bytes);
 for(auto pair:{std::pair<void*,unsigned char>{layer.device_k,3},{layer.device_v,5}}){
  auto* bytes=static_cast<unsigned char*>(pair.first);
  assert(std::all_of(bytes,bytes+initial_bytes,[&](unsigned char v){return v==pair.second;}));
  assert(std::all_of(bytes+initial_bytes,bytes+initial_bytes*2,[&](unsigned char v){return v==(pair.second==3?7:11);}));
  assert(std::all_of(bytes+initial_bytes*2,bytes+capacity*1024,[&](unsigned char v){return v==(pair.second==3?13:17);}));
 }
 hipFree(layer.device_allocation);hipFree(layer.device_decode_tail_allocation);assert(live.empty());
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / 'chunks')
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            '-x', 'c++', '-', '-o', exe], input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30, capture_output=True)


if __name__ == '__main__':
    unittest.main()
