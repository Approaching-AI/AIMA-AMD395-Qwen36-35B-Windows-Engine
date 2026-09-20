"""Run the actual checkpoint ownership, clone and query code with host storage."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


def run_cpp(source):
    with tempfile.TemporaryDirectory() as tmp:
        exe = str(Path(tmp) / "checkpoints")
        subprocess.run([
            "c++", "-std=c++17", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
            "-I", str(ROOT), "-x", "c++", "-", "-o", exe,
        ], input=source, text=True, check=True, timeout=45)
        result = subprocess.run([exe], timeout=30, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)


class ModelPrefixCheckpointTests(unittest.TestCase):
    def test_saved_state_owner_coverage_and_shadow_rollback(self):
        text = (ROOT / "native/providers/whole_provider.cpp").read_text()
        structs = text[text.index("struct Qwen36ResidentSessionLinearLayer {"):
                       text.index("Qwen36ResidentSessionState g_qwen36_resident_root_session;")]
        lookup = function(text, "const Qwen36ResidentPrefixCheckpoint *find_qwen36_prefix_checkpoint(")
        finish = function(text, "void finalize_qwen36_prefix_checkpoints(")
        hidden = function(
            text[text.rindex("void capture_qwen36_prefix_checkpoint_hidden("):],
            "void capture_qwen36_prefix_checkpoint_hidden(")
        start = text.index("class ScopedQwen36ResidentSessionShadowTransaction final {")
        shadow = text[start:text.index("\nbool run_qwen36_q16_serial_transaction_probe_if_requested(", start)]
        query = function(text, "QRT_PREFILL_DESCRIPTOR_BATCH_HIP_CALL qrt_qwen36_whole_provider_checkpoint_query_v1(")
        release = function(text, "bool release_qwen36_resident_session_locked()")
        run_cpp(r'''
#include "native/src/qrt.h"
#include "native/src/qrt_prefix_checkpoint.h"
#include "native/providers/prefix_checkpoint_policy.h"
#include "native/providers/gdn/fla_checkpoint.h"
#include "native/providers/gdn/sm121_q2_cache_lifetime.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <type_traits>
static bool fail_next_host_new=false;
static bool fail_all_host_new=false;
void* operator new(std::size_t bytes){
 if(fail_all_host_new)throw std::bad_alloc();
 if(fail_next_host_new){fail_next_host_new=false;throw std::bad_alloc();}
 if(void* value=std::malloc(bytes?bytes:1u))return value;
 throw std::bad_alloc();
}
void operator delete(void* value)noexcept{std::free(value);}
#if defined(__cpp_sized_deallocation)
void operator delete(void* value,std::size_t)noexcept{std::free(value);}
#endif
using hipError_t=int;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipMemcpyDeviceToDevice=2;
static unsigned alloc_calls=0,fail_alloc=0,copy_calls=0,fail_copy=0,device_syncs=0,cleanup_calls=0;
static bool fail_device_sync=false,other_stream_pending=false;
static std::vector<std::tuple<void*,const void*,size_t>> pending;
static std::unordered_set<void*> allocations;
int hipMalloc(void** p,size_t n){if(++alloc_calls==fail_alloc){*p=nullptr;return 1;}
 *p=std::malloc(n);assert(*p);allocations.insert(*p);return 0;}
int hipFree(void* p){assert(pending.empty()&&!other_stream_pending);assert(allocations.erase(p)==1);std::free(p);return 0;}
int hipMemcpy(void* d,const void* s,size_t n,int){
 if(++copy_calls==fail_copy){pending.emplace_back(d,s,n);return 1;}
 std::memcpy(d,s,n);return 0;}
int hipMemset(void* d,int c,size_t n){std::memset(d,c,n);return 0;}
int hipStreamSynchronize(void*){return 0;}
int hipDeviceSynchronize(){++device_syncs;if(fail_device_sync)return 1;
 for(auto copy:pending)std::memcpy(std::get<0>(copy),std::get<1>(copy),std::get<2>(copy));
 pending.clear();other_stream_pending=false;return 0;}
const char* hipGetErrorString(int){return "injected";}
enum class Qwen36ResidentSessionElementKind {kNone,kF32,kBf16};
enum class Qwen36ResidentDecodeActivationWorkspacePhase {kIdle,kBusy};
struct Qwen36ResidentDecodeActivationWorkspace {
 bool in_use=false,valid=true,q1_moe_priority_stream_requested=false,q1_moe_priority_stream_ready=false;
 void* q1_moe_priority_stream=nullptr; void* device_allocation=nullptr;size_t layout_bytes=0;
 Qwen36ResidentDecodeActivationWorkspacePhase phase=Qwen36ResidentDecodeActivationWorkspacePhase::kIdle;
};
namespace qrt_sm121_mtp {
// The actual immutable owner is exercised by test_mtp_drafter; this fixture
// observes session copying, partial-prefix clearing and rollback identity.
struct RequestCheckpoint {std::shared_ptr<const int> retained;};
}
''' + structs + r'''
Qwen36ResidentSessionState g_qwen36_resident_root_session;
Qwen36ResidentSessionState* g_qwen36_resident_active_session=&g_qwen36_resident_root_session;
#define g_qwen36_resident_session g_qwen36_resident_root_session
std::recursive_mutex g_qwen36_resident_session_mutex;
bool release_qwen36_resident_dual_attention_state_locked(){return true;}
bool release_qwen36_resident_decode_activation_workspace_locked(){++cleanup_calls;return true;}
bool release_qwen36_q1024_suffix_cache_locked(){++cleanup_calls;return true;}
bool raw_env_flag_enabled(const char*){return false;}
bool refresh_qwen36_resident_session_immutable_weight_views(Qwen36ResidentSessionState*){return true;}
uint16_t qrt_float_to_bf16(float f){uint32_t b;std::memcpy(&b,&f,4);return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
uint64_t qrt_fnv1a64_bytes(const void* p,size_t n){uint64_t h=14695981039346656037ULL;auto b=(const uint8_t*)p;
 for(size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ULL;}return h;}
std::string hex_u64(uint64_t x){return std::to_string(x);}
uint64_t qwen36_resident_session_linear_layer_mask(){uint64_t m=0;for(unsigned i=0;i<40;++i)if(i%4!=3)m|=1ULL<<i;return m;}
constexpr size_t kQwen36ResidentDecodeTailCapacityTokens=1536;
bool qwen36_resident_session_capture_is_active(){return true;}
struct PrefillLinearAttentionDescriptorBatchRun {
 struct {bool correctness_pass=true;size_t selected_token_count=3;
  std::vector<unsigned> selected_token_ids{63,127,128};
  std::vector<float> gpu_output=std::vector<float>(3*QRT_QWEN36_HIDDEN_SIZE,1.25f);} final_norm;
 struct {bool correctness_pass=true,first_generated_token_valid=true;} token_loop_validation;
 bool final_norm_attempted=true,token_loop_validation_attempted=true;
};
#define QRT_PREFILL_DESCRIPTOR_BATCH_HIP_CALL int
''' + lookup + finish + hidden + shadow + query + release + r'''
void* allocate(size_t bytes,int value){void* p=nullptr;assert(!hipMalloc(&p,bytes));std::memset(p,value,bytes);return p;}
void setup(bool contiguous){
 auto& s=g_qwen36_resident_session;s={};s.valid=s.provider_completed=true;s.prefix_tokens=129;
 s.owner_engine=reinterpret_cast<const qrt_engine_t*>(uintptr_t(0x1234));s.generation=17;
 s.current_token_id=77;s.current_token_valid=true;
 s.native_mtp_checkpoint.retained=std::make_shared<const int>(19);
 s.native_mtp_processed_inputs.assign(129,42);
 s.prefix_checkpoints=std::make_shared<Qwen36ResidentPrefixCheckpointStore>();
 auto& store=*s.prefix_checkpoints;store.owner_tokens.assign(129,42);store.owner_engine=s.owner_engine;
 store.owner_generation=s.generation;store.owner_digest=qrt_fnv1a64_bytes(store.owner_tokens.data(),129*4);
 s.prompt_token_ids_fnv1a64=store.owner_digest;store.count=2;
 for(size_t ci=0;ci<2;++ci){auto& cp=store.checkpoints[ci];cp.prefix_tokens=(ci+1)*64;
  cp.prompt_digest=qrt_fnv1a64_bytes(store.owner_tokens.data(),cp.prefix_tokens*4);
  cp.hidden_valid=true;cp.terminal_hidden.fill(1.25f);cp.linear_mask=qwen36_resident_session_linear_layer_mask();}
 for(unsigned l=0;l<40;++l){
  if(l%4!=3){
   auto init=[&](Qwen36ResidentSessionLinearLayer& v,size_t n,int fill){
    v.device_allocation=allocate(96,fill);v.device_recurrent_state=(float*)v.device_allocation;
    v.device_qkv_ring=(char*)v.device_allocation+64;v.recurrent_state_bytes=64;v.qkv_ring_bytes=32;
    v.prefix_tokens=n;v.valid=true;v.recurrent_state_key_major=true;v.qkv_element_kind=Qwen36ResidentSessionElementKind::kBf16;};
   init(s.linear_layers[l],129,90);
   init(store.checkpoints[0].linear_layers[l],64,10+int(l));
   init(store.checkpoints[1].linear_layers[l],128,20+int(l));
  }else{
   auto& v=s.full_attention_layers[l];v.valid=true;v.history_tokens=129;
   v.k_bytes=v.v_bytes=129*8;v.decode_tail_k_bytes=v.decode_tail_v_bytes=16*8;
   v.decode_tail_capacity_tokens=16;v.element_kind=Qwen36ResidentSessionElementKind::kBf16;
   v.decode_tail_contiguous=contiguous;
   v.device_allocation=allocate(v.k_bytes+v.v_bytes+(contiguous?256:0),11);
   v.device_k=v.device_allocation;v.device_v=(char*)v.device_k+v.k_bytes+(contiguous?128:0);
   std::memset(v.device_v,29,v.v_bytes);
   if(contiguous){v.device_decode_tail_k=(char*)v.device_k+v.k_bytes;v.device_decode_tail_v=(char*)v.device_v+v.v_bytes;}
   else{v.device_decode_tail_allocation=allocate(256,99);v.device_decode_tail_k=v.device_decode_tail_allocation;v.device_decode_tail_v=(char*)v.device_decode_tail_k+128;}
  }
 }
 finalize_qwen36_prefix_checkpoints();assert(store.checkpoints[0].valid&&store.checkpoints[1].valid);
}
void teardown(){
 auto& s=g_qwen36_resident_session;
 for(auto& v:s.linear_layers)if(v.device_allocation)hipFree(v.device_allocation);
 for(auto& v:s.full_attention_layers){if(v.device_allocation)hipFree(v.device_allocation);if(v.device_decode_tail_allocation)hipFree(v.device_decode_tail_allocation);}
 s={};assert(allocations.empty());
}
void recover_test_quarantine(){
 // The fake queue establishes completion. Production has no recovery/free API
 // for a shadow whose GPU completion remains unknown.
 assert(g_qwen36_resident_completion_unknown&&g_qwen36_resident_shadow_quarantine);
 fail_device_sync=false;assert(!hipDeviceSynchronize());
 auto* q=g_qwen36_resident_shadow_quarantine;
 assert(!q->next);g_qwen36_resident_session=q->original;
 for(const auto* group:{&q->linear,&q->full,&q->tail})for(void* p:*group)if(p)hipFree(p);
 g_qwen36_resident_shadow_quarantine=nullptr;
 auto held=std::move(q->retained);held.reset();g_qwen36_resident_completion_unknown=false;
}
int main(){
 (void)kQwen36DflashStateCommitTokenCount;(void)kQwen36DflashMultiStateCommitTokenCount;
 // Every supported plain-input length respects the FLA boundary and slot cap.
 std::vector<uint32_t> plain(8193,42);
 for(size_t n=0;n<=8193;++n){auto p=qrt_prefix_checkpoint::select(plain.data(),n);
  assert(p.count<=3);if(n<=64||n>8192){assert(!p.count);continue;}assert(p.count);
  for(size_t i=0;i<p.count;++i)assert(p.tokens[i]&&p.tokens[i]%64==0&&p.tokens[i]<n&&(i==0||p.tokens[i]>p.tokens[i-1]));}
 plain[79]=248046;plain[159]=248046;plain[160]=248045;plain[161]=74455;plain[162]=198;
 auto selected=qrt_prefix_checkpoint::select(plain.data(),257);assert(selected.tokens[0]==64&&selected.tokens[1]==128);
 for(bool contiguous:{false,true})for(size_t committed:{0u,3u}){
  setup(contiguous);auto& s=g_qwen36_resident_session;auto store=s.prefix_checkpoints;
  // Generated-token state differs from the saved checkpoint. Restoring a
  // branch must preserve both the immutable KV prefix and the advanced owner.
  s.committed_decode_token_count=committed;
  for(auto& layer:s.full_attention_layers)if(layer.valid)layer.decode_tail_token_count=committed;
  PrefillLinearAttentionDescriptorBatchRun batch;
  for(auto& cp:store->checkpoints)cp.hidden_valid=false;
  capture_qwen36_prefix_checkpoint_hidden(batch,129);
  assert(store->checkpoints[0].hidden_valid&&store->checkpoints[1].hidden_valid);
  batch.final_norm.gpu_output[0]=std::numeric_limits<float>::quiet_NaN();
  capture_qwen36_prefix_checkpoint_hidden(batch,129);assert(store->failed);store->failed=false;
  batch.final_norm.gpu_output[0]=1.25f;batch.final_norm.selected_token_ids[0]=62;
  capture_qwen36_prefix_checkpoint_hidden(batch,129);assert(store->failed);store->failed=false;
  batch.final_norm.selected_token_ids[0]=63;capture_qwen36_prefix_checkpoint_hidden(batch,129);
  finalize_qwen36_prefix_checkpoints();
  const auto base_count=allocations.size();auto original=s;
  // An actual metadata allocation in the session copy must not escape the
  // transaction constructor or submit any GPU clone before it is owned.
  {
   const auto allocated_before=alloc_calls,copied_before=copy_calls;
   const auto unchanged=[&]{
    assert(alloc_calls==allocated_before&&copy_calls==copied_before&&allocations.size()==base_count);
    assert(s.valid&&s.current_token_valid==original.current_token_valid&&s.current_token_id==original.current_token_id);
    assert(s.generation==original.generation&&s.prefix_tokens==original.prefix_tokens&&
           s.committed_decode_token_count==original.committed_decode_token_count&&s.prefix_checkpoints==original.prefix_checkpoints);
    assert(s.native_mtp_checkpoint.retained==original.native_mtp_checkpoint.retained&&
           s.native_mtp_processed_inputs==original.native_mtp_processed_inputs);
    for(unsigned l=0;l<40;++l){
     assert(s.linear_layers[l].device_allocation==original.linear_layers[l].device_allocation&&
            s.linear_layers[l].device_recurrent_state==original.linear_layers[l].device_recurrent_state&&
            s.linear_layers[l].device_qkv_ring==original.linear_layers[l].device_qkv_ring);
     const auto& a=s.full_attention_layers[l];const auto& b=original.full_attention_layers[l];
     assert(a.device_allocation==b.device_allocation&&a.device_decode_tail_allocation==b.device_decode_tail_allocation&&
            a.device_k==b.device_k&&a.device_v==b.device_v&&
            a.device_decode_tail_k==b.device_decode_tail_k&&a.device_decode_tail_v==b.device_decode_tail_v);
    }
   };
   std::string stage,error;bool escaped=false;fail_next_host_new=true;
   try{ScopedQwen36ResidentSessionShadowTransaction tx(s.generation,s.prompt_token_ids_fnv1a64,&stage,&error);
       assert(!tx.ready());}
   catch(const std::bad_alloc&){escaped=true;}
   assert(!escaped&&!fail_next_host_new&&stage=="qwen36_resident_shadow_metadata");
   unchanged();
   stage.clear();error.clear();escaped=false;
   // Persistent host exhaustion also includes writing the failure details.
   std::string empty_stage,empty_error;fail_all_host_new=true;
   try{ScopedQwen36ResidentSessionShadowTransaction tx(s.generation,s.prompt_token_ids_fnv1a64,&empty_stage,&empty_error);
       assert(!tx.ready());}
   catch(const std::bad_alloc&){escaped=true;}
   fail_all_host_new=false;
   assert(!escaped&&empty_stage.empty()&&empty_error.empty());
   unchanged();
  }
  std::vector<uint32_t> input(129,42);input[128]=7;
  qrt_prefix_checkpoint_query_v1_t q{};q.struct_size=sizeof(q);q.abi_version=1;q.owner_engine=s.owner_engine;
  q.owner_token_count=129;q.input_token_count=129;q.output_token_capacity=32;q.input_tokens=input.data();
  q.owner_generation=s.generation;q.owner_prompt_digest=s.prompt_token_ids_fnv1a64;
  qrt_prefix_checkpoint_match_v1_t result{};
  assert(qrt_qwen36_whole_provider_checkpoint_query_v1(&q,&result)&&result.prefix_token_count==128);
  {auto full=q;std::vector<uint32_t> extended(130,42);full.input_tokens=extended.data();full.input_token_count=130;
   assert(qrt_qwen36_whole_provider_checkpoint_query_v1(&full,&result));
   assert(result.prefix_token_count==(committed==0?129u:128u));}
  q.maximum_prefix_tokens=64;assert(qrt_qwen36_whole_provider_checkpoint_query_v1(&q,&result)&&result.prefix_token_count==64);q.maximum_prefix_tokens=0;
  for(unsigned layer=0;layer<40;++layer){auto& cp=store->checkpoints[1];
   if(layer%4!=3){cp.linear_layers[layer].valid=false;finalize_qwen36_prefix_checkpoints();assert(!cp.valid);cp.linear_layers[layer].valid=true;}
   else{s.full_attention_layers[layer].valid=false;finalize_qwen36_prefix_checkpoints();assert(!cp.valid);s.full_attention_layers[layer].valid=true;}
   finalize_qwen36_prefix_checkpoints();assert(cp.valid);}
  auto good=q;q.owner_generation++;assert(qrt_qwen36_whole_provider_checkpoint_query_v1(&q,&result)&&!result.prefix_token_count);q=good;
  q.owner_engine=reinterpret_cast<const qrt_engine_t*>(uintptr_t(0x5678));assert(qrt_qwen36_whole_provider_checkpoint_query_v1(&q,&result)&&!result.prefix_token_count);q=good;
  input[0]++;assert(qrt_qwen36_whole_provider_checkpoint_query_v1(&q,&result)&&!result.prefix_token_count);input[0]--;
  for(size_t prefix:{64u,128u}){
   const auto digest=qrt_fnv1a64_bytes(input.data(),prefix*4);std::string stage,error;
   {ScopedQwen36ResidentSessionShadowTransaction tx(17,digest,&stage,&error,prefix,input.data());
    assert(tx.ready()&&s.prefix_tokens==prefix&&s.prompt_token_ids_fnv1a64==digest&&!s.current_token_valid);
    assert(tx.base_committed_decode_token_count()==committed&&s.committed_decode_token_count==0u);
    assert(!s.native_mtp_checkpoint.retained&&s.native_mtp_processed_inputs.empty());
    for(unsigned l=0;l<40;++l)if(l%4!=3){auto& v=s.linear_layers[l];
     assert(v.prefix_tokens==prefix&&((unsigned char*)v.device_allocation)[0]==(prefix==64?10:20)+l);
     assert(v.device_allocation!=original.linear_layers[l].device_allocation);std::memset(v.device_allocation,222,96);
    }else{auto& v=s.full_attention_layers[l];assert(v.history_tokens==prefix&&v.k_bytes==prefix*8);
     assert(((unsigned char*)v.device_k)[prefix*8-1]==11&&((unsigned char*)v.device_v)[prefix*8-1]==29);
     if(!contiguous)assert(v.device_v==original.full_attention_layers[l].device_v);
     std::memset(v.device_decode_tail_k,202,128);}
    assert(!tx.commit(&stage,&error));assert(tx.rollback("test"));assert(s.prefix_tokens==129&&s.current_token_id==77&&s.committed_decode_token_count==committed);
   }
   assert(allocations.size()==base_count);
   assert(s.native_mtp_checkpoint.retained==original.native_mtp_checkpoint.retained&&
          s.native_mtp_processed_inputs==original.native_mtp_processed_inputs);
   // Every allocation failure restores owner metadata and pointers.
   for(unsigned at=1;at<=40;++at){fail_alloc=alloc_calls+at;
    {ScopedQwen36ResidentSessionShadowTransaction tx(17,digest,&stage,&error,prefix,input.data());assert(!tx.ready());}
    fail_alloc=0;assert(allocations.size()==base_count&&s.prefix_tokens==129&&s.valid&&s.committed_decode_token_count==committed);}
  }
  assert(s.prefix_checkpoints==store);
  original.prefix_checkpoints.reset();store.reset();teardown();
 }
 // Failed copies are allowed to have submitted work. Every partial destination
 // is tracked before submission and is freed only after an all-stream drain.
 for(bool contiguous:{false,true}){
  setup(contiguous);auto& s=g_qwen36_resident_session;const auto base=allocations.size();
  std::vector<uint32_t> input(129,42);const auto digest=qrt_fnv1a64_bytes(input.data(),128u*4u);
  const unsigned count=contiguous?50u:40u;std::string stage,error;
  for(unsigned at=1;at<=count;++at){
   fail_copy=copy_calls+at;
   {ScopedQwen36ResidentSessionShadowTransaction tx(17,digest,&stage,&error,128,input.data());assert(!tx.ready());}
   fail_copy=0;assert(pending.empty()&&allocations.size()==base&&s.valid&&s.prefix_tokens==129u);
  }
  // Same failures with unknown completion retain both original metadata and
  // partial clones even after the transaction and active-session scope end.
  for(unsigned at:{1u,20u,count}){
   std::weak_ptr<Qwen36ResidentPrefixCheckpointStore> weak=s.prefix_checkpoints;
   fail_copy=copy_calls+at;fail_device_sync=true;
   {ScopedQwen36ResidentSessionShadowTransaction tx(17,digest,&stage,&error,128,input.data());assert(!tx.ready());}
   fail_copy=0;assert(!s.valid&&g_qwen36_resident_completion_unknown&&!pending.empty());
   const auto retained=allocations.size();assert(retained>base&&!weak.expired());
   const auto before=alloc_calls;
   {ScopedQwen36ResidentSessionShadowTransaction tx(17,digest,&stage,&error,128,input.data());assert(!tx.ready());}
   assert(alloc_calls==before);cleanup_calls=0;
   assert(!release_qwen36_resident_session_locked()&&!cleanup_calls&&allocations.size()==retained);
   s={};assert(!weak.expired());recover_test_quarantine();assert(allocations.size()==base&&s.valid);
  }
  // Work on a non-default stream must also finish before a healthy rollback.
  {ScopedQwen36ResidentSessionShadowTransaction tx(17,s.prompt_token_ids_fnv1a64,&stage,&error);
   assert(tx.ready());other_stream_pending=true;assert(tx.rollback("all_streams"));}
  assert(!other_stream_pending&&allocations.size()==base);
  for(unsigned phase=0;phase<3u;++phase){
   const auto original=s.linear_layers[0].device_allocation;
   {ScopedQwen36ResidentSessionShadowTransaction tx(17,s.prompt_token_ids_fnv1a64,&stage,&error);
    assert(tx.ready()&&s.linear_layers[0].device_allocation!=original);
    other_stream_pending=true;fail_device_sync=true;
    if(phase==0u)assert(!tx.rollback("failed_fence"));
    if(phase==1u)assert(!tx.commit(&stage,&error));
   }
   assert(g_qwen36_resident_completion_unknown&&!s.valid&&allocations.count(original));
   assert(other_stream_pending&&allocations.size()==base+40u);
   recover_test_quarantine();assert(s.valid&&allocations.size()==base&&s.linear_layers[0].device_allocation==original);
  }
  teardown();
 }
 // The actual target pin blocks retirement/reentrant clones and commit. Drop
 // it before the enclosing transaction publishes or restores the cache.
 for(bool contiguous:{false,true}) {
  setup(contiguous);auto& s=g_qwen36_resident_session;std::string stage,error;
  {
   ScopedQwen36ResidentSessionShadowTransaction tx(s.generation,s.prompt_token_ids_fnv1a64,&stage,&error);
   assert(tx.ready());auto pin=tx.acquire_target_cache_lifetime();
   assert(pin&&pin->ready(&s)&&g_qwen36_target_cache_borrows==1u);
   const auto before=allocations.size();const auto cleanups=cleanup_calls;
   assert(!release_qwen36_resident_session_locked()&&allocations.size()==before&&cleanup_calls==cleanups&&s.valid);
   {ScopedQwen36ResidentSessionShadowTransaction nested(s.generation,s.prompt_token_ids_fnv1a64,&stage,&error);
    assert(!nested.ready()&&allocations.size()==before);}
   assert(!tx.commit(&stage,&error)&&tx.ready());
   pin.reset();assert(!g_qwen36_target_cache_borrows&&tx.commit(&stage,&error));
  }
  teardown();
  setup(contiguous);std::shared_ptr<const qrt_sm121_q2::ResidentCacheLifetime> escaped;
  {
   ScopedQwen36ResidentSessionShadowTransaction tx(s.generation,s.prompt_token_ids_fnv1a64,&stage,&error);
   assert(tx.ready());escaped=tx.acquire_target_cache_lifetime();assert(escaped);
  }
  assert(g_qwen36_resident_completion_unknown&&!escaped->ready(&s)&&g_qwen36_target_cache_borrows==1u);
  assert(!release_qwen36_resident_session_locked());
  escaped.reset();recover_test_quarantine();teardown();
 }
 assert(device_syncs>100u);
}
''')

    def test_quarantined_engine_release_preserves_nonowner_and_last_engine(self):
        text = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        release = function(text, 'qrt_qwen36_whole_provider_release_engine_v1(')
        run_cpp(r'''
#include "native/src/qrt.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <mutex>
#include <vector>
using Result=qrt_qwen36_whole_provider_engine_lifecycle_result_v1_t;
std::recursive_mutex g_qwen36_resident_session_mutex;
std::atomic<bool> g_qwen36_resident_completion_unknown{false};
std::atomic<size_t> g_qwen36_target_cache_borrows{0u};
std::vector<const qrt_engine_t*> g_qwen36_whole_provider_live_engines;
struct Session {const qrt_engine_t* owner_engine=nullptr;bool valid=false;} g_qwen36_resident_session;
unsigned session_releases=0,shared_releases=0;
uint64_t qrt_now_ns(){return 1;}
uint64_t qrt_elapsed_ns(uint64_t a,uint64_t b){return b-a;}
void init_qwen36_provider_engine_lifecycle_result(Result* r,unsigned operation){*r={};r->operation=operation;}
int fail_qwen36_provider_engine_lifecycle(Result* r,qrt_status_t status,const char* stage,const char*,uint64_t){
 r->status=status;std::strncpy(r->failure_stage,stage,sizeof(r->failure_stage)-1);return 0;}
void capture_qwen36_provider_engine_lifecycle_before_locked(const qrt_engine_t* e,Result* r){
 r->engine_session_owner_before=g_qwen36_resident_session.owner_engine==e;}
void capture_qwen36_provider_engine_lifecycle_after_locked(const qrt_engine_t*,Result* r){r->live_engine_count_after=g_qwen36_whole_provider_live_engines.size();}
bool release_qwen36_resident_session_locked(){++session_releases;if(g_qwen36_resident_completion_unknown)return false;g_qwen36_resident_session={};return true;}
uint64_t qwen36_resident_session_owned_bytes_locked(){return g_qwen36_resident_session.valid?1:0;}
void qrt_prefill_descriptor_batch_hip_release_v1(){++shared_releases;}
void emit_qwen36_provider_engine_lifecycle_marker(const char*,const Result&){}
''' + 'int ' + release + r'''
int main(){
 auto* owner=reinterpret_cast<const qrt_engine_t*>(uintptr_t(0x1000));
 auto* other=reinterpret_cast<const qrt_engine_t*>(uintptr_t(0x2000));
 Result result;g_qwen36_whole_provider_live_engines={owner,other};g_qwen36_resident_session={owner,true};
 g_qwen36_resident_completion_unknown=true;
 for(auto* engine:{owner,other}){
  assert(!qrt_qwen36_whole_provider_release_engine_v1(engine,&result));
  assert(!result.completed&&g_qwen36_whole_provider_live_engines.size()==2u&&!session_releases&&!shared_releases);
 }
 // A scoped branch can be quarantined while the root session is empty.
 g_qwen36_resident_session={};g_qwen36_whole_provider_live_engines={other};
 assert(!qrt_qwen36_whole_provider_release_engine_v1(other,&result));
 assert(g_qwen36_whole_provider_live_engines.size()==1u&&!session_releases&&!shared_releases);
 g_qwen36_resident_completion_unknown=false;g_qwen36_target_cache_borrows=1u;
 g_qwen36_whole_provider_live_engines={owner,other};g_qwen36_resident_session={owner,true};
 for(auto* engine:{owner,other}){
  assert(!qrt_qwen36_whole_provider_release_engine_v1(engine,&result));
  assert(!result.completed&&g_qwen36_whole_provider_live_engines.size()==2u&&!session_releases&&!shared_releases);
 }
 g_qwen36_resident_session={};g_qwen36_whole_provider_live_engines={other};
 assert(!qrt_qwen36_whole_provider_release_engine_v1(other,&result));
 assert(g_qwen36_whole_provider_live_engines.size()==1u&&!session_releases&&!shared_releases);
 g_qwen36_target_cache_borrows=0u;g_qwen36_whole_provider_live_engines={owner,other};g_qwen36_resident_session={owner,true};
 assert(qrt_qwen36_whole_provider_release_engine_v1(other,&result)&&result.completed);
 assert(result.shared_model_weights_retained&&!session_releases&&!shared_releases);
 assert(qrt_qwen36_whole_provider_release_engine_v1(owner,&result)&&result.completed);
 assert(result.resident_session_released&&result.shared_model_weights_released&&session_releases==1u&&shared_releases==1u);
 assert(g_qwen36_whole_provider_live_engines.empty());
}
''')

    def test_core_query_rejects_malformed_provider_results_without_mutation(self):
        text = (ROOT / "native/src/qrt.c").read_text()
        query = function(text, "static qrt_status_t qrt_engine_prefix_checkpoint_match_v1_unlocked(")
        run_cpp(r'''
#include "native/src/qrt.h"
#include "native/src/qrt_prefix_checkpoint.h"
#include <cassert>
#include <cstring>
#include <vector>
#define _WIN32
struct qrt_engine {
 bool ready=true,manifest_loaded=true,resident_prefix_cache_session_valid=true;
 uint32_t* resident_prefix_cache_tokens=nullptr;size_t resident_prefix_cache_token_count=129;
 uint64_t resident_prefix_cache_session_generation=7,resident_prefix_cache_prompt_token_ids_fnv1a64=9;
 qrt_prefix_checkpoint_query_fn qwen36_whole_provider_checkpoint_query=nullptr;
};
int qrt_win32_qwen36_whole_provider_backend_load(qrt_engine_t*){return 1;}
uint64_t qrt_fnv1a64_bytes(const void* p,size_t n){uint64_t h=14695981039346656037ULL;auto b=(const uint8_t*)p;
 for(size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ULL;}return h;}
unsigned mode=0,calls=0;
int provider(const qrt_prefix_checkpoint_query_v1_t* q,qrt_prefix_checkpoint_match_v1_t* m){
 ++calls;assert(q->owner_engine&&q->owner_token_count==129&&q->owner_generation==7);
 *m={};m->struct_size=sizeof(*m);m->abi_version=1;m->prefix_token_count=128;
 m->owner_generation=q->owner_generation;m->prefix_digest=qrt_fnv1a64_bytes(q->input_tokens,128*4);
 if(mode==1)m->struct_size--;if(mode==2)m->abi_version++;if(mode==3)m->reserved=1;
 if(mode==4)m->prefix_token_count=130;if(mode==5)m->owner_generation++;if(mode==6)m->prefix_digest++;
 if(mode==7)m->prefix_token_count=0;return mode==8?0:1;
}
''' + query + r'''
int main(){
 std::vector<uint32_t> a(129,42),b=a;qrt_engine_t e;e.resident_prefix_cache_tokens=a.data();
 size_t prefix=99;
 auto run=[&](size_t max=0){return qrt_engine_prefix_checkpoint_match_v1_unlocked(&e,b.data(),b.size(),32,max,&prefix);};
 assert(run()==QRT_STATUS_OK&&prefix==0);e.qwen36_whole_provider_checkpoint_query=provider;
 assert(run()==QRT_STATUS_OK&&prefix==128);
 for(mode=1;mode<=8;++mode){auto old=e;auto status=run();
  assert(prefix==0&&status==(mode==7?QRT_STATUS_OK:QRT_STATUS_UNSUPPORTED));
  assert(std::memcmp(&old,&e,sizeof(e))==0);}
 mode=0;assert(run(64)==QRT_STATUS_UNSUPPORTED&&prefix==0);
 b[64]=7;assert(run()==QRT_STATUS_UNSUPPORTED&&prefix==0);b=a;b[0]=7;
 unsigned before=calls;assert(run()==QRT_STATUS_OK&&prefix==0&&calls==before);
 b[0]=QRT_QWEN36_VOCAB_SIZE;assert(run()==QRT_STATUS_INVALID_ARGUMENT&&prefix==0);
}
''')
