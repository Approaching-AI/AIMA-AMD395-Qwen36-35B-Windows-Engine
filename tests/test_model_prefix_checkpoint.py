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
        subprocess.run([exe], check=True, timeout=30, capture_output=True)


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
        run_cpp(r'''
#include "native/src/qrt.h"
#include "native/src/qrt_prefix_checkpoint.h"
#include "native/providers/prefix_checkpoint_policy.h"
#include "native/providers/gdn/fla_checkpoint.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>
using hipError_t=int;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipMemcpyDeviceToDevice=2;
static unsigned alloc_calls=0,fail_alloc=0;
static std::unordered_set<void*> allocations;
int hipMalloc(void** p,size_t n){if(++alloc_calls==fail_alloc){*p=nullptr;return 1;}
 *p=std::malloc(n);assert(*p);allocations.insert(*p);return 0;}
int hipFree(void* p){assert(allocations.erase(p)==1);std::free(p);return 0;}
int hipMemcpy(void* d,const void* s,size_t n,int){std::memcpy(d,s,n);return 0;}
int hipMemset(void* d,int c,size_t n){std::memset(d,c,n);return 0;}
int hipStreamSynchronize(void*){return 0;}
const char* hipGetErrorString(int){return "injected";}
enum class Qwen36ResidentSessionElementKind {kNone,kF32,kBf16};
enum class Qwen36ResidentDecodeActivationWorkspacePhase {kIdle,kBusy};
struct Qwen36ResidentDecodeActivationWorkspace {
 bool in_use=false,valid=true,q1_moe_priority_stream_requested=false,q1_moe_priority_stream_ready=false;
 void* q1_moe_priority_stream=nullptr; void* device_allocation=nullptr;size_t layout_bytes=0;
 Qwen36ResidentDecodeActivationWorkspacePhase phase=Qwen36ResidentDecodeActivationWorkspacePhase::kIdle;
};
''' + structs + r'''
Qwen36ResidentSessionState g_qwen36_resident_root_session;
#define g_qwen36_resident_session g_qwen36_resident_root_session
std::recursive_mutex g_qwen36_resident_session_mutex;
bool release_qwen36_resident_dual_attention_state_locked(){return true;}
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
''' + lookup + finish + hidden + shadow + query + r'''
void* allocate(size_t bytes,int value){void* p=nullptr;assert(!hipMalloc(&p,bytes));std::memset(p,value,bytes);return p;}
void setup(bool contiguous){
 auto& s=g_qwen36_resident_session;s={};s.valid=s.provider_completed=true;s.prefix_tokens=129;
 s.owner_engine=reinterpret_cast<const qrt_engine_t*>(uintptr_t(0x1234));s.generation=17;
 s.current_token_id=77;s.current_token_valid=true;
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
int main(){
 (void)kQwen36DflashStateCommitTokenCount;(void)kQwen36DflashMultiStateCommitTokenCount;
 // Every supported plain-input length respects the FLA boundary and slot cap.
 std::vector<uint32_t> plain(8193,42);
 for(size_t n=0;n<=8193;++n){auto p=qrt_prefix_checkpoint::select(plain.data(),n);
  assert(p.count<=3);if(n<=64||n>8192){assert(!p.count);continue;}assert(p.count);
  for(size_t i=0;i<p.count;++i)assert(p.tokens[i]&&p.tokens[i]%64==0&&p.tokens[i]<n&&(i==0||p.tokens[i]>p.tokens[i-1]));}
 plain[79]=248046;plain[159]=248046;plain[160]=248045;plain[161]=74455;plain[162]=198;
 auto selected=qrt_prefix_checkpoint::select(plain.data(),257);assert(selected.tokens[0]==64&&selected.tokens[1]==128);
 for(bool contiguous:{false,true}){
  setup(contiguous);auto& s=g_qwen36_resident_session;auto store=s.prefix_checkpoints;
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
  std::vector<uint32_t> input(129,42);input[128]=7;
  qrt_prefix_checkpoint_query_v1_t q{};q.struct_size=sizeof(q);q.abi_version=1;q.owner_engine=s.owner_engine;
  q.owner_token_count=129;q.input_token_count=129;q.output_token_capacity=32;q.input_tokens=input.data();
  q.owner_generation=s.generation;q.owner_prompt_digest=s.prompt_token_ids_fnv1a64;
  qrt_prefix_checkpoint_match_v1_t result{};
  assert(qrt_qwen36_whole_provider_checkpoint_query_v1(&q,&result)&&result.prefix_token_count==128);
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
    for(unsigned l=0;l<40;++l)if(l%4!=3){auto& v=s.linear_layers[l];
     assert(v.prefix_tokens==prefix&&((unsigned char*)v.device_allocation)[0]==(prefix==64?10:20)+l);
     assert(v.device_allocation!=original.linear_layers[l].device_allocation);std::memset(v.device_allocation,222,96);
    }else{auto& v=s.full_attention_layers[l];assert(v.history_tokens==prefix&&v.k_bytes==prefix*8);
     assert(((unsigned char*)v.device_k)[prefix*8-1]==11&&((unsigned char*)v.device_v)[prefix*8-1]==29);
     if(!contiguous)assert(v.device_v==original.full_attention_layers[l].device_v);
     std::memset(v.device_decode_tail_k,202,128);}
    assert(!tx.commit(&stage,&error));assert(tx.rollback("test"));assert(s.prefix_tokens==129&&s.current_token_id==77);
   }
   assert(allocations.size()==base_count);
   // Every allocation failure restores owner metadata and pointers.
   for(unsigned at=1;at<=40;++at){fail_alloc=alloc_calls+at;
    {ScopedQwen36ResidentSessionShadowTransaction tx(17,digest,&stage,&error,prefix,input.data());assert(!tx.ready());}
    fail_alloc=0;assert(allocations.size()==base_count&&s.prefix_tokens==129&&s.valid);}
  }
  assert(s.prefix_checkpoints==store);
  original.prefix_checkpoints.reset();store.reset();teardown();
 }
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
