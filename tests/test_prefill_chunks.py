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
        coordinator = function(chunk, 'int run_qwen36_chunked_prefill(')
        source = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "native/src/qrt.h"
''' + attention_capacity() + r'''
enum hipError_t{hipSuccess,hipErrorInvalidValue};
const char* hipGetErrorString(hipError_t){return "injected";}
constexpr size_t kQwen36ResidentDecodeTailCapacityTokens=1536;
struct Linear{bool valid=true;size_t prefix_tokens=8192,decode_qkv_token_count=0,decode_recurrent_token_count=0;};
struct Attention{size_t decode_tail_k_bytes=1536*1024,decode_tail_v_bytes=1536*1024;};
struct Workspace{size_t full_attention_score_scratch_token_capacity=0;};
struct Session{
 bool valid=false,current_token_valid=false,last_decode_top2_valid=false,mtp_target_hidden_valid=false;
 size_t prefix_tokens=0,committed_decode_token_count=0,last_decode_top2_position=0;
 uint64_t prompt_token_ids_fnv1a64=0,generation=99,full_attention_decode_tail_bytes=10*1536*2048;
 uint64_t full_attention_kv_bytes=10*8192*2048;
 std::array<Linear,40> linear_layers{};std::array<Attention,40> full_attention_layers{};Workspace activation_workspace;
} g_qwen36_resident_session;
std::recursive_mutex g_qwen36_resident_session_mutex;
size_t g_qwen36_chunked_prefill_total_tokens=0;
struct ScopedQwen36PrefixBatchSuffix{inline static void* active=nullptr;};
uint64_t clock_ns=0;
uint64_t qrt_now_ns(){return ++clock_ns;}
uint64_t qrt_elapsed_ns(uint64_t a,uint64_t b){return b-a;}
uint64_t qrt_fnv1a64_bytes(const void* p,size_t n){
 uint64_t h=1469598103934665603ULL;auto* bytes=static_cast<const unsigned char*>(p);
 for(size_t i=0;i<n;++i){h^=bytes[i];h*=1099511628211ULL;}return h;
}
uint64_t qrt_fnv1a64_update_bytes(uint64_t h,const void* p,size_t n){return h^qrt_fnv1a64_bytes(p,n);}
bool raw_env_flag_enabled(const char*){return false;}
bool env_flag_enabled(const char*){return true;}
bool qwen36_resident_decode_activation_workspace_layout_valid(const Workspace&){return true;}
unsigned seeds=0,suffixes=0,callbacks=0,releases=0,fail_suffix=0;
bool fail_seed=false,cancel=false,bad_counter=false;
const uint32_t* actual_prompt=nullptr;size_t requested_total=0;
void qrt_qwen36_whole_provider_set_failure(qrt_qwen36_whole_provider_result_t* r,const std::string& stage,const std::string&,uint64_t start){
 r->completed=0;std::strncpy(r->failure_stage,stage.c_str(),sizeof(r->failure_stage)-1);r->wall_clock_ns=qrt_elapsed_ns(start,qrt_now_ns());
}
bool release_qwen36_resident_session_locked(){++releases;g_qwen36_resident_session=Session{};return true;}
int qrt_qwen36_whole_provider_prefill_v1(const qrt_qwen36_whole_provider_request_t* r,qrt_qwen36_whole_provider_result_t* out){
 ++seeds;assert(r->input_tokens==actual_prompt&&r->input_token_count==8192&&r->output_token_capacity==1);
 assert(!r->prefill_emit_callback&&(r->flags&QRT_QWEN36_WHOLE_PROVIDER_FLAG_PREFIX_SEED_CAPTURE));
 auto& s=g_qwen36_resident_session;s=Session{};s.valid=true;s.prefix_tokens=8192;
 s.prompt_token_ids_fnv1a64=r->expected_prompt_token_ids_fnv1a64;
 s.activation_workspace.full_attention_score_scratch_token_capacity=g_qwen36_chunked_prefill_total_tokens+1537;
 out->completed=1;out->output_token_count=1;out->output_tokens[0]=999;
 return !fail_seed;
}
hipError_t resize_qwen36_prefill_chunk_tail(Attention& a,size_t count){a.decode_tail_k_bytes=a.decode_tail_v_bytes=count*1024;return hipSuccess;}
hipError_t promote_qwen36_prefill_chunk_attention(Attention&,size_t,size_t){return hipSuccess;}
bool run_qwen36_resident_batch_suffix(const qrt_qwen36_whole_provider_prefix_request_v1_t& r,uint32_t* teachers,
 std::string* stage,std::string* failure,bool terminal_only,qrt_qwen36_whole_provider_result_t* out){
 ++suffixes;auto& s=g_qwen36_resident_session;
 assert(!teachers&&terminal_only&&r.expected_prefix_token_count==s.prefix_tokens);
 assert(r.suffix_tokens==actual_prompt+s.prefix_tokens&&r.suffix_token_count==std::min(size_t(8192),requested_total-s.prefix_tokens));
 if(suffixes==fail_suffix){*stage="injected_suffix";*failure="failed";return false;}
 for(unsigned i=0;i<40;++i)if(i%4!=3){auto& l=s.linear_layers[i];l.decode_qkv_token_count=l.decode_recurrent_token_count=r.suffix_token_count;}
 if(bad_counter)++s.linear_layers[12].decode_qkv_token_count;
 s.committed_decode_token_count=r.suffix_token_count;s.current_token_valid=s.last_decode_top2_valid=true;
 s.last_decode_top2_position=s.prefix_tokens+r.suffix_token_count-1;
 out->completed=1;out->output_token_count=1;out->output_tokens[0]=42;out->continuation.output_token_emitted=1;
 out->continuation.output_token_id=42;out->continuation.output_logit=5.5;out->output_tokens_fnv1a64=234;
 return true;
}
int QRT_CDECL emit(void*,uint32_t token,uint64_t elapsed){
 ++callbacks;assert(token==42&&elapsed>1&&g_qwen36_resident_session.prefix_tokens==requested_total);
 assert(!g_qwen36_resident_session.committed_decode_token_count);return !cancel;
}
''' + coordinator + r'''
int main(){
 std::vector<uint32_t> prompt(65536);for(size_t i=0;i<prompt.size();++i)prompt[i]=uint32_t(i%245000);
 actual_prompt=prompt.data();auto result=std::make_unique<qrt_qwen36_whole_provider_result_t>();
 auto run=[&](size_t total){
  requested_total=total;seeds=suffixes=callbacks=releases=0;*result={};result->preload_wall_clock_ns=789;
  qrt_qwen36_whole_provider_request_t r{};r.resident_engine=reinterpret_cast<qrt_engine_t*>(uintptr_t(16));
  r.input_tokens=prompt.data();r.input_token_count=total;r.output_token_capacity=512;r.prefill_emit_callback=emit;
  r.expected_prompt_token_ids_fnv1a64=qrt_fnv1a64_bytes(prompt.data(),total*4);
  const int ok=run_qwen36_chunked_prefill(r,result.get(),qrt_now_ns());assert(!g_qwen36_chunked_prefill_total_tokens);return ok;
 };
 for(size_t total:{16384u,17408u,32768u,65536u}){
  assert(run(total)==1&&callbacks==1&&seeds==1&&suffixes==(total+8191)/8192-1&&!releases);
  assert(result->resident_session_valid&&result->resident_session_prefix_token_count==total);
  assert(result->resident_session_generation==99&&result->output_tokens[0]==42&&result->output_token_capacity==512);
  assert(result->preload_wall_clock_ns==789&&result->prefill_emit_completed&&!result->prefill_emit_rejected);
  assert(result->prompt_token_ids_fnv1a64==qrt_fnv1a64_bytes(prompt.data(),total*4));
 }
 fail_seed=true;assert(!run(16384)&&!callbacks&&releases==1&&!g_qwen36_resident_session.valid);fail_seed=false;
 for(unsigned failure:{1u,2u}){fail_suffix=failure;assert(!run(32768)&&!callbacks&&releases==1&&!g_qwen36_resident_session.valid);}
 fail_suffix=0;bad_counter=true;assert(!run(16384)&&!callbacks&&releases==1);bad_counter=false;
 cancel=true;assert(!run(17408)&&callbacks==1&&releases==1&&result->prefill_emit_rejected&&!result->resident_session_valid);cancel=false;
 assert(!run(16385)&&!seeds&&!callbacks&&!releases);
 assert(run(16384)&&callbacks==1&&result->completed);
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
unsigned copies=0,fail_copy=0;
hipError_t hipMalloc(void** out,size_t bytes) {
 if(fail_allocate){*out=nullptr;return hipErrorOutOfMemory;}
 *out=std::malloc(bytes);assert(*out&&live.insert(*out).second);return hipSuccess;
}
hipError_t hipFree(void* p){assert(p&&live.erase(p)==1);std::free(p);return hipSuccess;}
hipError_t hipMemcpy(void* out,const void* in,size_t bytes,int kind){
 assert(kind==hipMemcpyDeviceToDevice);
 if(++copies==fail_copy)return hipErrorUnknown;
 std::memcpy(out,in,bytes);return hipSuccess;
}
''' + attention_capacity() + declarations + operations + r'''
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
