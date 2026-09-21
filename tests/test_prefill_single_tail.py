"""Execute the actual cold q1 bridge with bounded failures and private MTP rows."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class PrefillSingleTailTests(unittest.TestCase):
    def test_actual_single_input_and_publication(self):
        bridge = function((ROOT / 'native/providers/prefill_chunks.h').read_text(),
                          'bool run_qwen36_prefill_single_tail(')
        source = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "native/src/qrt.h"
#include "native/providers/mtp_target_rows.h"
#include "native/providers/native_mtp_target_stream.h"
enum hipError_t{hipSuccess,hipErrorInvalidValue};
constexpr int hipMemcpyDeviceToHost=1;
enum class Qwen36ResidentDecodeActivationWorkspacePhase{kIdle,kActive};
struct Workspace{
 uint64_t generation=77;bool in_use=false,layout=true;
 const uint16_t* device_norm_bf16=nullptr;
 Qwen36ResidentDecodeActivationWorkspacePhase phase=Qwen36ResidentDecodeActivationWorkspacePhase::kIdle;
};
struct Session{
 bool valid=true,provider_completed=true,current_token_valid=true,dflash_prefetched_valid=false,q2_prefetched_valid=false;
 size_t prefix_tokens=8192,committed_decode_token_count=0,last_decode_top2_position=8192;
 uint64_t generation=77,prompt_token_ids_fnv1a64=567;
 uint32_t current_token_id=999;
 bool last_decode_top2_valid=true;
 std::array<uint32_t,2> last_decode_top2_ids{220,64};
 std::array<float,2> last_decode_top2_logits{9.75f,9.6875f};
 Workspace activation_workspace;
}g_qwen36_resident_session;
std::array<uint16_t,2048> actual_norm;
unsigned fault=0,calls=0,copies=0;bool direct=true;
bool q1_decode_direct_output_plan_enabled(){return direct;}
bool qwen36_resident_decode_activation_workspace_layout_valid(const Workspace& w){return w.layout;}
const char* hipGetErrorString(hipError_t){return "injected copy failure";}
hipError_t hipMemcpy(void* to,const void* from,size_t n,int kind){
 ++copies;assert(from==actual_norm.data()&&n==4096&&kind==hipMemcpyDeviceToHost);
 if(fault==13)return hipErrorInvalidValue;std::memcpy(to,from,n);return hipSuccess;
}
namespace qrt_sm121_q1{
float widen(uint16_t value){uint32_t bits=uint32_t(value)<<16u;float out;std::memcpy(&out,&bits,4);return out;}
}
uint64_t qrt_fnv1a64_bytes(const void* p,size_t n){
 uint64_t h=1469598103934665603ULL;auto* bytes=static_cast<const unsigned char*>(p);
 for(size_t i=0;i<n;++i){h^=bytes[i];h*=1099511628211ULL;}return h;
}
uint64_t qrt_fnv1a64_update_bytes(uint64_t h,const void* p,size_t n){return h^qrt_fnv1a64_bytes(p,n);}
uint64_t qrt_now_ns(){static uint64_t n=1;return ++n;}
uint64_t qrt_elapsed_ns(uint64_t a,uint64_t b){return b-a;}
int qrt_qwen36_whole_provider_decode_v1(const qrt_qwen36_whole_provider_decode_request_v1_t* r,
 qrt_qwen36_whole_provider_decode_result_v1_t* out){
 ++calls;auto& s=g_qwen36_resident_session;
 assert(r->struct_size==sizeof(*r)&&r->abi_version==QRT_QWEN36_WHOLE_PROVIDER_DECODE_ABI_VERSION);
 assert(r->batch_size==1&&r->flags==QRT_QWEN36_WHOLE_PROVIDER_DECODE_FLAG_NONE&&r->output_token_capacity==2);
 assert(r->expected_prefix_token_count==8192&&r->expected_session_generation==77&&r->expected_prompt_token_ids_fnv1a64==567);
 assert(r->initial_output_token_id==63&&s.current_token_id==63&&!r->emit_callback&&!r->emit_user_data);
 assert(Qwen36NativeTargetOnly::matches(r)&&!Qwen36NativeTargetOnly::single_row_reference());
 const auto unrelated=*r;assert(!Qwen36NativeTargetOnly::matches(&unrelated));
 if(fault==15)throw std::runtime_error("injected target failure");
 out->completed=1;out->status=QRT_STATUS_OK;out->output_token_count=2;out->decode_token_count=1;
 out->output_tokens[0]=63;out->output_tokens[1]=220;s.current_token_id=220;s.committed_decode_token_count=1;
 if(fault==1){std::strcpy(out->failure_stage,"injected_target");std::strcpy(out->failure,"failed");return 0;}
 if(fault==2)out->completed=0;if(fault==3)out->output_tokens[0]=999;if(fault==4)out->decode_token_count=2;
 if(fault==5)++s.generation;if(fault==6)s.committed_decode_token_count=0;
 if(fault==7)--s.last_decode_top2_position;if(fault==8)s.last_decode_top2_ids[0]=64;
 if(fault==9)s.activation_workspace.device_norm_bf16=nullptr;if(fault==10)s.activation_workspace.in_use=true;
 if(fault==11)s.activation_workspace.phase=Qwen36ResidentDecodeActivationWorkspacePhase::kActive;
 if(fault==12)s.activation_workspace.layout=false;if(fault==14)actual_norm[2047]=0x7f80u;
 if(fault==16)s.valid=false;if(fault==17)out->output_token_count=1;
 if(fault==18)++s.prompt_token_ids_fnv1a64;if(fault==19)out->status=QRT_STATUS_UNSUPPORTED;
 if(fault==20)++s.activation_workspace.generation;
 return 1;
}
''' + bridge + r'''
int main(){
 std::vector<uint32_t> prompt(8193,32);prompt.back()=63;
 auto result=std::make_unique<qrt_qwen36_whole_provider_result_t>();
 qrt_qwen36_whole_provider_prefix_request_v1_t suffix{};
 suffix.expected_prefix_token_count=8192;suffix.suffix_token_count=1;suffix.suffix_tokens=prompt.data()+8192;
 std::string stage,failure;
 auto reset=[&]{g_qwen36_resident_session=Session{};calls=copies=0;
  for(size_t i=0;i<actual_norm.size();++i)actual_norm[i]=uint16_t(0x3f00u+i%128u);
  g_qwen36_resident_session.activation_workspace.device_norm_bf16=actual_norm.data();
  *result={};result->descriptor_result.wall_clock_ns=1234;stage.clear();failure.clear();};
 qrt_qwen36_whole_provider_decode_request_v1_t outer{};
 unsigned passed=0;
 for(fault=0;fault<=20;++fault){
  reset();qrt_mtp_target_rows::PrefillRows batch(prompt.data(),prompt.size(),8192,1);
  qrt_mtp_target_rows::Scope rows(&batch);
  Qwen36NativeTargetOnly prior(&outer);assert(Qwen36NativeTargetOnly::single_row_reference());
  bool ok=false,threw=false;
  try{ok=run_qwen36_prefill_single_tail(suffix,&stage,&failure,result.get());}catch(const std::runtime_error&){threw=true;}
  assert(Qwen36NativeTargetOnly::matches(&outer)&&Qwen36NativeTargetOnly::single_row_reference());
  assert(calls==1&&threw==(fault==15));
  if(!fault){
   assert(ok&&batch.published()&&batch.hidden()==std::vector<uint16_t>(actual_norm.begin(),actual_norm.end()));
   assert(batch.shifted_tokens()==std::vector<uint32_t>{220}&&batch.sampled_token()==220&&copies==1);
   assert(result->completed&&result->output_token_capacity==1&&result->output_token_count==1&&result->output_tokens[0]==220);
   assert(result->continuation.output_token_emitted&&result->continuation.output_token_id==220&&result->continuation.output_logit==9.75f);
   assert(result->continuation.lm_head_logits_fnv1a64==0&&result->continuation.sampler_fnv1a64);
   assert(result->descriptor_result.wall_clock_ns==0&&!result->prefill_emit_attempted&&!result->resident_session_valid);
  }else{assert(!ok&&!batch.published()&&batch.hidden().empty()&&batch.shifted_tokens().empty());}
  ++passed;
 }
 assert(!Qwen36NativeTargetOnly::single_row_reference()&&!Qwen36NativeTargetOnly::matches(&outer));
 fault=0;reset();direct=false;
 {qrt_mtp_target_rows::Scope no_rows(nullptr);assert(run_qwen36_prefill_single_tail(suffix,&stage,&failure,result.get())&&calls==1&&!copies);++passed;}
 reset();
 {qrt_mtp_target_rows::PrefillRows batch(prompt.data(),prompt.size(),8192,1);qrt_mtp_target_rows::Scope rows(&batch);
  assert(!run_qwen36_prefill_single_tail(suffix,&stage,&failure,result.get())&&!calls&&!batch.published());++passed;}
 direct=true;
 for(unsigned bad=0;bad<8;++bad){
  reset();auto candidate=suffix;
  qrt_mtp_target_rows::PrefillRows batch(prompt.data(),prompt.size(),8192,1);qrt_mtp_target_rows::Scope rows(&batch);
  if(bad==0)candidate.suffix_token_count=2;if(bad==1)candidate.suffix_tokens=nullptr;
  if(bad==2)++candidate.expected_prefix_token_count;if(bad==3)g_qwen36_resident_session.committed_decode_token_count=1;
  if(bad==4)g_qwen36_resident_session.dflash_prefetched_valid=true;if(bad==5)g_qwen36_resident_session.q2_prefetched_valid=true;
  if(bad==6)g_qwen36_resident_session.provider_completed=false;
  assert(!run_qwen36_prefill_single_tail(candidate,&stage,&failure,bad==7?nullptr:result.get())&&!calls&&!batch.published());++passed;
 }
 std::cout<<"single_tail_bridge=pass cases="<<passed<<" actual_input=63 private_sample=220 norm_values=2048\n";
}
'''
        with tempfile.TemporaryDirectory() as directory:
            src, exe = Path(directory) / 'bridge.cpp', Path(directory) / 'bridge'
            src.write_text(source)
            subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-I', str(ROOT), str(src), '-o', str(exe)],
                           check=True, timeout=40)
            subprocess.run([str(exe)], check=True, timeout=20)
