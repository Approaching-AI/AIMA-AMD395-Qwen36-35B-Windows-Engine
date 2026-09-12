"""Execute the actual suffix halo, ring, state layout and request validation."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class PrefixBatchSuffixTests(unittest.TestCase):
    def test_bf16_export_without_legacy_f32_prepare_and_submit_failures(self):
        header = (ROOT/'native/providers/prefix_batch_suffix.h').read_text()
        method = function(header, '    hipError_t attention(')
        source = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#define _WIN32 1
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorNotSupported,hipErrorUnknown};
constexpr int hipMemcpyDeviceToDevice=1;
unsigned copies=0,syncs=0,calls=0,fail_copy=0;bool fail_launch=false,missing_symbol=false;
hipError_t hipStreamSynchronize(void*){++syncs;return hipSuccess;}
hipError_t hipMemcpyAsync(void*,const void*,size_t bytes,int kind,void*){
 assert(bytes==1024u*1024u&&kind==1);return ++copies==fail_copy?hipErrorUnknown:hipSuccess;
}
int launch(const uint16_t*,const uint16_t*,const uint16_t*,const uint16_t*,const uint16_t*,float*,void*,unsigned prefix,unsigned tokens){
 ++calls;assert(copies==2&&prefix==16384&&tokens==1024);return fail_launch?hipErrorUnknown:hipSuccess;
}
struct Provider {std::mutex mutex;void* module=this;bool prepared=false;} provider;
Provider& ck_fmha_provider_state(){return provider;}
void* GetProcAddress(void* module,const char* name){assert(module==&provider&&std::strcmp(name,"qrt_ck_fmha_sm121_suffix_bf16_v1")==0);return missing_symbol?nullptr:reinterpret_cast<void*>(&launch);}
struct Layer {void* device_k=nullptr;void* device_v=nullptr;void* device_decode_tail_k=nullptr;void* device_decode_tail_v=nullptr;unsigned decode_tail_token_count=0;};
struct Session {std::array<Layer,40> full_attention_layers;};
struct Scope {Session* session;unsigned tokens=1024,prefix=16384;uint64_t attention_layers=0;std::string failure;
 bool reject(const char* s){failure=s;return false;}bool check(hipError_t s){return s==hipSuccess;}
''' + method + r'''
};
int main(){
 Session session{};uint16_t input[16]{};float output[16]{};
 auto call=[&](){copies=syncs=calls=0;session.full_attention_layers[3].decode_tail_token_count=0;
  Scope scope{&session,1024,16384,0,{}};auto status=scope.attention(3,input,input,input,output,1024);
  if(status==hipSuccess)assert(scope.attention_layers==8&&session.full_attention_layers[3].decode_tail_token_count==1024);
  else assert(!scope.attention_layers&&!session.full_attention_layers[3].decode_tail_token_count);
  return status;};
 assert(!provider.prepared&&call()==hipSuccess&&copies==2&&calls==1);
 provider.module=nullptr;assert(call()==hipErrorNotSupported&&!copies&&!calls);provider.module=&provider;
 missing_symbol=true;assert(call()==hipErrorNotSupported&&!copies&&!calls);missing_symbol=false;
 for(unsigned i=1;i<=2;++i){fail_copy=i;assert(call()==hipErrorUnknown&&copies==i&&!calls&&syncs==1);}
 fail_copy=0;fail_launch=true;assert(call()==hipErrorUnknown&&copies==2&&calls==1&&syncs==1);
 fail_launch=false;assert(call()==hipSuccess);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp)/'attention')
            subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
                '-fsanitize=undefined','-fno-sanitize-recover=all','-x','c++','-','-o',exe],
                input=source,text=True,check=True,timeout=30)
            subprocess.run([exe],check=True,timeout=15,capture_output=True)

    def test_original_ring_layouts_and_canonical_state(self):
        header = (ROOT/'native/providers/prefix_batch_suffix.h').read_text()
        whole = (ROOT/'native/providers/whole_provider.cpp').read_text()
        kernels = '\n'.join(function(header, '__global__ void '+name+'(') for name in (
            'qwen36_prefix_suffix_halo_kernel', 'qwen36_prefix_suffix_ring_kernel',
            'qwen36_prefix_suffix_canonical_state_kernel'))
        structures = '\n'.join(function(whole, name)+';' for name in (
            'enum class Qwen36ResidentSessionElementKind',
            'struct Qwen36ResidentSessionLinearLayer', 'struct Qwen36ResidentSessionFullAttentionLayer'))
        validation = function(header, '    bool validate()')
        source = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#define __global__
struct Dim {unsigned x=0;}; Dim blockIdx,threadIdx,blockDim{256};
float device_bf16_to_float(uint16_t value){uint32_t bits=uint32_t(value)<<16;float out;std::memcpy(&out,&bits,4);return out;}
uint16_t device_float_to_bf16(float value){uint32_t bits;std::memcpy(&bits,&value,4);return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);}
template<class F> void cells(size_t n,F fn){for(size_t i=0;i<n;++i){blockIdx.x=unsigned(i/256);threadIdx.x=unsigned(i%256);fn();}}
''' + kernels + structures + r'''
struct Session {
 bool valid=true; void* owner_engine=this; size_t prefix_tokens=16384,committed_decode_token_count=0;
 std::array<Qwen36ResidentSessionLinearLayer,40> linear_layers{};
 std::array<Qwen36ResidentSessionFullAttentionLayer,40> full_attention_layers{};
};
struct Validate {
 Validate* previous=nullptr;Session* session;unsigned prefix=16384,tokens=1024;std::string failure;
 bool reject(const char* s){failure=s;return false;}
''' + validation + r'''
};
int main(){
 const unsigned prefix=16384,tokens=1024;
 std::vector<float> qkv(size_t(tokens)*8192),ring(4u*8192),halo(size_t(tokens+3)*8192+256,-999.0f);
 std::vector<uint16_t> ring_bf16(4u*8192);
 for(size_t i=0;i<qkv.size();++i)qkv[i]=float(int(i%254)-127)/16.0f;
 for(size_t i=0;i<ring.size();++i){ring[i]=float(1000+i/8192)+float(i%16)/16;ring_bf16[i]=device_float_to_bf16(ring[i]);}
 const auto original_ring=ring;const auto original_bf16=ring_bf16;
 for(bool bf16:{false,true}){
  cells(halo.size(),[&]{qwen36_prefix_suffix_halo_kernel(qkv.data(),bf16?nullptr:ring.data(),bf16?ring_bf16.data():nullptr,halo.data(),prefix,tokens);});
  // The causal window for suffix row 0 is prefix positions 16381..16383 and its own new projection.
  for(unsigned row=0;row<3;++row)for(unsigned c=0;c<8192;++c)
   assert(halo[size_t(row)*8192+c]==(bf16?device_bf16_to_float(original_bf16[size_t(row+1)*8192+c]):original_ring[size_t(row+1)*8192+c]));
  assert(std::memcmp(halo.data()+3u*8192,qkv.data(),qkv.size()*4)==0);
  for(size_t i=size_t(tokens+3)*8192;i<halo.size();++i)assert(halo[i]==-999.0f);
  cells(4u*8192+256,[&]{qwen36_prefix_suffix_ring_kernel(qkv.data(),bf16?nullptr:ring.data(),bf16?ring_bf16.data():nullptr,prefix,tokens);});
  for(unsigned slot=0;slot<4;++slot)for(unsigned c=0;c<8192;++c){
   float expected=qkv[size_t(1020+slot)*8192+c];
   if(bf16)assert(ring_bf16[size_t(slot)*8192+c]==device_float_to_bf16(expected));
   else assert(ring[size_t(slot)*8192+c]==expected);
  }
  ring=original_ring;ring_bf16=original_bf16;
 }
 std::vector<float> state(524288),canonical(524288+256,-999.0f);
 for(unsigned i=0;i<state.size();++i)state[i]=float(i);
 for(bool key_major:{false,true}){
  cells(canonical.size(),[&]{qwen36_prefix_suffix_canonical_state_kernel(state.data(),canonical.data(),key_major);});
  for(unsigned h=0;h<32;++h)for(unsigned v=0;v<128;++v)for(unsigned k=0;k<128;++k)
   assert(canonical[h*16384+v*128+k]==float(h*16384+(key_major?k*128+v:v*128+k)));
  for(size_t i=state.size();i<canonical.size();++i)assert(canonical[i]==-999.0f);
 }
 Session session;
 for(unsigned i=0;i<40;++i){
  if(i%4!=3){auto& l=session.linear_layers[i];l.valid=true;l.device_recurrent_state=state.data();l.device_qkv_ring=ring.data();
   l.prefix_tokens=16384;l.recurrent_state_bytes=524288*4;l.qkv_element_kind=Qwen36ResidentSessionElementKind::kF32;l.qkv_ring_bytes=4*8192*4;
  }else{auto& l=session.full_attention_layers[i];l.valid=true;l.device_k=l.device_v=l.device_decode_tail_k=l.device_decode_tail_v=state.data();
   l.element_kind=Qwen36ResidentSessionElementKind::kBf16;l.history_tokens=16384;l.decode_tail_capacity_tokens=1536;
   l.k_bytes=l.v_bytes=16384*1024;l.decode_tail_k_bytes=l.decode_tail_v_bytes=1536*1024;}
 }
 Validate v{nullptr,&session,16384,1024,{}};assert(v.validate());
 for(unsigned i=0;i<40;++i){
  if(i%4!=3){auto& l=session.linear_layers[i];l.decode_recurrent_token_count=1;assert(!v.validate());l.decode_recurrent_token_count=0;
   l.qkv_ring_bytes-=4;assert(!v.validate());l.qkv_ring_bytes+=4;
   l.qkv_element_kind=Qwen36ResidentSessionElementKind::kBf16;l.qkv_ring_bytes/=2;assert(v.validate());
   l.qkv_element_kind=Qwen36ResidentSessionElementKind::kF32;l.qkv_ring_bytes*=2;
  }else{auto& l=session.full_attention_layers[i];l.decode_tail_token_count=1;assert(!v.validate());l.decode_tail_token_count=0;
   l.element_kind=Qwen36ResidentSessionElementKind::kF32;assert(!v.validate());l.element_kind=Qwen36ResidentSessionElementKind::kBf16;
   l.decode_tail_capacity_tokens=1023;assert(!v.validate());l.decode_tail_capacity_tokens=1536;}
 }
 v.tokens=1025;assert(!v.validate());v.tokens=1024;v.prefix=16383;assert(!v.validate());v.prefix=16384;
 v.previous=&v;assert(!v.validate());v.previous=nullptr;session.committed_decode_token_count=1;assert(!v.validate());
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp)/'prefix')
            subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
                '-fsanitize=undefined','-fno-sanitize-recover=all','-x','c++','-','-o',exe],
                input=source,text=True,check=True,timeout=30)
            subprocess.run([exe],check=True,timeout=20,capture_output=True)
