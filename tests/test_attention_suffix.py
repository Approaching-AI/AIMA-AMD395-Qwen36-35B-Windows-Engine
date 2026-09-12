"""Check actual compact-suffix staging, validation and failed-submit cleanup."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class AttentionSuffixTests(unittest.TestCase):
    def test_actual_staging_offsets_refresh_and_failure_drain(self):
        text = (ROOT/'native/providers/ck_fmha/qrt_ck_fmha_q8192_provider.cpp').read_text()
        workspace = function(text, 'struct Sm121SuffixWorkspace') + ';'
        implementation = function(text, 'int launch_sm121_suffix_attention(')
        source = r'''
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <mutex>
#include <set>
#include <utility>
using hipStream_t=void*;
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorNotSupported,hipErrorUnknown};
constexpr unsigned kQueryFeatures=4096,kKvFeatures=512,kSm121MaxTokens=32768;
constexpr int hipMemcpyDeviceToDevice=1;
''' + workspace + r'''
Sm121SuffixWorkspace g_sm121_suffix;
std::mutex g_sm121_suffix_mutex;
unsigned copies=0,launches=0,syncs=0,allocations=0,fail_copy=0,expected_prefix=0,expected_suffix=0;
bool fail_allocate=false,fail_launch=false,enabled=true;
const void* expected_inputs[5]{};
float* expected_output=nullptr;
hipStream_t expected_stream=reinterpret_cast<void*>(uintptr_t(16));
std::set<void*> live;
bool sm121_attention_enabled(unsigned n){assert(n<=32768);return enabled;}
hipError_t hipMalloc(void** p,size_t bytes){
 ++allocations;
 if(fail_allocate){*p=nullptr;return hipErrorUnknown;}
 *p=std::malloc(bytes);assert(*p);assert(live.insert(*p).second);return hipSuccess;
}
hipError_t hipFree(void* p){if(p){assert(live.erase(p)==1);std::free(p);}return hipSuccess;}
hipError_t hipStreamSynchronize(hipStream_t stream){assert(stream==expected_stream);++syncs;return hipSuccess;}
hipError_t hipMemcpyAsync(void* out,const void* in,size_t bytes,int kind,hipStream_t stream){
 assert(kind==hipMemcpyDeviceToDevice&&stream==expected_stream&&copies<5);
 auto* q=g_sm121_suffix.cells;
 auto* k=q+size_t(g_sm121_suffix.capacity_tokens)*4096;
 auto* v=k+size_t(g_sm121_suffix.capacity_tokens)*512;
 void* destinations[]={q+size_t(expected_prefix)*4096,k,v,k+size_t(expected_prefix)*512,v+size_t(expected_prefix)*512};
 const size_t sizes[]={size_t(expected_suffix)*8192,size_t(expected_prefix)*1024,size_t(expected_prefix)*1024,
                       size_t(expected_suffix)*1024,size_t(expected_suffix)*1024};
 assert(out==destinations[copies]&&in==expected_inputs[copies]&&bytes==sizes[copies]);
 return ++copies==fail_copy?hipErrorUnknown:hipSuccess;
}
int launch_sm121_attention(const uint16_t* q,const uint16_t* k,const uint16_t* v,float* out,
 hipStream_t stream,unsigned start,unsigned count,unsigned output_start){
 ++launches;assert(copies==5&&start==expected_prefix&&count==expected_suffix&&output_start==0);
 assert(q==g_sm121_suffix.cells&&k==q+size_t(g_sm121_suffix.capacity_tokens)*4096);
 assert(v==k+size_t(g_sm121_suffix.capacity_tokens)*512&&out==expected_output&&stream==expected_stream);
 return fail_launch?hipErrorUnknown:hipSuccess;
}
''' + implementation + r'''
int main(){
 // Only mock transport inspects these allocations; no data pages need touching.
 for(unsigned i=0;i<5;++i){expected_inputs[i]=std::malloc(i?32768u*1024u:1024u*8192u);assert(expected_inputs[i]);}
 expected_output=static_cast<float*>(std::malloc(1024u*4096u*4u));assert(expected_output);
 auto call=[&](unsigned prefix,unsigned suffix){
  copies=launches=syncs=0;expected_prefix=prefix;expected_suffix=suffix;
  return launch_sm121_suffix_attention((const uint16_t*)expected_inputs[0],(const uint16_t*)expected_inputs[1],
   (const uint16_t*)expected_inputs[2],(const uint16_t*)expected_inputs[3],(const uint16_t*)expected_inputs[4],
   expected_output,expected_stream,prefix,suffix);
 };
 for(auto shape:{std::pair<unsigned,unsigned>{0,1024},{16384,0},{16384,1025},{32768,1},{32700,69}}){
  assert(call(shape.first,shape.second)==hipErrorInvalidValue&&allocations==0&&copies==0&&launches==0);
 }
 const auto original=expected_inputs[0];
 for(const void* invalid:std::initializer_list<const void*>{nullptr,(const void*)expected_output,(const void*)(UINTPTR_MAX-1u),
                         (const void*)(reinterpret_cast<uintptr_t>(original)+1u)}){
  expected_inputs[0]=invalid;assert(call(16384,1024)==hipErrorInvalidValue&&allocations==0);
 }
 expected_inputs[0]=original;
 enabled=false;assert(call(16384,1024)==hipErrorNotSupported&&allocations==0);enabled=true;
 fail_allocate=true;assert(call(16384,1024)==hipErrorUnknown&&!g_sm121_suffix.cells&&copies==0);
 fail_allocate=false;assert(call(16384,1024)==hipSuccess&&copies==5&&launches==1&&live.size()==1);
 auto* retained=g_sm121_suffix.cells;unsigned before=allocations;
 // A second input identity and a shorter extent must refresh every consumed span.
 std::swap(expected_inputs[1],expected_inputs[2]);
 assert(call(7168,1)==hipSuccess&&allocations==before&&copies==5&&g_sm121_suffix.cells==retained);
 std::swap(expected_inputs[1],expected_inputs[2]);
 for(unsigned i=1;i<=5;++i){
  fail_copy=i;assert(call(16384,1024)==hipErrorUnknown&&copies==i&&launches==0&&syncs==1);
  assert(g_sm121_suffix.cells==retained&&live.size()==1);
 }
 fail_copy=0;fail_launch=true;
 assert(call(16384,1024)==hipErrorUnknown&&copies==5&&launches==1&&syncs==1);
 fail_launch=false;fail_allocate=true;
 assert(call(31744,1024)==hipErrorUnknown&&copies==0&&g_sm121_suffix.cells==retained&&live.size()==1);
 fail_allocate=false;assert(call(31744,1024)==hipSuccess&&g_sm121_suffix.capacity_tokens==32768&&live.size()==1);
 assert(call(16384,1024)==hipSuccess&&copies==5);
 hipFree(g_sm121_suffix.cells);assert(live.empty());
 for(auto p:expected_inputs)std::free(const_cast<void*>(p));std::free(expected_output);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp)/'suffix')
            subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
                '-fsanitize=undefined','-fno-sanitize-recover=all','-x','c++','-','-o',exe],
                input=source,text=True,check=True,timeout=30)
            subprocess.run([exe],check=True,timeout=15,capture_output=True)
