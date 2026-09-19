"""Check actual compact-suffix staging, validation and failed-submit cleanup."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import attention_capacity, compact_decode_policy, function, workspace_capacity_policy

ROOT = Path(__file__).resolve().parents[1]


class AttentionSuffixTests(unittest.TestCase):
    def test_actual_staging_offsets_refresh_and_failure_drain(self):
        text = (ROOT/'native/providers/ck_fmha/qrt_ck_fmha_q8192_provider.cpp').read_text()
        workspace = function(text, 'struct Sm121SuffixWorkspace') + ';'
        implementation = function(text, 'int launch_sm121_suffix_attention(')
        implementation = (ROOT/'native/providers/ck_fmha/discardable_workspace.h').read_text().replace(
            '#pragma once', '') + implementation
        source = r'''
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <set>
#include <string>
#include <utility>
using hipStream_t=void*;
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorNotSupported,hipErrorUnknown};
''' + attention_capacity() + workspace_capacity_policy() + compact_decode_policy() + r'''
constexpr unsigned kQueryFeatures=4096,kKvFeatures=512,kSm121MaxTokens=qrt_sm121_attention_capacity::kTokens;
constexpr unsigned kPrefillChunkTokens=8192;
constexpr int hipMemcpyDeviceToDevice=1;
''' + workspace + r'''
Sm121SuffixWorkspace g_sm121_suffix,g_sm121_compact_suffix;
std::mutex g_sm121_suffix_mutex;
unsigned copies=0,launches=0,syncs=0,allocations=0,fail_copy=0,expected_prefix=0,expected_suffix=0;
bool fail_allocate=false,fail_launch=false,enabled=true,expected_compact=false;
size_t last_allocation_bytes=0;
const void* expected_inputs[5]{};
float* expected_output=nullptr;
hipStream_t expected_stream=reinterpret_cast<void*>(uintptr_t(16));
std::set<void*> live;
bool sm121_attention_enabled(unsigned n){assert(n<=kSm121MaxTokens);return enabled;}
hipError_t hipMalloc(void** p,size_t bytes){
 ++allocations;last_allocation_bytes=bytes;
 if(fail_allocate){*p=nullptr;return hipErrorUnknown;}
 *p=std::malloc(bytes);assert(*p);assert(live.insert(*p).second);return hipSuccess;
}
hipError_t hipFree(void* p){if(p){assert(live.erase(p)==1);std::free(p);}return hipSuccess;}
hipError_t hipStreamSynchronize(hipStream_t stream){assert(stream==expected_stream);++syncs;return hipSuccess;}
hipError_t hipMemcpyAsync(void* out,const void* in,size_t bytes,int kind,hipStream_t stream){
 assert(kind==hipMemcpyDeviceToDevice&&stream==expected_stream&&copies<5);
 auto& owner=expected_compact?g_sm121_compact_suffix:g_sm121_suffix;
 auto* q=owner.cells;
 auto* k=q+(expected_compact?0u:size_t(owner.capacity_tokens)*4096u);
 auto* v=k+size_t(owner.capacity_tokens)*512;
 void* destinations[]={expected_compact?nullptr:q+size_t(expected_prefix)*4096,k,v,k+size_t(expected_prefix)*512,v+size_t(expected_prefix)*512};
 const size_t sizes[]={size_t(expected_suffix)*8192,size_t(expected_prefix)*1024,size_t(expected_prefix)*1024,
                       size_t(expected_suffix)*1024,size_t(expected_suffix)*1024};
 const unsigned index=copies+(expected_compact?1u:0u);
 assert(index<5u&&out==destinations[index]&&in==expected_inputs[index]&&bytes==sizes[index]);
 return ++copies==fail_copy?hipErrorUnknown:hipSuccess;
}
int launch_sm121_attention(const uint16_t* q,const uint16_t* k,const uint16_t* v,float* out,
 hipStream_t stream,unsigned start,unsigned count,unsigned output_start,unsigned query_origin){
 auto& owner=expected_compact?g_sm121_compact_suffix:g_sm121_suffix;
 ++launches;assert(copies==(expected_compact?4u:5u)&&start==expected_prefix&&count==expected_suffix&&output_start==0);
 assert(q==(expected_compact?expected_inputs[0]:owner.cells));
 assert(k==owner.cells+(expected_compact?0u:size_t(owner.capacity_tokens)*4096u));
 assert(query_origin==(expected_compact?expected_prefix:0u));
 assert(v==k+size_t(owner.capacity_tokens)*512&&out==expected_output&&stream==expected_stream);
 return fail_launch?hipErrorUnknown:hipSuccess;
}
''' + implementation + r'''
int main(){
 unsetenv("QRT_CK_SM121_COMPACT_DECODE_QUERY");
 unsetenv("QRT_CK_SM121_LONG_ATTENTION_PIPELINE");
 unsetenv("QRT_CK_SM121_WORKSPACE_RESERVE_TOKENS");
 // Only mock transport inspects these allocations; no data pages need touching.
 for(unsigned i=0;i<5;++i){expected_inputs[i]=std::malloc(i?size_t(kSm121MaxTokens)*1024u:8192u*8192u);assert(expected_inputs[i]);}
 expected_output=static_cast<float*>(std::malloc(8192u*4096u*4u));assert(expected_output);
 auto call=[&](unsigned prefix,unsigned suffix){
  copies=launches=syncs=0;expected_prefix=prefix;expected_suffix=suffix;
  return launch_sm121_suffix_attention((const uint16_t*)expected_inputs[0],(const uint16_t*)expected_inputs[1],
   (const uint16_t*)expected_inputs[2],(const uint16_t*)expected_inputs[3],(const uint16_t*)expected_inputs[4],
   expected_output,expected_stream,prefix,suffix);
 };
 for(auto shape:{std::pair<unsigned,unsigned>{0,1024},{16384,0},{16384,8193},
                 {kSm121MaxTokens,1},{kSm121MaxTokens-68,69},{UINT32_MAX,1024}}){
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
 assert(call(32768,1024)==hipErrorUnknown&&copies==0&&!g_sm121_suffix.cells&&!g_sm121_suffix.capacity_tokens&&live.empty());
 fail_allocate=false;assert(call(32768,1024)==hipSuccess&&g_sm121_suffix.capacity_tokens==33792&&live.size()==1);
 assert(call(32768,8192)==hipSuccess&&copies==5&&g_sm121_suffix.capacity_tokens==40960);
 assert(call(32768,1)==hipSuccess&&copies==5);
 retained=g_sm121_suffix.cells;fail_allocate=true;
 assert(call(65536,1024)==hipErrorUnknown&&copies==0&&!g_sm121_suffix.cells&&!g_sm121_suffix.capacity_tokens&&live.empty());
 fail_allocate=false;
 assert(call(65536,1024)==hipSuccess&&copies==5&&g_sm121_suffix.capacity_tokens==66560&&live.size()==1);
 assert(call(66560,512)==hipSuccess&&copies==5&&g_sm121_suffix.capacity_tokens==67072&&live.size()==1);
 assert(call(32700,69)==hipSuccess&&copies==5);
 assert(call(kSm121MaxTokens-1024,1024)==hipSuccess&&g_sm121_suffix.capacity_tokens==kSm121MaxTokens&&live.size()==1);
 assert(call(16384,1024)==hipSuccess&&copies==5);
 retained=g_sm121_suffix.cells;
 setenv("QRT_CK_SM121_LONG_ATTENTION_PIPELINE","1",1);expected_compact=true;
 fail_allocate=true;
 assert(call(8192,8192)==hipErrorUnknown&&!g_sm121_compact_suffix.cells&&g_sm121_suffix.cells==retained&&live.size()==1);
 fail_allocate=false;
 for(auto shape:{std::pair<unsigned,unsigned>{8192,8192},{16384,129},{32768,1024},
                 {131072,8192},{262144,1024},{kSm121MaxTokens-2u,2u}}){
  assert(call(shape.first,shape.second)==hipSuccess&&copies==4&&launches==1&&live.size()==2);
  assert(last_allocation_bytes==size_t(g_sm121_compact_suffix.capacity_tokens)*2048u);
  assert(g_sm121_suffix.cells==retained);
 }
 auto* compact=g_sm121_compact_suffix.cells;before=allocations;
 std::swap(expected_inputs[1],expected_inputs[2]);
 assert(call(262144,1024)==hipSuccess&&copies==4&&allocations==before&&g_sm121_compact_suffix.cells==compact);
 std::swap(expected_inputs[1],expected_inputs[2]);
 for(unsigned i=1;i<=4;++i){
  fail_copy=i;assert(call(262144,1024)==hipErrorUnknown&&copies==i&&!launches&&syncs==1);
 }
 fail_copy=0;fail_launch=true;
 assert(call(262144,1024)==hipErrorUnknown&&copies==4&&launches==1&&syncs==1);
 fail_launch=false;expected_compact=false;
 assert(call(262144,1)==hipSuccess&&copies==5&&allocations==before);
 unsetenv("QRT_CK_SM121_LONG_ATTENTION_PIPELINE");
 assert(call(262144,1024)==hipSuccess&&copies==5&&g_sm121_suffix.cells==retained&&g_sm121_compact_suffix.cells==compact);
 // Reserve compact KV once; every copy and launcher still uses true extents.
 hipFree(g_sm121_compact_suffix.cells);g_sm121_compact_suffix={};
 setenv("QRT_CK_SM121_LONG_ATTENTION_PIPELINE","1",1);expected_compact=true;
 const auto reservation=std::to_string(kSm121MaxTokens);
 for(const char* invalid:{"-1","+1","true","1 ","99999999999999999999"}){
  before=allocations;setenv("QRT_CK_SM121_WORKSPACE_RESERVE_TOKENS",invalid,1);
  assert(call(8192,129)==hipErrorInvalidValue&&allocations==before&&!copies&&!launches);
 }
 setenv("QRT_CK_SM121_WORKSPACE_RESERVE_TOKENS",reservation.c_str(),1);
 fail_allocate=true;
 assert(call(8192,129)==hipErrorUnknown&&!copies&&!g_sm121_compact_suffix.cells);
 fail_allocate=false;
 assert(call(8192,129)==hipSuccess&&g_sm121_compact_suffix.capacity_tokens==kSm121MaxTokens);
 compact=g_sm121_compact_suffix.cells;before=allocations;
 for(auto shape:{std::pair<unsigned,unsigned>{16384,8192},{131072,8192},{262144,1024},{kSm121MaxTokens-2u,2u}}){
  assert(call(shape.first,shape.second)==hipSuccess&&copies==4&&launches==1&&allocations==before&&
         g_sm121_compact_suffix.cells==compact&&g_sm121_suffix.cells==retained);
 }
 for(unsigned failure=1;failure<=4;++failure){
  fail_copy=failure;assert(call(16384,129)==hipErrorUnknown&&copies==failure&&!launches&&syncs==1);
 }
 fail_copy=0;fail_launch=true;
 assert(call(16384,129)==hipErrorUnknown&&copies==4&&launches==1&&syncs==1);
 fail_launch=false;
 unsetenv("QRT_CK_SM121_WORKSPACE_RESERVE_TOKENS");
 // Q1 borrows exactly the caller row and can reuse prior compact KV capacity.
 unsetenv("QRT_CK_SM121_LONG_ATTENTION_PIPELINE");
 for(const char* invalid:{"-1","2","01","1 ","true"}){
  before=allocations;setenv("QRT_CK_SM121_COMPACT_DECODE_QUERY",invalid,1);
  assert(call(262144,1)==hipErrorInvalidValue&&allocations==before&&!copies&&!launches);
 }
 setenv("QRT_CK_SM121_COMPACT_DECODE_QUERY","1",1);before=allocations;
 for(unsigned prefix:{1u,8191u,8192u,131072u,262144u,kSm121MaxTokens-1u}){
  assert(call(prefix,1)==hipSuccess&&copies==4&&launches==1&&allocations==before&&
         g_sm121_compact_suffix.cells==compact&&g_sm121_suffix.cells==retained);
 }
 for(unsigned failure=1;failure<=4;++failure){
  fail_copy=failure;assert(call(262144,1)==hipErrorUnknown&&copies==failure&&!launches&&syncs==1);
 }
 fail_copy=0;fail_launch=true;
 assert(call(262144,1)==hipErrorUnknown&&copies==4&&launches==1&&syncs==1);
 fail_launch=false;
 hipFree(g_sm121_compact_suffix.cells);g_sm121_compact_suffix={};
 fail_allocate=true;
 assert(call(131072,1)==hipErrorUnknown&&!copies&&!launches&&!g_sm121_compact_suffix.cells);
 fail_allocate=false;
 assert(call(131072,1)==hipSuccess&&copies==4&&last_allocation_bytes==size_t(139264u)*2048u);
 before=allocations;compact=g_sm121_compact_suffix.cells;
 assert(call(131073,1)==hipSuccess&&copies==4&&allocations==before&&g_sm121_compact_suffix.cells==compact);
 // A larger decode refreshes the replacement KV; failed growth leaves no owner.
 fail_allocate=true;
 assert(call(262144,1)==hipErrorUnknown&&!copies&&!g_sm121_compact_suffix.cells);
 fail_allocate=false;
 assert(call(262144,1)==hipSuccess&&copies==4&&last_allocation_bytes==size_t(kSm121MaxTokens)*2048u);
 before=allocations;compact=g_sm121_compact_suffix.cells;
 assert(call(kSm121MaxTokens-1u,1)==hipSuccess&&copies==4&&allocations==before&&g_sm121_compact_suffix.cells==compact);
 expected_compact=false;
 assert(call(16384,1024)==hipSuccess&&copies==5&&g_sm121_suffix.cells==retained);
 for(const char* off:{"","0"}){
  setenv("QRT_CK_SM121_COMPACT_DECODE_QUERY",off,1);
  assert(call(262144,1)==hipSuccess&&copies==5&&g_sm121_suffix.cells==retained);
 }
 unsetenv("QRT_CK_SM121_COMPACT_DECODE_QUERY");
 hipFree(g_sm121_suffix.cells);hipFree(g_sm121_compact_suffix.cells);assert(live.empty());
 for(auto p:expected_inputs)std::free(const_cast<void*>(p));std::free(expected_output);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp)/'suffix')
            subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
                '-fsanitize=undefined','-fno-sanitize-recover=all','-x','c++','-','-o',exe],
                input=source,text=True,check=True,timeout=30)
            result = subprocess.run([exe],check=True,timeout=15,capture_output=True,text=True)
            markers = [line for line in result.stderr.splitlines()
                       if line.startswith('SM121_COMPACT_SUFFIX_QUERY ')]
            self.assertEqual(len(markers), 12)
            self.assertTrue(all('query_staging_bytes=0 copies=4 original_q_view=1 stream_drained=1'
                                in line for line in markers))
            decode_markers = [line for line in result.stderr.splitlines()
                              if line.startswith('SM121_COMPACT_DECODE_QUERY ')]
            self.assertEqual(len(decode_markers), 10)
            self.assertTrue(all('suffix_tokens=1 ' in line and
                'original_single_query_arithmetic=1 stream_drained=1' in line for line in decode_markers))
