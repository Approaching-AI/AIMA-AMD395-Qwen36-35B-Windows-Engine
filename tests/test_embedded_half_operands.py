"""Lossless metadata encoding and bounded preparation submission contracts."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class EmbeddedHalfOperandsTests(unittest.TestCase):
    def test_every_bf16_payload_and_supported_scale(self):
        code = r'''
#include "native/providers/moe_accumulator/sm121_embedded_half_operands.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
namespace packed=qrt_sm121_embedded_half;
size_t checked=0,supported=0,unsupported=0;
void verify(const uint16_t* raw){
 unsigned lo=255,hi=0;bool valid=true,any=false,all=true;
 for(unsigned i=0;i<16;++i){
  const bool live=(raw[i]&0x7fff)!=0;all&=live;
  if(live){const unsigned e=(raw[i]>>7)&255;valid&=e>0&&e<255;lo=std::min(lo,e);hi=std::max(hi,e);any=true;}
 }
 valid&=!any||hi-lo<=29;
 packed::Row row;std::memset(&row,0xa5,sizeof(row));
 assert(packed::prepare(raw,&row)==valid);
 const auto original=qrt_sm121_scaled_half_products::prepare(raw);
 for(unsigned i=0;i<16;++i)assert(packed::original(row,i,valid)==raw[i]);
 if(valid){
  const unsigned meta=unsigned((any?int(hi)-142:-15)+142)|(all?256u:0u);
  for(unsigned lane=0;lane<4;++lane)assert(packed::metadata(row.pairs[lane*2],row.pairs[lane*2+1])==meta);
  for(unsigned i=0;i<8;++i){assert(!(original.pairs[i]&~packed::payload_mask));assert((row.pairs[i]&packed::payload_mask)==original.pairs[i]);}
  ++supported;
 }else{assert(!std::memcmp(row.pairs,original.pairs,sizeof(row)));++unsupported;}
 ++checked;
}
int main(){
 uint16_t row[16];
 for(unsigned bits=0;bits<65536;++bits){
  std::fill_n(row,16,uint16_t(bits));verify(row);
  const unsigned exponent=(bits>>7)&255;
  for(unsigned distance=0;distance<=30;++distance){
   const unsigned anchor=std::min(254u,exponent+distance);
   for(unsigned i=0;i<16;++i)row[i]=uint16_t((i&1?0x8000:0)|(anchor<<7)|(i*7&127));
   row[bits&15]=uint16_t(bits);verify(row);
   row[(bits+7)&15]=uint16_t(bits&0x8000);verify(row);
  }
 }
 for(unsigned signs=0;signs<65536;++signs){for(unsigned i=0;i<16;++i)row[i]=uint16_t(((signs>>i)&1)<<15);verify(row);}
 assert(supported&&unsupported);
 std::printf("{\"groups\":%zu,\"supported\":%zu,\"unsupported\":%zu,\"roundtrip_mismatches\":0}\n",checked,supported,unsupported);
}
'''
        self.compile_run(code)

    def test_preparation_capacities_and_submission_error(self):
        header = (ROOT / 'native/providers/moe_accumulator/sm121_embedded_half_projection.h').read_text()
        code = r'''
#include "native/providers/moe_accumulator/sm121_embedded_half_operands.h"
#include <cassert>
#include <cstddef>
#include <initializer_list>
namespace compact=qrt_sm121_embedded_half;
using Row=compact::Row;
enum hipError_t{hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
struct dim3{unsigned x;explicit dim3(unsigned a):x(a){}};
unsigned calls=0;bool fail=false;unsigned seen_rows=0;void* seen_stream=nullptr;
template<class... Args>void record(dim3 grid,dim3 block,unsigned shared,void* stream,Args...){++calls;assert(block.x==256&&!shared);seen_rows=grid.x;seen_stream=stream;}
hipError_t hipGetLastError(){return fail?hipErrorUnknown:hipSuccess;}
#define hipLaunchKernelGGL(kernel,...) record(__VA_ARGS__)
''' + function(header, 'inline hipError_t prepare(') + r'''
int main(){
 uint16_t input[1];Row output[1];unsigned flags[1];void* stream=reinterpret_cast<void*>(uintptr_t(123));
 auto run=[&](size_t iw=8976,size_t og=561,size_t fc=66){return prepare(input,iw,output,og,flags,fc,33,272,stream);};
 for(bool error:{false,true}){fail=error;calls=0;assert(run()==(error?hipErrorUnknown:hipSuccess));assert(calls==1&&seen_rows==33&&seen_stream==stream);}
 fail=false;calls=0;assert(run(8975)==hipErrorInvalidValue);assert(run(8976,560)==hipErrorInvalidValue);assert(run(8976,561,65)==hipErrorInvalidValue);
 for(unsigned width:{0u,15u,17u,8193u,UINT32_MAX})assert(prepare(input,SIZE_MAX,output,SIZE_MAX,flags,SIZE_MAX,33,width,stream)==hipErrorInvalidValue);
 for(unsigned rows:{0u,16385u,UINT32_MAX})assert(prepare(input,SIZE_MAX,output,SIZE_MAX,flags,SIZE_MAX,rows,272,stream)==hipErrorInvalidValue);
 assert(prepare(nullptr,SIZE_MAX,output,SIZE_MAX,flags,SIZE_MAX,33,272,stream)==hipErrorInvalidValue);
 assert(prepare(input,SIZE_MAX,nullptr,SIZE_MAX,flags,SIZE_MAX,33,272,stream)==hipErrorInvalidValue);
 assert(prepare(input,SIZE_MAX,output,SIZE_MAX,nullptr,SIZE_MAX,33,272,stream)==hipErrorInvalidValue);assert(!calls);
 assert(prepare(input,SIZE_MAX,output,SIZE_MAX,flags,SIZE_MAX,16384,8192,stream)==hipSuccess&&seen_rows==16384);
}
'''
        self.compile_run(code)

    def compile_run(self, code):
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / 'check')
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                            '-Wno-unknown-pragmas', '-fsanitize=address,undefined',
                            '-fno-sanitize-recover=all', '-I', str(ROOT), '-x', 'c++', '-', '-o', exe],
                           input=code, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=45)
