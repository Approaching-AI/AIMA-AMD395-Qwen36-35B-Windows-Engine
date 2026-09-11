"""Check K16 row layouts against model coordinates, including causal padding."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class IntegerAttentionPrepackingTests(unittest.TestCase):
    def test_qkv_probability_coordinates_and_partial_final_online_tile(self):
        source = (ROOT/'native/providers/ck_fmha/blackwell_attention.h').read_text()
        actual = '\n'.join(function(source, signature) for signature in (
            '__host__ __device__ inline size_t integer_row_count(',
            '__host__ __device__ inline size_t integer_row_input_index(',
        ))
        program = r'''
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#define __host__
#define __device__
constexpr unsigned kKvHeads=2,kQueryHeads=16,kHeadDim=256;
enum class IntegerRowKind { Query, Key, Value, Probability };
''' + actual + r'''
int main(){
 size_t checked=0,padding=0;
 for(unsigned tokens:{1u,15u,16u,17u,31u,32u,33u,63u,64u,65u,8191u,8192u,8193u,16384u}){
  const unsigned queries=tokens<32?tokens:32,first=tokens-queries;
  const unsigned groups=((tokens+31)/32)*2;
  assert(integer_row_count(IntegerRowKind::Key,tokens,queries)==size_t(tokens)*512/16);
  assert(integer_row_count(IntegerRowKind::Value,tokens,queries)==size_t(groups)*512);
  for(unsigned t=0;t<tokens;++t)for(unsigned h=0;h<2;++h)for(unsigned d=0;d<256;++d){
   const size_t row=(h*16u+d/16u)*size_t(tokens)+t;
   assert(integer_row_input_index(IntegerRowKind::Key,row,d%16,tokens,first,queries)==(size_t(t)*2+h)*256+d);++checked;
  }
  for(unsigned t=0;t<groups*16;++t)for(unsigned h=0;h<2;++h)for(unsigned d=0;d<256;++d){
   const size_t row=(t/16u*2u+h)*size_t(256)+d;
   const size_t expect=t<tokens?(size_t(t)*2+h)*256+d:size_t(-1);
   assert(integer_row_input_index(IntegerRowKind::Value,row,t%16,tokens,first,queries)==expect);
   ++checked;padding+=expect==size_t(-1);
  }
  for(unsigned q=0;q<queries;++q)for(unsigned h=0;h<16;++h){
   for(unsigned d=0;d<256;++d){
    const size_t row=(size_t(q)*16+h)*16+d/16;
    assert(integer_row_input_index(IntegerRowKind::Query,row,d%16,tokens,first,queries)==((size_t(first)+q)*16+h)*256+d);++checked;
   }
   for(unsigned key=0;key<groups*16;++key){
    const size_t row=(size_t(q)*16+h)*groups+key/16;
    const size_t expect=key<tokens && key<=first+q?(size_t(q)*16+h)*tokens+key:size_t(-1);
    assert(integer_row_input_index(IntegerRowKind::Probability,row,key%16,tokens,first,queries)==expect);
    ++checked;padding+=expect==size_t(-1);
   }
  }
  for(auto kind:{IntegerRowKind::Query,IntegerRowKind::Key,IntegerRowKind::Value,IntegerRowKind::Probability}){
   assert(integer_row_input_index(kind,integer_row_count(kind,tokens,queries),0,tokens,first,queries)==size_t(-1));
   assert(integer_row_input_index(kind,0,16,tokens,first,queries)==size_t(-1));
  }
 }
 assert(padding>0);std::printf("prepacked_integer_coordinates=%zu causal_or_online_padding=%zu pass\n",checked,padding);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-integer-rows-') as tmp:
            exe = str(Path(tmp)/'rows')
            subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-Wall','-Wextra','-Werror',
                            '-fsanitize=undefined','-x','c++','-','-o',exe],
                           input=program,text=True,check=True,timeout=30)
            subprocess.run([exe],check=True,timeout=30)


if __name__ == '__main__':
    unittest.main()
