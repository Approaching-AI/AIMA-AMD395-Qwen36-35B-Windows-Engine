"""Read mixed original cache layouts without copying or mutating committed rows."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class Q2CacheViewTests(unittest.TestCase):
    def test_prefix_decode_and_private_rows(self):
        code=r'''
#include "native/providers/gdn/sm121_q2_cache_view.h"
#include <cassert>
#include <vector>
using namespace qrt_sm121_q2;
uint16_t expected(unsigned token,unsigned cell,bool value){return uint16_t(0x3e00u+token*128u+(cell%64u)+(value?64u:0u));}
struct PlaneOwner {
 std::vector<uint16_t> bf16;
 std::vector<float> f32;
 CachePlane view;
 PlaneOwner(unsigned first,unsigned tokens,unsigned stride,unsigned bytes):view{nullptr,nullptr,tokens,tokens+2u,stride,bytes}{
  const size_t elements=size_t(view.capacity)*1024u;
  bf16.assign(elements,0x7fc1u);f32.assign(elements,qrt_sm121_q1::widen(0x7fc1u));
  const size_t value_offset=stride==1024u?512u:size_t(view.capacity)*512u;
  for(unsigned row=0;row<tokens;++row)for(unsigned col=0;col<512u;++col)for(bool value:{false,true}){
   const size_t i=size_t(row)*stride+col+(value?value_offset:0u);const auto bits=expected(first+row,col,value);
   bf16[i]=bits;f32[i]=qrt_sm121_q1::widen(bits);
  }
  view.keys=bytes==2u?static_cast<const void*>(bf16.data()):static_cast<const void*>(f32.data());
  view.values=bytes==2u?static_cast<const void*>(bf16.data()+value_offset):static_cast<const void*>(f32.data()+value_offset);
 }
};
int main(){
 for(unsigned prefix:{0u,1u,5u})for(unsigned tail:{0u,1u,3u})for(unsigned ps:{512u,1024u})for(unsigned ds:{512u,1024u})
 for(unsigned pb:{2u,4u})for(unsigned db:{2u,4u}){
  PlaneOwner p(0,prefix,ps,pb),d(prefix,tail,ds,db);std::vector<uint16_t> staged(2048u);
  for(unsigned row=0;row<2u;++row)for(unsigned col=0;col<512u;++col)for(bool value:{false,true})
   staged[row*1024u+col+(value?512u:0u)]=expected(prefix+tail+row,col,value);
  CacheView v{prefix?p.view:CachePlane{},tail?d.view:CachePlane{},staged.data()};assert(valid_cache_view(v));
  const auto before_p=p.bf16,before_d=d.bf16,before_s=staged;const auto pf=p.f32,df=d.f32;
  for(unsigned token=0;token<prefix+tail+2u;++token)for(unsigned col=0;col<512u;++col){
   assert(v.key(token,col/256u,col%256u)==expected(token,col,false));
   assert(v.value(token,col/256u,col%256u)==expected(token,col,true));
  }
  assert(v.key(prefix+tail+2u,0,0)==0x7fc0u && v.key(0,2,0)==0x7fc0u && v.value(0,0,256u)==0x7fc0u);
  assert(before_p==p.bf16 && before_d==d.bf16 && before_s==staged);
  assert(!std::memcmp(pf.data(),p.f32.data(),pf.size()*4u) && !std::memcmp(df.data(),d.f32.data(),df.size()*4u));
 }
 uint16_t staged[2048]{};CacheView v{{},{},staged};assert(valid_cache_view(v));
 for(unsigned invalid:{0u,1u,3u,8u}){auto b=v;b.prefix={staged,staged,1,1,512u,invalid};assert(!valid_cache_view(b));}
 for(unsigned stride:{0u,511u,513u,~0u}){auto b=v;b.prefix={staged,staged,1,1,stride,2u};assert(!valid_cache_view(b));}
 auto b=v;b.prefix.tokens=1u;assert(!valid_cache_view(b));b=v;b.staged=nullptr;assert(!valid_cache_view(b));
 b=v;b.prefix={staged,staged+512u,1u,2u,1024u,2u};assert(!valid_cache_view(b));
 b=v;b.decoded={staged,staged+512u,1u,2u,1024u,2u};assert(!valid_cache_view(b));
 b=v;b.prefix={reinterpret_cast<void*>(~uintptr_t(1)),reinterpret_cast<void*>(~uintptr_t(1)),1u,1u,512u,2u};assert(!valid_cache_view(b));
 const auto fake=[](unsigned slot){return reinterpret_cast<const void*>((uintptr_t(1)<<36u)+(uintptr_t(slot)<<32u));};
 v.prefix={fake(1),fake(2),262144u,262144u,512u,2u};v.decoded={fake(3),fake(4),1534u,1536u,512u,4u};assert(valid_cache_view(v));
 b=v;++b.decoded.tokens;assert(!valid_cache_view(b));b=v;b.prefix.tokens=263679u;b.prefix.capacity=263680u;assert(!valid_cache_view(b));
 b=v;b.prefix.capacity=~0u;assert(!valid_cache_view(b));
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-q2-cache-') as temporary:
            src=Path(temporary)/'cache.cpp';src.write_text(code);exe=Path(temporary)/'cache'
            subprocess.run([os.getenv('CXX','c++'),'-std=c++17','-O1','-Wall','-Wextra','-Werror',
                '-ffp-contract=off','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I',str(ROOT),str(src),'-o',str(exe)],check=True,timeout=60)
            subprocess.run([str(exe)],check=True,timeout=15)


if __name__=='__main__':
    unittest.main()
