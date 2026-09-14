from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class StagedHalfPvLaunchTests(unittest.TestCase):
    def test_actual_preparation_tail_capacity_and_replay_bounds(self):
        text = (ROOT / 'native/providers/ck_fmha/staged_half_pv_replay.h').read_text()
        kernel = function(text, '__global__ void prepare_rows(')
        prepare = function(text, 'inline int prepare(')
        launch = function(text, 'inline int launch(')
        source = r'''
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#define __global__
#define HIP_KERNEL_NAME(x) x
#define hipLaunchKernelGGL(...) launch_mock(__VA_ARGS__)
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
dim3 blockIdx,blockDim,threadIdx;
using hipStream_t=void*;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipErrorUnknown=2;
constexpr unsigned heads=16,kv_heads=2,dimensions=256,threads=256,maximum_tokens=264736;
struct Row { uint32_t pairs[8],control; };
namespace staged { namespace half {
Row prepare(const uint16_t* data){Row row{};std::memcpy(row.pairs,data,32);row.control=0x12345678;return row;}
} }
unsigned calls=0,last_grid=0,last_start=0,last_queries=0,last_stride=0,last_value_stride=0;
bool execute=false,failed=false;
int hipGetLastError(){return failed?hipErrorUnknown:hipSuccess;}
void launch_mock(void(*kernel)(const uint16_t*,unsigned,unsigned,unsigned,Row*),dim3 grid,dim3 block,
 unsigned shared,hipStream_t stream,const uint16_t* input,unsigned tokens,unsigned start,unsigned count,Row* output){
 assert(!shared&&stream==reinterpret_cast<void*>(17)&&block.x==256);++calls;last_grid=grid.x;
 if(execute){blockDim=block;for(unsigned b=0;b<grid.x;++b)for(unsigned t=0;t<block.x;++t){blockIdx.x=b;threadIdx.x=t;kernel(input,tokens,start,count,output);}}
}
template<bool Audit> constexpr int replay=Audit?3:2;
void launch_mock(int kind,dim3 grid,dim3 block,unsigned shared,hipStream_t stream,
 const Row* p,const Row* v,const float* scales,float* out,unsigned start,unsigned count,unsigned output,
 unsigned stride,unsigned value_stride,const unsigned char*,float*,float*,const unsigned* indices,
 const unsigned* selected,unsigned long long* audit){
 assert((kind==2||kind==3)&&p&&v&&scales&&out&&indices&&selected&&(!shared)&&block.x==256&&stream==reinterpret_cast<void*>(17));
 assert(kind!=3||audit);assert(output+count<=maximum_tokens);++calls;last_grid=grid.x;
 last_start=start;last_queries=count;last_stride=stride;last_value_stride=value_stride;
}
'''
        source += '\ntemplate<bool Probability>\n' + kernel
        source += '\ntemplate<bool Probability>\n' + prepare
        source += '\ntemplate<bool Audit=false>\n' + launch
        source += r'''
template<bool Probability>void check_prepare(unsigned tokens,unsigned start,unsigned count){
 const unsigned rows=Probability?count*heads:512,groups=(tokens+15)/16;
 const size_t records=size_t(rows)*groups;constexpr size_t guard=19;
 std::vector<uint16_t> input(size_t(rows)*tokens+2*guard,0x5a5a);
 for(size_t i=guard;i+guard<input.size();++i)input[i]=uint16_t(i*173+i/17);
 const auto before=input;
 Row sentinel;std::memset(&sentinel,0xa5,sizeof(sentinel));
 std::vector<Row> output(records+2*guard,sentinel);execute=true;calls=0;failed=false;
 assert(prepare<Probability>(input.data()+guard,tokens,start,count,output.data()+guard,records,reinterpret_cast<void*>(17))==0);
 assert(calls==1&&last_grid==(records+255)/256&&input==before);
 for(unsigned row=0;row<rows;++row)for(unsigned group=0;group<groups;++group){
  uint16_t expected[16]{};const unsigned extent=Probability?start+row/heads+1:tokens;
  for(unsigned i=0;i<16;++i)if(group*16+i<extent)expected[i]=input[guard+size_t(row)*tokens+group*16+i];
  const Row encoded=staged::half::prepare(expected);
  assert(!std::memcmp(&output[guard+size_t(row)*groups+group],&encoded,sizeof(Row)));
 }
 for(size_t i=0;i<guard;++i)assert(!std::memcmp(&output[i],&sentinel,sizeof(Row))&&!std::memcmp(&output[guard+records+i],&sentinel,sizeof(Row)));
 execute=false;calls=0;
 assert(prepare<Probability>(input.data()+guard,tokens,start,count,output.data()+guard,records-1,reinterpret_cast<void*>(17))==1&&calls==0);
 failed=true;assert(prepare<Probability>(input.data()+guard,tokens,start,count,output.data()+guard,records,reinterpret_cast<void*>(17))==2&&calls==1);failed=false;
}
int main(){
 for(unsigned n:{1u,2u,15u,16u,17u,31u,32u,33u,129u}){
  check_prepare<false>(n,0,0);
  for(unsigned q:{1u,std::min(32u,n),std::min(128u,n)})check_prepare<true>(n,n-q,q);
 }
 uint16_t input=0;Row p{},v{};float out=0,scale=1;unsigned index=0,count=0;unsigned long long audit=0;auto stream=reinterpret_cast<void*>(17);
 execute=false;failed=false;calls=0;
 assert(prepare<true>(&input,maximum_tokens,maximum_tokens-128,128,&p,size_t(-1),stream)==0);
 assert(last_grid==(size_t(128)*16*((maximum_tokens+15)/16)+255)/256);
 for(unsigned n:{0u,maximum_tokens+1,~0u})assert(prepare<false>(&input,n,0,0,&v,size_t(-1),stream)==1);
 assert(prepare<true>(&input,33,33,1,&p,size_t(-1),stream)==1);
 assert(prepare<true>(&input,33,32,2,&p,size_t(-1),stream)==1);
 assert(prepare<true>(&input,256,0,129,&p,size_t(-1),stream)==1);
 assert(prepare<false>(&input,33,1,0,&p,size_t(-1),stream)==1);
 assert(prepare<false>(&input,33,0,1,&p,size_t(-1),stream)==1);
 assert(prepare<false>(nullptr,33,0,0,&p,size_t(-1),stream)==1);
 for(unsigned start:{0u,31u,8191u,65536u,131072u,263168u,264735u})for(unsigned q:{1u,2u,32u,128u}){
  if(q>maximum_tokens-start)continue;
  const unsigned stride=start+q;const size_t np=size_t(q)*16*((stride+15)/16),nv=size_t(512)*((maximum_tokens+15)/16);
  calls=0;assert(launch(&p,np,&v,nv,&scale,&out,start,q,maximum_tokens-q,stride,maximum_tokens,nullptr,nullptr,nullptr,&index,&count,stream)==0);
  assert(calls==1&&last_grid==std::min(1024u,q*64)&&last_start==start&&last_queries==q&&last_stride==stride&&last_value_stride==maximum_tokens);
  calls=0;assert(launch(&p,np-1,&v,nv,&scale,&out,start,q,0,stride,maximum_tokens,nullptr,nullptr,nullptr,&index,&count,stream)==1&&calls==0);
  assert(launch(&p,np,&v,nv-1,&scale,&out,start,q,0,stride,maximum_tokens,nullptr,nullptr,nullptr,&index,&count,stream)==1&&calls==0);
 }
 auto submit=[&](unsigned start,unsigned q,unsigned output,unsigned stride,unsigned vs){calls=0;return launch(&p,size_t(-1),&v,size_t(-1),&scale,&out,start,q,output,stride,vs,nullptr,nullptr,nullptr,&index,&count,stream);};
 assert(submit(0,0,0,1,1)==1&&calls==0);assert(submit(0,129,0,256,256)==1&&calls==0);
 assert(submit(1,1,0,1,1)==1&&calls==0);assert(submit(1,2,0,2,2)==1&&calls==0);
 assert(submit(0,1,maximum_tokens,1,1)==1&&calls==0);assert(submit(0,2,maximum_tokens-1,2,2)==1&&calls==0);
 assert(submit(0,1,0,maximum_tokens+1,maximum_tokens+1)==1&&calls==0);
 assert(submit(0,1,0,33,32)==1&&calls==0);assert(submit(0,1,0,1,maximum_tokens+1)==1&&calls==0);
 assert(launch<true>(&p,1<<20,&v,1<<20,&scale,&out,0,1,0,1,1,nullptr,nullptr,nullptr,&index,&count,stream)==1);
 assert(launch<true>(&p,1<<20,&v,1<<20,&scale,&out,0,1,0,1,1,nullptr,nullptr,nullptr,&index,&count,stream,&audit)==0);
 failed=true;assert(submit(0,1,0,1,1)==2&&calls==1);
}
'''
        compiler = shutil.which('clang++') or shutil.which('c++')
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'check.cpp').write_text(source)
            subprocess.run([compiler, '-std=c++17', '-O1', '-g', '-Wall', '-Wextra',
                            '-Werror', '-Wno-unknown-pragmas', '-fsanitize=address,undefined',
                            str(path / 'check.cpp'), '-o', str(path / 'check')], check=True, timeout=60)
            subprocess.run([str(path / 'check')], check=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
