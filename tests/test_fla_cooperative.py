"""Execute cooperative FLA tensor transport and ownership with threaded CTAs.

The CPU dot is a transport stand-in; native full-component comparisons own
SM121 arithmetic acceptance. Actual kernel bodies run with fewer CPU lanes.
"""
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]

class CooperativeFlaTests(unittest.TestCase):
    def test_actual_kernels_tail_checkpoint_alias_and_column_ownership(self):
        s = (ROOT / 'native/providers/gdn/blackwell_cooperative.cpp').read_text()
        kernels = '\n'.join(function(s, '__global__ void ' + name + '(')
                            for name in ('wu_kernel', 'scores_kernel', 'output_kernel', 'state_kernel'))
        source = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#define __global__
#define __shared__ static
constexpr unsigned threads=32,lanes=4,groups=threads/lanes,tile_columns=8,state_columns=4;
struct Dim { unsigned x=0,y=0,z=0; };
thread_local Dim blockIdx,threadIdx;
std::mutex barrier_lock;
std::condition_variable barrier_changed;
unsigned arrived=0,generation=0;
void __syncthreads() {
 std::unique_lock<std::mutex> l(barrier_lock);unsigned old=generation;
 if(++arrived==threads){arrived=0;++generation;barrier_changed.notify_all();}
 else barrier_changed.wait(l,[&]{return generation!=old;});
}
float from_bf16(uint16_t b){uint32_t u=uint32_t(b)<<16;float f;std::memcpy(&f,&u,4);return f;}
uint16_t to_bf16(float f){uint32_t u;std::memcpy(&u,&f,4);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
float exponential(float x,const unsigned char*){return std::exp2(x*1.4426950408889634074f);}
float reference_dot(const uint16_t* a,const uint16_t* b,unsigned count){float f=0;for(unsigned i=0;i<count;++i)f+=from_bf16(a[i])*from_bf16(b[i]);return f;}
float dot(const uint16_t* a,const uint16_t* b,unsigned count){return (threadIdx.x&(lanes-1))?0.0f:reference_dot(a,b,count);}
struct Pool {
 std::mutex lock;std::condition_variable changed;unsigned epoch=0,finished=0;bool stop=false;
 std::function<void()> job;Dim index;std::vector<std::thread> workers;
 Pool(){for(unsigned lane=0;lane<threads;++lane)workers.emplace_back([&,lane]{
  unsigned seen=0;for(;;){std::unique_lock<std::mutex> l(lock);changed.wait(l,[&]{return stop||epoch!=seen;});
   if(stop)return;seen=epoch;auto work=job;Dim idx=index;l.unlock();threadIdx.x=lane;blockIdx=idx;work();
   l.lock();++finished;changed.notify_all();}});}
 void run(Dim idx,std::function<void()> work){std::unique_lock<std::mutex> l(lock);job=work;index=idx;finished=0;++epoch;changed.notify_all();changed.wait(l,[&]{return finished==threads;});}
 ~Pool(){{std::lock_guard<std::mutex> l(lock);stop=true;changed.notify_all();}for(auto& t:workers)t.join();}
} pool;
''' + kernels + r'''
constexpr uint16_t guard=0x5a5a;
constexpr float fguard=12345.0f;
bool owned(unsigned column,unsigned width){return column<width||column>=128-width;}
bool head_owned(unsigned head){return head==0||head==31;}
void fill(std::vector<uint16_t>& a){for(size_t i=0;i<a.size();++i)a[i]=to_bf16(float(int(i*7%17)-8)/64);}
void check(unsigned count){
 const unsigned chunks=(count+63)/64;
 std::vector<uint16_t> k(count*2048),q(count*2048),v(count*4096),b(count*32),a(count*2048),w(count*4096,guard);
 std::vector<float> g(count*32);
 fill(k);fill(q);fill(v);fill(b);fill(a);
 for(size_t i=0;i<g.size();++i)g[i]=-float(i/32%64)/256;
 const auto input_v=v,input_k=k,input_q=q,input_b=b,input_a=a; const auto input_g=g;
 // Reverse output-column order exercises the exact production U=V alias.
 for(unsigned chunk=0;chunk<chunks;++chunk)for(unsigned head:{0u,31u})for(unsigned tile:{15u,0u})
  pool.run({tile,head,chunk},[&]{wu_kernel(k.data(),v.data(),b.data(),a.data(),g.data(),w.data(),v.data(),count,nullptr);});
 for(unsigned token=0;token<count;++token)for(unsigned head=0;head<32;++head)for(unsigned col=0;col<128;++col){
  size_t index=(size_t(token)*32+head)*128+col;
  if(!head_owned(head)||!owned(col,8)){assert(w[index]==guard&&v[index]==input_v[index]);continue;}
  unsigned offset=token/64*64,valid=std::min(64u,count-offset);float sw=0,su=0;
  for(unsigned r=0;r<valid;++r){
   float beta=from_bf16(b[(offset+r)*32+head]);
   float key=from_bf16(to_bf16(from_bf16(k[((offset+r)*16+head/2)*128+col])*beta));
   key=from_bf16(to_bf16(key*exponential(g[(offset+r)*32+head],nullptr)));
   float value=from_bf16(to_bf16(from_bf16(input_v[((offset+r)*32+head)*128+col])*beta));
   float inv=from_bf16(a[(size_t(token)*32+head)*64+r]);sw+=inv*key;su+=inv*value;
  }
  assert(w[index]==to_bf16(sw)&&v[index]==to_bf16(su));
 }
 assert(k==input_k&&q==input_q&&b==input_b&&a==input_a&&g==input_g);
 // State owns four full value rows. Independently update their entire K128
 // state through two chunks; check every checkpoint and all untouched cells.
 fill(w);v=input_v;
 std::vector<float> state(32*128*128),expected;
 for(size_t i=0;i<state.size();++i)state[i]=float(int(i*3%13)-6)/128;
 expected=state;
 std::vector<uint16_t> h(size_t(chunks)*524288,guard),want_h=h,new_v(count*4096,guard),want_new=new_v;
 for(unsigned head:{0u,31u})for(unsigned col=0;col<128;++col){if(!owned(col,4))continue;
  size_t base=(head*128+col)*128;
  for(unsigned offset=0;offset<count;offset+=64){unsigned valid=std::min(64u,count-offset);uint16_t rounded[128],residual[64]{};
   for(unsigned key=0;key<128;++key){rounded[key]=to_bf16(expected[base+key]);want_h[size_t(offset/64)*524288+base+key]=rounded[key];}
   for(unsigned token=0;token<valid;++token){size_t pos=offset+token,index=(pos*32+head)*128+col;
    float sum=reference_dot(w.data()+(pos*32+head)*128,rounded,128);
    float value=from_bf16(v[index])-sum;want_new[index]=to_bf16(value);
    residual[token]=to_bf16(value*exponential(g[(offset+valid-1)*32+head]-g[pos*32+head],nullptr));
   }
   for(unsigned key=0;key<128;++key){float sum=0;for(unsigned token=0;token<valid;++token)
    sum+=from_bf16(k[((offset+token)*16+head/2)*128+key])*from_bf16(residual[token]);
    expected[base+key]=fmaf(expected[base+key],exponential(g[(offset+valid-1)*32+head],nullptr),sum);
   }
  }
 }
 for(unsigned head:{0u,31u})for(unsigned tile:{0u,31u})
  pool.run({tile,head,0},[&]{state_kernel(k.data(),v.data(),w.data(),g.data(),h.data(),new_v.data(),state.data(),count,nullptr);});
 assert(h==want_h&&new_v==want_new&&state==expected&&v==input_v);
 // Scores execute complete logical rows, including causal zeros and the tail.
 std::vector<uint16_t> scores(count*2048,guard),want_scores=scores;
 for(unsigned chunk=0;chunk<chunks;++chunk)for(unsigned head:{0u,31u}){
  unsigned valid=std::min(64u,count-chunk*64);
  for(unsigned block=0;block<valid*64/groups;++block)
   pool.run({block,head,chunk},[&]{scores_kernel(q.data(),k.data(),g.data(),scores.data(),count,nullptr);});
  for(unsigned t=0;t<valid;++t)for(unsigned r=0;r<64;++r){unsigned pos=chunk*64+t,source=chunk*64+r;
   float sum=r>t?0:reference_dot(q.data()+(pos*16+head/2)*128,k.data()+(source*16+head/2)*128,128);
   want_scores[(pos*32+head)*64+r]=r>t?0:to_bf16(sum*exponential(g[pos*32+head]-g[source*32+head],nullptr));
  }
 }
 assert(scores==want_scores);
 // Output checkpoint and V transposes use the same columns at each chunk.
 fill(h);fill(v);std::vector<float> output(count*4096,fguard),want_output=output;
 for(unsigned token=0;token<count;++token)for(unsigned head:{0u,31u})for(unsigned col=0;col<128;++col){if(!owned(col,8))continue;
  unsigned offset=token/64*64,valid=std::min(64u,count-offset);float local=0;
  float old=reference_dot(q.data()+(token*16+head/2)*128,h.data()+size_t(token/64)*524288+(head*128+col)*128,128);
  for(unsigned source=0;source<valid;++source)local+=from_bf16(scores[(token*32+head)*64+source])*from_bf16(v[((offset+source)*32+head)*128+col]);
  constexpr float scale=0.08838834764831845f;float prior=old*exponential(g[token*32+head],nullptr);
  want_output[(token*32+head)*128+col]=from_bf16(to_bf16(fmaf(local,scale,prior*scale)));
 }
 for(unsigned chunk=0;chunk<chunks;++chunk)for(unsigned head:{0u,31u})for(unsigned tile:{0u,15u})
  pool.run({tile,head,chunk},[&]{output_kernel(q.data(),v.data(),h.data(),g.data(),scores.data(),output.data(),count,nullptr);});
 assert(output==want_output&&k==input_k&&q==input_q&&g==input_g);
}
int main(){check(1);check(65);}
'''
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'test.cpp').write_text(source)
            build = subprocess.run(['c++','-std=c++17','-O1','-ffp-contract=off','-pthread',
                '-fsanitize=address,undefined',str(p/'test.cpp'),'-o',str(p/'test')],capture_output=True,text=True,timeout=30)
            self.assertEqual(build.returncode,0,build.stderr)
            run = subprocess.run([str(p/'test')],capture_output=True,text=True,timeout=40)
            self.assertEqual(run.returncode,0,run.stderr)
