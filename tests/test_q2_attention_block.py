"""Exercise private two-row attention routing, bounds and every partial failure."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class Q2AttentionBlockTests(unittest.TestCase):
    def test_private_history_and_submission_failures(self):
        header = (ROOT/'native/providers/gdn/sm121_q2_attention_block.h').read_text()
        gate = (ROOT/'native/providers/gdn/sm121_mtp_gate.h').read_text()
        actual = ('namespace qrt_sm121_mtp {' + function(gate, 'inline hipError_t launch_gate(') + '}\n'
            'namespace qrt_sm121_q2 { namespace attention_block_detail {' +
            function(header, 'inline hipError_t project_input(') + '}\n' +
            function(header, 'inline hipError_t launch_attention_block(') + '}')
        code = r'''
#include "native/providers/gdn/sm121_q2_attention_block_layout.h"
#include <cassert>
#include <tuple>
#include <initializer_list>
using namespace qrt_sm121_q2;
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
struct dim3 {unsigned x,y,z;explicit dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
namespace qrt_sm121_mtp {void query_rows(){} void publish_attention_context(){} void gate_contexts(){}}
namespace qrt_sm121_q2::attention_block_detail {void private_key_values(){} void private_scores(){}}
namespace qrt_sm121_q1_moe {template<unsigned K>void projection(){}}
namespace qrt_blackwell_attention {
template<bool A,bool B=false,bool C=false,bool D=false,bool E=false,bool F=false,bool G=false,bool H=false>
void blackwell_exact_attention_kernel(){}
}
unsigned launches=0,fail_at=0;
const AttentionBlockViews* expected=nullptr;
const AttentionBlockTables* tables=nullptr;
hipStream_t expected_stream=reinterpret_cast<void*>(0x1234);
template<class... Args>void record(void(*kernel)(),dim3 grid,dim3 block,size_t shared,hipStream_t stream,Args... args){
 assert(!shared && stream==expected_stream && grid.z==1u && block.y==1u && block.z==1u);
 const auto a=std::make_tuple(args...);const auto& v=*expected;const auto& t=*tables;
 if constexpr(sizeof...(Args)==4){
  const bool input=launches<6u;const unsigned part=launches/2u,row=launches%2u;
  assert(input||launches==12u||launches==13u);
  assert(kernel==(input?qrt_sm121_q1_moe::projection<2048u>:qrt_sm121_q1_moe::projection<4096u>));
  const unsigned n=input?(part?512u:8192u):2048u;
  const uint16_t* weights=input?(part?(part==1u?v.k_weights:v.v_weights):v.q_weights):v.output_weights;
  const uint16_t* x=input?v.normalized_input+row*2048u:v.gated+row*4096u;
  uint16_t* y=input?(part?v.kv_projected+row*1024u+(part-1u)*512u:v.q_projected+row*8192u):v.output+row*2048u;
  assert(grid.x==(n+15u)/16u && grid.y==1u && block.x==256u);
  assert(std::get<0>(a)==x && std::get<1>(a)==weights && std::get<2>(a)==y && std::get<3>(a)==n);
 }else if constexpr(sizeof...(Args)==8){
  assert(launches==6u && kernel==qrt_sm121_mtp::query_rows && grid.x==2u && grid.y==16u && block.x==64u);
  assert(std::get<0>(a)==v.q_projected && std::get<1>(a)==v.q_norm_weights);
  assert(std::get<2>(a)==t.rsqrt && std::get<3>(a)==t.rope && std::get<4>(a)==v.first_position);
  assert(std::get<5>(a)==v.queries && std::get<6>(a)==v.gates && std::get<7>(a)==v.q_norm);
 }else if constexpr(sizeof...(Args)==7){
  assert(launches==7u && kernel==attention_block_detail::private_key_values && grid.x==2u && grid.y==2u && block.x==256u);
  assert(std::get<0>(a)==v.kv_projected && std::get<1>(a)==v.k_norm_weights);
  assert(std::get<2>(a)==t.rsqrt && std::get<3>(a)==t.rope && std::get<4>(a)==v.first_position);
  assert(std::get<5>(a)==v.staged_kv && std::get<6>(a)==v.k_norm);
 }else if constexpr(sizeof...(Args)==6){
  assert(launches==8u && kernel==attention_block_detail::private_scores && grid.x==2u*v.score_stride && grid.y==1u && block.x==256u);
  assert(std::get<0>(a)==v.queries && std::get<1>(a)==v.history && std::get<2>(a)==v.staged_kv);
  assert(std::get<3>(a)==v.scores && std::get<4>(a)==v.first_position && std::get<5>(a)==v.score_stride);
 }else if constexpr(sizeof...(Args)==15){
  assert((kernel==qrt_blackwell_attention::blackwell_exact_attention_kernel<true,true,true,false,false,false,false,true>));
  assert(launches==9u && grid.x==16u && grid.y==2u && block.x==256u);
  assert(!std::get<0>(a) && !std::get<1>(a) && std::get<2>(a)==v.history+512u);
  assert(std::get<3>(a)==v.float_context && std::get<4>(a)==v.first_position && !std::get<5>(a));
  assert(std::get<6>(a)==t.exp2 && !std::get<7>(a) && !std::get<8>(a));
  assert(std::get<9>(a) && std::get<10>(a)==t.reciprocal && std::get<11>(a)==v.scores);
  assert(std::get<12>(a)==v.score_stride && std::get<13>(a)==v.staged_kv+512u && std::get<14>(a)==v.first_position);
 }else if constexpr(sizeof...(Args)==3){
  assert(launches==10u && kernel==qrt_sm121_mtp::publish_attention_context && grid.x==32u && grid.y==1u && block.x==256u);
  assert(std::get<0>(a)==v.float_context && std::get<1>(a)==v.context && std::get<2>(a)==8192u);
 }else{
  static_assert(sizeof...(Args)==5);
  assert(launches==11u && kernel==qrt_sm121_mtp::gate_contexts && grid.x==32u && grid.y==1u && block.x==256u);
  assert(std::get<0>(a)==v.context && std::get<1>(a)==v.gates && std::get<2>(a)==t.sigmoid);
  assert(std::get<3>(a)==v.gated && std::get<4>(a)==8192u);
 }
 ++launches;
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
inline hipError_t hipGetLastError(){return launches==fail_at?hipErrorUnknown:hipSuccess;}
''' + actual + r'''
template<class T>T* fake(unsigned slot){return reinterpret_cast<T*>((uintptr_t(1)<<36u)+(uintptr_t(slot)<<32u));}
int main(){
 AttentionBlockViews v;AttentionBlockTables t;unsigned slot=1;
 const uint16_t* AttentionBlockViews::* reads[]={&AttentionBlockViews::normalized_input,&AttentionBlockViews::q_weights,
  &AttentionBlockViews::k_weights,&AttentionBlockViews::v_weights,&AttentionBlockViews::output_weights,
  &AttentionBlockViews::q_norm_weights,&AttentionBlockViews::k_norm_weights,&AttentionBlockViews::history};
 uint16_t* AttentionBlockViews::* writes[]={&AttentionBlockViews::q_projected,&AttentionBlockViews::kv_projected,
  &AttentionBlockViews::q_norm,&AttentionBlockViews::k_norm,&AttentionBlockViews::queries,&AttentionBlockViews::gates,
  &AttentionBlockViews::staged_kv,&AttentionBlockViews::context,&AttentionBlockViews::gated,&AttentionBlockViews::output};
 for(auto m:reads)v.*m=fake<uint16_t>(slot++);
 for(auto m:writes)v.*m=fake<uint16_t>(slot++);
 v.scores=fake<float>(slot++);v.float_context=fake<float>(slot++);v.history_capacity=263680u;
 t.rsqrt=fake<unsigned char>(slot++);t.exp2=fake<unsigned char>(slot++);t.reciprocal=fake<unsigned char>(slot++);
 t.rope=fake<uint16_t>(slot++);t.rope_rows=263680u;t.sigmoid=fake<uint16_t>(slot++);
 expected=&v;tables=&t;
 for(unsigned first:{0u,31u,7169u,8194u,262143u,263678u}){
  v.first_position=first;v.score_stride=(first+33u)&~31u;
  for(unsigned fail=0;fail<=14u;++fail){
   launches=0;fail_at=fail;
   assert(launch_attention_block(v,t,expected_stream)==(fail?hipErrorUnknown:hipSuccess));
   assert(launches==(fail?fail:14u));
  }
  for(unsigned rows:{1u,2u}){auto a=accepted_attention(v,rows);assert(a.key_values==v.staged_kv && a.first_position==first && a.rows==rows);}
 }
 const auto reject=[&](const AttentionBlockViews& b,const AttentionBlockTables& z){
  launches=0;fail_at=0;assert(launch_attention_block(b,z,expected_stream)==hipErrorInvalidValue && !launches);
 };
 for(auto m:reads){auto b=v;b.*m=nullptr;reject(b,t);b=v;b.*m=reinterpret_cast<const uint16_t*>(~uintptr_t(1));reject(b,t);}
 for(auto m:writes){
  for(auto r:reads){auto b=v;b.*m=const_cast<uint16_t*>(v.*r)+1u;reject(b,t);}
  for(auto w:writes)if(w!=m){auto b=v;b.*m=(v.*w)+1u;reject(b,t);}
  auto b=v;b.*m=nullptr;reject(b,t);b=v;b.*m=reinterpret_cast<uint16_t*>(~uintptr_t(1));reject(b,t);
  b=v;b.*m=reinterpret_cast<uint16_t*>(v.scores)+1u;reject(b,t);
  b=v;b.*m=reinterpret_cast<uint16_t*>(v.float_context)+1u;reject(b,t);
  b=v;b.*m=const_cast<uint16_t*>(t.rope)+1u;reject(b,t);
  b=v;b.*m=const_cast<uint16_t*>(t.sigmoid)+1u;reject(b,t);
 }
 for(auto r:reads){auto b=v;b.scores=reinterpret_cast<float*>(const_cast<uint16_t*>(v.*r))+1u;reject(b,t);}
 auto b=v;b.float_context=v.scores+1u;reject(b,t);b=v;b.scores=nullptr;reject(b,t);
 b=v;b.history_capacity=0;reject(b,t);b=v;b.history_capacity=v.first_position-1u;reject(b,t);
 b=v;b.history_capacity=263681u;reject(b,t);
 for(unsigned first:{263679u,~0u}){b=v;b.first_position=first;reject(b,t);}
 for(unsigned stride:{0u,32u,263679u,263712u,~0u}){b=v;b.score_stride=stride;reject(b,t);}
 auto z=t;z.rope_rows=v.first_position+1u;reject(v,z);z=t;z.rope_rows=~0u;reject(v,z);
 for(const auto* pointer:{t.rsqrt,t.exp2,t.reciprocal}){
  b=v;b.scores=reinterpret_cast<float*>(const_cast<unsigned char*>(pointer))+1u;reject(b,t);
 }
 z=t;z.rsqrt=nullptr;reject(v,z);z=t;z.exp2=nullptr;reject(v,z);z=t;z.reciprocal=nullptr;reject(v,z);
 z=t;z.rope=nullptr;reject(v,z);z=t;z.sigmoid=nullptr;reject(v,z);
 // Read-only weights may share storage. They cannot overlap any producer.
 b=v;b.v_weights=b.k_weights;assert(valid_attention_block(b,t));
 for(unsigned rows:{0u,3u,~0u})assert(!accepted_attention(v,rows).key_values);
 b=v;b.staged_kv=nullptr;assert(!accepted_attention(b,1u).key_values);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-q2-attention-') as temporary:
            source=Path(temporary)/'attention.cpp';source.write_text(code);exe=Path(temporary)/'attention'
            subprocess.run([os.getenv('CXX','c++'),'-std=c++17','-O1','-Wall','-Wextra','-Werror',
                '-ffp-contract=off','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                '-I',str(ROOT),str(source),'-o',str(exe)],check=True,timeout=60)
            subprocess.run([str(exe)],check=True,timeout=15)


if __name__ == '__main__':
    unittest.main()
