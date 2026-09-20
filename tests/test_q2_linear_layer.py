"""Check complete private-layer routing and rejection before any producer runs."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Q2LinearLayerTests(unittest.TestCase):
    def test_private_layer_graph_and_partial_failure(self):
        header = (ROOT / 'native/providers/gdn/sm121_q2_linear_layer.h').read_text()
        actual = '\n'.join(line for line in header.splitlines()
                           if not line.startswith(('#pragma once', '#include')))
        moe = (ROOT / 'native/providers/gdn/sm121_mtp_moe.h').read_text()
        types = moe[moe.index('struct MoeWeights {'):moe.index('namespace mtp_moe_detail {')]
        prelude = r'''
#include "native/providers/gdn/sm121_q2_linear_block_layout.h"
#include "native/providers/gdn/sm121_mtp_moe_layout.h"
#include <cassert>
#include <vector>
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
inline hipStream_t wanted_stream=reinterpret_cast<void*>(0x1234);
struct Event {unsigned kind;const void* input;const void* residual;const void* weight;const void* table;void* output;void* next_residual;};
std::vector<Event> events;unsigned fail_phase=0,expected_blocks=0;
hipError_t finish(){return events.size()==fail_phase?hipErrorUnknown:hipSuccess;}
namespace qrt_sm121_mtp {
TYPES
inline hipError_t launch_residual_normalize(const uint16_t* input,const uint16_t* residual,const uint16_t* weight,
 const unsigned char* table,unsigned rows,uint16_t* output,uint16_t* next_residual,hipStream_t stream){
 assert(rows==2u && stream==wanted_stream);
 events.push_back({0,input,residual,weight,table,output,next_residual});return finish();
}
inline hipError_t launch_moe(const uint16_t* input,const MoeWeights& weights,const MoeTables& tables,
 void* workspace,size_t bytes,unsigned rows,unsigned blocks,hipStream_t stream){
 assert(rows==2u && blocks==expected_blocks && stream==wanted_stream && bytes>=moe_workspace_bytes(2u));
 events.push_back({2,input,weights.routed_down,weights.router,tables.router_exp_fraction,workspace,nullptr});return finish();
}
}
namespace qrt_sm121_q2 {
template<class Element>hipError_t launch_linear_block(const LinearBlockViews<Element>& v,const LinearBlockTables& t,hipStream_t stream){
 assert(stream==wanted_stream && valid_linear_block(v,t));
 events.push_back({1,v.normalized_input,v.recurrent.initial_state,v.qkv_weights,t.recurrent.rsqrt,v.output,v.convolution.staged_rings});return finish();
}
}
'''.replace('TYPES', types)
        main = r'''
using namespace qrt_sm121_q2;
template<class T>T* fake(unsigned slot){return reinterpret_cast<T*>((uintptr_t(1)<<36u)+(uintptr_t(slot)<<32u));}
template<class Element>void exercise(){
 LinearLayerViews<Element> v;auto& l=v.linear;
 v.normalized_input=fake<uint16_t>(1);l.normalized_input=v.normalized_input;
 l.qkv_weights=fake<uint16_t>(2);l.z_weights=fake<uint16_t>(3);l.a_weights=fake<uint16_t>(4);l.b_weights=fake<uint16_t>(5);
 l.output_weights=fake<uint16_t>(6);l.norm_weights=fake<uint16_t>(7);
 l.qkv=fake<uint16_t>(8);l.z=fake<uint16_t>(9);l.a=fake<uint16_t>(10);l.b=fake<uint16_t>(11);l.gated=fake<uint16_t>(12);l.output=fake<uint16_t>(13);
 l.convolution={l.qkv,fake<Element>(14),fake<uint16_t>(15),fake<unsigned char>(16),fake<Element>(17),fake<uint16_t>(18),8192u};
 l.recurrent={l.convolution.staged_convolution,l.a,l.b,fake<float>(19),fake<float>(20),fake<uint16_t>(21),false};
 LinearLayerTables t{{{fake<float>(22),fake<float>(23),fake<unsigned char>(24),fake<unsigned char>(25)},fake<float>(26)},
  {fake<uint16_t>(41),fake<uint16_t>(42),fake<uint32_t>(43)}};
 v.hidden=fake<uint16_t>(27);v.residual=fake<uint16_t>(28);v.input_norm_weights=fake<uint16_t>(29);v.post_norm_weights=fake<uint16_t>(30);
 v.input_residual=fake<uint16_t>(31);v.moe_input=fake<uint16_t>(32);v.output_residual=fake<uint16_t>(33);
 v.moe_weights={fake<uint16_t>(35),fake<uint16_t>(36),fake<uint16_t>(37),fake<uint16_t>(38),fake<uint16_t>(39),fake<uint16_t>(40)};
 v.moe_workspace=fake<unsigned char>(44);v.moe_workspace_bytes=qrt_sm121_mtp::moe_workspace_bytes(2u);
 for(bool key_major:{false,true})for(size_t position:{size_t(0),size_t(7169),size_t(262143)})for(unsigned blocks:{1u,7u,1024u}){
  l.recurrent.key_major=key_major;l.convolution.first_position=position;expected_blocks=blocks;
  for(unsigned fault=0;fault<=4u;++fault){events.clear();fail_phase=fault;
   assert(launch_linear_layer(v,t,blocks,wanted_stream)==(fault?hipErrorUnknown:hipSuccess));
   assert(events.size()==(fault?fault:4u));
   const auto& a=events[0];assert(a.kind==0 && a.input==v.hidden && a.residual==v.residual && a.weight==v.input_norm_weights &&
    a.table==t.linear.recurrent.rsqrt && a.output==v.normalized_input && a.next_residual==v.input_residual);
   if(events.size()>1){const auto& b=events[1];assert(b.kind==1 && b.input==v.normalized_input && b.output==l.output && b.residual==l.recurrent.initial_state);}
   if(events.size()>2){const auto& c=events[2];assert(c.kind==0 && c.input==l.output && c.residual==v.input_residual && c.weight==v.post_norm_weights &&
    c.table==t.linear.recurrent.rsqrt && c.output==v.moe_input && c.next_residual==v.output_residual);}
   if(events.size()>3){const auto& d=events[3];assert(d.kind==2 && d.input==v.moe_input && d.output==v.moe_workspace &&
    d.weight==v.moe_weights.router && d.residual==v.moe_weights.routed_down && d.table==t.moe.router_exp_fraction);}
  }
 }
 const auto reject=[&](LinearLayerViews<Element> bad,LinearLayerTables tables,unsigned blocks=1024u){
  events.clear();fail_phase=0;assert(launch_linear_layer(bad,tables,blocks,wanted_stream)==hipErrorInvalidValue);assert(events.empty());
 };
 reject(v,t,0);reject(v,t,4097);
 auto bad=v;bad.linear.normalized_input=fake<uint16_t>(45);reject(bad,t);
 const uint16_t* LinearLayerViews<Element>::* reads[]={&LinearLayerViews<Element>::hidden,&LinearLayerViews<Element>::residual,
  &LinearLayerViews<Element>::input_norm_weights,&LinearLayerViews<Element>::post_norm_weights};
 uint16_t* LinearLayerViews<Element>::* writes[]={&LinearLayerViews<Element>::normalized_input,&LinearLayerViews<Element>::input_residual,
  &LinearLayerViews<Element>::moe_input,&LinearLayerViews<Element>::output_residual};
 for(auto write:writes){
  bad=v;bad.*write=nullptr;bad.linear.normalized_input=bad.normalized_input;reject(bad,t);
  for(auto read:reads){bad=v;bad.*write=const_cast<uint16_t*>(v.*read)+1u;bad.linear.normalized_input=bad.normalized_input;reject(bad,t);}
  for(auto other:writes)if(write!=other){bad=v;bad.*write=(v.*other)+1u;bad.linear.normalized_input=bad.normalized_input;reject(bad,t);}
  bad=v;bad.*write=reinterpret_cast<uint16_t*>(v.moe_workspace)+1u;bad.linear.normalized_input=bad.normalized_input;reject(bad,t);
  bad=v;bad.*write=reinterpret_cast<uint16_t*>(v.linear.recurrent.staged_states)+1u;bad.linear.normalized_input=bad.normalized_input;reject(bad,t);
  bad=v;bad.*write=const_cast<uint16_t*>(v.moe_weights.routed_gate_up)+(1u<<28u);bad.linear.normalized_input=bad.normalized_input;reject(bad,t);
  bad=v;bad.*write=reinterpret_cast<uint16_t*>(const_cast<float*>(t.linear.recurrent.g))+1u;bad.linear.normalized_input=bad.normalized_input;reject(bad,t);
  bad=v;bad.*write=reinterpret_cast<uint16_t*>(~uintptr_t(1));bad.linear.normalized_input=bad.normalized_input;reject(bad,t);
 }
 for(auto read:reads){bad=v;bad.*read=nullptr;reject(bad,t);bad=v;bad.*read=v.linear.gated+1u;reject(bad,t);}
 bad=v;bad.moe_workspace_bytes-=1;reject(bad,t);
 bad=v;bad.moe_workspace=static_cast<unsigned char*>(v.moe_workspace)+1;reject(bad,t);
 bad=v;bad.moe_workspace=const_cast<float*>(l.recurrent.initial_state);reject(bad,t);
 bad=v;bad.moe_workspace=const_cast<uint16_t*>(l.qkv_weights)+128u;reject(bad,t);
 bad=v;bad.moe_workspace=reinterpret_cast<void*>(~uintptr_t(255));reject(bad,t);
 bad=v;bad.moe_weights.routed_gate_up=nullptr;reject(bad,t);
 bad=v;bad.moe_weights.routed_down=v.linear.gated;reject(bad,t);
 auto broken=t;broken.moe.silu=nullptr;reject(v,broken);
 broken=t;broken.moe.sigmoid=v.normalized_input;reject(v,broken);
 broken=t;broken.moe.router_exp_fraction=reinterpret_cast<const uint32_t*>(v.linear.output);reject(v,broken);
}
int main(){exercise<float>();exercise<uint16_t>();}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-q2-layer-') as temporary:
            source = Path(temporary) / 'layer.cpp'
            exe = Path(temporary) / 'layer'
            source.write_text(prelude + actual + main)
            subprocess.run([os.getenv('CXX', 'c++'), '-std=c++17', '-O1', '-Wall', '-Wextra',
                            '-Werror', '-ffp-contract=off', '-fsanitize=address,undefined',
                            '-fno-sanitize-recover=all', '-I', str(ROOT), str(source), '-o', str(exe)],
                           check=True, timeout=60)
            subprocess.run([str(exe)], check=True, timeout=20)


if __name__ == '__main__':
    unittest.main()
