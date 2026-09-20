"""Check complete private-layer routing and rejection before any producer runs."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Q2AttentionLayerTests(unittest.TestCase):
    def test_private_layer_graph_and_partial_failure(self):
        header = (ROOT / 'native/providers/gdn/sm121_q2_attention_layer.h').read_text()
        actual = '\n'.join(line for line in header.splitlines()
                           if not line.startswith(('#pragma once', '#include')))
        moe = (ROOT / 'native/providers/gdn/sm121_mtp_moe.h').read_text()
        types = moe[moe.index('struct MoeWeights {'):moe.index('namespace mtp_moe_detail {')]
        prelude = r'''
#include "native/providers/gdn/sm121_q2_attention_block_layout.h"
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
hipError_t launch_attention_block(const AttentionBlockViews& v,const AttentionBlockTables& t,hipStream_t stream){
 assert(stream==wanted_stream && valid_attention_block(v,t));
 events.push_back({1,v.normalized_input,v.history,v.q_weights,t.rsqrt,v.output,v.staged_kv});return finish();
}
}
'''.replace('TYPES', types)
        main = r'''
using namespace qrt_sm121_q2;
template<class T>T* fake(unsigned slot){return reinterpret_cast<T*>((uintptr_t(1)<<36u)+(uintptr_t(slot)<<32u));}
int main(){
 AttentionLayerViews v;auto& l=v.attention;AttentionLayerTables t;unsigned slot=1u;
 const uint16_t* AttentionBlockViews::* block_reads[]={&AttentionBlockViews::q_weights,&AttentionBlockViews::k_weights,
  &AttentionBlockViews::v_weights,&AttentionBlockViews::output_weights,&AttentionBlockViews::q_norm_weights,
  &AttentionBlockViews::k_norm_weights,&AttentionBlockViews::history};
 uint16_t* AttentionBlockViews::* block_writes[]={&AttentionBlockViews::q_projected,&AttentionBlockViews::kv_projected,
  &AttentionBlockViews::q_norm,&AttentionBlockViews::k_norm,&AttentionBlockViews::queries,&AttentionBlockViews::gates,
  &AttentionBlockViews::staged_kv,&AttentionBlockViews::context,&AttentionBlockViews::gated,&AttentionBlockViews::output};
 for(auto m:block_reads)l.*m=fake<uint16_t>(slot++);
 for(auto m:block_writes)l.*m=fake<uint16_t>(slot++);
 l.scores=fake<float>(slot++);l.float_context=fake<float>(slot++);l.history_capacity=263680u;
 t.attention={fake<unsigned char>(slot++),fake<unsigned char>(slot++),fake<unsigned char>(slot++),fake<uint16_t>(slot++),263680u,fake<uint16_t>(slot++)};
 const uint16_t* AttentionLayerViews::* reads[]={&AttentionLayerViews::hidden,&AttentionLayerViews::residual,
  &AttentionLayerViews::input_norm_weights,&AttentionLayerViews::post_norm_weights};
 uint16_t* AttentionLayerViews::* writes[]={&AttentionLayerViews::normalized_input,&AttentionLayerViews::input_residual,
  &AttentionLayerViews::moe_input,&AttentionLayerViews::output_residual};
 for(auto m:reads)v.*m=fake<uint16_t>(slot++);
 for(auto m:writes)v.*m=fake<uint16_t>(slot++);
 l.normalized_input=v.normalized_input;
 v.moe_weights={fake<uint16_t>(slot++),fake<uint16_t>(slot++),fake<uint16_t>(slot++),fake<uint16_t>(slot++),fake<uint16_t>(slot++),fake<uint16_t>(slot++)};
 t.moe={fake<uint16_t>(slot++),fake<uint16_t>(slot++),fake<uint32_t>(slot++)};
 v.moe_workspace=fake<unsigned char>(slot++);v.moe_workspace_bytes=qrt_sm121_mtp::moe_workspace_bytes(2u);
 for(unsigned position:{0u,7169u,262143u,263678u})for(unsigned blocks:{1u,7u,1024u}){
  l.first_position=position;l.score_stride=(position+33u)&~31u;expected_blocks=blocks;
  for(unsigned fault=0;fault<=4u;++fault){events.clear();fail_phase=fault;
   assert(launch_attention_layer(v,t,blocks,wanted_stream)==(fault?hipErrorUnknown:hipSuccess));
   assert(events.size()==(fault?fault:4u));
   const auto& a=events[0];assert(a.kind==0 && a.input==v.hidden && a.residual==v.residual && a.weight==v.input_norm_weights &&
    a.table==t.attention.rsqrt && a.output==v.normalized_input && a.next_residual==v.input_residual);
   if(events.size()>1){const auto& b=events[1];assert(b.kind==1 && b.input==v.normalized_input && b.output==l.output && b.residual==l.history && b.next_residual==l.staged_kv);}
   if(events.size()>2){const auto& c=events[2];assert(c.kind==0 && c.input==l.output && c.residual==v.input_residual && c.weight==v.post_norm_weights &&
    c.table==t.attention.rsqrt && c.output==v.moe_input && c.next_residual==v.output_residual);}
   if(events.size()>3){const auto& d=events[3];assert(d.kind==2 && d.input==v.moe_input && d.output==v.moe_workspace &&
    d.weight==v.moe_weights.router && d.residual==v.moe_weights.routed_down && d.table==t.moe.router_exp_fraction);}
  }
 }
 const auto reject=[&](AttentionLayerViews bad,AttentionLayerTables tables,unsigned blocks=1024u){
  events.clear();fail_phase=0;assert(launch_attention_layer(bad,tables,blocks,wanted_stream)==hipErrorInvalidValue);assert(events.empty());
 };
 reject(v,t,0);reject(v,t,4097);
 auto bad=v;bad.attention.normalized_input=fake<uint16_t>(slot);reject(bad,t);
 for(auto write:writes){
  bad=v;bad.*write=nullptr;bad.attention.normalized_input=bad.normalized_input;reject(bad,t);
  for(auto read:reads){bad=v;bad.*write=const_cast<uint16_t*>(v.*read)+1u;bad.attention.normalized_input=bad.normalized_input;reject(bad,t);}
  for(auto other:writes)if(write!=other){bad=v;bad.*write=(v.*other)+1u;bad.attention.normalized_input=bad.normalized_input;reject(bad,t);}
  for(auto read:block_reads){bad=v;bad.*write=const_cast<uint16_t*>(l.*read)+1u;bad.attention.normalized_input=bad.normalized_input;reject(bad,t);}
  for(auto out:block_writes){bad=v;bad.*write=(l.*out)+1u;bad.attention.normalized_input=bad.normalized_input;reject(bad,t);}
  bad=v;bad.*write=reinterpret_cast<uint16_t*>(v.moe_workspace)+1u;bad.attention.normalized_input=bad.normalized_input;reject(bad,t);
  bad=v;bad.*write=const_cast<uint16_t*>(v.moe_weights.routed_gate_up)+(1u<<28u);bad.attention.normalized_input=bad.normalized_input;reject(bad,t);
  bad=v;bad.*write=reinterpret_cast<uint16_t*>(~uintptr_t(1));bad.attention.normalized_input=bad.normalized_input;reject(bad,t);
 }
 for(auto read:reads){
  bad=v;bad.*read=nullptr;reject(bad,t);
  for(auto write:block_writes){bad=v;bad.*read=(l.*write)+1u;reject(bad,t);}
 }
 bad=v;bad.moe_workspace_bytes-=1;reject(bad,t);
 bad=v;bad.moe_workspace=static_cast<unsigned char*>(v.moe_workspace)+1;reject(bad,t);
 bad=v;bad.moe_workspace=const_cast<uint16_t*>(l.history);reject(bad,t);
 bad=v;bad.moe_workspace=const_cast<uint16_t*>(l.q_weights)+128u;reject(bad,t);
 bad=v;bad.moe_workspace=reinterpret_cast<void*>(~uintptr_t(255));reject(bad,t);
 bad=v;bad.moe_weights.routed_gate_up=nullptr;reject(bad,t);
 bad=v;bad.moe_weights.routed_down=l.gated;reject(bad,t);
 auto broken=t;broken.moe.silu=nullptr;reject(v,broken);
 broken=t;broken.moe.sigmoid=v.normalized_input;reject(v,broken);
 broken=t;broken.moe.router_exp_fraction=reinterpret_cast<const uint32_t*>(l.output);reject(v,broken);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-q2-attention-layer-') as temporary:
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
