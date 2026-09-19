"""Check one-copy destinations across original shard boundaries."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function
import test_resident_text_shards as text_shard_harness

ROOT = Path(__file__).resolve().parents[1]


def layout_host_source():
    core = (ROOT / 'native/src/qwen36_baseline.c').read_text()
    names = '\n'.join(function(core, signature) for signature in (
        'static qrt_status_t qwen36_write_name(',
        'static qrt_status_t qwen36_write_fixed_name(',
        'qrt_status_t qrt_qwen36_tensor_name('))
    return ('#include "native/src/qrt.h"\n#include <cstdio>\n#include <cstring>\n' + names +
            '\n#include "tests/native/resident_ordered_shard_layout_host.cpp"\n')


class ResidentOrderedShardTests(unittest.TestCase):
    def test_actual_cross_shard_copy_alias_and_cleanup(self):
        whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        actual = '\n'.join(function(whole, signature) for signature in (
            'bool assign_resident_model_ordered_shard_layout(',
            'bool bind_resident_model_ordered_fixed_arena(\n'
            '    const std::vector<uint64_t>& offsets, uint64_t bytes, uint8_t** output) {',
            'void release_q1_decode_order_fixed_bf16_arena() {'))
        globals_ = r'''
#include <iostream>
struct WholeRepeatedLayerFixedWeightEntry {
 std::string tensor_name; bool borrowed=true; uint16_t* device_weights=nullptr; uint64_t bytes=0;
};
std::vector<WholeRepeatedLayerFixedWeightEntry> g_whole_repeated_layer_fixed_weights;
uint8_t* g_q1_decode_order_fixed_bf16_storage=nullptr;
bool g_q1_decode_order_fixed_bf16_storage_borrowed=false;
uint64_t g_q1_decode_order_fixed_bf16_storage_bytes=0;
uint32_t g_q1_decode_order_fixed_bf16_entry_count=0;
bool g_q1_decode_order_fixed_bf16_active=false;
'''
        main = r'''
 {
 using namespace qrt_resident_ordered_shard;
 const std::vector<Input> inputs={
  {8197,{{"cold_a",128,256},{"hot_a",4090,107},{"mtp.omitted",4200,100},{"last",8188,9}}},
  {5003,{{"hot_b",0,700},{"cold_b",1024,512},{"hot_c",4000,1000}}}};
 const std::vector<std::string> order={"hot_b","hot_a","hot_c","last"};
 unsigned observed_failures=0;
 for(unsigned injection=0;injection<6;++injection){
  auto& owner=g_resident_model_shard_store;
  owner.ordered_fixed=owner.text_only=true;owner.model_dir="ordered";
  assert(build(inputs,order,true,&owner.ordered_plan));
  owner.ordered_fixed_names=order;owner.ordered_headers={"header-a","header-b"};
  std::vector<unsigned char> fixed(owner.ordered_plan.fixed_bytes+128,0xa5);
  std::array<std::vector<unsigned char>,2> disk,cold;
  owner.fixed_device_arena=fixed.data()+64;
  ResidentModelShardStoreShard bad;bad.file_bytes=8197;
  assert(!assign_resident_model_ordered_shard_layout(&owner,bad,0,8197,"changed",&stage,&error));
  assert(!assign_resident_model_ordered_shard_layout(&owner,bad,0,8196,"header-a",&stage,&error));
  assert(!assign_resident_model_ordered_shard_layout(&owner,bad,2,8197,"header-a",&stage,&error));
  assert(bad.layout.spans.empty()&&bad.fixed_layout.spans.empty()&&owner.metrics.omitted_tensor_count==0);
  for(unsigned i=0;i<2;++i){
   disk[i].resize(inputs[i].file_bytes);
   for(size_t k=0;k<disk[i].size();++k)disk[i][k]=static_cast<unsigned char>(k*11+k/5+i*23);
   cold[i].assign(owner.ordered_plan.shards[i].ordinary.device_bytes+128,0xa5);
   ResidentModelShardStoreShard item;item.file_bytes=inputs[i].file_bytes;item.device_base=cold[i].data()+64;
   assert(assign_resident_model_ordered_shard_layout(&owner,item,i,item.file_bytes,owner.ordered_headers[i],&stage,&error));
   owner.shards.push_back(item);
   for(const auto& tensor:inputs[i].tensors){
    ResidentModelShardStoreTensor metadata;metadata.shard_index=i;metadata.absolute_begin=tensor.begin;metadata.bytes=tensor.bytes;
    owner.tensors[tensor.name]=metadata;
   }
  }
  assert(owner.metrics.omitted_tensor_count==1&&owner.metrics.omitted_tensor_bytes==100);
  owner.slots[0].stream=reinterpret_cast<void*>(uintptr_t(1));
  copies=events=waits=0;pending.clear();fail_copy=fail_event=fail_wait=0;
  if(injection==1)fail_copy=2;
  if(injection==2)fail_event=2;
  if(injection==3)fail_wait=1;
  if(injection==4)fail_event=1;
  if(injection==5)fail_copy=8;
  bool ok=true;
  for(unsigned i=0;i<2&&ok;++i){
   const uint64_t streamed=disk[i].size()/4096*4096;
   for(uint64_t first=0;first<streamed&&ok;first+=4096){
    owner.slots[0].host=disk[i].data()+first;owner.slots[0].chunk_index=first/4096;
    ok=copy_resident_model_shard_store_chunk(&owner.shards[i],&owner.slots[0],streamed,0,&owner.metrics,&stage,&error);
   }
   if(ok){
    ok=copy_resident_model_shard_range(owner.shards[i],disk[i].data()+streamed,streamed,disk[i].size()-streamed,
        owner.slots[0].stream,"tail",&stage,&error);
    if(ok)drain();
   }
  }
  assert(ok==(injection==0));
  if(!ok)++observed_failures;
  if(ok){
   owner.valid=true;
   for(unsigned i=0;i<2;++i)for(const auto& tensor:inputs[i].tensors){
    const uint16_t* pointer=nullptr;
    const bool found=try_resident_model_shard_store_device_bf16_view("ordered",tensor.name,tensor.bytes,&pointer,&error);
    assert(found==qrt_resident_text_shard::keep(tensor.name));
    if(found){
     assert(!std::memcmp(pointer,disk[i].data()+tensor.begin,tensor.bytes));
     std::vector<unsigned char> slice(tensor.bytes/2+1);
     assert(copy_resident_model_shard_store_host_slice("ordered",tensor.name,0,slice.size(),slice.data(),&error));
     assert(!std::memcmp(slice.data(),disk[i].data()+tensor.begin,slice.size()));
    }
   }
   std::vector<uint64_t> offsets;
   for(const auto& name:order){
    const auto& tensor=owner.tensors.at(name);uint64_t offset=0;
    assert(qrt_resident_text_shard::offset(owner.shards[tensor.shard_index].fixed_layout,tensor.absolute_begin,tensor.bytes,&offset));
    offsets.push_back(offset);
    g_whole_repeated_layer_fixed_weights.push_back({name,true,reinterpret_cast<uint16_t*>(fixed.data()+64+offset),tensor.bytes});
   }
   uint8_t* pointer=nullptr;
   assert(bind_resident_model_ordered_fixed_arena(offsets,owner.ordered_plan.fixed_bytes,&pointer));
   assert(pointer==fixed.data()+64);
   auto broken=offsets;++broken[1];
   assert(!bind_resident_model_ordered_fixed_arena(broken,owner.ordered_plan.fixed_bytes,&pointer));
   g_whole_repeated_layer_fixed_weights[0].tensor_name="wrong";
   assert(!bind_resident_model_ordered_fixed_arena(offsets,owner.ordered_plan.fixed_bytes,&pointer));
   g_whole_repeated_layer_fixed_weights[0].tensor_name=order[0];
   g_whole_repeated_layer_fixed_weights[0].borrowed=false;
   assert(!bind_resident_model_ordered_fixed_arena(offsets,owner.ordered_plan.fixed_bytes,&pointer));
   g_whole_repeated_layer_fixed_weights.clear();
   g_q1_decode_order_fixed_bf16_storage=fixed.data()+64;
   g_q1_decode_order_fixed_bf16_storage_borrowed=true;
   g_q1_decode_order_fixed_bf16_storage_bytes=owner.ordered_plan.fixed_bytes;
   g_q1_decode_order_fixed_bf16_active=true;g_q1_decode_order_fixed_bf16_entry_count=4;
   unsigned before=freed;release_q1_decode_order_fixed_bf16_arena();
   assert(freed==before&&!g_q1_decode_order_fixed_bf16_storage&&!g_q1_decode_order_fixed_bf16_storage_borrowed);
   assert(!g_q1_decode_order_fixed_bf16_active&&!g_q1_decode_order_fixed_bf16_storage_bytes&&!g_q1_decode_order_fixed_bf16_entry_count);
  }
  unsigned before=freed;release_resident_model_shard_store();
  assert(pending.empty()&&freed==before+3&&!owner.fixed_device_arena&&!owner.valid&&!owner.ordered_fixed);
  assert(owner.ordered_plan.shards.empty()&&owner.ordered_headers.empty()&&owner.ordered_fixed_names.empty());
  for(unsigned k=0;k<64;++k){
   assert(fixed[k]==0xa5&&fixed[fixed.size()-1-k]==0xa5);
   for(unsigned i=0;i<2;++i)assert(cold[i][k]==0xa5&&cold[i][cold[i].size()-1-k]==0xa5);
  }
 }
 assert(observed_failures==5);
 // The retained independently owned arena still frees exactly once.
 unsigned before=freed;
 g_q1_decode_order_fixed_bf16_storage=reinterpret_cast<uint8_t*>(uintptr_t(1));
 release_q1_decode_order_fixed_bf16_arena();assert(freed==before+1);
 release_q1_decode_order_fixed_bf16_arena();assert(freed==before+1);
 std::puts("{\"kind\":\"resident_ordered_shard_runtime_host\",\"failed_owner_cases\":5,\"mismatches\":0}");
 }
'''
        text_shard_harness.ResidentTextShardTests().compile_run(
            text_shard_harness.runtime_host_source(globals_, actual, main))

    def test_disjoint_owners_original_bytes_holes_and_failed_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'test.cpp'
            source.write_text(layout_host_source())
            executable = directory / 'test'
            result = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(ROOT),
                str(source), '-o', str(executable)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"mismatches":0', result.stdout)


if __name__ == '__main__':
    unittest.main()
