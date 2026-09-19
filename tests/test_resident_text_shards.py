"""Exercise packed shard bytes and the actual runtime copy/view functions."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


def runtime_host_source(extra_globals="", extra_functions="", extra_main=""):
    whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
    declarations = '\n'.join(function(whole, name) + ';' for name in (
        'struct ResidentModelShardStoreMetrics',
        'struct ResidentModelShardStoreShard',
        'struct ResidentModelShardStoreTensor',
        'struct ResidentModelShardStore {'))
    functions = '\n'.join(function(whole, name) for name in (
        'bool resident_model_shard_device_location(',
        'bool copy_resident_model_shard_range(',
        'bool copy_resident_model_shard_store_chunk(',
        'bool try_resident_model_shard_store_device_bf16_view(',
        'bool copy_resident_model_shard_store_host_slice(',
        'void release_resident_model_shard_store('))
    source = r'''
#include "native/providers/resident_text_shard_layout.h"
#include "native/providers/resident_ordered_shard_layout.h"
#include <cassert>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <array>
#include <cstdio>
struct TensorLocation{std::string shard;uint64_t header_len=0,data_begin=0,data_end=0;};
using hipStream_t=void*;using hipEvent_t=void*;
using HANDLE=void*;
#define _WIN32 1
#define INVALID_HANDLE_VALUE reinterpret_cast<void*>(uintptr_t(-1))
constexpr unsigned MEM_RELEASE=1;
constexpr size_t kResidentModelShardStoreQueueDepth=4;
enum hipError_t{hipSuccess,hipErrorInvalidValue};
enum hipMemcpyKind{hipMemcpyHostToDevice,hipMemcpyDeviceToHost};
constexpr uint64_t kResidentModelShardStoreIoChunkBytes=4096;
struct ResidentModelShardStoreIoSlot{uint64_t chunk_index=0;void* host=nullptr;
 hipStream_t stream=nullptr;hipEvent_t copy_start=nullptr,copy_end=nullptr;bool copy_pending=false;
 bool read_pending=false;HANDLE read_file=INVALID_HANDLE_VALUE,read_event=nullptr;unsigned read_overlapped=0;};
uint64_t qrt_now_ns(){static uint64_t value=0;return ++value;}
uint64_t qrt_elapsed_ns(uint64_t a,uint64_t b){return b-a;}
const char* hipGetErrorString(hipError_t){return "injected";}
struct Pending{void* destination;const void* source;size_t bytes;};
std::vector<Pending> pending;
unsigned copies=0,fail_copy=0,events=0,fail_event=0,waits=0,fail_wait=0;
hipError_t hipMemcpyAsync(void* to,const void* from,size_t n,hipMemcpyKind,hipStream_t){
 if(++copies==fail_copy)return hipErrorInvalidValue;
 pending.push_back({to,from,n});return hipSuccess;
}
hipError_t hipMemcpy(void* to,const void* from,size_t n,hipMemcpyKind){std::memcpy(to,from,n);return hipSuccess;}
hipError_t hipEventRecord(hipEvent_t,hipStream_t){return ++events==fail_event?hipErrorInvalidValue:hipSuccess;}
void drain(){for(auto p:pending)std::memcpy(p.destination,p.source,p.bytes);pending.clear();}
unsigned freed=0,cancelled=0,drained=0;
hipError_t hipStreamSynchronize(hipStream_t){++drained;drain();return hipSuccess;}
hipError_t hipFree(void*){assert(pending.empty());++freed;return hipSuccess;}
hipError_t hipHostFree(void*){assert(pending.empty());return hipSuccess;}
hipError_t hipHostUnregister(void*){assert(pending.empty());return hipSuccess;}
bool VirtualFree(void*,size_t,unsigned){assert(pending.empty());return true;}
void cancel_and_drain_resident_model_shard_store_read(HANDLE,unsigned*){++cancelled;}
bool CloseHandle(HANDLE){return true;}
hipError_t hipEventDestroy(hipEvent_t){return hipSuccess;}
hipError_t hipStreamDestroy(hipStream_t){return hipSuccess;}
hipError_t hipEventSynchronize(hipEvent_t){if(++waits==fail_wait)return hipErrorInvalidValue;drain();return hipSuccess;}
hipError_t hipEventElapsedTime(float* value,hipEvent_t,hipEvent_t){*value=0.125f;return hipSuccess;}
bool check_hip(hipError_t value,const char* stage,std::string* where,std::string* message){
 if(value==hipSuccess)return true;*where=stage;*message="injected";return false;
}
''' + declarations + r'''
ResidentModelShardStore g_resident_model_shard_store;
''' + extra_globals + '\n' + functions + '\n' + extra_functions + r'''
int main(){
 std::vector<unsigned char> source(8197),device(4096+128,0xa5);
 for(size_t i=0;i<source.size();++i)source[i]=static_cast<unsigned char>(i*19+i/7);
 ResidentModelShardStoreShard shard;shard.device_base=device.data()+64;shard.file_bytes=source.size();
 assert(qrt_resident_text_shard::build(source.size(),
  {{128,512,true},{640,512,false},{1152,512,true},{4096,512,false},{8188,9,true}},true,&shard.layout));
 ResidentModelShardStoreIoSlot slot;slot.host=source.data();
 ResidentModelShardStoreMetrics metrics;std::string stage,error;
 for(unsigned failure=0;failure<5;++failure){
  std::fill(device.begin(),device.end(),0xa5);pending.clear();copies=events=waits=0;
  fail_copy=failure==1?2:0;fail_event=failure==2?2:0;fail_wait=failure==3?1:0;
  if(failure==4)fail_event=1;
  slot.chunk_index=0;slot.copy_pending=false;
  const bool ok=copy_resident_model_shard_store_chunk(&shard,&slot,8192,0,&metrics,&stage,&error);
  assert(ok==(failure==0));
  if(!ok)assert(!stage.empty());
  if(!ok){
   auto& owner=g_resident_model_shard_store;owner.valid=true;owner.text_only=true;
   owner.shards.push_back(shard);owner.slots[0].stream=reinterpret_cast<void*>(uintptr_t(1));
   owner.slots[0].read_pending=true;owner.slots[0].read_file=reinterpret_cast<void*>(uintptr_t(2));
   const unsigned before=freed,before_cancel=cancelled,before_drain=drained;
   release_resident_model_shard_store();
   assert(pending.empty()&&freed==before+1&&cancelled==before_cancel+1&&drained==before_drain+1);
   assert(owner.shards.empty()&&owner.tensors.empty()&&!owner.valid&&!owner.text_only);
  }
  if(ok){
   assert(!slot.copy_pending&&copies==2&&pending.empty());
   assert(!std::memcmp(device.data()+64,source.data()+128,512));
   assert(!std::memcmp(device.data()+64+512,source.data()+1152,512));
  }
  for(unsigned i=0;i<64;++i)assert(device[i]==0xa5&&device[device.size()-1-i]==0xa5);
 }
 fail_copy=fail_event=fail_wait=0;pending.clear();copies=events=waits=0;
 slot.chunk_index=0;slot.host=source.data();
 assert(copy_resident_model_shard_store_chunk(&shard,&slot,8192,0,&metrics,&stage,&error));
 copies=0;
 slot.chunk_index=1;slot.host=source.data()+4096;
 assert(copy_resident_model_shard_store_chunk(&shard,&slot,8192,0,&metrics,&stage,&error));
 assert(copies==1&&!std::memcmp(device.data()+64+1024,source.data()+8188,4));
 // The genuine five-byte buffered tail continues the same compact span.
 assert(qrt_resident_text_shard::copy(shard.layout,8192,5,[&](auto to,auto from,auto bytes){
  return hipMemcpyAsync(device.data()+64+to,source.data()+8192+from,bytes,hipMemcpyHostToDevice,nullptr)==hipSuccess;
 }));drain();
 assert(!std::memcmp(device.data()+64+1024,source.data()+8188,9));
 auto& store=g_resident_model_shard_store;store.valid=true;store.text_only=true;store.model_dir="model";
 store.shards.push_back(shard);
 ResidentModelShardStoreTensor a;a.absolute_begin=1152;a.bytes=512;a.shard_index=0;
 store.tensors["text"]=a;a.absolute_begin=640;store.tensors["mtp"]=a;
 const uint16_t* view=reinterpret_cast<const uint16_t*>(uintptr_t(1));
 assert(try_resident_model_shard_store_device_bf16_view("model","text",512,&view,&error));
 assert(view==reinterpret_cast<const uint16_t*>(device.data()+64+512));
 assert(!try_resident_model_shard_store_device_bf16_view("model","mtp",512,&view,&error)&&!view);
 assert(!try_resident_model_shard_store_device_bf16_view("other","text",512,&view,&error)&&!view);
 std::array<unsigned char,128> result{};
 assert(copy_resident_model_shard_store_host_slice("model","text",23,result.size(),result.data(),&error));
 assert(!std::memcmp(result.data(),source.data()+1152+23,result.size()));
 result.fill(0xb7);
 assert(!copy_resident_model_shard_store_host_slice("model","mtp",0,result.size(),result.data(),&error));
 assert(!copy_resident_model_shard_store_host_slice("model","text",UINT64_MAX,result.size(),result.data(),&error));
 assert(std::all_of(result.begin(),result.end(),[](auto x){return x==0xb7;}));
 // Ordinary/host-backed storage retains its original absolute mapping.
 assert(qrt_resident_text_shard::build(source.size(),{{128,512,true}},false,&store.shards[0].layout));
 store.shards[0].host_mapped=true;store.shards[0].host_base=source.data();store.shards[0].device_base=source.data();
 assert(copy_resident_model_shard_store_host_slice("model","text",23,result.size(),result.data(),&error));
 assert(!std::memcmp(result.data(),source.data()+1152+23,result.size()));
 release_resident_model_shard_store();
 assert(store.shards.empty()&&store.tensors.empty()&&!store.valid&&!store.text_only);
''' + extra_main + r'''
 std::puts("{\"kind\":\"resident_text_shard_runtime_host\",\"actual_runtime_functions\":6,\"mismatches\":0}");
}
'''
    return source


class ResidentTextShardTests(unittest.TestCase):
    def compile_run(self, source):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            unit = directory / 'test.cpp'
            unit.write_text(source)
            binary = directory / 'test'
            compiled = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            '-I', str(ROOT), str(unit), '-o', str(binary)],
                           capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(binary)], check=True, capture_output=True,
                                    text=True, timeout=60)
            self.assertIn('"mismatches":0', result.stdout)

    def test_generated_bytes_holes_failed_copies_and_wide_offsets(self):
        self.compile_run('#include "tests/native/resident_text_shard_layout_host.cpp"\n')

    def test_actual_runtime_copy_and_tensor_views(self):
        self.compile_run(runtime_host_source())


if __name__ == '__main__':
    unittest.main()
