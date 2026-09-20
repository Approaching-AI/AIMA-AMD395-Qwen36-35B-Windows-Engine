"""Run the actual resident metadata/acquire/release code with tracked owners."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def definition(source, signature):
    start = source.index(signature)
    opening = source.index('{', start)
    depth = 0
    tokens = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]'
    for token in re.finditer(tokens, source[opening:]):
        value = token.group()
        depth += (value == '{') - (value == '}')
        if not depth:
            return source[start:opening + token.end()]
    raise AssertionError(signature)


class MtpResidentWeightTests(unittest.TestCase):
    def test_original_shapes_and_actual_resident_release_lifetime(self):
        whole = (ROOT/'native/providers/whole_provider.cpp').read_text()
        core = (ROOT/'native/src/qrt.c').read_text()
        core_functions = '\n'.join(definition(core, signature) for signature in (
            'static const char *qrt_json_skip_ws(', 'static int qrt_json_parse_string(',
            'static const char *qrt_find_token_until(', 'static const char *qrt_json_skip_value(',
            'static int qrt_json_read_data_offsets_until('))
        structs = '\n'.join(definition(whole, 'struct '+name+' {')+';' for name in (
            'TensorLocation','ResidentModelShardStoreMetrics','ResidentModelShardStoreShard',
            'ResidentModelShardStoreTensor','ResidentModelShardStore'))
        actual = '\n'.join(definition(whole, signature) for signature in (
            'bool resident_model_shard_device_location(', 'bool parse_resident_model_shard_header(',
            'bool parse_qwen36_mtp_tensor_shape(',
            'std::shared_ptr<const qrt_sm121_mtp::ModelWeightSource> acquire_qwen36_mtp_model_weight_source(',
            'void release_resident_model_shard_store() {'))
        source = r'''
#include <algorithm>
#include <atomic>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
using hipError_t=int;using hipStream_t=void*;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipErrorOutOfMemory=2,hipMemcpyDeviceToDevice=4;
static std::map<void*,size_t> allocations;
static unsigned frees=0;
static hipError_t hipFree(void* pointer){assert(allocations.erase(pointer)==1);++frees;return hipSuccess;}
hipError_t hipMalloc(void**,size_t);hipError_t hipMemcpyAsync(void*,const void*,size_t,int,hipStream_t);
hipError_t hipStreamSynchronize(hipStream_t);
static hipError_t hipHostUnregister(void*){assert(false);return hipSuccess;}
static hipError_t hipHostFree(void*){assert(false);return hipSuccess;}
constexpr int MEM_RELEASE=1;
static int VirtualFree(void*,size_t,int){assert(false);return 0;}
constexpr unsigned QRT_LOAD_JSON_STRING_CAPACITY=512;
#include "native/providers/resident_ordered_shard_layout.h"
#include "sm121_mtp_resident_weights.h"
''' + core_functions + '\n' + structs + r'''
ResidentModelShardStore g_resident_model_shard_store;
std::atomic<uint64_t> g_qwen36_mtp_weight_storage_epoch{1};
''' + actual + r'''
using namespace qrt_sm121_mtp;
static std::string metadata(const ModelTensorSpec& spec,size_t first){
    std::ostringstream out;out<<"{\"dtype\": \"BF16\" , \"shape\" : [";
    for(unsigned i=0;i<spec.rank;++i)out<<(i?", ":"")<<spec.shape[i];
    out<<" ] , \"data_offsets\": ["<<first<<", "<<first+spec.bytes()<<"]}";return out.str();
}
static void initialize(){
    auto& store=g_resident_model_shard_store;
    assert(!store.valid&&!store.mtp_weight_storage&&store.shards.empty());
    const uintptr_t base=UINT64_C(0x100000000)* (1+4*g_qwen36_mtp_weight_storage_epoch.load());
    size_t total=0;std::ostringstream json;json<<'{';
    for(size_t i=0;i<model_weight_specs.size();++i){
        const auto& spec=model_weight_specs[i];
        json<<(i?",":"")<<'"'<<spec.name<<"\":"<<metadata(spec,total);total+=spec.bytes();
    }json<<'}';const auto header=json.str();
    uint64_t bytes=0,count=0;std::string failure;
    assert(parse_resident_model_shard_header(header,"original-shard",header.size(),8+header.size()+total,0,
        &store.tensors,&bytes,&count,&failure));
    assert(count==21&&bytes==total);
    const size_t fixed=model_weight_specs.back().bytes(),ordinary=total-fixed;
    ResidentModelShardStoreShard shard;shard.device_base=reinterpret_cast<void*>(base);
    shard.fixed_device_base=reinterpret_cast<void*>(base+UINT64_C(0x200000000));
    shard.layout.file_bytes=shard.fixed_layout.file_bytes=8+header.size()+total;
    shard.layout.device_bytes=ordinary;shard.fixed_layout.device_bytes=fixed;
    shard.layout.spans.push_back({8+header.size(),0,ordinary});
    shard.fixed_layout.spans.push_back({8+header.size()+ordinary,0,fixed});
    store.fixed_device_arena=shard.fixed_device_base;store.ordered_plan.fixed_bytes=fixed;
    assert(allocations.emplace(shard.device_base,ordinary).second);
    assert(allocations.emplace(shard.fixed_device_base,fixed).second);
    store.shards.push_back(shard);store.model_dir="original-model";
    store.valid=store.text_only=store.ordered_fixed=store.include_mtp=true;
}
static std::shared_ptr<const ModelWeightSource> acquire(){
    std::string stage,failure;return acquire_qwen36_mtp_model_weight_source("original-model",&stage,&failure);
}
int main(){
    // Original JSON whitespace/order is accepted. Every malformed shape is
    // rejected without mutating a caller's prior view.
    const std::string good=" { \"shape\" : [ 2048, 4096 ] , \"dtype\" : \"BF16\" } ";
    ModelTensorView out;out.bytes=16777216;assert(parse_qwen36_mtp_tensor_shape(good,&out));
    assert(out.rank==2&&out.shape[0]==2048&&out.shape[1]==4096&&out.bf16&&out.contiguous);
    const auto previous=out;
    for(const char* bad:{"", "{}", "{\"shape\":[2048,4096]}",
        "{\"dtype\":\"BF16\"}","{\"dtype\":\"F32\",\"shape\":[2048,4096]}",
        "{\"dtype\":\"BF16\",\"shape\":[2048,4096],\"shape\":[2048,4096]}",
        "{\"dtype\":\"BF16\",\"dtype\":\"BF16\",\"shape\":[2048,4096]}",
        "{\"dtype\":\"BF16\",\"shape\":[]}","{\"dtype\":\"BF16\",\"shape\":[0]}",
        "{\"dtype\":\"BF16\",\"shape\":[-1]}","{\"dtype\":\"BF16\",\"shape\":[+1]}",
        "{\"dtype\":\"BF16\",\"shape\":[01]}","{\"dtype\":\"BF16\",\"shape\":[1e3]}",
        "{\"dtype\":\"BF16\",\"shape\":[1,2,3,4]}","{\"dtype\":\"BF16\",\"shape\":[1,]}",
        "{\"dtype\":\"BF16\",\"shape\":[2048 4096]}","{\"dtype\":\"BF16\",\"shape\":[2048,4096],}",
        "{\"dtype\":\"BF16\",\"shape\":[184467440737095516160]}",
        "{\"dtype\":\"BF16\",\"shape\":[18446744073709551615,2]}",
        "{\"dtype\":\"BF16\",\"shape\":[2048,4096]}garbage"}){
        assert(!parse_qwen36_mtp_tensor_shape(bad,&out));
        assert(out.rank==previous.rank&&out.shape==previous.shape&&out.bytes==previous.bytes);
    }
    for(size_t i=0;i<good.find_last_of('}');++i)assert(!parse_qwen36_mtp_tensor_shape(good.substr(0,i),&out));
    // An unadopted ownership candidate must leave the original store alone.
    initialize();auto& store=g_resident_model_shard_store;const auto before=frees;
    {ResidentWeightStorage candidate;
        assert(candidate.describe(store.shards[0].device_base,store.shards[0].layout.device_bytes));
        assert(!candidate.describe(store.shards[0].device_base,1));
        assert(!candidate.describe(reinterpret_cast<void*>(UINTPTR_MAX-1),4));
        assert(!candidate.contains(store.shards[0].device_base,store.shards[0].layout.device_bytes+1));
    }assert(frees==before&&allocations.size()==2);
    for(size_t i=0;i<model_weight_specs.size();++i){
        auto& tensor=store.tensors.at(model_weight_specs[i].name);const auto original=tensor.mtp_metadata_json;
        tensor.mtp_metadata_json="{\"dtype\":\"BF16\",\"shape\":[1]}";
        assert(!acquire()&&!store.mtp_weight_storage&&allocations.size()==2);tensor.mtp_metadata_json=original;
    }
    store.include_mtp=false;assert(!acquire());store.include_mtp=true;
    store.text_only=false;assert(!acquire());store.text_only=true;
    store.ordered_fixed=false;assert(!acquire());store.ordered_fixed=true;
    store.shards[0].host_base=reinterpret_cast<void*>(1);assert(!acquire());store.shards[0].host_base=nullptr;
    auto* fixed=store.fixed_device_arena;store.fixed_device_arena=store.shards[0].device_base;
    assert(!acquire()&&!store.mtp_weight_storage);store.fixed_device_arena=fixed;
    auto source=acquire();assert(source&&store.mtp_weight_storage&&store.mtp_weight_storage->adopted());
    auto second=acquire();assert(second&&store.mtp_weight_storage->allocations()==2);
    const auto epoch=source->epoch();ModelTensorView view;
    for(const auto& spec:model_weight_specs){
        assert(source->tensor(spec.name,&view)&&view.bytes==spec.bytes()&&view.shape==spec.shape);
    }
    std::weak_ptr<ResidentWeightStorage> old_owner=store.mtp_weight_storage;
    release_resident_model_shard_store();
    assert(!store.valid&&store.tensors.empty()&&store.shards.empty()&&!store.mtp_weight_storage);
    assert(allocations.size()==2&&!old_owner.expired()&&source->epoch()==epoch+1);
    assert(!source->tensor(model_weight_specs[0].name,&view)&&!view.device);
    // A new model can acquire independent ownership before old readers die.
    initialize();auto replacement=acquire();assert(replacement&&allocations.size()==4);
    source.reset();assert(allocations.size()==4&&!old_owner.expired());
    second.reset();assert(allocations.size()==2&&old_owner.expired());
    release_resident_model_shard_store();assert(allocations.size()==2);
    replacement.reset();assert(allocations.empty());
    // The default, unleased release still frees each original allocation once.
    initialize();assert(!store.mtp_weight_storage);release_resident_model_shard_store();
    assert(allocations.empty());
    // Snapshot names survive original map/header destruction even without
    // changing the global epoch (the source owns canonical immutable names).
    auto storage=std::make_shared<ResidentWeightStorage>();
    void* base=reinterpret_cast<void*>(UINT64_C(0x8000000000));allocations[base]=4096;
    assert(storage->describe(base,4096)&&storage->adopt());
    ResidentModelWeightSource::Views views{};std::string temporary_name=model_weight_specs[1].name;
    views[1]={temporary_name.c_str(),static_cast<const uint16_t*>(base),1,{2048,0,0},4096,epoch,true,true};
    std::atomic<uint64_t> generation{epoch};
    {ResidentModelWeightSource snapshot(storage,&generation,epoch,views);temporary_name.clear();temporary_name.shrink_to_fit();
        assert(snapshot.tensor(model_weight_specs[1].name,&view)&&view.device==base);
    }storage.reset();assert(allocations.empty());
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ('sm121_mtp_model_weights.h','sm121_mtp_resident_weights.h'):
                header = (ROOT/'native/providers/gdn'/name).read_text().replace('#include <hip/hip_runtime.h>', '')
                (directory/name).write_text(header)
            path = directory/'test.cpp'; path.write_text(source)
            exe = directory/'test'
            build = subprocess.run(['c++','-std=c++17','-O1','-Wall','-Wextra','-Werror',
                '-fsanitize=address,undefined','-fno-sanitize-recover=all','-I',str(directory),
                '-I',str(ROOT),str(path),'-o',str(exe)],capture_output=True,text=True,timeout=60)
            self.assertEqual(build.returncode,0,build.stderr)
            run = subprocess.run([str(exe)],capture_output=True,text=True,timeout=30)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)


if __name__ == '__main__':
    unittest.main()
