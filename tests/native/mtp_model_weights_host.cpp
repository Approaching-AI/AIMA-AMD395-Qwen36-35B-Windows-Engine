#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "mtp_weight_types.h"
using hipError_t = int;
using hipStream_t = void*;
constexpr int hipSuccess=0, hipErrorInvalidValue=1, hipErrorOutOfMemory=2;
constexpr int hipMemcpyDeviceToDevice=4, injected=99;
struct Copy { void* destination; const void* source; size_t bytes; };
static std::vector<Copy> pending;
static std::map<void*,size_t> allocations;
static unsigned allocation_calls=0, copy_calls=0, sync_calls=0, fail_copy=0;
static bool fail_allocation=false, fail_completion=false;
static uint64_t* change_epoch_after_sync=nullptr;
static hipStream_t expected_stream=reinterpret_cast<void*>(0x1234);
static bool inside(const void* pointer, const void* base, size_t bytes) {
    const uintptr_t p=reinterpret_cast<uintptr_t>(pointer),b=reinterpret_cast<uintptr_t>(base);
    return p>=b&&p-b<bytes;
}
static hipError_t hipMalloc(void** out,size_t bytes) {
    ++allocation_calls;if(fail_allocation)return hipErrorOutOfMemory;
    *out=std::malloc(bytes);assert(*out);std::memset(*out,0x5a,bytes);allocations[*out]=bytes;
    return hipSuccess;
}
static hipError_t hipFree(void* pointer) {
    assert(allocations.count(pointer));
    for(const auto& copy:pending)assert(!inside(copy.destination,pointer,allocations.at(pointer)));
    allocations.erase(pointer);std::free(pointer);return hipSuccess;
}
static hipError_t hipMemcpyAsync(void* out,const void* in,size_t bytes,int kind,hipStream_t stream) {
    assert(kind==hipMemcpyDeviceToDevice&&stream==expected_stream);
    pending.push_back({out,in,bytes});return ++copy_calls==fail_copy?injected:hipSuccess;
}
static hipError_t hipStreamSynchronize(hipStream_t stream) {
    assert(stream==expected_stream);++sync_calls;if(fail_completion)return injected;
    for(const auto& copy:pending)std::memcpy(copy.destination,copy.source,copy.bytes);
    pending.clear();if(change_epoch_after_sync)++*change_epoch_after_sync;
    return hipSuccess;
}
#include "sm121_mtp_model_weights.h"
using namespace qrt_sm121_mtp;
struct Source final:ModelWeightSource {
    std::array<std::vector<uint16_t>,4> parts;
    std::array<uint16_t,21> markers{};
    mutable uint64_t generation=7;
    mutable unsigned lookups=0;
    int corrupt_index=-1, corrupt_kind=0;
    bool change_during_lookup=false;
    Source() {
        for(unsigned i=0;i<4;++i)parts[i].assign(mtp_model_weight_detail::part_elements,uint16_t(0x1100+i));
    }
    ~Source() {
        for(const auto& part:parts)for(const auto& copy:pending)
            assert(!inside(copy.source,part.data(),part.size()*sizeof(uint16_t)));
    }
    uint64_t epoch() const noexcept override { return generation; }
    const uint16_t* pointer(unsigned index) const {
        constexpr unsigned packed_indices[]={5,6,14,15};
        for(unsigned i=0;i<4;++i)if(index==packed_indices[i])return parts[i].data();
        return markers.data()+index;
    }
    bool tensor(const char* name,ModelTensorView* out) const override {
        const auto found=std::find_if(model_weight_specs.begin(),model_weight_specs.end(),
            [&](const auto& item){return !std::strcmp(name,item.name);});
        assert(found!=model_weight_specs.end());const unsigned index=unsigned(found-model_weight_specs.begin());
        ++lookups;*out={found->name,pointer(index),found->rank,found->shape,found->bytes(),generation,true,true};
        if(change_during_lookup&&index==20)++generation;
        if(int(index)!=corrupt_index)return true;
        switch(corrupt_kind){
        case 0:return false;
        case 1:out->name="wrong.tensor";break;
        case 2:out->device=nullptr;break;
        case 3:out->device=reinterpret_cast<const uint16_t*>(reinterpret_cast<uintptr_t>(out->device)+1);break;
        case 4:++out->rank;break;
        case 5:++out->shape[0];break;
        case 6:--out->bytes;break;
        case 7:out->bf16=false;break;
        case 8:out->contiguous=false;break;
        case 9:++out->epoch;break;
        case 10:throw std::runtime_error("lookup failed");
        case 11:throw std::bad_alloc();
        default:assert(false);
        }
        return true;
    }
};
static void reset(){
    assert(pending.empty());allocation_calls=copy_calls=sync_calls=fail_copy=0;
    fail_allocation=fail_completion=false;change_epoch_after_sync=nullptr;
}
static ModelWeightStep prepare(ModelWeights& owner,const std::shared_ptr<Source>& source) {
    return owner.prepare(source,7,expected_stream);
}
static DrafterWeights weights(const ModelWeightBinding& binding,uint64_t epoch=7) {
    DrafterWeights out;assert(binding.weights(epoch,&out));return out;
}
static void verify_mapping(const DrafterWeights& w,const Source& source) {
    const uint16_t* borrowed[]={w.prompt.embeddings,w.prompt.embedding_norm,w.prompt.hidden_norm,
        w.prompt.fusion,w.prompt.input_norm,w.prompt.key_norm,w.query,w.query_norm,w.output,
        w.post_attention_norm,w.moe.router,w.moe.shared_gate,w.moe.shared_down,w.moe.routed_gate_up,
        w.moe.routed_down,w.final_norm,w.lm_head};
    const unsigned indices[]={0,1,2,3,4,7,8,9,10,11,12,13,16,17,18,19,20};
    for(unsigned i=0;i<17;++i)assert(borrowed[i]==source.pointer(indices[i]));
    assert(!w.prompt.split1024_pre_fc_norm);
    const uint16_t* parts[]={w.prompt.kv_projection,w.prompt.kv_projection+1048576,
        w.moe.shared_gate_up,w.moe.shared_gate_up+1048576};
    for(unsigned i=0;i<4;++i){
        assert(parts[i]!=source.parts[i].data());
        assert(std::equal(source.parts[i].begin(),source.parts[i].end(),parts[i]));
    }
}
int main(){
    // Every original tensor must have an exact identity, layout and lease.
    auto source=std::make_shared<Source>();
    for(int index=0;index<21;++index)for(int kind=0;kind<12;++kind){
        reset();source->corrupt_index=index;source->corrupt_kind=kind;
        ModelWeights owner;auto result=prepare(owner,source);
        assert(result.status==(kind==11?hipErrorOutOfMemory:hipErrorInvalidValue));
        assert(!result.completion_unknown&&!owner.binding(7).valid(7)&&allocation_calls==0&&copy_calls==0);
    }
    source->corrupt_index=-1;reset();{
        ModelWeights owner;assert(owner.prepare({},7,expected_stream).status==hipErrorInvalidValue);
        assert(owner.prepare(source,0,expected_stream).status==hipErrorInvalidValue);
        assert(owner.prepare(source,8,expected_stream).status==hipErrorInvalidValue);
        source->change_during_lookup=true;assert(prepare(owner,source).status==hipErrorInvalidValue);
        assert(allocation_calls==0);source->generation=7;source->change_during_lookup=false;
    }
    ModelWeightBinding retained;
    std::weak_ptr<Source> lifetime=source;
    reset();{
        ModelWeights owner;assert(prepare(owner,source).status==hipSuccess);
        assert(allocation_calls==1&&copy_calls==4&&sync_calls==1&&allocations.size()==1);
        retained=owner.binding(7);assert(retained.allocated_bytes()==8388608);
        auto original=weights(retained);verify_mapping(original,*source);
        assert(!owner.binding(8).valid(8));
        for(unsigned failed=0;failed<6;++failed){
            reset();fail_allocation=failed==0;fail_copy=failed>=1&&failed<=4?failed:0;
            change_epoch_after_sync=failed==5?&source->generation:nullptr;
            auto result=prepare(owner,source);
            assert(result.status!=hipSuccess&&!result.completion_unknown&&!owner.quarantined());
            assert(pending.empty()&&allocations.size()==1);
            if(failed==5){assert(!retained.valid(7));source->generation=7;}
            assert(weights(owner.binding(7)).prompt.kv_projection==original.prompt.kv_projection);
        }
        reset();auto replacement=std::make_shared<Source>();
        assert(prepare(owner,replacement).status==hipSuccess&&allocations.size()==2);
        assert(weights(owner.binding(7)).prompt.kv_projection!=original.prompt.kv_projection);
        assert(weights(retained).prompt.kv_projection==original.prompt.kv_projection);
        source.reset();assert(!lifetime.expired());verify_mapping(weights(retained),*lifetime.lock());
        DrafterWeights invalid=original;assert(!retained.weights(8,&invalid)&&!invalid.prompt.embeddings&&!invalid.lm_head);
        assert(!retained.weights<DrafterWeights>(7,nullptr));
    }
    assert(allocations.size()==1&&!lifetime.expired());retained={};
    assert(allocations.empty()&&lifetime.expired());
    // Both ordinary and partially failed enqueues retain every borrowed input
    // after a failed fence, even when source and owner objects leave scope.
    for(unsigned failed=0;failed<=4;++failed){
        reset();std::weak_ptr<Source> borrowed;
        {ModelWeights owner;auto model=std::make_shared<Source>();borrowed=model;
            fail_copy=failed;fail_completion=true;
            auto result=prepare(owner,model);
            assert(result.status==injected&&result.completion_unknown&&owner.quarantined());
            assert(!owner.binding(7).valid(7));unsigned calls=copy_calls;
            assert(prepare(owner,model).completion_unknown&&copy_calls==calls);
        }
        assert(!borrowed.expired()&&!pending.empty());fail_completion=false;
        assert(hipStreamSynchronize(expected_stream)==hipSuccess);
    }
    // A downstream drafter fence can quarantine an already completed binding.
    reset();std::weak_ptr<Source> downstream;
    {ModelWeights owner;auto model=std::make_shared<Source>();downstream=model;
        assert(prepare(owner,model).status==hipSuccess);auto view=owner.binding(7);
        view.quarantine();view.quarantine();assert(!view.valid(7)&&owner.quarantined());
        DrafterWeights invalid;assert(!view.weights(7,&invalid)&&!invalid.prompt.embeddings);
        assert(prepare(owner,model).completion_unknown);
    }
    assert(!downstream.expired()&&allocations.size()==6&&pending.empty());
    // Simulate process device teardown only after every queued copy completed.
    while(!allocations.empty())hipFree(allocations.begin()->first);
}
