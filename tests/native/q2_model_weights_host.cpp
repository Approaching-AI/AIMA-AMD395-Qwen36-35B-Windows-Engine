#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "target_weight_types.h"
using hipError_t=int; using hipStream_t=void*;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipErrorOutOfMemory=2,hipMemcpyDeviceToDevice=4,injected=99;
struct Copy {void* out;const void* in;size_t bytes;};
static std::vector<Copy> pending;
static std::map<void*,size_t> allocations;
static unsigned copies=0,allocations_called=0,syncs=0,fail_copy=0;
static bool fail_alloc=false,fail_fence=false;
static uint64_t* change_epoch=nullptr;
static hipStream_t stream=reinterpret_cast<void*>(0x1234u);
static bool inside(const void* pointer,const void* base,size_t bytes){
    const auto p=reinterpret_cast<uintptr_t>(pointer),b=reinterpret_cast<uintptr_t>(base);
    return p>=b&&p-b<bytes;
}
static hipError_t hipMalloc(void** out,size_t bytes){
    ++allocations_called;if(fail_alloc)return hipErrorOutOfMemory;
    *out=std::malloc(bytes);assert(*out);allocations[*out]=bytes;return hipSuccess;
}
static hipError_t hipFree(void* pointer){
    assert(allocations.count(pointer));
    for(const auto& c:pending)assert(!inside(c.out,pointer,allocations.at(pointer)));
    allocations.erase(pointer);std::free(pointer);return hipSuccess;
}
static hipError_t hipMemcpyAsync(void* out,const void* in,size_t bytes,int kind,hipStream_t actual){
    assert(kind==hipMemcpyDeviceToDevice&&actual==stream);
    pending.push_back({out,in,bytes});return ++copies==fail_copy?injected:hipSuccess;
}
static hipError_t hipStreamSynchronize(hipStream_t actual){
    assert(actual==stream);++syncs;if(fail_fence)return injected;
    for(const auto& c:pending)std::memcpy(c.out,c.in,c.bytes);
    pending.clear();if(change_epoch)++*change_epoch;return hipSuccess;
}
#include "sm121_q2_resident_weights.h"
using namespace qrt_sm121_q2;
using qrt_sm121_mtp::ModelTensorView;
struct Source final:qrt_sm121_mtp::ModelWeightSource{
    std::vector<uint16_t> packed_inputs;
    std::array<uint16_t,target_weight_count> markers{};
    mutable uint64_t generation=7;
    int corrupt=-1,kind=0;
    bool change_on_lookup=false;
    Source():packed_inputs(target_weight_detail::shared_part_elements+80u*32u){
        for(size_t i=0;i<packed_inputs.size();++i)packed_inputs[i]=uint16_t(i*37u+11u);
    }
    ~Source(){for(const auto& c:pending)assert(!inside(c.in,packed_inputs.data(),packed_inputs.size()*2u));}
    uint64_t epoch()const noexcept override{return generation;}
    const uint16_t* pointer(const TensorSpec& spec)const{
        if(spec.role==WeightRole::SharedGateProjection||spec.role==WeightRole::SharedUpProjection)
            return packed_inputs.data()+(spec.layer*2u+(spec.role==WeightRole::SharedUpProjection))*32u;
        return markers.data()+(&spec-target_weight_specs().data());
    }
    bool tensor(const char* name,ModelTensorView* out)const override{
        const auto& specs=target_weight_specs();
        const auto it=std::find_if(specs.begin(),specs.end(),[&](const auto& s){return s.name==name;});
        assert(it!=specs.end());const auto& s=*it;
        *out={s.name.c_str(),pointer(s),s.rank,s.shape,s.bytes(),generation,true,true};
        if(change_on_lookup&&s.role==WeightRole::KeyNorm&&s.layer==39)++generation;
        if(it-specs.begin()!=corrupt)return true;
        switch(kind){
        case 0:return false;
        case 1:out->name="wrong.tensor";break;
        case 2:out->device=nullptr;break;
        case 3:out->device=reinterpret_cast<const uint16_t*>(reinterpret_cast<uintptr_t>(out->device)+1u);break;
        case 4:++out->rank;break;
        case 5:++out->shape[0];break;
        case 6:--out->bytes;break;
        case 7:out->bf16=false;break;
        case 8:out->contiguous=false;break;
        case 9:++out->epoch;break;
        case 10:throw std::runtime_error("original lookup failure");
        case 11:throw std::bad_alloc();
        case 12:out->device=reinterpret_cast<const uint16_t*>(UINTPTR_MAX-1u);break;
        default:assert(false);
        }
        return true;
    }
};
static void reset(){assert(pending.empty());copies=allocations_called=syncs=fail_copy=0;fail_alloc=fail_fence=false;change_epoch=nullptr;}
static const uint16_t* expected(const Source& source,unsigned layer,WeightRole role){
    for(const auto& spec:target_weight_specs())if(spec.layer==layer&&spec.role==role)return source.pointer(spec);
    return nullptr;
}
static void verify(const ModelWeightBinding& binding,const Source& source){
    assert(binding.valid(7)&&binding.allocated_bytes()==167772160u);
    std::set<std::string> names;
    for(const auto& spec:target_weight_specs()){
        assert(names.insert(spec.name).second);
        assert(binding.tensor(7,spec.layer,spec.role)==source.pointer(spec));
    }
    assert(names.size()==633u);
    const auto ptr=[&](unsigned l,WeightRole r){return expected(source,l,r);};
    for(unsigned layer=0;layer<40;++layer){
        LinearLayerViews<uint16_t> linear;AttentionLayerViews attention;
        linear.hidden=attention.hidden=reinterpret_cast<const uint16_t*>(0x2000u);
        if(layer%4u!=3u){
            assert(binding.linear_weights(7,layer,&linear)&&!binding.attention_weights(7,layer,&attention));
            assert(linear.linear.qkv_weights==ptr(layer,WeightRole::Qkv));
            assert(linear.linear.z_weights==ptr(layer,WeightRole::Z));
            assert(linear.linear.a_weights==ptr(layer,WeightRole::A));
            assert(linear.linear.b_weights==ptr(layer,WeightRole::B));
            assert(linear.linear.output_weights==ptr(layer,WeightRole::Output));
            assert(linear.linear.norm_weights==ptr(layer,WeightRole::LinearNorm));
            assert(linear.linear.convolution.weights==ptr(layer,WeightRole::Convolution));
        }else{
            assert(binding.attention_weights(7,layer,&attention)&&!binding.linear_weights(7,layer,&linear));
            assert(attention.attention.q_weights==ptr(layer,WeightRole::Query));
            assert(attention.attention.k_weights==ptr(layer,WeightRole::Key));
            assert(attention.attention.v_weights==ptr(layer,WeightRole::Value));
            assert(attention.attention.output_weights==ptr(layer,WeightRole::Output));
            assert(attention.attention.q_norm_weights==ptr(layer,WeightRole::QueryNorm));
            assert(attention.attention.k_norm_weights==ptr(layer,WeightRole::KeyNorm));
        }
        const auto& moe=layer%4u!=3u?linear.moe_weights:attention.moe_weights;
        assert((layer%4u!=3u?linear.input_norm_weights:attention.input_norm_weights)==ptr(layer,WeightRole::InputNorm));
        assert((layer%4u!=3u?linear.post_norm_weights:attention.post_norm_weights)==ptr(layer,WeightRole::PostNorm));
        assert(moe.router==ptr(layer,WeightRole::Router)&&moe.shared_gate==ptr(layer,WeightRole::SharedGate));
        assert(moe.shared_down==ptr(layer,WeightRole::SharedDown)&&moe.routed_gate_up==ptr(layer,WeightRole::RoutedGateUp));
        assert(moe.routed_down==ptr(layer,WeightRole::RoutedDown)&&moe.shared_gate_up==binding.shared_gate_up(7,layer));
        assert(std::equal(ptr(layer,WeightRole::SharedGateProjection),
            ptr(layer,WeightRole::SharedGateProjection)+target_weight_detail::shared_part_elements,moe.shared_gate_up));
        assert(std::equal(ptr(layer,WeightRole::SharedUpProjection),
            ptr(layer,WeightRole::SharedUpProjection)+target_weight_detail::shared_part_elements,
            moe.shared_gate_up+target_weight_detail::shared_part_elements));
        assert(linear.hidden==reinterpret_cast<const uint16_t*>(0x2000u)&&attention.hidden==linear.hidden);
    }
    assert(!binding.tensor(8,0,WeightRole::Qkv)&&!binding.tensor(7,41,WeightRole::Qkv));
    assert(!binding.tensor(7,0,WeightRole::Count)&&!binding.tensor(7,3,WeightRole::Qkv));
    assert(!binding.shared_gate_up(7,40)&&!binding.shared_gate_up(8,0));
}
int main(){
    auto source=std::make_shared<Source>();
    for(int index=0;index<633;++index)for(int kind=0;kind<13;++kind){
        reset();source->corrupt=index;source->kind=kind;ModelWeights owner;
        const auto r=owner.prepare(source,7,stream);
        assert(r.status==(kind==11?hipErrorOutOfMemory:hipErrorInvalidValue)&&!r.completion_unknown);
        assert(!owner.binding(7).valid(7)&&allocations_called==0&&copies==0);
    }
    source->corrupt=-1;reset();{
        ModelWeights owner;
        assert(owner.prepare({},7,stream).status==hipErrorInvalidValue);
        assert(owner.prepare(source,0,stream).status==hipErrorInvalidValue);
        assert(owner.prepare(source,8,stream).status==hipErrorInvalidValue);
        source->change_on_lookup=true;assert(owner.prepare(source,7,stream).status==hipErrorInvalidValue);
        assert(!allocations_called);source->generation=7;source->change_on_lookup=false;
    }
    ModelWeightBinding retained;std::weak_ptr<Source> weak=source;
    reset();{
        ModelWeights owner;assert(owner.prepare(source,7,stream).status==hipSuccess);
        assert(allocations_called==1&&copies==80&&syncs==1);retained=owner.binding(7);verify(retained,*source);
        const auto* pack=retained.shared_gate_up(7,0);
        for(unsigned failed=0;failed<=81;++failed){
            reset();fail_alloc=failed==0;fail_copy=failed>=1&&failed<=80?failed:0;
            change_epoch=failed==81?&source->generation:nullptr;
            const auto r=owner.prepare(source,7,stream);
            assert(r.status!=hipSuccess&&!r.completion_unknown&&pending.empty()&&allocations.size()==1u);
            if(failed==81){assert(!retained.valid(7));source->generation=7;}
            assert(owner.binding(7).shared_gate_up(7,0)==pack);
        }
        reset();auto second=std::make_shared<Source>();assert(owner.prepare(second,7,stream).status==hipSuccess);
        assert(owner.binding(7).shared_gate_up(7,0)!=pack&&allocations.size()==2u);
        source.reset();assert(!weak.expired());verify(retained,*weak.lock());
    }
    assert(allocations.size()==1u&&!weak.expired());retained={};assert(allocations.empty()&&weak.expired());
    // The real resident source owns immutable names and validates containment
    // independently of tensor shape validation in ModelWeights::prepare.
    reset();{
        auto storage=std::make_shared<qrt_sm121_mtp::ResidentWeightStorage>();
        void* allocation=nullptr;assert(hipMalloc(&allocation,4096u)==hipSuccess&&storage->describe(allocation,4096u));
        std::atomic<uint64_t> epoch{7};ResidentTargetWeightSource::Views views{};
        std::string transient=target_weight_specs()[1].name;
        views[1]={transient.c_str(),static_cast<const uint16_t*>(allocation),1,{2048,0,0},4096u,7,true,true};
        auto model=std::make_shared<ResidentTargetWeightSource>(storage,&epoch,7,views);
        ModelTensorView v;assert(!model->tensor(transient.c_str(),&v));assert(storage->adopt());
        transient.clear();transient.shrink_to_fit();
        assert(model->tensor(target_weight_specs()[1].name.c_str(),&v)&&v.device==allocation);
        assert(!model->tensor("foreign",&v)&&!v.device);
        epoch=8;assert(!model->tensor(target_weight_specs()[1].name.c_str(),&v));epoch=7;
        storage.reset();assert(allocations.size()==1);model.reset();assert(allocations.empty());
    }
    for(unsigned failed:{0u,1u,80u}){
        reset();std::weak_ptr<Source> borrowed;
        {ModelWeights owner;auto model=std::make_shared<Source>();borrowed=model;fail_copy=failed;fail_fence=true;
            const auto r=owner.prepare(model,7,stream);
            assert(r.status==injected&&r.completion_unknown&&owner.quarantined()&&!owner.binding(7).valid(7));
            const unsigned before=copies;assert(owner.prepare(model,7,stream).completion_unknown&&before==copies);
        }
        assert(!borrowed.expired()&&!pending.empty());fail_fence=false;assert(hipStreamSynchronize(stream)==hipSuccess);
    }
    reset();std::weak_ptr<Source> downstream;
    {ModelWeights owner;auto model=std::make_shared<Source>();downstream=model;
        assert(owner.prepare(model,7,stream).status==hipSuccess);auto binding=owner.binding(7);
        binding.quarantine();binding.quarantine();assert(owner.quarantined()&&!binding.valid(7));
    }
    assert(!downstream.expired()&&allocations.size()==4u&&pending.empty());
    while(!allocations.empty())hipFree(allocations.begin()->first);
}
