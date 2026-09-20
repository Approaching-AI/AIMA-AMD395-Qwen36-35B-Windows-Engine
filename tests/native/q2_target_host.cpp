#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
using hipError_t = int;
using hipStream_t = void*;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipErrorOutOfMemory=2,hipMemcpyDeviceToDevice=3,hipMemcpyDeviceToHost=4,fault=99;
static hipError_t hipMalloc(void**,size_t);
static hipError_t hipFree(void*);
static hipError_t hipHostMalloc(void**,size_t);
static hipError_t hipHostFree(void*);
static hipError_t hipMemcpyAsync(void*,const void*,size_t,int,hipStream_t);
static hipError_t hipMemsetAsync(void*,int,size_t,hipStream_t);
static hipError_t hipStreamSynchronize(hipStream_t);
#include "sm121_q2_publication.h"
using namespace qrt_sm121_q2;
using qrt_sm121_mtp::ModelTensorView;
static hipStream_t wanted_stream=reinterpret_cast<void*>(0x1234u);
struct Allocation {size_t bytes;bool pinned;};
struct Event {std::vector<const void*> borrowed;std::function<void()> run;};
static std::map<void*,Allocation> allocations;
static std::vector<Event> pending;
static unsigned calls=0,fail_at=0,syncs=0,allocate_calls=0,fail_allocate=0,layer_calls=0;
static int invalid_layer=-1;
static unsigned bad_head=0;
static bool failed_fence=false;
static uint64_t* change_at_completion=nullptr;
template<class T>static T* fake(unsigned slot){return reinterpret_cast<T*>((uintptr_t(1)<<44u)+(uintptr_t(slot)<<32u));}
static bool inside(const void* pointer,const void* base,size_t bytes){
    const auto p=reinterpret_cast<uintptr_t>(pointer),b=reinterpret_cast<uintptr_t>(base);
    return p>=b&&p-b<bytes;
}
static hipError_t enqueue(Event event){pending.push_back(std::move(event));return ++calls==fail_at?fault:hipSuccess;}
static hipError_t allocate(void** pointer,size_t bytes,bool pinned){
    if(++allocate_calls==fail_allocate)return hipErrorOutOfMemory;
    assert(!posix_memalign(pointer,256u,(bytes+255u)&~size_t(255u)));
    allocations[*pointer]={bytes,pinned};return hipSuccess;
}
static hipError_t hipMalloc(void** p,size_t bytes){return allocate(p,bytes,false);}
static hipError_t hipHostMalloc(void** p,size_t bytes){return allocate(p,bytes,true);}
static hipError_t release(void* p,bool pinned){
    assert(allocations.count(p)&&allocations.at(p).pinned==pinned);
    for(const auto& e:pending)for(const auto* read:e.borrowed)assert(!inside(read,p,allocations.at(p).bytes));
    allocations.erase(p);std::free(p);return hipSuccess;
}
static hipError_t hipFree(void* p){return release(p,false);}
static hipError_t hipHostFree(void* p){return release(p,true);}
static hipError_t hipMemcpyAsync(void* out,const void* input,size_t bytes,int kind,hipStream_t stream){
    assert(stream==wanted_stream&&(kind==hipMemcpyDeviceToDevice||kind==hipMemcpyDeviceToHost));
    return enqueue({{out,input},[=]{
        // Model identities are virtual spans. Pack contents are immaterial to
        // orchestration; embedding row identity is propagated through all layers.
        if(inside(input,fake<uint16_t>(0),size_t(2048u)*248320u*2u)){
            assert(bytes==4096u);
            const size_t token=(reinterpret_cast<uintptr_t>(input)-reinterpret_cast<uintptr_t>(fake<uint16_t>(0)))/4096u;
            std::fill_n(static_cast<uint16_t*>(out),2048u,uint16_t(token));
        }else if(reinterpret_cast<uintptr_t>(input)>=(uintptr_t(1)<<44u))std::memset(out,0,bytes);
        else std::memcpy(out,input,bytes);
    }});
}
static hipError_t hipMemsetAsync(void* p,int value,size_t bytes,hipStream_t stream){
    assert(stream==wanted_stream);return enqueue({{p},[=]{std::memset(p,value,bytes);}});
}
static void drain(){for(const auto& e:pending)e.run();pending.clear();}
static hipError_t hipStreamSynchronize(hipStream_t stream){
    assert(stream==wanted_stream);++syncs;if(failed_fence)return fault;
    drain();if(change_at_completion)++*change_at_completion;return hipSuccess;
}
struct Weights final:qrt_sm121_mtp::ModelWeightSource{
    uint64_t generation=7;
    ~Weights(){assert(pending.empty());}
    uint64_t epoch()const noexcept override{return generation;}
    bool tensor(const char* name,ModelTensorView* output)const override{
        const auto& specs=target_weight_specs();
        auto it=std::find_if(specs.begin(),specs.end(),[&](const auto& s){return s.name==name;});
        assert(it!=specs.end());const auto& s=*it;
        *output={s.name.c_str(),fake<uint16_t>(unsigned(it-specs.begin())),s.rank,s.shape,s.bytes(),generation,true,true};return true;
    }
};
struct Source final:TargetCacheOwner{
    mutable TargetSnapshot state;
    mutable bool isolated=false;
    bool decline=false,throws=false;
    bool publication_allowed=true,publication_throws=false;
    mutable unsigned invalidations=0;
    std::vector<void*> owned_caches;
    struct Guard {unsigned char* base;size_t bytes;};
    std::vector<Guard> guards;
    std::function<void(TargetSnapshot&)> corrupt;
    explicit Source(unsigned first=8192u){
        state.owner=this;state.model_epoch=7;state.processed_tokens=first;state.current_token=144;
        unsigned slot=1000;
        auto& t=state.tables;
        t.beta=fake<float>(slot++);t.gated_silu=fake<float>(slot++);t.convolution_silu=fake<unsigned char>(slot++);
        t.attention={fake<unsigned char>(slot++),fake<unsigned char>(slot++),fake<unsigned char>(slot++),
            fake<uint16_t>(slot++),263680u,fake<uint16_t>(slot++)};
        t.moe={fake<uint16_t>(slot++),fake<uint16_t>(slot++),fake<uint32_t>(slot++)};
        for(unsigned layer=0;layer<40u;++layer){
            if(layer%4u==3u){
                const unsigned prefix=first/2u;
                state.attention[layer].prefix={fake<uint16_t>(slot++),fake<float>(slot++),prefix,prefix+1u,512u,2u};
                state.attention[layer].decoded={fake<float>(slot++),fake<float>(slot++),first-prefix,first-prefix+2u,512u,4u};
            }else{
                state.linear[layer]={fake<float>(slot++),fake<uint16_t>(slot++),layer%2u?4u:2u,bool(layer%2u)};
                t.g[layer]=fake<float>(slot++);
            }
        }
    }
    void writable_caches(unsigned element_bytes){
        const auto buffer=[&](size_t bytes){
            void* p=nullptr;assert(hipMalloc(&p,bytes+512u)==hipSuccess);
            std::memset(p,0x5a,bytes+512u);owned_caches.push_back(p);guards.push_back({static_cast<unsigned char*>(p),bytes});
            return static_cast<unsigned char*>(p)+256u;
        };
        for(unsigned layer=0;layer<40u;++layer){
            if(layer%4u==3u){
                auto& p=state.attention[layer].decoded;p.element_bytes=element_bytes;
                p.keys=buffer(size_t(p.capacity)*512u*element_bytes);
                p.values=buffer(size_t(p.capacity)*512u*element_bytes);
            }else{
                auto& p=state.linear[layer];p.state=reinterpret_cast<float*>(buffer(state_elements*4u));
                p.ring=buffer(ring_elements*p.ring_element_bytes);
            }
        }
    }
    void check_guards()const{
        for(const auto& g:guards)for(size_t i=0;i<256u;++i){assert(g.base[i]==0x5au);assert(g.base[256u+g.bytes+i]==0x5au);}
    }
    ~Source(){assert(pending.empty());check_guards();for(auto* p:owned_caches)(void)hipFree(p);}
    bool snapshot(TargetSnapshot* result)const override{
        if(throws)throw std::runtime_error("snapshot");if(decline)return false;
        *result=state;if(corrupt)corrupt(*result);return true;
    }
    bool matches(const TargetSnapshot& s)const noexcept override{
        return !isolated&&s.owner==state.owner&&s.generation==state.generation&&s.model_epoch==state.model_epoch&&
            s.processed_tokens==state.processed_tokens&&s.current_token==state.current_token;
    }
    void quarantine()const noexcept override{isolated=true;}
    bool prepare_publication(const TargetSnapshot& s,unsigned rows)const override{
        assert(rows==1u||rows==2u);if(publication_throws)throw std::bad_alloc();
        return publication_allowed&&matches(s);
    }
    void invalidate()const noexcept override{isolated=true;++invalidations;}
};
hipError_t publication_detail::copy_plane(const uint16_t* input,void* output,unsigned rows,unsigned stride,unsigned bytes,hipStream_t stream){
    assert(stream==wanted_stream&&stride==512u&&(bytes==2u||bytes==4u));
    return enqueue({{input,output},[=]{for(unsigned row=0;row<rows;++row)for(unsigned col=0;col<512u;++col){
        const uint16_t value=input[size_t(row)*1024u+col];
        if(bytes==2u)static_cast<uint16_t*>(output)[size_t(row)*stride+col]=value;
        else{uint32_t bits=uint32_t(value)<<16u;std::memcpy(static_cast<float*>(output)+size_t(row)*stride+col,&bits,4u);}
    }}});
}
static uint16_t marker(unsigned layer,unsigned row,bool residual){return uint16_t((residual?0x3e00u:0x3f00u)+layer*2u+row);}
template<class V>static void layer_result(const V& v,unsigned layer){
    qrt_sm121_mtp::MoeBuffers moe;
    assert(qrt_sm121_mtp::bind_moe_buffers(v.moe_workspace,v.moe_workspace_bytes,2u,&moe));
    for(unsigned row=0;row<2u;++row){
        const uint16_t expected=layer?marker(layer-1u,row,false):uint16_t(row?255u:144u);
        const uint16_t residual=layer?marker(layer-1u,row,true):0u;
        assert(v.hidden[row*2048u]==expected&&v.hidden[(row+1u)*2048u-1u]==expected);
        assert(v.residual[row*2048u]==residual&&v.residual[(row+1u)*2048u-1u]==residual);
        std::fill_n(moe.output+row*2048u,2048u,marker(layer,row,false));
        std::fill_n(v.output_residual+row*2048u,2048u,marker(layer,row,true));
    }
    *moe.invalid=invalid_layer==int(layer)?1u:0u;
}
namespace qrt_sm121_q2 {
template<class Element>hipError_t launch_linear_layer(const LinearLayerViews<Element>& v,const LinearLayerTables& t,
    unsigned blocks,hipStream_t stream){
    const unsigned layer=layer_calls++;assert(layer%4u!=3u&&stream==wanted_stream&&blocks&&valid_linear_layer(v,t));
    return enqueue({{v.hidden,v.residual,v.moe_workspace,v.output_residual,v.linear.recurrent.staged_states,v.linear.convolution.staged_rings},[=]{
        layer_result(v,layer);
        for(unsigned row=0;row<2u;++row){
            float* state=v.linear.recurrent.staged_states+size_t(row)*state_elements;
            std::fill_n(state,state_elements,float(layer*2u+row+1u));
            Element* ring=v.linear.convolution.staged_rings+size_t(row)*ring_elements;
            std::fill_n(ring,ring_elements,Element(layer*2u+row+1u));
        }
    }});
}
hipError_t launch_attention_layer(const AttentionLayerViews& v,const AttentionLayerTables& t,unsigned blocks,hipStream_t stream){
    const unsigned layer=layer_calls++;assert(layer%4u==3u&&stream==wanted_stream&&blocks&&valid_attention_layer(v,t));
    return enqueue({{v.hidden,v.residual,v.moe_workspace,v.output_residual,v.attention.staged_kv},[=]{
        layer_result(v,layer);
        for(unsigned row=0;row<2u;++row)std::fill_n(v.attention.staged_kv+row*1024u,1024u,uint16_t(layer*2u+row+1u));
    }});
}
hipError_t launch_target_head(const uint16_t* weights,const uint16_t* input,uint16_t* logits,uint32_t* tokens,
    float* values,uint32_t* invalid,unsigned blocks,hipStream_t stream){
    assert(layer_calls==40u&&blocks&&stream==wanted_stream&&weights);
    return enqueue({{input,logits,tokens,values,invalid},[=]{
        assert(input[0]==0x3f80u&&input[2048u]==0x4000u);
        tokens[0]=bad_head==2?248320u:255u;tokens[1]=82u;
        values[0]=bad_head==3?INFINITY:17.875f;values[1]=9.5625f;*invalid=bad_head==1?1u:0u;
        logits[0]=0x3f80u;logits[496639u]=0x4000u;
        if(bad_head==4)const_cast<uint16_t*>(input)[4095]=0x7fc0u;
    }});
}
}
namespace qrt_sm121_mtp {
hipError_t launch_residual_normalize(const uint16_t* input,const uint16_t* residual,const uint16_t* weights,
    const unsigned char* table,unsigned rows,uint16_t* output,uint16_t* next_residual,hipStream_t stream){
    assert(layer_calls==40u&&weights&&table&&rows==2u&&stream==wanted_stream);
    return enqueue({{input,residual,output,next_residual},[=]{
        for(unsigned row=0;row<2u;++row){
            assert(input[row*2048u]==marker(39,row,false)&&residual[row*2048u]==marker(39,row,true));
            std::fill_n(output+row*2048u,2048u,uint16_t(row?0x4000u:0x3f80u));
            std::fill_n(next_residual+row*2048u,2048u,marker(39,row,true));
        }
    }});
}
}
static void reset(){
    assert(pending.empty());calls=fail_at=syncs=allocate_calls=fail_allocate=layer_calls=bad_head=0;
    invalid_layer=-1;failed_fence=false;change_at_completion=nullptr;
}
static void verify(const TargetResult& result,unsigned first){
    assert(result.ready()&&result.host()&&result.frontier()->processed_tokens==first&&result.inputs()->at(1)==255u);
    const auto& h=*result.host();assert(h.tokens[0]==255u&&h.tokens[1]==82u&&h.logits[0]==17.875f&&h.logits[1]==9.5625f);
    for(const auto flag:h.invalid)assert(!flag);
    for(unsigned row=0;row<2u;++row)assert(h.normalized[row*2048u]==uint16_t(row?0x4000u:0x3f80u));
    assert(result.vocabulary_logits()[496639u]==0x4000u);
    for(unsigned layer=0;layer<40u;++layer)for(unsigned rows=0;rows<4u;++rows){
        auto linear=result.linear(layer,rows);auto attention=result.attention(layer,rows);
        if(rows<1u||rows>2u){assert(!linear.state&&!attention.key_values);continue;}
        if(layer%4u==3u){
            assert(!linear.state&&attention.key_values&&attention.rows==rows&&attention.first_position==first);
            for(unsigned row=0;row<rows;++row)assert(attention.key_values[row*1024u]==layer*2u+row+1u);
        }else{
            assert(!attention.key_values&&linear.state&&linear.first_position==first&&linear.rows==rows);
            assert(linear.state[0]==float(layer*2u+rows)&&linear.state[state_elements-1u]==float(layer*2u+rows));
            if(linear.ring_element_bytes==2u)assert(static_cast<const uint16_t*>(linear.ring)[ring_elements-1u]==layer*2u+rows);
            else assert(static_cast<const float*>(linear.ring)[ring_elements-1u]==float(layer*2u+rows));
            assert(linear.key_major==bool(layer%2u));
        }
    }
    assert(!result.linear(40,1).state&&!result.attention(40,1).key_values);
}
static void recover_test_quarantine(){
    // Only the test queue can prove completion here. Production intentionally
    // offers no release path for allocations with an unknown GPU fence.
    failed_fence=false;drain();
    auto* p=target_detail::quarantined_head.exchange(nullptr);
    while(p){auto* next=p->quarantine_next;auto held=std::move(p->quarantine_hold);held.reset();p=next;}
    auto* w=target_weight_detail::quarantined_head.exchange(nullptr);
    while(w){auto* next=w->quarantine_next;auto held=std::move(w->quarantine_hold);held.reset();w=next;}
}
int main(){
    {
        auto original=std::make_shared<Weights>();ModelWeights model;
        assert(model.prepare(original,7,wanted_stream).status==hipSuccess);auto binding=model.binding(7);
        {
            Target target;auto source=std::make_shared<Source>();TargetResult result;
            reset();assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipSuccess);
            assert(calls==89u&&syncs==1u&&layer_calls==40u);verify(result,8192u);
            const void* retained=result.linear(0,1).state;
            result={};reset();assert(target.evaluate(binding,source,{144,255},&result,257,wanted_stream).status==hipSuccess);
            assert(!allocate_calls&&result.linear(0,1).state==retained);verify(result,8192u);
            auto previous=result;result={};reset();
            assert(target.evaluate(binding,source,{144,255},&result,4096,wanted_stream).status==hipSuccess);
            assert(allocate_calls==2u&&result.linear(0,1).state!=retained);verify(previous,8192u);verify(result,8192u);
            previous={};
            // A late layer borrowing an earlier private result would be missed
            // by isolated layer validation. Check the entire private allocation.
            const auto* alias=result.linear(0,1).state;result={};
            source->corrupt=[&](auto& s){s.linear[38].state=alias;};reset();
            assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipErrorInvalidValue&&calls==0u);
            source->corrupt={};
            for(unsigned failure=1;failure<=89u;++failure){
                reset();fail_at=failure;const auto step=target.evaluate(binding,source,{144,255},&result,1024,wanted_stream);
                assert(step.status==fault&&!step.completion_unknown&&!result.ready()&&calls==failure&&syncs==1u&&pending.empty());
            }
            for(int layer=0;layer<40;++layer){
                reset();invalid_layer=layer;const auto step=target.evaluate(binding,source,{144,255},&result,1024,wanted_stream);
                assert(step.status==hipErrorInvalidValue&&std::string(step.stage)=="target_numerical_validity"&&!result.ready());
            }
            for(unsigned bad=1;bad<=4u;++bad){
                reset();bad_head=bad;assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipErrorInvalidValue&&!result.ready());
            }
            for(unsigned layer=0;layer<40u;++layer){
                source->corrupt=[=](auto& s){if(layer%4u==3u)++s.attention[layer].decoded.tokens;else s.tables.g[layer]=nullptr;};
                reset();assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipErrorInvalidValue&&calls==0u);
            }
            source->corrupt={};
            for(unsigned which=0;which<9u;++which){
                source->corrupt=[=](auto& s){switch(which){
                    case 0:s.owner=nullptr;break;case 1:s.tables.beta=nullptr;break;
                    case 2:s.tables.attention.rope_rows=8193u;break;case 3:s.linear[38].ring_element_bytes=3u;break;
                    case 4:s.attention[39].staged=fake<uint16_t>(1900u);break;
                    case 5:s.linear[38].ring=nullptr;break;case 6:s.tables.moe.sigmoid=nullptr;break;
                    case 7:s.linear[38].state=reinterpret_cast<float*>(UINTPTR_MAX-3u);break;
                    case 8:s.attention[39].prefix.capacity=263681u;break;
                }};
                reset();assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipErrorInvalidValue&&calls==0u);
            }
            source->corrupt={};
            for(unsigned blocks:{0u,4097u}){reset();assert(target.evaluate(binding,source,{144,255},&result,blocks,wanted_stream).status==hipErrorInvalidValue&&!calls);}
            for(auto inputs:{std::array<uint32_t,2>{0,255},std::array<uint32_t,2>{144,248320}}){
                reset();assert(target.evaluate(binding,source,inputs,&result,1024,wanted_stream).status==hipErrorInvalidValue&&!calls);}
            reset();source->decline=true;assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipErrorInvalidValue&&!calls);source->decline=false;
            reset();source->throws=true;assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipErrorInvalidValue&&!calls);source->throws=false;
            reset();change_at_completion=&source->state.generation;
            assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipErrorInvalidValue&&!result.ready());
            reset();change_at_completion=&original->generation;
            assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipErrorInvalidValue&&!result.ready());
            --original->generation;reset();
            assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipSuccess);
            ++source->state.processed_tokens;assert(!result.ready()&&!result.host()&&!result.linear(0,1).state);--source->state.processed_tokens;
            assert(result.ready());++original->generation;assert(!result.ready());--original->generation;
        }
        for(unsigned first:{0u,262143u,263678u}){
            reset();Target target;TargetResult result;auto source=std::make_shared<Source>(first);
            assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipSuccess);verify(result,first);
        }
        for(unsigned failure:{1u,2u}){
            reset();Target target;TargetResult result;auto source=std::make_shared<Source>();fail_allocate=failure;
            const size_t before=allocations.size();
            assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipErrorOutOfMemory&&!calls&&!result.ready());
            assert(allocations.size()==before);
        }
        // An immutable result owns its complete private storage and source even
        // after the producer goes away.
        reset();TargetResult saved;std::weak_ptr<const Source> weak;
        {Target target;auto source=std::make_shared<Source>();weak=source;
            assert(target.evaluate(binding,source,{144,255},&saved,1024,wanted_stream).status==hipSuccess);}
        assert(!weak.expired());verify(saved,8192u);saved={};assert(weak.expired());
    }
    assert(allocations.empty());
    for(unsigned bytes:{2u,4u})for(unsigned accepted:{1u,2u}){
        reset();auto original=std::make_shared<Weights>();ModelWeights model;
        assert(model.prepare(original,7,wanted_stream).status==hipSuccess);auto binding=model.binding(7);
        auto source=std::make_shared<Source>(7u);source->writable_caches(bytes);Target target;TargetResult result;
        reset();assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipSuccess);
        auto other=std::make_shared<Source>(7u);reset();
        assert(CachePublisher::publish(result,*other,accepted,wanted_stream).status==hipErrorInvalidValue&&!calls);
        for(unsigned rows:{0u,3u})assert(CachePublisher::publish(result,*source,rows,wanted_stream).status==hipErrorInvalidValue&&!calls);
        source->publication_allowed=false;
        assert(CachePublisher::publish(result,*source,accepted,wanted_stream).status==hipErrorInvalidValue&&!calls);
        source->publication_allowed=true;source->publication_throws=true;
        assert(CachePublisher::publish(result,*source,accepted,wanted_stream).status==hipErrorOutOfMemory&&!calls);
        source->publication_throws=false;
        assert(!result.quarantine_borrower(hipSuccess));
        assert(CachePublisher::publish(result,*source,accepted,wanted_stream).status==hipSuccess);
        assert(calls==80u&&syncs==1u&&pending.empty()&&result.cache_published()&&!source->invalidations);
        // Only the accepted state/ring and its K/V rows may be selected. The
        // rejected second candidate, committed history and unused tail survive.
        for(unsigned layer=0;layer<40u;++layer){
            if(layer%4u!=3u){
                const auto& p=source->state.linear[layer];
                for(size_t i=0;i<state_elements;++i)assert(p.state[i]==float(layer*2u+accepted));
                for(size_t i=0;i<ring_elements;++i){
                    if(p.ring_element_bytes==2u)assert(static_cast<const uint16_t*>(p.ring)[i]==layer*2u+accepted);
                    else assert(static_cast<const float*>(p.ring)[i]==float(layer*2u+accepted));
                }
            }else{
                const auto& p=source->state.attention[layer].decoded;
                for(const void* data:{p.keys,p.values})for(unsigned row=0;row<p.capacity;++row)for(unsigned col=0;col<512u;++col){
                    const bool written=row>=p.tokens&&row<p.tokens+accepted;
                    const uint16_t value=uint16_t(layer*2u+(row-p.tokens)+1u);
                    if(bytes==2u)assert(static_cast<const uint16_t*>(data)[size_t(row)*512u+col]==(written?value:0x5a5au));
                    else{uint32_t actual=0;std::memcpy(&actual,static_cast<const float*>(data)+size_t(row)*512u+col,4u);
                        assert(actual==(written?uint32_t(value)<<16u:0x5a5a5a5au));}
                }
            }
        }
        const auto before=calls;
        assert(CachePublisher::publish(result,*source,accepted,wanted_stream).status==hipErrorInvalidValue&&calls==before);
        source->check_guards();
        // Same producer storage may be reused only after the old result dies.
        result={};reset();
        assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipSuccess);
        assert(!result.cache_published());
        reset();assert(CachePublisher::publish(result,*source,accepted,wanted_stream).status==hipSuccess);
        if(bytes==2u&&accepted==1u)for(unsigned failure=1u;failure<=80u;++failure){
            result={};source->isolated=false;reset();
            assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipSuccess);
            reset();fail_at=failure;
            const auto step=CachePublisher::publish(result,*source,1u,wanted_stream);
            assert(step.status==fault&&!step.completion_unknown&&source->isolated&&!result.ready());
            assert(calls==failure&&syncs==1u&&pending.empty());source->check_guards();
        }
    }
    assert(allocations.empty());
    for(unsigned bad=0;bad<8u;++bad){
        reset();auto original=std::make_shared<Weights>();ModelWeights model;
        assert(model.prepare(original,7,wanted_stream).status==hipSuccess);auto binding=model.binding(7);
        auto source=std::make_shared<Source>(7u);source->writable_caches(2u);auto& s=source->state;
        switch(bad){
            case 0:s.attention[39].decoded.capacity=s.attention[39].decoded.tokens+1u;break;
            case 1:s.linear[38].state=s.linear[0].state;break;
            case 2:s.attention[3].prefix.keys=s.linear[0].state;break;
            case 3:s.tables.beta=s.linear[0].state;break;
            case 4:s.linear[0].state=reinterpret_cast<const float*>(fake<uint16_t>(0));break;
            case 5:s.attention[39].decoded.stride=1024u;break;
            case 6:s.attention[39].decoded.keys=s.attention[39].decoded.values;break;
            case 7:s.tables.g[38]=s.linear[0].state;break;
        }
        Target target;TargetResult result;reset();
        assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipSuccess);
        reset();assert(CachePublisher::publish(result,*source,2u,wanted_stream).status==hipErrorInvalidValue);
        assert(calls==0u&&syncs==0u&&!source->invalidations);source->check_guards();
    }
    assert(allocations.empty());
    for(unsigned failure:{0u,17u,80u}){
        reset();std::weak_ptr<Source> weak_source;std::weak_ptr<Weights> weak_weights;
        {
            auto original=std::make_shared<Weights>();weak_weights=original;ModelWeights model;
            assert(model.prepare(original,7,wanted_stream).status==hipSuccess);auto binding=model.binding(7);
            auto source=std::make_shared<Source>(7u);weak_source=source;source->writable_caches(4u);
            Target target;TargetResult result;
            reset();assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).status==hipSuccess);
            reset();failed_fence=true;fail_at=failure;
            const auto step=CachePublisher::publish(result,*source,2u,wanted_stream);
            assert(step.status==fault&&step.completion_unknown&&!result.ready()&&source->isolated&&!binding.valid(7)&&target.quarantined());
            const auto submitted=calls;
            assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).completion_unknown&&calls==submitted);
        }
        assert(!weak_source.expired()&&!weak_weights.expired()&&!pending.empty());
        recover_test_quarantine();assert(weak_source.expired()&&weak_weights.expired()&&allocations.empty());
    }
    // An independent observer can report an unresolved later borrow even if
    // the mutable frontier has changed since the original producer completed.
    {
        reset();std::weak_ptr<Source> weak_source;std::weak_ptr<Weights> weak_weights;
        {
            auto original=std::make_shared<Weights>();weak_weights=original;ModelWeights model;
            assert(model.prepare(original,7,wanted_stream).status==hipSuccess);auto binding=model.binding(7);
            auto source=std::make_shared<Source>();weak_source=source;TargetResult saved;
            {Target target;assert(target.evaluate(binding,source,{144,255},&saved,1024,wanted_stream).status==hipSuccess);}
            const auto* read=saved.linear(0,1).state;enqueue({{read},[read]{assert(read[0]==1.0f);}});
            ++source->state.current_token;assert(!saved.ready());
            assert(saved.quarantine_borrower(fault)&&source->isolated);
        }
        assert(!weak_source.expired()&&!weak_weights.expired()&&!pending.empty());
        recover_test_quarantine();assert(weak_source.expired()&&weak_weights.expired()&&allocations.empty());
    }
    for(unsigned submission_failure:{0u,44u}){
        reset();std::weak_ptr<Source> weak_source;std::weak_ptr<Weights> weak_weights;
        {
            auto original=std::make_shared<Weights>();weak_weights=original;ModelWeights model;
            assert(model.prepare(original,7,wanted_stream).status==hipSuccess);auto binding=model.binding(7);
            auto source=std::make_shared<Source>();weak_source=source;Target target;TargetResult result;
            reset();failed_fence=true;fail_at=submission_failure;
            const auto step=target.evaluate(binding,source,{144,255},&result,1024,wanted_stream);
            assert(step.status==fault&&step.completion_unknown&&target.quarantined()&&source->isolated&&!result.ready());
            const auto before=calls;assert(target.evaluate(binding,source,{144,255},&result,1024,wanted_stream).completion_unknown&&calls==before);
        }
        assert(!weak_source.expired()&&!weak_weights.expired()&&!pending.empty()&&!allocations.empty());
        recover_test_quarantine();assert(weak_source.expired()&&weak_weights.expired()&&allocations.empty());
    }
    std::cout<<"all 89 submission failures drained; 40 per-layer invalid flags retained; private owners and unknown completion checked\n";
    std::cout<<"all 80 publication failures drained; accepted cache rows and later borrowed lifetimes checked\n";
}
