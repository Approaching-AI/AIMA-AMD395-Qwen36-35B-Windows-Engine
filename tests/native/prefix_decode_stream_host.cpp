#include "native/src/qrt.h"
#include "native/providers/prefix_decode_stream.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct Session {
    bool valid=true,current_token_valid=true;
    uint64_t generation=7u,prompt_token_ids_fnv1a64=19u;
    size_t committed_decode_token_count=17u;
    uint32_t current_token_id=42u;
    std::array<uint64_t,40u> cache{};
    bool operator==(const Session& other)const{
        return valid==other.valid&&current_token_valid==other.current_token_valid&&generation==other.generation&&
            prompt_token_ids_fnv1a64==other.prompt_token_ids_fnv1a64&&committed_decode_token_count==other.committed_decode_token_count&&
            current_token_id==other.current_token_id&&cache==other.cache;
    }
} g_qwen36_resident_session;
static std::recursive_mutex g_qwen36_resident_session_mutex;
static unsigned produced=0u,callbacks=0u,late_callbacks=0u,rollbacks=0u,spans=0u,fault=0u;
static uint32_t cancel_at=UINT32_MAX;
static uint64_t ticks=0u;
static bool teacher_mode=false,stream_requested=true,inside_decode=false;
static const qrt_qwen36_whole_provider_decode_request_v1_t* active_decode=nullptr;
using Event=std::tuple<uint32_t,uint32_t,uint64_t,uint64_t>;
static std::vector<Event> events;
static uint64_t qrt_now_ns(){ticks+=10u;return ticks;}
static uint64_t qrt_elapsed_ns(uint64_t begin,uint64_t end){assert(end>=begin);return end-begin;}
static uint64_t qrt_fnv1a64_bytes(const void* pointer,size_t bytes){
    uint64_t hash=14695981039346656037ull;const auto* p=static_cast<const uint8_t*>(pointer);
    for(size_t i=0;i<bytes;++i){hash^=p[i];hash*=1099511628211ull;}return hash;
}
// The test decoder owns explicit mutable host cache values. The actual
// provider rollback lambda below must restore the complete source on failure.
struct Shadow {
    Session saved=g_qwen36_resident_session;
    bool active=true;
    bool rollback(const char*){assert(active);active=false;++rollbacks;g_qwen36_resident_session=saved;return true;}
    ~Shadow(){assert(!active);}
};
static int QRT_CDECL receive(void*,uint64_t generation,uint32_t index,uint32_t token,uint64_t step,uint64_t end){
    assert(generation==7u&&index==callbacks&&token==1000u+index&&step);
    assert(g_qwen36_resident_session.committed_decode_token_count==1024u+produced);
    for(unsigned layer=0;layer<40u;++layer)
        assert(g_qwen36_resident_session.cache[layer]==uint64_t(layer)+1024u+produced);
    if(index&&(!inside_decode||produced!=index))++late_callbacks;
    events.emplace_back(index,token,step,end);++callbacks;
    if(fault==13u&&index==2u)throw std::runtime_error("user callback");
    if(fault==14u&&index==2u){
        assert(active_decode&&active_decode->emit_callback);
        assert(!active_decode->emit_callback(active_decode->emit_user_data,generation,3u,1003u,10u,end+10u));
    }
    return index!=cancel_at;
}
static int qrt_qwen36_whole_provider_decode_v1(
    const qrt_qwen36_whole_provider_decode_request_v1_t* request,
    qrt_qwen36_whole_provider_decode_result_v1_t* output){
    ++spans;inside_decode=true;active_decode=request;
    assert(request->initial_output_token_id==1000u+produced&&request->output_token_capacity>=2u);
    output->output_tokens[0]=request->initial_output_token_id;
    uint64_t elapsed=0u;
    for(unsigned i=1u;i<request->output_token_capacity;++i){
        ++produced;++g_qwen36_resident_session.committed_decode_token_count;
        for(auto& value:g_qwen36_resident_session.cache)++value;
        const uint32_t token=1000u+produced;
        g_qwen36_resident_session.current_token_id=token;
        const uint64_t step=100u+produced;elapsed+=step;ticks+=step;
        output->output_tokens[i]=token;output->token_step_elapsed_ns[i]=step;output->token_end_elapsed_ns[i]=elapsed;
        uint64_t callback_generation=7u,callback_step=step,callback_end=elapsed;
        uint32_t callback_index=i,callback_token=token;
        if(produced==2u){
            switch(fault){case 1:callback_generation++;break;case 2:callback_index++;break;
                case 3:callback_token=QRT_QWEN36_VOCAB_SIZE;break;case 4:callback_step=0u;break;
                case 5:callback_end=100u;break;default:break;}
        }
        if(fault==6u&&produced==64u)callback_end=UINT64_MAX;
        int accepted=1;
        if(request->emit_callback&&!(fault==7u&&produced==2u)){
            accepted=request->emit_callback(request->emit_user_data,callback_generation,callback_index,
                callback_token,callback_step,callback_end);
            if(fault==8u&&produced==2u&&accepted)
                accepted=request->emit_callback(request->emit_user_data,7u,i,token,step,elapsed);
        }
        if((!accepted&&fault!=11u)||(fault==12u&&produced==2u)){
            output->status=QRT_STATUS_UNSUPPORTED;
            std::strcpy(output->failure_stage,"test_decode_cancelled");std::strcpy(output->failure,"producer stopped");
            inside_decode=false;active_decode=nullptr;return 0;
        }
    }
    output->completed=1u;output->status=QRT_STATUS_OK;
    output->output_token_count=output->timing_count=request->output_token_capacity;output->tpot_elapsed_ns=elapsed;
    if(fault==9u)output->output_tokens[1]++;
    if(fault==10u)output->token_step_elapsed_ns[1]++;
    inside_decode=false;active_decode=nullptr;return 1;
}
static int invoke(const qrt_qwen36_whole_provider_prefix_request_v1_t* request,
    qrt_qwen36_resident_prefix_cache_result_v1_t* out_result,bool legacy=false){
    Shadow transaction;
    g_qwen36_resident_session.committed_decode_token_count=1024u;
    g_qwen36_resident_session.current_token_id=1000u;
    for(unsigned layer=0u;layer<40u;++layer)g_qwen36_resident_session.cache[layer]=layer+1024u;
    const auto set_failure=[&](qrt_status_t status,const std::string& stage,const std::string& message){
        out_result->status=status;out_result->completed=0u;
        std::snprintf(out_result->failure_stage,sizeof(out_result->failure_stage),"%s",stage.c_str());
        std::snprintf(out_result->failure,sizeof(out_result->failure),"%s",message.c_str());return 0;
    };
    qrt_qwen36_whole_provider_decode_request_v1_t decode_request{};
    qrt_qwen36_whole_provider_decode_result_v1_t decode_result{};
    const bool gb10_continuation_teacher_forced=teacher_mode;
    std::vector<uint32_t> gb10_teacher_forced_continuation_tokens(request->output_token_capacity);
    for(unsigned i=0;i<request->output_token_capacity;++i)gb10_teacher_forced_continuation_tokens[i]=1000u+i;
    out_result->output_tokens[0]=1000u;out_result->output_token_step_elapsed_ns[0]=1000007u;
    out_result->output_token_end_elapsed_ns[0]=0u;
    uint32_t completed_output_tokens=1u;
    if(request->emit_callback)assert(request->emit_callback(request->emit_user_data,7u,0u,1000u,1000007u,0u));
    if(legacy){
#include "prefix_legacy_decode_span.inc"
    }else{
#include "prefix_live_decode_span.inc"
    }
    assert(transaction.rollback("complete"));out_result->state_restored=1u;
    out_result->completed=1u;out_result->status=QRT_STATUS_OK;return 1;
}
static void reset(){
    produced=callbacks=late_callbacks=rollbacks=spans=fault=0u;cancel_at=UINT32_MAX;ticks=0u;
    teacher_mode=false;stream_requested=true;inside_decode=false;active_decode=nullptr;events.clear();
    g_qwen36_resident_session={};for(unsigned i=0;i<40u;++i)g_qwen36_resident_session.cache[i]=i;
}
static qrt_qwen36_whole_provider_prefix_request_v1_t request(unsigned count){
    qrt_qwen36_whole_provider_prefix_request_v1_t r{};r.expected_session_generation=7u;
    r.expected_prefix_token_count=16384u;r.expected_prompt_token_ids_fnv1a64=19u;
    r.output_token_capacity=count;r.emit_callback=stream_requested?receive:nullptr;return r;
}
int main(){
    unsigned success=0u,cancellations=0u,invalid=0u;
    for(unsigned count:{1u,2u,32u,64u,65u,128u,512u})for(bool streamed:{false,true}){
        reset();stream_requested=streamed;const auto r=request(count);const auto before=g_qwen36_resident_session;
        auto result=std::make_unique<qrt_qwen36_resident_prefix_cache_result_v1_t>();
        assert(invoke(&r,result.get())&&result->output_token_count==count&&produced==count-1u);
        assert(callbacks==(streamed?count:0u)&&!late_callbacks&&rollbacks==1u&&g_qwen36_resident_session==before);
        for(unsigned i=0;i<count;++i){
            assert(result->output_tokens[i]==1000u+i);
            if(streamed)assert(events[i]==Event(i,result->output_tokens[i],result->output_token_step_elapsed_ns[i],result->output_token_end_elapsed_ns[i]));
        }
        ++success;
    }
    // The previous real span loop buffers 63 produced tokens before forwarding
    // token 1. It succeeds numerically but fails the observed live-delivery test.
    reset();auto r=request(65u);auto result=std::make_unique<qrt_qwen36_resident_prefix_cache_result_v1_t>();
    assert(invoke(&r,result.get(),true)&&late_callbacks==64u&&spans==2u);
    for(unsigned index:{1u,2u,62u,63u,64u,65u,127u,511u}){
        reset();cancel_at=index;r=request(512u);const auto before=g_qwen36_resident_session;*result={};
        assert(!invoke(&r,result.get())&&produced==index&&callbacks==index+1u&&!late_callbacks);
        assert(rollbacks==1u&&result->state_restored&&g_qwen36_resident_session==before);++cancellations;
    }
    for(unsigned mode=1u;mode<=14u;++mode){
        reset();fault=mode;if(mode==11u)cancel_at=2u;r=request(128u);*result={};
        const auto before=g_qwen36_resident_session;
        assert(!invoke(&r,result.get())&&rollbacks==1u&&result->state_restored&&g_qwen36_resident_session==before);
        assert(!late_callbacks);if(mode==11u)assert(callbacks==3u&&produced==63u);++invalid;
    }
    reset();teacher_mode=true;r=request(65u);*result={};
    assert(invoke(&r,result.get())&&spans==64u&&callbacks==65u&&!late_callbacks);++success;
    std::cout<<"live prefix decode spans, cancellation and original owner restoration pass cases="<<success
        <<" cancellations="<<cancellations<<" malformed_or_failed_producers="<<invalid<<" legacy_delayed_callbacks=64\n";
}
