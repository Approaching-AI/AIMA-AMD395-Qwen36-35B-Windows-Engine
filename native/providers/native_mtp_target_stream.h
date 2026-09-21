#pragma once
#include "../src/qrt.h"
#include <array>

// Only the exact private request may bypass native MTP dispatch. A callback
// that enters another decode cannot inherit the target-only route.
class Qwen36NativeTargetOnly final {
public:
    explicit Qwen36NativeTargetOnly(const qrt_qwen36_whole_provider_decode_request_v1_t* request)
        : prior_(active_) { active_=request; }
    ~Qwen36NativeTargetOnly(){active_=prior_;}
    Qwen36NativeTargetOnly(const Qwen36NativeTargetOnly&)=delete;
    Qwen36NativeTargetOnly& operator=(const Qwen36NativeTargetOnly&)=delete;
    static bool matches(const qrt_qwen36_whole_provider_decode_request_v1_t* request){
        return request && request==active_;
    }
private:
    inline static thread_local const qrt_qwen36_whole_provider_decode_request_v1_t* active_=nullptr;
    const qrt_qwen36_whole_provider_decode_request_v1_t* prior_;
};

// Forward the ordinary target's real callbacks while the outer native-MTP
// shadow remains live. Record producer clocks for return-value validation;
// caller clocks measure arrival in the enclosing decode request.
class Qwen36NativeTargetStream final {
public:
    using Clock=uint64_t(*)();
    Qwen36NativeTargetStream(const qrt_qwen36_whole_provider_decode_request_v1_t& request,
        qrt_qwen36_whole_provider_decode_result_v1_t& output,uint32_t base,uint64_t started,
        uint64_t& previous,Clock clock)
        : request_(request),output_(output),base_(base),started_(started),previous_(previous),clock_(clock) {}
    bool bind(qrt_qwen36_whole_provider_decode_request_v1_t* inner){
        if(bound_ || !inner || !clock_ || !base_ || base_>=request_.output_token_capacity ||
            request_.output_token_capacity>QRT_QWEN36_WHOLE_PROVIDER_DECODE_MAX_OUTPUT_TOKENS ||
            inner->output_token_capacity!=request_.output_token_capacity-base_+1u ||
            inner->expected_session_generation!=request_.expected_session_generation ||
            inner->initial_output_token_id!=output_.output_tokens[base_-1u] ||
            inner->emit_callback || inner->emit_user_data)return reject("invalid retired target span");
        inner->emit_callback=forward;inner->emit_user_data=this;bound_=true;return true;
    }
    bool complete(const qrt_qwen36_whole_provider_decode_result_v1_t& result){
        const uint32_t count=request_.output_token_capacity-base_;
        if(!bound_ || failure_ || emitted_!=count || !result.completed || result.status!=QRT_STATUS_OK ||
            result.struct_size!=sizeof(result) || result.abi_version!=QRT_QWEN36_WHOLE_PROVIDER_DECODE_ABI_VERSION ||
            result.batch_size!=1u || result.prefill_token_count!=1u || result.decode_token_count!=count ||
            result.output_token_capacity!=count+1u || result.output_token_count!=count+1u ||
            result.timing_count!=count+1u || result.session_generation!=request_.expected_session_generation ||
            result.output_tokens[0]!=output_.output_tokens[base_-1u] ||
            result.token_end_elapsed_ns[0] || result.token_step_elapsed_ns[0] ||
            result.tpot_elapsed_ns!=producer_end_ || result.wall_clock_ns<producer_end_)
            return reject("retired target returned an incomplete span");
        for(uint32_t i=1u;i<=count;++i)
            if(result.output_tokens[i]!=output_.output_tokens[base_+i-1u] ||
                result.token_end_elapsed_ns[i]!=ends_[i] || result.token_step_elapsed_ns[i]!=steps_[i])
                return reject("retired target return differs from its live stream");
        return true;
    }
    const char* failure()const{return failure_?failure_:"retired target stream is incomplete";}
private:
    static int QRT_CDECL forward(void* context,uint64_t generation,uint32_t index,uint32_t token,
        uint64_t step,uint64_t end)noexcept{
        return context?static_cast<Qwen36NativeTargetStream*>(context)->emit(generation,index,token,step,end):0;
    }
    int emit(uint64_t generation,uint32_t index,uint32_t token,uint64_t step,uint64_t end)noexcept{
        if(!bound_ || failure_ || in_callback_ || generation!=request_.expected_session_generation ||
            index!=emitted_+1u || index>=request_.output_token_capacity-base_+1u ||
            token>=QRT_QWEN36_VOCAB_SIZE || !step || end<=producer_end_ || end-producer_end_!=step)
            return reject("retired target callback has an invalid identity, index or clock");
        const uint64_t now=clock_();
        if(now<started_ || now-started_<=previous_)return reject("retired target arrival clock is invalid");
        const uint32_t at=base_+index-1u;const uint64_t elapsed=now-started_,duration=elapsed-previous_;
        output_.output_tokens[at]=token;output_.token_step_elapsed_ns[at]=duration;
        output_.token_end_elapsed_ns[at]=elapsed;previous_=elapsed;
        steps_[index]=step;ends_[index]=end;producer_end_=end;++emitted_;in_callback_=true;
        int accepted=1;
        try{if(request_.emit_callback)accepted=request_.emit_callback(request_.emit_user_data,generation,at,token,duration,elapsed);}
        catch(...){in_callback_=false;return reject("retired target callback threw");}
        in_callback_=false;
        if(!accepted || failure_)return reject("retired target callback cancelled");
        return 1;
    }
    bool reject(const char* reason)noexcept{if(!failure_)failure_=reason;return false;}
    const qrt_qwen36_whole_provider_decode_request_v1_t& request_;
    qrt_qwen36_whole_provider_decode_result_v1_t& output_;
    uint32_t base_,emitted_=0u;
    uint64_t started_,&previous_,producer_end_=0u;
    Clock clock_;
    std::array<uint64_t,QRT_QWEN36_WHOLE_PROVIDER_DECODE_MAX_OUTPUT_TOKENS> ends_{},steps_{};
    const char* failure_=nullptr;
    bool bound_=false,in_callback_=false;
};
