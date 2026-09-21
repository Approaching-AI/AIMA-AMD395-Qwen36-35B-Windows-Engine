#pragma once
#include "../src/qrt.h"

// A synchronous bridge for one decode span inside a prefix transaction.
// The outer request retains its shadow until this bridge and the decoder
// return. No token is re-emitted after the completed span is copied out.
class Qwen36PrefixDecodeStream final {
public:
    Qwen36PrefixDecodeStream(const qrt_qwen36_whole_provider_prefix_request_v1_t& request,
        qrt_qwen36_resident_prefix_cache_result_v1_t& output,uint32_t base,uint32_t count,uint64_t prior_end)
        : request_(request),output_(output),base_(base),count_(count),prior_end_(prior_end) {}
    bool bind(qrt_qwen36_whole_provider_decode_request_v1_t* decode) {
        if(bound_ || failure_ || !decode || !base_ || !count_ || count_>=QRT_QWEN36_WHOLE_PROVIDER_DECODE_MAX_OUTPUT_TOKENS ||
            request_.output_token_capacity>QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS ||
            base_>=request_.output_token_capacity || count_>request_.output_token_capacity-base_ ||
            !request_.expected_session_generation || decode->output_token_capacity!=count_+1u ||
            decode->expected_session_generation!=request_.expected_session_generation ||
            decode->emit_callback || decode->emit_user_data)
            return reject("prefix decode stream has an invalid span or target generation");
        if(request_.emit_callback){decode->emit_callback=forward;decode->emit_user_data=this;}
        bound_=true;return true;
    }
    bool complete(const qrt_qwen36_whole_provider_decode_result_v1_t& result) {
        if(!bound_ || failure_)return false;
        if(!request_.emit_callback)return true;
        if(emitted_!=count_ || result.output_token_count!=count_+1u || result.timing_count!=count_+1u)
            return reject("decode provider did not deliver the complete live prefix stream");
        for(uint32_t i=1u;i<=count_;++i){
            const uint32_t index=base_+i-1u;
            if(result.token_end_elapsed_ns[i]>UINT64_MAX-prior_end_ ||
                output_.output_tokens[index]!=result.output_tokens[i] ||
                output_.output_token_step_elapsed_ns[index]!=result.token_step_elapsed_ns[i] ||
                output_.output_token_end_elapsed_ns[index]!=prior_end_+result.token_end_elapsed_ns[i])
                return reject("returned decode differs from its emitted prefix stream");
        }
        return true;
    }
    const char* failure()const{return failure_?failure_:"prefix decode stream is incomplete";}
private:
    static int QRT_CDECL forward(void* context,uint64_t generation,uint32_t index,uint32_t token,
        uint64_t step,uint64_t end) noexcept {
        if(!context)return 0;
        return static_cast<Qwen36PrefixDecodeStream*>(context)->emit(generation,index,token,step,end);
    }
    int emit(uint64_t generation,uint32_t index,uint32_t token,uint64_t step,uint64_t end) noexcept {
        if(!bound_ || failure_ || in_callback_)return reject("prefix decode stream received a late or reentrant callback");
        if(generation!=request_.expected_session_generation || index!=emitted_+1u || index>count_ ||
            token>=QRT_QWEN36_VOCAB_SIZE || !step || end<step || end<=last_end_ ||
            end>UINT64_MAX-prior_end_)
            return reject("prefix decode callback has an invalid identity, index or clock");
        const uint32_t output_index=base_+index-1u;
        output_.output_tokens[output_index]=token;
        output_.output_token_step_elapsed_ns[output_index]=step;
        output_.output_token_end_elapsed_ns[output_index]=prior_end_+end;
        ++emitted_;last_end_=end;in_callback_=true;
        int accepted=0;
        try {
            accepted=request_.emit_callback(request_.emit_user_data,generation,output_index,token,step,prior_end_+end);
        }catch(...){in_callback_=false;return reject("prefix decode callback threw an exception");}
        in_callback_=false;
        if(!accepted || failure_)return reject("prefix decode callback cancelled the live stream");
        return 1;
    }
    bool reject(const char* reason)noexcept{if(!failure_)failure_=reason;return false;}
    const qrt_qwen36_whole_provider_prefix_request_v1_t& request_;
    qrt_qwen36_resident_prefix_cache_result_v1_t& output_;
    uint32_t base_,count_,emitted_=0u;
    uint64_t prior_end_,last_end_=0u;
    const char* failure_=nullptr;
    bool bound_=false,in_callback_=false;
};
