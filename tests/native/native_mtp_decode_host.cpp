#include "native/src/qrt.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "native/providers/mtp_decode_rows.h"
using hipError_t=int;
constexpr int hipSuccess=0;
struct Step {int status=0;const char* stage="complete";bool completion_unknown=false;};
static std::string fault_stage;
static bool fault_unknown=false,throw_prepare=false;
static unsigned fault_batch=1,batch_index=0,pins=0,targets=0,rollbacks=0,commits=0,aborts=0,callbacks=0,cancel_at=0;
static unsigned target_committed=0,drafter_committed=0;
static uint64_t clock_tick=10;
static bool fails(const char* stage){return fault_stage==stage&&(batch_index?batch_index:1u)==fault_batch;}
static Step step(const char* stage){return fails(stage)?Step{1,stage,fault_unknown}:Step{};}
static uint64_t qrt_now_ns(){clock_tick+=10;return clock_tick;}
static uint64_t qrt_elapsed_ns(uint64_t begin,uint64_t end){assert(end>=begin);return end-begin;}
static uint64_t qrt_fnv1a64_bytes(const void* data,size_t bytes){
    uint64_t value=14695981039346656037ull;const auto* p=static_cast<const uint8_t*>(data);
    for(size_t i=0;i<bytes;++i){value^=p[i];value*=1099511628211ull;}return value;
}
namespace qrt_sm121_q1 {
static float widen(uint16_t value){uint32_t bits=uint32_t(value)<<16u;float result;std::memcpy(&result,&bits,4u);return result;}
}
namespace qrt_sm121_mtp {
struct RequestCheckpoint {bool valid=false;std::vector<uint32_t> inputs;uint32_t current=0;};
struct TargetFrontier {
    const void* owner=nullptr;uint64_t generation=0,model_epoch=0;
    const uint32_t* processed_inputs=nullptr;size_t processed_count=0;uint32_t current_token=0;
};
struct TargetBatch {uint64_t first_position=0;unsigned rows=0;std::array<uint32_t,2> inputs{};};
struct AcceptedTarget {unsigned rows=0;std::array<uint32_t,2> outputs{};bool drafter_retired=false;};
}
struct Session {
    const qrt_engine_t* owner_engine=reinterpret_cast<const qrt_engine_t*>(uintptr_t(0x1234));
    uint64_t generation=7u;std::string model_dir="original-model";
    size_t prefix_tokens=7u;uint32_t current_token_id=144u;
    bool valid=true,route_active=true;
    std::vector<uint32_t> native_mtp_processed_inputs{1,2,3,4,5,6,7};
    qrt_sm121_mtp::RequestCheckpoint native_mtp_checkpoint{true,native_mtp_processed_inputs,current_token_id};
} g_qwen36_resident_session;
static std::atomic<bool> g_qwen36_resident_completion_unknown{false};
class ScopedQwen36ResidentSessionShadowTransaction {
    Session saved=g_qwen36_resident_session;bool active=false;
public:
    ScopedQwen36ResidentSessionShadowTransaction(uint64_t generation,uint64_t,std::string*,std::string*){
        assert(generation==saved.generation&&!pins&&!targets);active=!fails("shadow_begin");
    }
    bool ready()const{return active;}
    bool commit(std::string* stage,std::string* message){
        assert(active&&!pins&&!targets);
        if(fails("shadow_commit")){*stage="shadow_commit";*message="injected";return false;}
        active=false;++commits;return true;
    }
    bool rollback(const char*){
        assert(active&&!pins&&!targets);active=false;++rollbacks;g_qwen36_resident_session=saved;
        return !g_qwen36_resident_completion_unknown;
    }
    ~ScopedQwen36ResidentSessionShadowTransaction(){if(active)(void)rollback("destructor");}
};
namespace qrt_sm121_q2 {struct TargetResult;struct TargetSnapshot {uint64_t model_epoch=11u;};}
struct Owner {
    Owner(){++pins;}
    ~Owner(){--pins;}
    bool snapshot(qrt_sm121_q2::TargetSnapshot* result){*result={};return !fails("snapshot");}
    void quarantine(){g_qwen36_resident_completion_unknown=true;}
    bool commit_metadata(const qrt_sm121_q2::TargetResult&,unsigned,const std::array<uint32_t,2>&);
};
static std::shared_ptr<Owner> acquire_qwen36_target_cache_owner(
    const ScopedQwen36ResidentSessionShadowTransaction&,std::string*,std::string*){
    return fails("owner")?nullptr:std::make_shared<Owner>();
}
struct OriginalModel {};
static std::shared_ptr<OriginalModel> acquire_qwen36_target_model_weight_source(const std::string&,std::string*,std::string*){
    return fails("source")?nullptr:std::make_shared<OriginalModel>();
}
namespace qrt_sm121_q2 {
struct Binding {};
struct ModelWeights {
    Step prepare(std::shared_ptr<OriginalModel>,uint64_t epoch){assert(epoch==11u);return step("pack");}
    Binding binding(uint64_t epoch)const{assert(epoch==11u);return {};}
};
struct Host {std::array<uint32_t,2> tokens{};std::array<float,2> logits{};std::array<uint16_t,4096> normalized{};};
struct TargetResult {
    Host values;std::array<uint32_t,2> scheduled{};std::shared_ptr<Owner> owner;bool published=false;
    const Host* host()const{return &values;}
    const std::array<uint32_t,2>* inputs()const{return &scheduled;}
};
struct Target {
    Target(){++targets;}~Target(){--targets;}
    Step evaluate(Binding,std::shared_ptr<Owner> owner,const std::array<uint32_t,2>& inputs,
        TargetResult* result,unsigned blocks){
        assert(blocks==1024u&&pins==1u&&inputs[0]==g_qwen36_resident_session.current_token_id);
        result->owner=owner;result->scheduled=inputs;
        if(fails("result"))result->scheduled[1]^=1u;
        result->values.tokens={100u+batch_index*2u,101u+batch_index*2u};
        result->values.logits={2.0f,3.0f};result->values.normalized.fill(0x3f80u);
        return step("target");
    }
};
struct CachePublisher {
    static Step publish(TargetResult& result,const Owner& owner,unsigned rows){
        assert(result.owner.get()==&owner&&rows>=1u&&rows<=2u&&drafter_committed==target_committed);
        result.published=true;return step("publish");
    }
};
}
bool Owner::commit_metadata(const qrt_sm121_q2::TargetResult& result,unsigned rows,const std::array<uint32_t,2>& outputs){
    assert(result.published&&result.owner.get()==this);
    if(fails("metadata"))return false;
    auto& s=g_qwen36_resident_session;
    s.native_mtp_processed_inputs.insert(s.native_mtp_processed_inputs.end(),result.scheduled.begin(),result.scheduled.begin()+rows);
    s.current_token_id=outputs[rows-1u];s.native_mtp_checkpoint={};++target_committed;
    return true;
}
namespace qrt_sm121_mtp {
class Request {
    std::vector<uint32_t> inputs;
    uint32_t current=0;TargetBatch batch;AcceptedTarget accepted;size_t remaining=0;
    bool pending=false,prepared=false,unknown=false;
public:
    Step restore(const RequestCheckpoint& saved,const TargetFrontier& actual,unsigned capacity){
        assert(saved.valid&&saved.inputs.size()==actual.processed_count&&capacity>=actual.processed_count);
        assert(actual.owner==g_qwen36_resident_session.owner_engine&&actual.generation==7u&&actual.model_epoch==11u);
        assert(std::equal(saved.inputs.begin(),saved.inputs.end(),actual.processed_inputs)&&saved.current==actual.current_token);
        inputs=saved.inputs;current=actual.current_token;return step("restore");
    }
    bool begin(const TargetFrontier& actual,size_t left,TargetBatch* output){
        ++batch_index;assert(!pending&&left&&actual.processed_count==inputs.size()&&actual.current_token==current);
        assert(std::equal(inputs.begin(),inputs.end(),actual.processed_inputs));
        if(fails("begin"))return false;
        batch={inputs.size(),2u,{current,batch_index%2u?100u+batch_index*2u:999u}};
        remaining=left;pending=true;*output=batch;return true;
    }
    Step prepare(const uint32_t* supplied,const uint32_t* samples,size_t rows,
        const std::vector<unsigned>& positions,const std::vector<float>& normalized,uint64_t epoch,
        AcceptedTarget* output,void* stream,unsigned blocks){
        assert(pending&&!prepared&&rows==2u&&epoch==11u&&!stream&&blocks==1024u);
        assert(supplied[0]==batch.inputs[0]&&supplied[1]==batch.inputs[1]);
        assert(normalized.size()==4096u&&normalized.front()==1.0f&&normalized.back()==1.0f);
        if(throw_prepare)throw std::bad_alloc();
        const auto status=step("prepare");unknown=status.completion_unknown;
        if(status.status)return status;
        qrt_mtp_target_rows::DecodeRows accepted_rows;
        assert(accepted_rows.capture(batch.first_position,supplied,samples,rows,remaining,positions,normalized));
        accepted.rows=static_cast<unsigned>(accepted_rows.rows());
        std::copy(accepted_rows.shifted_tokens().begin(),accepted_rows.shifted_tokens().end(),accepted.outputs.begin());
        accepted.drafter_retired=fails("acceptance");
        prepared=true;*output=accepted;return {};
    }
    bool commit(const TargetFrontier& actual){
        assert(pending&&prepared&&target_committed==drafter_committed+1u);
        if(fails("receipt"))return false;
        assert(actual.processed_count==inputs.size()+accepted.rows&&actual.current_token==accepted.outputs[accepted.rows-1u]);
        assert(std::equal(inputs.begin(),inputs.end(),actual.processed_inputs));
        assert(std::equal(batch.inputs.begin(),batch.inputs.begin()+accepted.rows,actual.processed_inputs+inputs.size()));
        inputs.insert(inputs.end(),batch.inputs.begin(),batch.inputs.begin()+accepted.rows);
        current=actual.current_token;pending=prepared=false;++drafter_committed;return true;
    }
    bool abort(uint64_t epoch){assert(epoch==11u);++aborts;pending=prepared=false;return true;}
    bool quarantined()const{return unknown;}
    Step save(RequestCheckpoint* checkpoint,const TargetFrontier& actual){
        assert(!pending&&inputs.size()==actual.processed_count&&actual.current_token==current);
        *checkpoint={true,inputs,current};return step("save");
    }
};
}
#include "native_mtp_decode_actual.h"

static std::vector<uint32_t> emitted;
static int emit(void*,uint64_t generation,uint32_t index,uint32_t token,uint64_t,uint64_t end){
    ++callbacks;assert(generation==7u&&index==callbacks&&end>0u&&target_committed==drafter_committed);
    emitted.push_back(token);return index!=cancel_at;
}
static void reset_case(){
    assert(!pins&&!targets);g_qwen36_resident_session={};g_qwen36_resident_completion_unknown=false;
    batch_index=rollbacks=commits=aborts=callbacks=cancel_at=target_committed=drafter_committed=0u;
    fault_stage.clear();fault_unknown=throw_prepare=false;fault_batch=1u;clock_tick=10;emitted.clear();
}
static qrt_qwen36_whole_provider_decode_request_v1_t request(unsigned capacity){
    qrt_qwen36_whole_provider_decode_request_v1_t r{};r.expected_session_generation=7u;
    r.expected_prefix_token_count=7u;r.expected_prompt_token_ids_fnv1a64=9u;r.initial_output_token_id=144u;
    r.output_token_capacity=capacity;r.emit_callback=emit;return r;
}
int main(){
    for(unsigned capacity:{2u,3u,4u,5u,8u,64u}) {
        reset_case();const auto r=request(capacity);qrt_qwen36_whole_provider_decode_result_v1_t output{};
        output.output_tokens[0]=144u;std::string stage,error;
        assert(run_qwen36_native_mtp_decode(r,&output,10u,&stage,&error));
        assert(output.completed&&output.status==QRT_STATUS_OK&&output.output_token_count==capacity&&
            output.decode_token_count==capacity-1u&&output.timing_count==capacity);
        assert(callbacks==capacity-1u&&commits==1u&&!rollbacks&&!pins&&!targets&&!aborts);
        assert(std::equal(emitted.begin(),emitted.end(),output.output_tokens+1u));
        const auto& s=g_qwen36_resident_session;
        assert(s.native_mtp_processed_inputs.size()==7u+capacity-1u&&s.native_mtp_checkpoint.valid);
        assert(s.native_mtp_checkpoint.inputs==s.native_mtp_processed_inputs&&s.native_mtp_checkpoint.current==s.current_token_id);
        assert(output.token_end_elapsed_ns[0]==0u&&output.token_step_elapsed_ns[0]==0u);
        uint64_t elapsed=0;for(unsigned i=1;i<capacity;++i){elapsed+=output.token_step_elapsed_ns[i];assert(elapsed==output.token_end_elapsed_ns[i]);}
        assert(elapsed==output.tpot_elapsed_ns&&elapsed<=output.wall_clock_ns);
    }
    for(const auto* stage:{"shadow_begin","owner","snapshot","source","pack","restore","begin","target","result",
        "prepare","acceptance","publish","metadata","receipt","save","shadow_commit"}) {
        reset_case();fault_stage=stage;if(fault_stage=="save"||fault_stage=="shadow_commit")fault_batch=2u;
        auto r=request(4u);qrt_qwen36_whole_provider_decode_result_v1_t output{};std::string failure,error;
        assert(!run_qwen36_native_mtp_decode(r,&output,10u,&failure,&error)&&!output.completed&&!commits&&!pins&&!targets);
        assert(g_qwen36_resident_session.native_mtp_processed_inputs.size()==7u&&g_qwen36_resident_session.current_token_id==144u);
        const bool mutated=fault_stage=="publish"||fault_stage=="metadata"||fault_stage=="receipt"||fault_stage=="save"||fault_stage=="shadow_commit";
        assert(g_qwen36_resident_session.valid!=mutated);
    }
    for(const auto* stage:{"pack","restore","target","prepare","publish","save"}) {
        reset_case();fault_stage=stage;fault_unknown=true;if(fault_stage=="save")fault_batch=2u;
        auto r=request(4u);qrt_qwen36_whole_provider_decode_result_v1_t output{};std::string failure,error;
        assert(!run_qwen36_native_mtp_decode(r,&output,10u,&failure,&error));
        assert(g_qwen36_resident_completion_unknown&&!g_qwen36_resident_session.valid&&!commits&&rollbacks==1u);
    }
    for(unsigned cancel:{1u,2u,3u}) {
        reset_case();cancel_at=cancel;auto r=request(4u);qrt_qwen36_whole_provider_decode_result_v1_t output{};std::string stage,error;
        assert(!run_qwen36_native_mtp_decode(r,&output,10u,&stage,&error));
        assert(callbacks==cancel&&!g_qwen36_resident_session.valid&&rollbacks==1u&&stage=="mtp_native_decode_callback");
    }
    reset_case();throw_prepare=true;
    {auto r=request(4u);qrt_qwen36_whole_provider_decode_result_v1_t output{};std::string stage,error;
     assert(!run_qwen36_native_mtp_decode(r,&output,10u,&stage,&error)&&stage=="mtp_native_decode_host_allocation");
     assert(rollbacks==1u&&!callbacks&&g_qwen36_resident_session.valid);}
    reset_case();g_qwen36_resident_session.native_mtp_processed_inputs.resize(262143u);
    {auto r=request(4u);qrt_qwen36_whole_provider_decode_result_v1_t output{};std::string stage,error;
     assert(!run_qwen36_native_mtp_decode(r,&output,10u,&stage,&error)&&stage=="mtp_native_decode_frontier");
     assert(rollbacks==1u&&!pins&&!targets&&!callbacks);}
    std::cout<<"native MTP decode request ordering and failure recovery pass\n";
}
