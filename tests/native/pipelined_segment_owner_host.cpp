// Exercise the real host owner against a dependency-graph HIP model. This
// checks ordering/cleanup, not numerical GPU behavior or elapsed performance.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include "../../native/providers/gdn/pipelined_segment_policy.h"
#include "../../native/providers/gdn/fla_checkpoint.h"
using hipError_t=int;
constexpr int hipSuccess=0,hipStreamNonBlocking=1,hipEventDisableTiming=1;
struct Stream { unsigned last=0; };
struct Event { unsigned node=0; };
using hipStream_t=Stream*;
using hipEvent_t=Event*;
std::vector<std::set<unsigned>> ancestry(1);
Stream caller;
unsigned synchronizations=0,live_streams=0,live_events=0,allocations=0;
int event_failure=-1,phase_failure=-1;
unsigned enqueue(hipStream_t stream,unsigned dependency=0) {
    assert(stream);
    auto parents=ancestry[stream->last];parents.insert(stream->last);
    parents.insert(ancestry[dependency].begin(),ancestry[dependency].end());parents.insert(dependency);
    ancestry.push_back(parents);return stream->last=unsigned(ancestry.size()-1);
}
hipError_t hipStreamCreateWithFlags(hipStream_t* p,unsigned) { *p=new Stream;++live_streams;return 0; }
hipError_t hipStreamDestroy(hipStream_t p) { delete p;--live_streams;return 0; }
hipError_t hipStreamSynchronize(hipStream_t) { ++synchronizations;return 0; }
hipError_t hipEventCreate(hipEvent_t* p) {
    if(event_failure==0)return 7;
    if(event_failure>0)--event_failure;
    *p=new Event;++live_events;return 0;
}
hipError_t hipEventCreateWithFlags(hipEvent_t* p,unsigned) { return hipEventCreate(p); }
hipError_t hipEventDestroy(hipEvent_t p) { delete p;--live_events;return 0; }
hipError_t hipEventRecord(hipEvent_t event,hipStream_t stream) {event->node=enqueue(stream);return 0;}
hipError_t hipStreamWaitEvent(hipStream_t stream,hipEvent_t event,unsigned) {
    assert(event && event->node);enqueue(stream,event->node);return 0;
}
hipError_t hipEventSynchronize(hipEvent_t event) {assert(event && event->node);++synchronizations;return 0;}
hipError_t hipEventElapsedTime(float* ms,hipEvent_t begin,hipEvent_t end) {
    assert(ancestry[end->node].count(begin->node));*ms=1.0f;return 0;
}
constexpr unsigned kSegmentTokens=1024u,kQkvRows=8192u,kGateRows=64u,kValueFeatures=4096u;
struct SegmentStorage { unsigned* compact_qkv=nullptr; };
struct ProviderState:SegmentStorage {char error[768]{};} g_state;
void set_error(const char* stage,hipError_t) {std::snprintf(g_state.error,sizeof(g_state.error),"%s",stage);}
void set_error_text(const char* stage) {set_error(stage,0);}
bool ensure_scratch(unsigned) {
    if(!g_state.compact_qkv) {g_state.compact_qkv=new unsigned(++allocations);}
    return true;
}
bool ensure_blackwell_state_scratch() {return true;}
void release_scratch() {delete g_state.compact_qkv;g_state.compact_qkv=nullptr;}
bool blackwell_batched_enabled() {return true;}
bool blackwell_state_enabled() {return true;}
namespace qrt_fla_blackwell_cooperative {bool enabled() {return true;}}
struct BlackwellSegmentGuard {
    unsigned operations=0;
    explicit BlackwellSegmentGuard(hipStream_t) {}
};
struct History {unsigned prepared=0,state=0,output=0;};
std::map<unsigned,History> history;
unsigned prior_state=0,phase_count=0,first_input=0;
int launch_segment_async(const float*,const float*,float*,float*,void* opaque,int count,
                         bool reset,int,qrt_fla_checkpoint::Segment,unsigned phase) {
    assert(count>0 && count<=1024 && count%64==0);
    const unsigned node=enqueue(static_cast<Stream*>(opaque));
    assert(ancestry[node].count(first_input));
    auto& h=history[*g_state.compact_qkv];
    if(phase==1u) {
        if(h.prepared)assert(h.output && ancestry[node].count(h.output));
        h.prepared=node;h.state=h.output=0;
    } else if(phase==2u) {
        assert(h.prepared && ancestry[node].count(h.prepared));
        if(prior_state)assert(ancestry[node].count(prior_state));
        if(reset)assert(phase_count%24u==1u);
        h.state=prior_state=node;
    } else {
        assert(phase==3u && h.state && ancestry[node].count(h.state));h.output=node;
    }
    ++phase_count;
    if(phase_failure==0)return 0;
    if(phase_failure>0)--phase_failure;
    return 1;
}
#include "../../native/providers/gdn/pipelined_segment_owner.h"
int main() {
    namespace policy=qrt_fla_pipeline_policy;
    unsetenv("QRT_FLA_GDN_PIPELINED_SEGMENTS");assert(policy::mode()==0);
    for(const char* value:{"","0","1","2","3","01","-1","true"}) {
        setenv("QRT_FLA_GDN_PIPELINED_SEGMENTS",value,1);
        const int expected=!value[0] || !std::strcmp(value,"0") ? 0 :
            !std::strcmp(value,"1") ? 1 : !std::strcmp(value,"2") ? 2 : -1;
        assert(policy::mode()==expected);
    }
    unsetenv("QRT_FLA_GDN_PIPELINED_SEGMENTS");
    unsigned eligibility=0;
    for(int mode=-1;mode<=3;++mode)for(unsigned tokens:{0u,1u,1024u,1025u,7169u,8192u,65536u,65537u})
        for(unsigned flags=0;flags<16;++flags) {
            const bool actual=policy::selected(mode,flags&1u,tokens,flags&2u,flags&4u,flags&8u);
            const bool expected=(mode==1 || mode==2) && tokens>1024u && tokens<=65536u && flags==9u;
            assert(actual==expected);++eligibility;
        }
    // A pre-existing default owner must survive every slot binding and cleanup.
    ensure_scratch(64u);const auto original=g_state.compact_qkv;
    // Uninitialized host arrays provide valid pointer spans; no fake GPU
    // operation reads their data or claims numerical execution.
    std::unique_ptr<float[]> raw(new float[8192u*kQkvRows]);
    std::unique_ptr<float[]> gates(new float[8192u*kGateRows]);
    std::unique_ptr<float[]> output(new float[8192u*kValueFeatures]);
    std::unique_ptr<float[]> state(new float[524288u]);
    unsigned runs=0;
    for(int mode:{1,2})for(unsigned tokens:{64u,1024u,2048u,7168u,8192u}) {
        phase_count=0;first_input=enqueue(&caller);
        assert(launch_pipeline_window(raw.get(),gates.get(),output.get(),state.get(),&caller,tokens,true,mode));
        assert(phase_count==3u*((tokens+1023u)/1024u));
        assert(g_state.compact_qkv==original);
        for(const auto& item:history)if(item.second.output)
            assert(ancestry[caller.last].count(item.second.output));
        ++runs;
    }
    // Fail after work was enqueued on each possible phase. All streams must
    // drain before the owner's error return; the default owner stays intact.
    for(int failure=0;failure<6;++failure) {
        history.clear();prior_state=0;phase_count=0;first_input=enqueue(&caller);phase_failure=failure;
        const unsigned before=synchronizations;
        assert(!launch_pipeline_window(raw.get(),gates.get(),output.get(),state.get(),&caller,8192u,false,2));
        assert(synchronizations>=before+3u && g_state.compact_qkv==original);
        phase_failure=-1;release_pipeline();
        assert(!live_streams && !live_events);
    }
    for(int failure:{0,1,12,48}) {
        event_failure=failure;assert(!ensure_pipeline());
        assert(!live_streams && !live_events && g_state.compact_qkv==original);event_failure=-1;
    }
    release_pipeline();release_scratch();
    std::printf("{\"kind\":\"pipelined_segment_owner_host\",\"eligibility_cases\":%u,\"complete_dependency_graphs\":%u,\"partial_submission_failures\":6,\"event_allocation_failures\":4,\"ring_reuse_and_ordered_state\":true,\"caller_dependency_and_completion\":true,\"all_streams_drained_on_error\":true,\"original_storage_preserved\":true}\n",eligibility,runs);
}
