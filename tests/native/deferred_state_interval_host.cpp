#include "../../native/providers/gdn/deferred_state_interval.h"
#include <array>
#include <cassert>
#include <cstdio>

namespace route=qrt_fla_deferred_state;
namespace bound=qrt_sm121_pv_bound;
using route::Interval;
struct History {
    std::array<Interval,17> states{};
    std::array<float,16> decays{},increments{};
    std::array<bool,16> flags{};
    Interval state(unsigned step)const{return states.at(step);}
    bool exact(unsigned step)const{return flags.at(step);}
    float increment(unsigned step)const{assert(flags.at(step));return increments.at(step);}
    float decay(unsigned step)const{return decays.at(step);}
    void cache(unsigned step,float value){assert(!flags.at(step));flags.at(step)=true;increments.at(step)=value;}
    void replace(unsigned step,Interval value){states.at(step)=value;}
};
struct Replay {
    const std::array<float,16>& source;
    std::array<unsigned,16> calls{};
    float operator()(unsigned step){assert(calls.at(step)==0u);++calls[step];return source.at(step);}
};
uint32_t random_word(uint32_t& x){x^=x<<13u;x^=x>>17u;return x^=x<<5u;}
float sample(uint32_t& seed){
    const auto r=random_word(seed);return bound::value((r&0x807fffffu)|((116u+(r>>24u)%16u)<<23u));
}
int main(){
    uint32_t seed=0x3958123u;
    uint64_t boundaries=0u,checkpoint_replays=0u,final_replays=0u,no_replay=0u,full=0u,depth=0u;
    constexpr unsigned cases=65536u;
    const float coefficients[]={0.0f,0x1p-20f,0.01f,0.1f,0.5f,0.98f,1.0f,1.00390625f};
    for(unsigned test=0u;test<cases;++test){
        History history;std::array<float,16> increments{},original{};Replay replay{increments};
        const unsigned steps=1u+test%16u;
        float exact=test%13u?sample(seed):bound::value(test&1u?0x80000000u:0u);
        history.states[0]={exact,exact};
        for(unsigned step=0u;step<steps;++step){
            history.decays[step]=coefficients[(test/16u+step/3u)%8u];
            increments[step]=test%17u?sample(seed):bound::value((step&1u)?0x80000000u:0u);
            exact=std::fma(exact,history.decays[step],increments[step]);original[step]=exact;
            const float error=bound::absolute(increments[step])*0x1p-18f+0x1p-120f;
            const float center=increments[step]+((random_word(seed)&1u)?error*0.25f:-error*0.25f);
            Interval enclosure{bound::next(center-error,false),bound::next(center+error,true)};
            assert(increments[step]>=enclosure.lower && increments[step]<=enclosure.upper);
            if(test%19u==0u && step%3u==0u)enclosure=route::interval::invalid();
            history.states[step+1u]=route::advance(history.states[step],history.decays[step],enclosure);
            if(route::interval::valid(history.states[step+1u])){
                assert(exact>=history.states[step+1u].lower && exact<=history.states[step+1u].upper);
            }
            const auto result=route::resolve(history,replay,step+1u,route::Boundary::bf16_checkpoint);
            assert(result.ready && bound::bf16(result.state.lower)==bound::bf16(exact));
            assert(result.state.lower<=exact && result.state.upper>=exact);
            checkpoint_replays+=result.original_dots;no_replay+=result.depth==0u;full+=result.complete_replay;
            depth+=result.depth;++boundaries;
        }
        const auto final=route::resolve(history,replay,steps,route::Boundary::fp32_state);
        assert(final.ready && bound::bits(final.state.lower)==bound::bits(original[steps-1u]));
        assert(bound::bits(final.state.upper)==bound::bits(original[steps-1u]));
        final_replays+=final.original_dots;full+=final.complete_replay;depth+=final.depth;++boundaries;
        unsigned calls=0u;for(unsigned step=0u;step<steps;++step){assert(replay.calls[step]<=1u);calls+=replay.calls[step];}
        assert(calls<=steps);
    }
    // Invalid/nonmonotone inputs decline interval propagation, and full
    // original replay retains exact exceptional endpoints without claiming a
    // finite certificate. An invalid seed cannot masquerade as an exact one.
    const float inf=bound::value(0x7f800000u);
    assert(!route::interval::valid(route::advance({1,2},-1,{0,1})));
    assert(!route::interval::valid(route::advance({1,2},inf,{0,1})));
    for(float increment:{inf,-inf,bound::value(0x7fc00000u)}){
        History h;std::array<float,16> source{};source[0]=increment;Replay replay{source};
        h.states[0]={1,1};h.states[1]=route::interval::invalid();h.decays[0]=0.5f;
        const auto r=route::resolve(h,replay,1u,route::Boundary::fp32_state);
        assert(r.ready && r.complete_replay && r.original_dots==1u);
        assert(bound::bits(r.state.lower)==bound::bits(std::fma(1.0f,0.5f,increment)));
    }
    History invalid;std::array<float,16> source{};Replay replay{source};
    invalid.states[0]={-1,1};invalid.states[1]=route::interval::invalid();invalid.decays[0]=1;
    assert(!route::resolve(invalid,replay,1u,route::Boundary::fp32_state).ready);
    std::printf("{\"kind\":\"deferred_state_interval_host\",\"sequences\":%u,\"boundaries\":%llu,\"checkpoint_replayed_dots\":%llu,\"final_replayed_dots\":%llu,\"checkpoints_without_refinement\":%llu,\"full_tail_resolutions\":%llu,\"total_refinement_depth\":%llu,\"checkpoint_bf16_differences\":0,\"final_fp32_differences\":0,\"interval_escapes\":0,\"cached_dots_replayed_once\":true,\"invalid_ranges_and_exceptional_points_checked\":true,\"native_increment_bound_qualified\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
        cases,(unsigned long long)boundaries,(unsigned long long)checkpoint_replays,(unsigned long long)final_replays,
        (unsigned long long)no_replay,(unsigned long long)full,(unsigned long long)depth);
}
