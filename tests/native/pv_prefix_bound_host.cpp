#include "../../native/providers/moe_accumulator/sm121_pv_prefix_bound.h"
#include <array>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace p = qrt_sm121_pv_prefix_bound;
namespace b = qrt_sm121_pv_bound;
namespace f = qrt_sm121_pv_final_bound;
int main() try {
    std::mt19937 random(0x61716f31u);
    uint64_t comparisons = 0, weight_checks = 0, interval_checks = 0;
    for (unsigned trial = 0; trial < 4096u; ++trial) {
        const unsigned groups = 2u + 2u * (random() % 256u);
        std::array<float,513> error{}, state{}, native{};
        std::array<float,256> alpha{};
        const int scale = int(random()%180u)-120;
        for (unsigned g = 0; g < groups; ++g) {
            float e=error[g], s=state[g], carry=native[g];
            if (!(g&1u)) {
                const float choices[]={0.0f,1.0f,0.5f,0.99609375f,0x1p-100f,0x1p-126f};
                const float a=(trial%3u)?choices[random()%6u]:float(random()&0xffffffu)*0x1p-24f;
                alpha[g/2u]=a;e=b::rescale(e,carry,a);s=f::rescale(s,carry,a);
                carry=f::multiply(carry,a);
            }
            const float dot=std::ldexp(float(1u+random()%65535u),scale-16);
            error[g+1u]=b::group(e,carry,dot);state[g+1u]=f::group(s,carry,dot);
            native[g+1u]=f::add(carry,(random()&1u)?dot:-dot);
            if (!(g&1u)) continue;
            const double low=p::prefix_lower(state[g+1u],g+1u);
            const float high=f::finalize(state[g+1u],g+1u);
            if (low<0 || low>double(error[g+1u]) || high<error[g+1u])
                throw std::runtime_error("prefix envelope sandwich failure");
            ++comparisons;
        }
        p::Weight weight{1,1};long double actual_weight=1;
        for (unsigned prefix=groups-2u; prefix; prefix-=2u) {
            weight=p::multiply(weight,alpha[prefix/2u]);actual_weight*=alpha[prefix/2u];
            if (!p::valid(weight) || (long double)weight.lower>actual_weight ||
                (long double)weight.upper<actual_weight)throw std::runtime_error("alpha product not enclosed");
            ++weight_checks;
            const double remaining=p::up(double(f::finalize(state[groups],groups))-
                p::down(weight.lower*p::prefix_lower(state[prefix],prefix)));
            const long double original_remaining=(long double)error[groups]-actual_weight*error[prefix];
            if ((long double)remaining<original_remaining)throw std::runtime_error("suffix old envelope lost");
            // Inject a known exact-prefix discrepancy. Test both extreme
            // suffix local errors of the independently unrolled old envelope.
            for (float sign : {-1.0f,1.0f}) {
                const float exact=f::add(native[prefix],sign*error[prefix]*0.5f);
                const auto interval=p::suffix({native[prefix],state[prefix]},
                    {native[groups],state[groups]},exact,weight,prefix,groups);
                const long double center=(long double)native[groups]+actual_weight*
                    ((long double)exact-native[prefix]);
                if ((long double)interval.lower>center-original_remaining ||
                    (long double)interval.upper<center+original_remaining)
                    throw std::runtime_error("signed suffix interval failure");
                ++interval_checks;
            }
        }
    }
    float output=123.0f;
    if (p::certified({-INFINITY,INFINITY},1,&output) ||
        p::certified({1,1},-1,&output) || p::valid(p::multiply({1,1},1.5f)) ||
        p::prefix_lower(INFINITY,2)!=-1 || p::prefix_lower(1,513)!=-1)
        throw std::runtime_error("invalid envelope accepted");
    // Check exact midpoint ties, signed zero and exponent boundaries without
    // using certificate internals as the expected BF16 answer.
    uint64_t midpoint_checks=0;
    for (uint32_t word=0x0080u;word<0x7f7fu;++word) for (unsigned sign=0;sign<2;++sign) {
        const float a=b::value((word|(sign?0x8000u:0u))<<16u);
        const float c=b::value(((word+1u)|(sign?0x8000u:0u))<<16u);
        const double midpoint=(double(a)+double(c))*0.5;
        const double radius=std::fabs(double(a)-double(c))*0.001;
        if (p::certified({midpoint-radius,midpoint+radius},1,&output))
            throw std::runtime_error("midpoint straddle accepted");
        const double narrow=std::fabs(double(a))*0x1p-18;
        if (!p::certified({double(a)-narrow,double(a)+narrow},1,&output) ||
            b::bf16(output)!=(word|(sign?0x8000u:0u)))
            throw std::runtime_error("interior interval rejected or changed");
        ++midpoint_checks;
    }
    std::printf("{\"kind\":\"pv_prefix_bound_host\",\"envelope_sandwiches\":%llu,\"alpha_products\":%llu,\"suffix_intervals\":%llu,\"midpoint_checks\":%llu,\"mismatches\":0}\n",
        (unsigned long long)comparisons,(unsigned long long)weight_checks,
        (unsigned long long)interval_checks,(unsigned long long)midpoint_checks);
    return 0;
} catch (const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
