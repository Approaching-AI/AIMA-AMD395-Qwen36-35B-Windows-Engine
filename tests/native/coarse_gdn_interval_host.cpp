#include "../../native/providers/gdn/coarse_interval.h"
#include "../../native/providers/moe_accumulator/sm121_group16_modulo.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
namespace coarse=qrt_sm121_coarse_projection_bound;
namespace bridge=qrt_fla_coarse_interval;
namespace consumer=qrt_fla_consumer_interval;
namespace interval=qrt_sm121_projection_interval;
namespace bound=qrt_sm121_pv_bound;
namespace original=qrt_q1_moe_hawkeye;
uint32_t random_state=0x3958192u;
uint32_t random_word(){random_state^=random_state<<13u;random_state^=random_state>>17u;return random_state^=random_state<<5u;}
float add(float a,float b){volatile float result=a+b;return result;}
struct Dot{interval::Interval range;float exact;};
size_t checkpoints=0u;
Dot dot(unsigned sample,unsigned width,unsigned mode){
    original::Value carry{0u,-133,false};coarse::State state;
    const unsigned ae=80u+random_word()%87u,be=80u+random_word()%87u;
    for(unsigned base=0u;base<width;base+=64u){
        float signed_sum=0.0f,absolute_sum=0.0f;
        for(unsigned group=0u;group<4u;++group){
            original::Value terms[17];terms[0]=carry;double sum=0.0,absolute=0.0;
            for(unsigned i=0u;i<16u;++i){
                uint16_t a=uint16_t((random_word()&0x807fu)|((ae+random_word()%9u)<<7u));
                uint16_t b=uint16_t((random_word()&0x807fu)|((be+random_word()%9u)<<7u));
                if(sample%7u==0u){a=uint16_t(0x3f81u|((i&1u)<<15u));b=0x3f85u;}
                if(sample%11u==0u && i%3u==0u)a=uint16_t((sample&1u)<<15u);
                if(sample%13u==0u){a=0x3fffu;b=0x3fffu;}
                if(sample%19u==0u){a=uint16_t((80u<<7u)|(random_word()&0x807fu));b=uint16_t((174u<<7u)|(random_word()&0x807fu));}
                if(sample%23u==0u)a=0u;
                assert(coarse::eligible(a)&&coarse::eligible(b));
                const double product=double(bound::value(uint32_t(a)<<16u))*double(bound::value(uint32_t(b)<<16u));
                sum+=product;absolute+=std::abs(product);terms[i+1u]=original::multiply_bf16(a,b,-133);
            }
            carry=original::group_sum<26,-133>(terms,17u);
            const double perturbation=mode==0u?0.0:(mode==1u?0.75:-0.75)*0x1p-19*absolute;
            const float native=float(sum+perturbation),positive=float(absolute-perturbation);
            assert(std::abs(double(native)-sum)<=0x1p-19*absolute);
            assert(std::abs(double(positive)-absolute)<=0x1p-19*absolute);
            signed_sum=add(signed_sum,native);absolute_sum=add(absolute_sum,positive);
        }
        state=coarse::advance<4u>(state,signed_sum,absolute_sum);
        const auto range=bridge::range(state,true);const float exact=original::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
        assert(interval::valid(range)&&range.lower<=exact&&exact<=range.upper);++checkpoints;
    }
    return {bridge::range(state,true),original::value_to_float(qrt_sm121_group16::finish_accumulator(carry))};
}
int main(){
    size_t wu=0u,residual=0u,output=0u,rejections=0u;
    for(unsigned mode=0u;mode<3u;++mode)for(unsigned sample=0u;sample<8192u;++sample){
        const auto prior=dot(sample,128u,mode),local=dot(sample+1u,64u,mode);
        uint16_t result=0xa5a5u,updated=0xa5a5u,scaled=0xa5a5u;
        if(consumer::rounded(local.range,&result)){assert(result==bound::bf16(local.exact));++wu;}
        else assert(result==0xa5a5u);
        const float u=sample%3u?bound::value((random_word()&0x80000000u)|((100u+random_word()%55u)<<23u)|((random_word()&127u)<<16u)):prior.exact;
        const float decay=sample%7u?bound::value(((90u+random_word()%40u)<<23u)|(random_word()&0x7fffffu)):sample%2u?0.0f:1.0f;
        if(consumer::residual(prior.range,u,decay,&updated,&scaled)){
            const float difference=consumer::subtract(u,prior.exact);
            assert(updated==bound::bf16(difference)&&scaled==bound::bf16(consumer::multiply(difference,decay)));++residual;
        }else assert(updated==0xa5a5u&&scaled==0xa5a5u);
        result=0xa5a5u;
        if(consumer::output(prior.range,local.range,decay,&result)){
            constexpr float scale=0.08838834764831845f;
            assert(result==bound::bf16(std::fma(local.exact,scale,consumer::multiply(consumer::multiply(prior.exact,decay),scale))));++output;
        }else assert(result==0xa5a5u);
    }
    for(const auto bad:{coarse::State{0.0f,-1.0f},coarse::State{INFINITY,0.0f},coarse::State{0.0f,INFINITY},coarse::State{NAN,0.0f},coarse::State{0.0f,NAN},coarse::State{0x1.fffffep127f,0x1.fffffep127f}}){
        assert(!interval::valid(bridge::range(bad,true)));++rejections;
    }
    for(unsigned x=0u;x<65536u;++x)if(!coarse::eligible(uint16_t(x))){
        uint16_t result=0xa5a5u;
        assert(!consumer::rounded(bridge::range({1.0f,0.0f},false),&result)&&result==0xa5a5u);++rejections;
    }
    for(float zero:{-0.0f,0.0f}){
        uint16_t result=0xa5a5u;assert(!consumer::rounded(bridge::range({zero,0.0f},true),&result));assert(result==0xa5a5u);++rejections;
    }
    assert(wu>1000u&&residual>1000u&&output>1000u&&checkpoints==73728u);
    std::printf("{\"kind\":\"coarse_gdn_interval_host\",\"cases\":24576,\"native_error_modes\":3,\"canonical_prefixes\":%zu,\"wu_admitted\":%zu,\"residual_admitted\":%zu,\"output_admitted\":%zu,\"exceptional_rejections\":%zu,\"undercoverage\":0,\"false_admissions\":0,\"rejection_outputs_unchanged\":true,\"hardware_error_bound_proven\":false}\n",checkpoints,wu,residual,output,rejections);
}
