#include "../../native/providers/gdn/absolute_dot_bound.h"
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
namespace norm=qrt_fla_absolute_dot;
namespace original=qrt_q1_moe_hawkeye;
namespace bits=qrt_sm121_pv_bound;
namespace consumer=qrt_fla_consumer_interval;
static uint32_t seed=0x3958192u;
static uint32_t random_word(){seed^=seed<<13u;seed^=seed>>17u;return seed^=seed<<5u;}
static float reference(const uint16_t* a,const uint16_t* b,unsigned count){
    original::Value carry{0u,-133,false};
    for(unsigned base=0u;base<count;base+=16u){
        original::Value terms[17];terms[0]=carry;
        for(unsigned i=0u;i<16u;++i)terms[i+1u]=original::multiply_bf16(a[base+i],b[base+i],-133);
        carry=original::group_sum<26,-133>(terms,17u);
    }
    return original::value_to_float(original::group_sum<26,-133>(&carry,1u));
}
int main(){
    size_t state=0u,output=0u,enclosed=0u,zeros=0u;
    for(unsigned trial=0u;trial<32768u;++trial){
        const unsigned count=trial%3u==0u?16u:trial%3u==1u?64u:128u;
        uint16_t a[128]{},b[128]{};norm::Row ar,br;
        long double absolute=0.0L,asum=0.0L,bsum=0.0L;
        for(unsigned i=0u;i<count;++i){
            const unsigned x=random_word(),y=random_word();
            unsigned ae=x%175u,be=y%175u;
            if(trial%8u==1u){ae=0u;be=1u;}
            if(trial%8u==2u){ae=174u;be=174u;}
            if(trial%8u==3u){ae=100u;be=100u;}
            a[i]=uint16_t((x&0x807fu)|(ae<<7u));b[i]=uint16_t((y&0x807fu)|(be<<7u));
            if(trial%8u==0u)b[i]&=0x8000u;
            if(trial%8u==4u){a[i]&=0x7fffu;b[i]&=0x7fffu;}
            if(trial%8u==5u && (i&1u)){a[i]=a[i-1u];b[i]=b[i-1u]^0x8000u;}
            norm::include(ar,a[i]);norm::include(br,b[i]);
            const long double av=std::fabs(bits::value(uint32_t(a[i])<<16u));
            const long double bv=std::fabs(bits::value(uint32_t(b[i])<<16u));
            absolute+=av*bv;asum+=av;bsum+=bv;
        }
        const auto an=norm::finish(ar),bn=norm::finish(br);
        const auto interval=norm::enclose(an,bn);
        assert(an.count==count && bn.count==count);
        assert(an.sum>=asum && bn.sum>=bsum);
        assert(qrt_sm121_projection_interval::valid(interval) && interval.upper>=absolute);
        const float exact=reference(a,b,count);
        assert(exact>=interval.lower && exact<=interval.upper);++enclosed;
        if(bn.maximum==0.0f){assert(bits::bits(exact)==0u && bits::bits(interval.lower)==0u && bits::bits(interval.upper)==0u);++zeros;}
        const float u=bits::value(uint32_t(uint16_t((random_word()&0x807fu)|(127u<<7u)))<<16u);
        const float decay=trial%5u==0u?0.0f:trial%5u==1u?1.0f:0.3125f;
        uint16_t updated=0xa5a5u,scaled=0xa5a5u,result=0xa5a5u;
        if(consumer::residual(interval,u,decay,&updated,&scaled)){
            const float difference=consumer::subtract(u,exact);
            assert(updated==bits::bf16(difference) && scaled==bits::bf16(consumer::multiply(difference,decay)));++state;
        }else assert(updated==0xa5a5u && scaled==0xa5a5u);
        if(consumer::output(interval,{u,u},decay,&result)){
            const float prior=consumer::multiply(consumer::multiply(exact,decay),0.08838834764831845f);
            assert(result==bits::bf16(std::fma(u,0.08838834764831845f,prior)));++output;
        }else assert(result==0xa5a5u);
    }
    size_t rejected=0u;
    for(uint16_t bad:{uint16_t(175u<<7u),uint16_t(176u<<7u),uint16_t(254u<<7u),uint16_t(0xff7f),uint16_t(0x7f80),uint16_t(0x7fc1),uint16_t(0xff80)}){
        norm::Row row;norm::include(row,0x3f80);norm::include(row,bad);norm::include(row,0);
        assert(!norm::finish(row).count);++rejected;
    }
    norm::Row empty,wide,one,two;
    for(unsigned i=0u;i<129u;++i)norm::include(wide,0);
    norm::include(one,0);norm::include(two,0);norm::include(two,0);
    for(auto interval:{norm::enclose(norm::finish(empty),norm::finish(empty)),
        norm::enclose(norm::finish(wide),norm::finish(wide)),
        norm::enclose(norm::finish(one),norm::finish(two))}){
        assert(!qrt_sm121_projection_interval::valid(interval));
        uint16_t a=0xa5a5u,b=a;
        assert(!consumer::residual(interval,1,1,&a,&b) && a==0xa5a5u && b==0xa5a5u);++rejected;
    }
    assert(state>4096u && output>4096u && zeros==4096u);
    std::printf("{\"dots\":%zu,\"zero_dots\":%zu,\"state_admitted\":%zu,\"output_admitted\":%zu,\"invalid_rejections\":%zu,\"false_admissions\":0,\"bound_violations\":0}\n",enclosed,zeros,state,output,rejected);
}
