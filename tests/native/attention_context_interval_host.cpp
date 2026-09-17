#include "../../native/providers/ck_fmha/attention_context_interval.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace context=qrt_attention_context_interval;
namespace rcp=qrt_sm121_attention_rcp;
uint32_t state=0x3958192u;
uint32_t random_word(){state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;}
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
float product(float a,float b){volatile float result=a*b;return result;}
float quotient(float a,float b){volatile float result=a/b;return result;}
uint16_t round_bf16(float x){
    // Independent retained-word/remainder RNE, rather than the certificate's
    // add-and-shift conversion. Inputs in the admitted audit are finite.
    const uint32_t word=rcp::bits(x),tail=word&0xffffu;uint32_t kept=word>>16u;
    kept+=tail>0x8000u || (tail==0x8000u && (kept&1u));return uint16_t(kept);
}
int main(int argc,char** argv)try{
    require(argc==2,"requires SHA-verified original attention reciprocal table");
    std::ifstream file(argv[1],std::ios::binary|std::ios::ate);
    require(file && file.tellg()==std::streamoff(rcp::table_bytes),"reciprocal span");
    std::vector<unsigned char> table(rcp::table_bytes);file.seekg(0);
    require(bool(file.read(reinterpret_cast<char*>(table.data()),table.size())) && rcp::valid_layout(table.data(),table.size()),"reciprocal layout");
    const auto before=table;uint64_t admitted=0u,rejected=0u,points=0u,denominator_points=0u;
    for(unsigned trial=0u;trial<65536u;++trial){
        const uint32_t denominator_bits=((127u+trial%19u)<<23u)|((random_word()&0x7fffffu)&~63u);
        const uint32_t low_den=denominator_bits,high_den=denominator_bits+32u;
        const float center_den=rcp::value(denominator_bits+16u);
        const float inverse=rcp::evaluate(table.data(),center_den);
        const uint32_t target_bits=(random_word()&0x80000000u)|((100u+random_word()%50u)<<23u)|
            ((random_word()&127u)<<16u)|(trial&1u?0x8000u:0x1800u);
        const float center_num=quotient(rcp::value(target_bits),inverse);
        float numerators[9];numerators[4]=center_num;
        for(unsigned i=1u;i<=4u;++i){
            numerators[4u-i]=std::nextafter(numerators[5u-i],-INFINITY);
            numerators[4u+i]=std::nextafter(numerators[3u+i],INFINITY);
        }
        const context::Interval numerator{numerators[0],numerators[8]},denominator{rcp::value(low_den),rcp::value(high_den)};
        const auto ri=context::reciprocal(denominator,table.data());
        for(uint32_t d=low_den;d<=high_den;++d){
            const float value=rcp::evaluate(table.data(),rcp::value(d));
            require(value>=ri.low && value<=ri.high,"reciprocal escaped interval");++denominator_points;
        }
        uint16_t result=0xa5a5u;
        if(context::stable(numerator,denominator,table.data(),&result)){
            ++admitted;
            for(float n:numerators)for(uint32_t d=low_den;d<=high_den;++d){
                const float expected=product(n,rcp::evaluate(table.data(),rcp::value(d)));
                require(std::isfinite(expected) && result==round_bf16(expected),"false context admission");++points;
            }
        }else{++rejected;require(result==0xa5a5u,"decline changed output");}
    }
    require(admitted>16000u && rejected>16000u,"missing admitted or uncertain coverage");
    unsigned exceptional=0u;
    for(auto n:{context::Interval{1.0f,-1.0f},context::Interval{NAN,0},context::Interval{-INFINITY,0},
        context::Interval{0,INFINITY},context::Interval{0x1p-126f,0x1p-126f},context::Interval{rcp::value(0x7f7fffffu),rcp::value(0x7f7fffffu)}}){
        const context::Interval d=exceptional==4u?context::Interval{2,2}:context::Interval{1,2};
        uint16_t out=0xa5a5u;require(!context::stable(n,d,table.data(),&out) && out==0xa5a5u,"exceptional numerator admitted");++exceptional;
    }
    for(auto d:{context::Interval{0,1},context::Interval{1,0},context::Interval{1,0x1p19f},
        context::Interval{NAN,2},context::Interval{1,INFINITY}}){
        uint16_t out=0xa5a5u;require(!context::stable({1,1},d,table.data(),&out) && out==0xa5a5u,"exceptional denominator admitted");++exceptional;
    }
    for(float zero:{0.0f,-0.0f}){
        uint16_t out=0xa5a5u;require(context::stable({zero,zero},{1,8192},table.data(),&out) && out==round_bf16(zero),"signed zero point failed");
    }
    uint16_t out=0xa5a5u;
    require(!context::stable({-0.0f,0.0f},{1,2},table.data(),&out) && out==0xa5a5u,"mixed signed zero admitted");
    require(!context::stable({1,1},{1,2},nullptr,&out) && out==0xa5a5u,"null table admitted");
    require(!context::stable({1,1},{1,2},table.data(),nullptr),"null output admitted");
    require(table==before,"reciprocal table changed");
    std::printf("{\"kind\":\"joint_context_interval_host\",\"cases\":65536,\"admitted\":%llu,\"declined\":%llu,\"checked_cartesian_points\":%llu,\"reciprocal_points\":%llu,\"exceptional_declines\":%u,\"false_admissions\":0,\"signed_zero_checked\":true,\"decline_outputs_unchanged\":true,\"immutable_table\":true,\"native_qualified\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",(unsigned long long)admitted,(unsigned long long)rejected,(unsigned long long)points,(unsigned long long)denominator_points,exceptional);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
