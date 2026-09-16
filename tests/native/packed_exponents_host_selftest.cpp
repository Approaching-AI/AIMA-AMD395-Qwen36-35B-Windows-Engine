#include "../../native/providers/moe_accumulator/sm121_packed_exponents.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
namespace core=qrt_sm121_packed_exponents;
uint32_t state=0x3958192u;
uint32_t random_word(){state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;}
void verify(const uint16_t* words,const core::Metadata& metadata){
    int maximum=0;bool good=true;
    for(unsigned i=0u;i<16u;++i){const unsigned e=(words[i]>>7u)&255u;good&=!(words[i]&32767u)||(e>=64u&&e<=190u);if(words[i]&32767u)maximum=std::max(maximum,int(e));}
    assert(metadata.maximum==(good?maximum:-1));
    for(unsigned i=0u;i<16u;++i){const unsigned expected=(words[i]&32767u)?unsigned(std::min(15,maximum-int((words[i]>>7u)&255u))):15u;assert(((metadata.deficits[i/6u]>>(i%6u*5u))&31u)==expected);}
    assert(!(metadata.deficits[0]>>30u) && !(metadata.deficits[1]>>30u) && !(metadata.deficits[2]>>20u));
}
int main(){
    for(unsigned first=0u;first<65536u;first+=16u){uint16_t words[16];for(unsigned i=0u;i<16u;++i)words[i]=uint16_t(first+i);const auto metadata=core::prepare(words);verify(words,metadata);}
    unsigned accepted=0u,declined=0u;
    for(unsigned sample=0u;sample<1048576u;++sample){
        uint16_t a[16],b[16];
        for(unsigned i=0u;i<16u;++i){
            const unsigned ea=sample%8u<4u?120u+random_word()%15u:64u+random_word()%127u;
            const unsigned eb=sample%8u<4u?120u+random_word()%15u:64u+random_word()%127u;
            a[i]=uint16_t(ea<<7u|(random_word()&0x807fu));b[i]=uint16_t(eb<<7u|(random_word()&0x807fu));
            if(sample%8u==1u && i%3u==0u)a[i]=i%2u?0x8000u:0u;
            if(sample%8u==2u)a[i]=i%2u?0x8000u:0u;
            if(sample%8u==5u){a[i]=uint16_t((i==0u?190u:64u)<<7u);b[i]=uint16_t((i==1u?190u:64u)<<7u);}
            if(sample%8u==6u && i==9u)a[i]=uint16_t(random_word());
            if(sample%8u==7u){a[i]=uint16_t(random_word());b[i]=uint16_t(random_word());}
        }
        uint16_t before_a[16],before_b[16];std::memcpy(before_a,a,sizeof(a));std::memcpy(before_b,b,sizeof(b));
        const auto am=core::prepare(a),bm=core::prepare(b);verify(a,am);verify(b,bm);
        assert(!std::memcmp(a,before_a,sizeof(a)) && !std::memcmp(b,before_b,sizeof(b)));
        int actual=12345,expected=-133;
        for(unsigned i=0u;i<16u;++i)if((a[i]&32767u)&&(b[i]&32767u))expected=std::max(expected,int((a[i]>>7u)&255u)+int((b[i]>>7u)&255u)-254);
        if(core::maximum(am,bm,&actual)){assert(actual==expected);++accepted;}
        else{assert(actual==12345);++declined;}
    }
    assert(accepted && declined);
    std::printf("{\"kind\":\"packed_exponents_host\",\"bf16_encodings\":65536,\"group_pairs\":1048576,\"accepted\":%u,\"declined\":%u,\"accepted_maximum_mismatches\":0,\"immutable_inputs\":true,\"metadata_bytes\":16,\"row_bytes\":48,\"native_executed\":false}\n",accepted,declined);
}
