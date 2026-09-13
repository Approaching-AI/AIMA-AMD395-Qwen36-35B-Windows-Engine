#include "../../native/providers/moe_accumulator/sm121_decoded_bf16.h"
#include <cstdio>
#include <cstring>
#include <initializer_list>

int main() {
    namespace old=qrt_sm121_float_alignment;
    namespace decoded=qrt_sm121_decoded_bf16;
    const uint16_t edges[]={0u,0x8000u,uint16_t(64u<<7u),uint16_t((64u<<7u)|127u),
        uint16_t(127u<<7u),uint16_t((127u<<7u)|127u),uint16_t(190u<<7u),uint16_t((190u<<7u)|127u)};
    unsigned values=0u;uint64_t groups=0u,products=0u;
    for(unsigned raw=0u;raw<65536u;++raw) {
        const uint16_t x=uint16_t(raw);
        if(!old::eligible(x))continue;
        ++values;
        const float f=decoded::value(x);uint32_t bits;std::memcpy(&bits,&f,4u);
        if(bits!=(raw<<16u))return 1;
        if((x&0x7fffu) ? decoded::exponent(x)!=int((x>>7u)&255u)-127
            : decoded::exponent(x)!=decoded::zero_exponent)return 2;
        for(unsigned mode=0u;mode<16u;++mode) {
            old::Group a,b;
            for(unsigned i=0u;i<16u;++i) {
                const uint16_t left=mode&1u?edges[(mode+i)%8u]:x;
                const uint16_t right=uint16_t(edges[(mode+i/2u)%8u]^((mode&2u)?0x8000u:0u));
                a.set(i,left,right);
                decoded::set(b,i,decoded::value(left),decoded::value(right),
                    decoded::exponent(left),decoded::exponent(right));
                ++products;
            }
            if(std::memcmp(a.products,b.products,sizeof(a.products)) ||
                a.maximum!=b.maximum || a.first_negative!=b.first_negative)return 3;
            // Same product bits, maximum and sign must preserve the original
            // admission decision and aligned result for tiny/large/zero carry.
            for(int exponent:{-133,-110,-63,0,63,127,130}) {
                qrt_q1_moe_hawkeye::Value carry{exponent==-133 || (mode&4u)?0u:0x00801234u,
                    int16_t(exponent),(mode&8u)!=0u};
                qrt_sm121_group16::AlignedSum sa{},sb{};
                const bool ca=old::sum(carry,a,&sa),cb=old::sum(carry,b,&sb);
                if(ca!=cb || (ca && (sa.max_exponent!=sb.max_exponent ||
                    sa.value.magnitude!=sb.value.magnitude || sa.value.negative!=sb.value.negative)))return 4;
            }
            ++groups;
        }
    }
    std::printf("{\"eligible_bf16_encodings\":%u,\"groups\":%llu,\"products\":%llu,\"carry_cases_per_group\":7,\"raw_mismatches\":0}\n",
        values,(unsigned long long)groups,(unsigned long long)products);
    return 0;
}
