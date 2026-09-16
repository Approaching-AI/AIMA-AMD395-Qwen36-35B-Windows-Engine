#include "../../native/providers/moe_accumulator/sm121_centered_karatsuba_core.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
namespace core=qrt_sm121_centered_karatsuba;
struct Row{int high[4]{},low[4]{};};
int signed_value(uint16_t x){return x&32768u?int(x)-65536:int(x);}
double half(uint16_t x){return (x&32768u?-1.0:1.0)*std::ldexp(double((x&1023u)+1024u),int((x>>10u)&31u)-25);}
void set(Row& row,unsigned i,uint16_t x){const unsigned shift=i%4u*8u;row.high[i/4u]=int(uint32_t(row.high[i/4u])|uint32_t(x>>8u)<<shift);row.low[i/4u]=int(uint32_t(row.low[i/4u])|uint32_t(x&255u)<<shift);}
uint32_t state=0x3958192u;
uint32_t random_word(){state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;}
int main(){
    unsigned encodings=0u;
    for(unsigned first=0u;first<65536u;first+=16u){
        Row row{};int sum=0;
        for(unsigned i=0u;i<16u;++i)set(row,i,uint16_t(first+i));
        const auto before=row;const auto prepared=core::prepare(row);
        for(unsigned i=0u;i<16u;++i){const int x=int(first+i);const int expected=(x>>8)-(x&32768?256:0)+(x&255)-128;
            assert(expected>=-256 && expected<=254 && core::digit(uint16_t(x))==expected);
            assert((prepared.combined[i]&0x7fffu)?half(prepared.combined[i])==expected:expected==0);sum+=expected;++encodings;}
        assert(sum==prepared.sum && !std::memcmp(&row,&before,sizeof(row)));
    }
    for(unsigned sample=0u;sample<262144u;++sample){
        int hh=0,ll=0,ss=0,sa=0,sb=0;int64_t expected=0;
        for(unsigned i=0u;i<16u;++i){
            uint16_t a=uint16_t(random_word()),b=uint16_t(random_word());
            if(sample%11u==0u){a=0x8000u;b=sample%2u?0x7fffu:0x8000u;}
            if(sample%13u==0u){a=0x7fffu;b=0x7fffu;}
            if(sample%17u==0u)b=i%2u?a:uint16_t(0u-a);
            const int ah=signed_value(a)/256-(signed_value(a)<0 && (a&255u)?1:0),bh=signed_value(b)/256-(signed_value(b)<0 && (b&255u)?1:0);
            const int al=a&255u,bl=b&255u,as=core::digit(a),bs=core::digit(b);
            hh+=ah*bh;ll+=al*bl;ss+=as*bs;sa+=as;sb+=bs;expected+=int64_t(signed_value(a))*signed_value(b);
        }
        assert(core::reconstruct(hh,ll,ss,sa,sb)==expected);
    }
    std::printf("{\"kind\":\"centered_karatsuba_host\",\"signed16_encodings\":%u,\"integer_dots\":262144,\"reconstruction_mismatches\":0,\"prepared_row_bytes\":36,\"native_matrix_executed\":false,\"hardware_model_proven\":false}\n",encodings);
}
