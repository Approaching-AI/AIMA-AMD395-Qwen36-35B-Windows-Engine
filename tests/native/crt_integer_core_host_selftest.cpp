#include "../../native/providers/moe_accumulator/sm121_crt_integer_core.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
namespace core=qrt_sm121_crt_integer;
struct Row{int high[4]{},low[4]{};};
int mod(int64_t x,int p){int r=int(x%p);return r<0?r+p:r;}
float half(uint16_t x){return x&0x7fffu?(x&32768u?-1.0f:1.0f)*std::ldexp(float((x&1023u)+1024u),int((x>>10u)&31u)-25):0.0f;}
uint32_t state=0x3958192u;
uint32_t random_word(){state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;}
int main(){
    std::vector<int> valid;
    for(unsigned first=0u;first<65536u;first+=16u){
        Row row{};
        for(unsigned i=0u;i<16u;++i){const unsigned x=first+i,shift=i%4u*8u;row.high[i/4u]=int(uint32_t(row.high[i/4u])|((x>>8u)<<shift));row.low[i/4u]=int(uint32_t(row.low[i/4u])|((x&255u)<<shift));}
        const Row before=row;const auto prepared=core::prepare(row);
        assert(!std::memcmp(&before,&row,sizeof(row)));
        for(unsigned i=0u;i<16u;++i){
            const unsigned x=first+i;const int value=x&32768u?int(x)-65536:int(x);
            bool expected=false;
            for(unsigned shift=0u;shift<8u;++shift)for(unsigned m=0u;m<256u;++m)
                expected |= value==int(m<<shift) || value==-int(m<<shift);
            assert(core::eligible(value)==expected);
            if(expected){assert(half(prepared.values[i])==float(value));valid.push_back(value);}
            else assert(prepared.values[i]==0x7e00u);
            assert(((prepared.residue255[i/4u]>>(i%4u*8u))&255u)==unsigned(mod(value,255)));
            assert(((prepared.residue256[i/4u]>>(i%4u*8u))&255u)==unsigned(mod(value,256)));
        }
    }
    assert(valid.size()==2303u);
    size_t recovered=0;
    for(int64_t value=-65280;value<130560;++value)for(float error:{-32000.25f,-3920.0f,0.0f,3920.0f,32000.25f}){
        const float approximate=float(double(value)+error);int64_t result=INT64_MIN;
        assert(std::abs(double(approximate)-double(value))<32640.0);
        assert(core::recover(approximate,unsigned(mod(value,255)),unsigned(mod(value,256)),&result));
        assert(result==value);++recovered;
    }
    for(unsigned sample=0u;sample<262144u;++sample){
        int64_t total=0;uint32_t residue255=0,residue256=0;
        for(unsigned i=0u;i<16u;++i){
            int a=valid[random_word()%valid.size()],b=valid[random_word()%valid.size()];
            if(sample%11u==0u){a=32640;b=sample%2u?32640:-32640;}
            if(sample%13u==0u){a=32640;b=i%2u?-a:a;}
            total+=int64_t(a)*b;residue255+=unsigned(mod(a,255)*mod(b,255));residue256+=unsigned(mod(a,256)*mod(b,256));
        }
        assert(core::reduce255(residue255)==unsigned(mod(total,255)));
        for(int error:{-30000,0,30000}){
            const float approximate=float(double(total)+error);int64_t result=INT64_MIN;
            assert(std::abs(double(approximate)-double(total))<32640.0);
            assert(core::recover(approximate,residue255,residue256,&result) && result==total);++recovered;
        }
        const uint32_t word=random_word();assert(core::reduce255(word)==word%255u);
        assert(core::base_residue(total)==mod(total,255));
    }
    for(float approximate:{32640.0f,-32640.0f,std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),float(core::maximum_dot+131072)}){
        int64_t result=123;assert(!core::recover(approximate,0,0,&result) && result==123);
    }
    for(int64_t value:{core::maximum_dot+1,-core::maximum_dot-1}){
        int64_t result=123;assert(!core::recover(float(value),unsigned(mod(value,255)),unsigned(mod(value,256)),&result) && result==123);
    }
    std::printf("{\"kind\":\"crt_integer_host\",\"signed16_encodings\":65536,\"eligible_encodings\":%zu,\"integer_dots\":262144,\"nearest_recoveries\":%zu,\"mismatches\":0,\"prepared_row_bytes\":64,\"native_matrix_executed\":false,\"hardware_model_proven\":false}\n",valid.size(),recovered);
}
