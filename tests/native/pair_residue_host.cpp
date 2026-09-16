#include "../../native/providers/moe_accumulator/sm121_pair_residue.h"
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <stdexcept>
namespace pair=qrt_sm121_pair_residue;
void require(bool okay) { if(!okay) throw std::runtime_error("pair residue host mismatch"); }
uint32_t next(uint32_t& r) {r^=r<<13u;r^=r>>17u;r^=r<<5u;return r;}
int main() try {
    uint32_t rng=0x823159a7u;uint64_t checked=0;
    for(unsigned i=0;i<524288u;++i) {
        unsigned m[4],s[4];bool n[4];int32_t v[4];
        for(unsigned j=0;j<4;++j) {m[j]=128u+(next(rng)&127u);s[j]=next(rng)&7u;n[j]=(next(rng)&1u)!=0u;v[j]=n[j]?-int32_t(m[j]<<s[j]):int32_t(m[j]<<s[j]);}
        const auto left=pair::prepare(m[0],s[0],n[0],m[1],s[1],n[1]);
        const auto right=pair::prepare(m[2],s[2],n[2],m[3],s[3],n[3]);
        for(unsigned j=0;j<4;++j) {
            const uint16_t h=uint16_t((j<2?left.half:right.half)>>((j&1u)*16u));
            const int32_t magnitude=int32_t((1024u+(h&1023u))<<(((h>>10u)&31u)-22u))>>3u;
            require((h&32768u?-magnitude:magnitude)==v[j]);
        }
        const int32_t p0=v[0]*v[2],p1=v[1]*v[3],exact=p0+p1;
        const uint32_t residue=((left.bytes&255u)*(right.bytes&255u)+((left.bytes>>8u)&255u)*((right.bytes>>8u)&255u))&255u;
        require(residue==(uint32_t(exact)&255u));
        int32_t restored=0;require(pair::recover(float(exact),residue,&restored)&&restored==exact);
        for(unsigned shift=0;shift<=8;++shift) {require(pair::aligned(restored,left,right,shift)==uint32_t(p0/int32_t(1u<<shift)+p1/int32_t(1u<<shift)));++checked;}
    }
    for(int32_t exact=-1024;exact<=1024;++exact) for(float error:{-127.75f,-127.0f,-0.25f,0.0f,0.25f,127.0f,127.75f}) {
        int32_t result=0;require(pair::recover(float(exact)+error,uint32_t(exact)&255u,&result)&&result==exact);
    }
    int32_t result=123;
    require(!pair::recover(128.0f,0u,&result)&&result==123);
    require(!pair::recover(-128.0f,0u,&result)&&result==123);
    require(!pair::recover(std::numeric_limits<float>::infinity(),0u,&result));
    require(!pair::recover(std::numeric_limits<float>::quiet_NaN(),0u,&result));
    require(!pair::recover(0.0f,0u,nullptr));
    std::printf("{\"kind\":\"pair_residue_host\",\"pairs\":524288,\"aligned_pairs\":%llu,\"mismatches\":0,\"native_executed\":false}\n",(unsigned long long)checked);
    return 0;
} catch(const std::exception& error) {std::fprintf(stderr,"%s\n",error.what());return 1;}
