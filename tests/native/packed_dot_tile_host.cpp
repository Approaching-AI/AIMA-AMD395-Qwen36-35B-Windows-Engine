#include "../../native/providers/moe_accumulator/sm121_packed_dot_tile.h"
#include "float_alignment_cases.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace tile=qrt_sm121_packed_dot_tile;
namespace original=qrt_q1_moe_hawkeye;
namespace cases=qrt_float_alignment_cases;
uint64_t checked=0u,outputs=0u,fast_tiles=0u,replayed_tiles=0u,late_restarts=0u;
template<unsigned Width,unsigned Rows,unsigned Columns>
void check(unsigned trials){
    constexpr unsigned lp=Width/2u+1u,rp=Columns+1u;
    for(unsigned trial=0u;trial<trials;++trial){
        std::array<uint32_t,Rows*lp> a;std::array<uint32_t,Width/2u*rp> b;
        a.fill(0xa5a5a5a5u);b.fill(0x5a5a5a5au);
        unsigned af[Rows],bf[Columns];
        for(unsigned r=0u;r<Rows;++r){
            af[r]=1u;
            for(unsigned i=0u;i<Width;i+=2u){
                auto x=cases::input(trial+r,i/16u,i%16u).left;
                auto y=cases::input(trial+r,i/16u,i%16u+1u).left;
                if(trial%257u==1u)x=y=i<16u?0x3f80u:0x5f7fu;
                a[r*lp+i/2u]=uint32_t(x)|(uint32_t(y)<<16u);
                af[r]&=unsigned(qrt_sm121_float_alignment::eligible(x)&&qrt_sm121_float_alignment::eligible(y));
            }
            if(trial%5u==0u)af[r]=0u;
        }
        for(unsigned c=0u;c<Columns;++c){
            bf[c]=1u;
            for(unsigned i=0u;i<Width;i+=2u){
                auto x=cases::input(trial+c+3u,i/16u,i%16u).right;
                auto y=cases::input(trial+c+3u,i/16u,i%16u+1u).right;
                if(trial%257u==1u)x=y=i<16u?0x3f80u:0x5f7fu;
                b[i/2u*rp+c]=uint32_t(x)|(uint32_t(y)<<16u);
                bf[c]&=unsigned(qrt_sm121_float_alignment::eligible(x)&&qrt_sm121_float_alignment::eligible(y));
            }
            if(trial%7u==0u)bf[c]=0u;
        }
        const auto saved_a=a;
        const auto saved_b=b;
        original::Value trace[Width/16u*Rows*Columns];
        for(auto& v:trace)v={123u,300,true};
        tile::Result<Rows,Columns> fast;
        if(tile::try_tile<Width,lp,rp,Rows,Columns,true>(a.data(),b.data(),af,bf,&fast,trace))++fast_tiles;
        else{
            ++replayed_tiles;
            late_restarts+=trace[0].exponent!=300;
        }
        const auto result=tile::dot<Width,lp,rp,Rows,Columns,true>(a.data(),b.data(),af,bf,trace);
        if(a!=saved_a || b!=saved_b)throw std::runtime_error("tile input modified");
        for(unsigned r=0u;r<Rows;++r)for(unsigned c=0u;c<Columns;++c){
            original::Value carry{0u,-133,false};
            for(unsigned base=0u;base<Width;base+=16u){
                original::Value terms[17];terms[0]=carry;
                for(unsigned i=0u;i<16u;++i){
                    const uint16_t x=uint16_t(a[r*lp+(base+i)/2u]>>((i%2u)*16u));
                    const uint16_t y=uint16_t(b[(base+i)/2u*rp+c]>>((i%2u)*16u));
                    terms[i+1u]=original::multiply_bf16(x,y,-133);
                }
                carry=original::group_sum<26,-133>(terms,17u);
                const auto got=trace[(base/16u*Rows+r)*Columns+c];
                if(carry.significand!=got.significand || carry.exponent!=got.exponent ||
                    (carry.significand && carry.negative!=got.negative))throw std::runtime_error("tile K16 carry mismatch");
                ++checked;
            }
            const float expected=original::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
            if(std::memcmp(&expected,&result.values[r][c],sizeof(float)))throw std::runtime_error("tile output mismatch");
            ++outputs;
        }
    }
}
int main()try{
    check<64u,1u,2u>(4096u);check<64u,2u,2u>(4096u);
    check<128u,1u,2u>(4096u);check<128u,2u,2u>(4096u);
    check<256u,2u,1u>(4096u);check<2048u,2u,2u>(64u);
    if(!fast_tiles || !replayed_tiles || !late_restarts)throw std::runtime_error("missing acceptance/restart coverage");
    std::printf("{\"kind\":\"packed_dot_tile_host\",\"ordered_k16_states\":%llu,\"raw_outputs\":%llu,\"accepted_fast_tiles\":%llu,\"replayed_tiles\":%llu,\"late_restarts\":%llu,\"mismatches\":0,\"inputs_and_padding_unchanged\":true,\"native_qualification\":false,\"inference_acceptance\":false}\n",(unsigned long long)checked,(unsigned long long)outputs,(unsigned long long)fast_tiles,(unsigned long long)replayed_tiles,(unsigned long long)late_restarts);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
