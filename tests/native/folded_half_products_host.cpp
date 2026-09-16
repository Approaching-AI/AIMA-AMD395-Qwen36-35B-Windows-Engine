#include "../../native/providers/moe_accumulator/sm121_folded_half_products.h"
#include "folded_half_reference.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
namespace f = qrt_sm121_folded_half_products;
namespace old = qrt_q1_moe_hawkeye;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
bool same(old::Value a, old::Value b) { return a.significand == b.significand && a.exponent == b.exponent && a.negative == b.negative; }
int main() try {
    size_t roundtrips=0u, metadata=0u, folded=0u, fallback=0u, products=0u, states=0u;
    for (unsigned word=0u;word<65536u;++word) for (unsigned mode=0u;mode<3u;++mode) {
        uint16_t input[16];
        for(unsigned i=0u;i<16u;++i) input[i]=uint16_t(mode==0u||i==7u?word:mode==1u?0x3f80u:(i&1u?0x8000u:0u));
        const auto encoded=f::prepare(input); const auto legacy=f::half::prepare(input);
        require(f::legacy_control(encoded.control)==legacy.control,"compressed unit or mask changed");
        require(!std::memcmp(encoded.pairs,legacy.pairs,sizeof(encoded.pairs)),"prepared operand changed");
        unsigned low=31u;
        for(unsigned i=0u;i<16u;++i) {
            require(f::original(encoded,i)==input[i],"BF16 roundtrip changed"); ++roundtrips;
            const unsigned e=(input[i]>>7u)&255u;
            if(input[i]&0x7fffu) low=std::min(low,unsigned(int(e)-112-f::half::unit(legacy)));
        }
        if(f::half::unit(legacy)!=-32768) require(f::minimum(encoded.control)==(low==31u?0u:low),"minimum exponent changed");
        ++metadata;
    }
    for(unsigned width:{16u,272u,2048u,4096u,4112u,8192u}) for(unsigned row=0u;row<512u;++row) {
        old::Value carry{0u,-133,false};
        for(unsigned group=0u;group<width/16u;++group) {
            uint16_t a[16],b[16]; old::Value terms[17];terms[0]=carry;
            for(unsigned i=0u;i<16u;++i) {const auto pair=qrt_strong_replay_cases::input(row,group,i);a[i]=pair.x;b[i]=pair.y;terms[i+1u]=old::multiply_bf16(a[i],b[i],-133);}
            const auto left=f::prepare(a),right=f::prepare(b);
            int maximum=0;
            for(unsigned i=0u;i<16u;++i) maximum=std::max(maximum,int(((left.pairs[i/2u]>>(i%2u*16u+10u))&31u)+((right.pairs[i/2u]>>(i%2u*16u+10u))&31u)));
            const auto plan=f::plan(carry,left.control,right.control,maximum);
            const auto expected=qrt_folded_half_reference::expected(carry,a,b);
            require(plan.allowed==expected.allowed,"fold predicate differs from original exponents");
            const auto reference=old::group_sum<26,-133>(terms,17u);
            if(plan.allowed) {
                require(plan.left==expected.left&&plan.maximum==expected.maximum&&plan.power==expected.power,"fold plan changed");
                const unsigned shift=unsigned(plan.maximum-carry.exponent);
                const uint32_t aligned=shift>=32u?0u:(carry.significand<<2u)>>shift;
                uint32_t total=carry.negative?0u-aligned:aligned;
                for(unsigned pair=0u;pair<8u;++pair) {
                    const uint32_t x=left.pairs[pair]+(plan.left?plan.delta:0u),y=right.pairs[pair]+(plan.left?0u:plan.delta);
                    const float values[2]{f::half::product<false>(x,y),f::half::product<true>(x,y)};
                    for(unsigned i=0u;i<2u;++i) {
                        const int32_t value=int32_t(values[i]);const auto term=terms[1u+2u*pair+i];
                        const unsigned distance=unsigned(plan.maximum-term.exponent);
                        const uint64_t magnitude=distance>=32u?0u:(uint64_t(term.significand)<<2u)>>distance;
                        require(int64_t(value)==(term.negative?-int64_t(magnitude):int64_t(magnitude)),"aligned product differs from original integer primitive");
                        total+=uint32_t(value);++products;
                    }
                }
                const auto sum=qrt_sm121_group16::decode_modulo_sum(total,((a[0]^b[0])&0x8000u)!=0u);
                const auto actual=qrt_sm121_canonical::normalize(sum.magnitude,sum.negative,plan.maximum);
                require(same(actual,reference),"folded original carry changed");++folded;
            } else ++fallback;
            carry=reference;++states;
        }
    }
    require(folded&&fallback,"missing path coverage");
    std::printf("{\"kind\":\"folded_half_products_host\",\"metadata_rows\":%zu,\"roundtrip_words\":%zu,\"ordered_states\":%zu,\"folded_groups\":%zu,\"fallback_groups\":%zu,\"aligned_products\":%zu,\"mismatches\":0,\"native_executed\":false}\n",metadata,roundtrips,states,folded,fallback,products);
    return 0;
} catch(const std::exception& error) {std::fprintf(stderr,"%s\n",error.what());return 1;}
