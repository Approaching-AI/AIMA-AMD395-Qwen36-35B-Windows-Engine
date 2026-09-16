#include "../../native/providers/moe_accumulator/sm121_pair_projection_products.h"
#include "pair_projection_reference.h"
#include <cstdio>
#include <stdexcept>
namespace products=qrt_sm121_pair_projection_products;
namespace q=qrt_q1_moe_hawkeye;
void require(bool value) {if(!value)throw std::runtime_error("pair projection host mismatch");}
int main() try {
    uint64_t rows=0,words=0,states=0,recovered=0,fallback=0;
    for(unsigned mode=0;mode<3;++mode)for(unsigned word=0;word<65536u;++word) {
        uint16_t input[16];for(unsigned i=0;i<16;++i)input[i]=uint16_t(mode==0u||i==7u?word:mode==1u?0x3f80u:(i&1u?0x8000u:0u));
        const auto row=products::prepare(input);
        for(unsigned i=0;i<16;++i){require(products::original(row,i)==input[i]);++words;}
        ++rows;
    }
    for(unsigned row=0;row<4096u;++row) {
        q::Value carry{0u,-133,false};
        for(unsigned group=0;group<257u;++group) {
            uint16_t a[16],b[16];q::Value terms[17];terms[0]=carry;
            for(unsigned i=0;i<16;++i){const auto p=qrt_strong_replay_cases::input(row,group,i);a[i]=p.x;b[i]=p.y;terms[1u+i]=q::multiply_bf16(a[i],b[i],-133);}
            const auto left=products::prepare(a),right=products::prepare(b);
            unsigned used=0;const auto actual=products::accumulate(carry,left,right,&used);
            const auto expected=q::group_sum<26,-133>(terms,17u);
            require(actual.significand==expected.significand&&actual.exponent==expected.exponent&&actual.negative==expected.negative);
            require(used==qrt_pair_projection_reference::expected(carry,a,b).recovered_pairs);
            carry=expected;recovered+=used;fallback+=8u-used;++states;
        }
    }
    require(recovered&&fallback);
    std::printf("{\"kind\":\"pair_projection_host\",\"metadata_rows\":%llu,\"roundtrip_words\":%llu,\"ordered_states\":%llu,\"recovered_pairs\":%llu,\"fallback_pairs\":%llu,\"mismatches\":0,\"native_executed\":false}\n",(unsigned long long)rows,(unsigned long long)words,(unsigned long long)states,(unsigned long long)recovered,(unsigned long long)fallback);
    return 0;
} catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
