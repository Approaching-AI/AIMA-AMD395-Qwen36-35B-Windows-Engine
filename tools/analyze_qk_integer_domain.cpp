// Read-only captured-operand audit. No GPU, model inference or timing claim.
#include "../native/providers/moe_accumulator/sm121_compact_integer_dot4.h"
#include "../native/providers/moe_accumulator/sm121_integer_core.h"
#include "../native/providers/moe_accumulator/sm121_narrow_f32_carry.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
namespace original=qrt_q1_moe_hawkeye;
namespace compact=qrt_sm121_compact_integer_dot4;
namespace core=qrt_sm121_integer_core;
namespace narrow=qrt_sm121_narrow_f32_carry;
std::vector<uint16_t> read(const char* path,size_t words){
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streamoff(words*2u))throw std::runtime_error("unexpected original capture size");
    std::vector<uint16_t> v(words);f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),words*2u);
    if(!f)throw std::runtime_error("short capture read");return v;
}
uint32_t random_state=0x8192395u;
uint32_t random_word(){random_state^=random_state<<13u;random_state^=random_state>>17u;random_state^=random_state<<5u;return random_state;}
int signed_word(uint16_t x){return x&0x8000u?int(x)-65536:int(x);}
int main(int argc,char** argv)try{
    if(argc!=3)throw std::runtime_error("usage: analyze_qk_integer_domain Q_BF16 K_BF16");
    constexpr unsigned source_tokens=7169u,tokens=8192u,dots=65536u;
    const auto q=read(argv[1],size_t(source_tokens)*4096u),k=read(argv[2],size_t(source_tokens)*512u);
    uint64_t groups=0u,narrow_groups=0u,compact_groups=0u,old_certified=0u,exact_certified=0u;
    uint64_t row_trailing_certified=0u,paired_trailing_certified=0u,full_exact_dots=0u;
    uint64_t full_compact_dots=0u,full_narrow_dots=0u,group_core_exceptions=0u,zero_products=0u;
    std::array<uint64_t,16> exact_by_group{},compact_by_group{};
    std::array<uint64_t,16> heads{};
    for(unsigned dot=0u;dot<dots;++dot){
        const uint64_t cell=random_word()%(uint64_t(tokens)*(tokens+1u)/2u);
        unsigned query=unsigned((std::sqrt(double(8u*cell+1u))-1.0)/2.0);
        while(uint64_t(query+1u)*(query+2u)/2u<=cell)++query;
        while(uint64_t(query)*(query+1u)/2u>cell)--query;
        unsigned key=unsigned(cell-uint64_t(query)*(query+1u)/2u),head=random_word()%16u;
        if(dot%1024u==0u){const unsigned edges[]={0u,63u,64u,127u,255u,1023u,4095u,7168u,7169u,8191u};query=edges[(dot/1024u)%10u];key=(dot/1024u)&1u?query:0u;}
        ++heads[head];const unsigned qi=query<source_tokens?query:query-source_tokens,ki=key<source_tokens?key:key-source_tokens;
        const uint16_t* a=q.data()+(size_t(qi)*16u+head)*256u;
        const uint16_t* b=k.data()+(size_t(ki)*2u+head/8u)*256u;
        original::Value carry{0u,-133,false};bool all_exact=true,all_compact=true,all_narrow=true;
        for(unsigned g=0u;g<16u;++g){
            const auto left=compact::prepare(a+g*16u),right=compact::prepare(b+g*16u);
            core::Row ca{},cb{};original::Value terms[17];terms[0]=carry;
            int maximum=std::max(-133,int(carry.exponent));bool normal=true,nonzero=false;
            for(unsigned i=0u;i<16u;++i){
                const uint16_t x=a[g*16u+i],y=b[g*16u+i];ca.original[i]=x;cb.original[i]=y;
                normal&=narrow::eligible(x)&&narrow::eligible(y);
                terms[i+1u]=original::multiply_bf16(x,y,-133);
                if(terms[i+1u].significand){maximum=std::max(maximum,int(terms[i+1u].exponent));nonzero=true;}
            }
            core::prepare(ca);core::prepare(cb);
            group_core_exceptions+=bool((ca.exceptions|cb.exceptions)&ca.nonzero&cb.nonzero);
            zero_products+=!nonzero;narrow_groups+=normal;all_narrow&=normal;
            const bool lossless=compact::unit(left)&&compact::unit(right);
            compact_groups+=lossless;compact_by_group[g]+=lossless;all_compact&=lossless;
            bool admitted=false;
            if(lossless){
                int64_t mathematical=0;unsigned min_a=31u,min_b=31u,min_pair=62u;
                for(unsigned i=0u;i<16u;++i){
                    const int x=signed_word(compact::word(left,i)),y=signed_word(compact::word(right,i));
                    mathematical+=int64_t(x)*y;
                    const unsigned tx=qrt_sm121_integer_parts::trailing_bits(uint32_t(x)),ty=qrt_sm121_integer_parts::trailing_bits(uint32_t(y));
                    min_a=std::min(min_a,tx);min_b=std::min(min_b,ty);min_pair=std::min(min_pair,tx+ty);
                }
                qrt_sm121_group16::AlignedSum old_sum;
                old_certified+=qrt_sm121_integer_parts::sum_exact_integer_product(carry,mathematical,ca.unit,ca.maximum,cb.unit,cb.maximum,&old_sum,ca.trailing,cb.trailing);
                const int shift=maximum-(int(compact::unit(left)+compact::unit(right))-254)-11;
                bool exact=shift>=-25&&shift<30;
                if(exact&&shift>0)for(unsigned i=0u;i<16u;++i){
                    const int64_t p=int64_t(signed_word(compact::word(left,i)))*signed_word(compact::word(right,i));
                    exact&=(uint64_t(p<0?-p:p)&((uint64_t(1)<<unsigned(shift))-1u))==0u;
                }
                const bool paired=shift>=-25&&shift<30&&(shift<=0||min_pair>=unsigned(shift));
                if(paired!=exact)throw std::runtime_error("paired trailing certificate differs");
                row_trailing_certified+=shift>=-25&&shift<30&&(shift<=0||min_a+min_b>=unsigned(shift));
                paired_trailing_certified+=paired;
                if(exact){
                    const int64_t products=shift<=0?mathematical*(int64_t(1)<<unsigned(-shift)):
                        mathematical<0?-int64_t(uint64_t(-mathematical)>>unsigned(shift)):int64_t(uint64_t(mathematical)>>unsigned(shift));
                    const unsigned distance=unsigned(maximum-carry.exponent);
                    const uint32_t aligned=distance>=32u?0u:(carry.significand<<2u)>>distance;
                    const uint32_t modulo=uint32_t(products)+(carry.negative?0u-aligned:aligned);
                    const auto sum=qrt_sm121_group16::decode_modulo_sum(modulo,((a[g*16u]^b[g*16u])&0x8000u)!=0u);
                    const auto result=qrt_sm121_canonical::normalize(sum.magnitude,sum.negative,maximum);
                    const auto expected=original::group_sum<26,-133>(terms,17u);
                    if(result.significand!=expected.significand||result.exponent!=expected.exponent||result.negative!=expected.negative)throw std::runtime_error("certified raw carry differs");
                    admitted=true;
                }
            }
            all_exact&=admitted;exact_certified+=admitted;exact_by_group[g]+=admitted;
            carry=original::group_sum<26,-133>(terms,17u);++groups;
        }
        full_exact_dots+=all_exact;full_compact_dots+=all_compact;full_narrow_dots+=all_narrow;
    }
    std::printf("{\"kind\":\"captured_qk_integer_domain_audit\",\"source_tokens\":7169,\"tokens\":8192,\"repeated_rows\":1023,\"sampled_dots\":%u,\"groups\":%llu,\"narrow_groups\":%llu,\"lossless_compact_groups\":%llu,\"old_upper_alignment_certified_groups\":%llu,\"exact_alignment_no_remainder_groups\":%llu,\"paired_trailing_certified_groups\":%llu,\"row_minimum_trailing_certified_groups\":%llu,\"groups_with_core_exceptions\":%llu,\"zero_product_groups\":%llu,\"fully_narrow_dots\":%llu,\"fully_compact_dots\":%llu,\"fully_exact_dots\":%llu,\"raw_carry_mismatches\":0,\"exact_by_group\":[",dots,(unsigned long long)groups,(unsigned long long)narrow_groups,(unsigned long long)compact_groups,(unsigned long long)old_certified,(unsigned long long)exact_certified,(unsigned long long)paired_trailing_certified,(unsigned long long)row_trailing_certified,(unsigned long long)group_core_exceptions,(unsigned long long)zero_products,(unsigned long long)full_narrow_dots,(unsigned long long)full_compact_dots,(unsigned long long)full_exact_dots);
    for(unsigned g=0u;g<16u;++g)std::printf("%s%llu",g?",":"",(unsigned long long)exact_by_group[g]);
    std::printf("],\"compact_by_group\":[");for(unsigned g=0u;g<16u;++g)std::printf("%s%llu",g?",":"",(unsigned long long)compact_by_group[g]);
    std::printf("],\"heads\":[");for(unsigned g=0u;g<16u;++g)std::printf("%s%llu",g?",":"",(unsigned long long)heads[g]);
    std::printf("],\"host_only\":true,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
