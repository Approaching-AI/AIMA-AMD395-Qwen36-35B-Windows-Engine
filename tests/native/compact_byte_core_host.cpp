#include "sm121_compact_byte_core.h"
#include "float_alignment_cases.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>
namespace compact=qrt_sm121_compact_byte_core;
namespace wide=qrt_q1_moe_hawkeye;
using Value=wide::Value;
bool same(Value a,Value b){return a.significand==b.significand && a.exponent==b.exponent && a.negative==b.negative;}
int independent_core(uint16_t word,int unit){
    if(unit<0 || !(word&0x7fffu))return 0;
    const int distance=int((word>>7u)&255u)-unit;
    const unsigned significand=128u|(word&127u);
    const int value=distance<=-8?0:distance<0?int(significand>>unsigned(-distance)):int(significand<<unsigned(distance));
    return word&0x8000u?-value:value;
}
int decode_half(uint16_t word){
    const unsigned exponent=(word>>10u)&31u;
    if(!exponent){assert(!(word&1023u));return 0;}
    const unsigned significand=1024u|(word&1023u);
    const int shift=int(exponent)-25;
    const int value=shift>=0?int(significand<<unsigned(shift)):int(significand>>unsigned(-shift));
    return word&0x8000u?-value:value;
}
int main(){
    size_t reconstructed=0u,without_original_read=0u,exceptions=0u,invalid_rows=0u;
    // All BF16 encodings, each in five row contexts. Strided original storage
    // is separate from the compact representation and includes untouched gaps.
    for(unsigned bits=0u;bits<65536u;++bits)for(unsigned mode=0u;mode<5u;++mode){
        uint16_t source[16],strided[16u*7u];
        for(auto& x:strided)x=0xa5a5u;
        for(unsigned i=0u;i<16u;++i){
            source[i]=mode==0u?uint16_t(bits):uint16_t(((bits+i*977u)&0x807fu)|((mode==1u?127u:mode==2u?254u:mode==3u?6u:1u)<<7u));
            if(i==bits%16u)source[i]=uint16_t(bits);
            strided[i*7u]=source[i];
        }
        const auto row=compact::prepare(source),before=row;
        assert(row.padding==0u);invalid_rows+=compact::unit(row)<0;
        for(unsigned i=0u;i<16u;++i){
            assert(compact::original(row,strided,7u,i)==source[i]);++reconstructed;
            const int expected=independent_core(source[i],compact::unit(row));
            assert(decode_half(row.half[i])==expected);
            assert(((uint32_t(row.low[i/4u])>>(8u*(i%4u)))&255u)==(uint32_t(expected)&255u));
            assert((row.half[i]&0x8000u)==(source[i]&0x8000u));
            if(compact::unit(row)>=0 && !(compact::exceptions(row)&(1u<<i))){
                assert(compact::original(row,nullptr,0u,i)==source[i]);++without_original_read;
            }else ++exceptions;
        }
        assert(!std::memcmp(&row,&before,sizeof(row)));
        for(unsigned i=0u;i<16u*7u;++i)assert(strided[i]==(i%7u?0xa5a5u:source[i/7u]));
    }
    size_t cases=0u,accepted=0u,rejected=0u,compensated=0u,fallback_checks=0u;
    const Value marker{0xa5a5a5a5u,-17,true};
    for(unsigned row=0u;row<4096u;++row)for(unsigned group=0u;group<16u;++group){
        uint16_t a[16],b[16],strided[16u*3u]{};Value terms[17];
        for(unsigned i=0u;i<16u;++i){const auto p=qrt_float_alignment_cases::input(row,group,i);a[i]=p.left;b[i]=strided[i*3u]=p.right;terms[i+1u]=wide::multiply_bf16(a[i],b[i],-133);}
        const auto left=compact::prepare(a),right=compact::prepare(b),before_left=left,before_right=right;
        int64_t mathematical=0;
        for(unsigned i=0u;i<16u;++i)mathematical+=int64_t(independent_core(a[i],compact::unit(left)))*independent_core(b[i],compact::unit(right));
        assert(mathematical>=-compact::small::maximum_dot && mathematical<=compact::small::maximum_dot);
        const int bound=int(compact::maximum(left)+compact::maximum(right))-254;
        for(int exponent:{bound-1,bound,bound+1,bound+3,bound+8,bound+13,bound+17,-133,-126,-102,-101,121,127,128}){
            terms[0]={exponent==-133?0u:0x800000u|((row*7919u+group*37u)&0x7fffffu),int16_t(exponent),bool((row+group)&1u)};
            const auto expected=wide::group_sum<26,-133>(terms,17u);
            Value actual=marker;unsigned pairs=0xa5a5a5a5u;
            const bool used=compact::dominant(terms[0],left,right,a,1u,strided,3u,int32_t(mathematical),&actual,&pairs);
            if(used){
                if(!same(actual,expected)){std::fprintf(stderr,"compact mismatch row=%u group=%u exponent=%d units=%d,%d expected=%x/%d/%u actual=%x/%d/%u\n",row,group,exponent,compact::unit(left),compact::unit(right),expected.significand,expected.exponent,unsigned(expected.negative),actual.significand,actual.exponent,unsigned(actual.negative));return 1;}
                assert(pairs<=16u);++accepted;compensated+=pairs;
            }else{assert(same(actual,marker) && pairs==0xa5a5a5a5u);++rejected;}
            const auto fallback=compact::fallback(terms[0],left,right,a,1u,strided,3u);
            assert(same(fallback,expected));++fallback_checks;
            assert(!compact::dominant(terms[0],left,right,a,1u,strided,3u,int32_t(mathematical),nullptr));
            Value invalid=terms[0];invalid.significand=0x1000000u;actual=marker;
            assert(!compact::dominant(invalid,left,right,a,1u,strided,3u,int32_t(mathematical),&actual) && same(actual,marker));
            actual=marker;assert(!compact::dominant(terms[0],left,right,a,1u,strided,3u,INT32_MAX,&actual) && same(actual,marker));
            ++cases;
        }
        assert(!std::memcmp(&left,&before_left,sizeof(left)) && !std::memcmp(&right,&before_right,sizeof(right)));
    }
    // Positive and negative sums in the unsigned modulo overlap interval.
    for(unsigned negative=0u;negative<2u;++negative){
        uint16_t a[16],b[16];Value terms[17];
        for(unsigned i=0u;i<16u;++i){a[i]=uint16_t(0x3fffu|(negative?0x8000u:0u));b[i]=0x3fffu;terms[i+1u]=wide::multiply_bf16(a[i],b[i],-133);}
        const auto left=compact::prepare(a),right=compact::prepare(b);terms[0]={0xffffffu,0,bool(negative)};
        Value actual=marker;assert(compact::dominant(terms[0],left,right,a,1u,b,1u,negative?-compact::small::maximum_dot:compact::small::maximum_dot,&actual));
        assert(same(actual,wide::group_sum<26,-133>(terms,17u)));
    }
    assert(accepted>100000u && rejected>100000u && compensated>10000u);
    std::printf("{\"kind\":\"compact_byte_core_host\",\"row_bytes\":%zu,\"representation_checks\":%zu,\"reconstructed_without_original_read\":%zu,\"exception_or_invalid_reads\":%zu,\"invalid_rows\":%zu,\"cases\":%zu,\"accepted\":%zu,\"rejected\":%zu,\"compensated_pairs\":%zu,\"fallback_checks\":%zu,\"maximum_modulo_overlap_cases\":2,\"raw_mismatches\":0,\"rejected_outputs_unchanged\":true,\"immutable_rows\":true,\"native_matrix_checked\":false}\n",sizeof(compact::Row),reconstructed,without_original_read,exceptions,invalid_rows,cases,accepted,rejected,compensated,fallback_checks);
}
