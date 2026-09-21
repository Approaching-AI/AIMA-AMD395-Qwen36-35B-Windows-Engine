#include "../../native/providers/moe_accumulator/sm121_compact_matrix_group.h"
#include "../../native/providers/moe_accumulator/sm121_compact_matrix_metadata.h"
#include "narrow_half_cases.h"
#include <cassert>
#include <cstdio>
#include <vector>
namespace matrix=qrt_sm121_compact_matrix_group;
namespace original=qrt_q1_moe_hawkeye;
int signed_word(uint16_t x){return x&0x8000u?int(x)-65536:int(x);}
uint32_t rng=0x3958192u;
uint32_t next(){rng^=rng<<13u;rng^=rng>>17u;rng^=rng<<5u;return rng;}
int main(){
    uint64_t metadata=0u,groups=0u,accepted=0u,rejected=0u,queue=0u;
    for(unsigned attempt=0u;attempt<65536u;++attempt){
        const uint32_t mask=attempt<32u?(1u<<attempt):next();unsigned rank=0u;
        for(unsigned bit=0u;bit<32u;++bit)if(mask&(1u<<bit)){
            assert(matrix::select_bit(mask,rank)==bit);++rank;++queue;
        }
    }
    const unsigned widths[]={16u,32u,256u,2048u,4096u,8192u};
    for(unsigned width:widths)for(unsigned row=0u;row<256u;++row){
        std::vector<uint16_t>a(width),b(width);std::vector<matrix::Row>pa(width/16u),pb(width/16u);bool narrow=true;
        for(unsigned g=0u;g<width/16u;++g){
            for(unsigned i=0u;i<16u;++i){const auto p=qrt_narrow_half_cases::input(row,g,i);a[g*16u+i]=p.x;b[g*16u+i]=p.y;
                narrow&=qrt_sm121_narrow_f32_carry::eligible(p.x)&&qrt_sm121_narrow_f32_carry::eligible(p.y);}
            pa[g]=matrix::prepare(a.data()+g*16u);pb[g]=matrix::prepare(b.data()+g*16u);
            for(unsigned side=0u;side<2u;++side){const auto& p=side?pb[g]:pa[g];const auto& raw=side?b:a;
                for(unsigned i=0u;i<16u;++i){
                    assert(matrix::compact::original(p.encoded,i)==raw[g*16u+i]);
                    if(matrix::compact::unit(p.encoded)){
                        const unsigned value=matrix::compact::word(p.encoded,i);unsigned trailing=15u;
                        if(value){trailing=0u;while(!((value>>trailing)&1u))++trailing;}
                        assert(((p.trailing[i/8u]>>(4u*(i%8u)))&15u)==trailing);
                    }
                    ++metadata;
                }
            }
        }
        if(!narrow)continue;
        original::Value carry{0u,-133,false};float carried=0.0f;
        for(unsigned g=0u;g<width/16u;++g){
            original::Value terms[17];terms[0]=carry;int64_t mathematical=0;
            for(unsigned i=0u;i<16u;++i){
                terms[i+1u]=original::multiply_bf16(a[g*16u+i],b[g*16u+i],-133);
                mathematical+=int64_t(signed_word(matrix::compact::word(pa[g].encoded,i)))*signed_word(matrix::compact::word(pb[g].encoded,i));
            }
            const auto expected=original::group_sum<26,-133>(terms,17u);
            float result=matrix::f32::alignment::from_bits(0x4f395819u);
            const bool admitted=matrix::accumulate(carried,pa[g],pb[g],mathematical,&result);
            namespace slim=qrt_sm121_compact_matrix_metadata;
            float slim_result=matrix::f32::alignment::from_bits(0x4f395819u);
            const bool slim_admitted=slim::accumulate(carried,slim::prepare(pa[g]),
                slim::prepare(pb[g]),mathematical,&slim_result);
            assert(slim_admitted==admitted);
            assert(matrix::f32::bits(slim_result)==matrix::f32::bits(result));
            if(admitted){
                assert(matrix::f32::bits(result)==matrix::f32::bits(original::value_to_float(expected)));++accepted;
            }else{
                assert(matrix::f32::bits(result)==0x4f395819u);++rejected;
                qrt_sm121_float_alignment::Group group;
                for(unsigned i=0u;i<16u;++i)group.set(i,a[g*16u+i],b[g*16u+i]);
                result=qrt_sm121_narrow_f32_carry::accumulate(carried,group);
                assert(matrix::f32::bits(result)==matrix::f32::bits(original::value_to_float(expected)));
            }
            carry=expected;carried=result;++groups;
        }
    }
    assert(accepted&&rejected);
    std::printf("{\"kind\":\"compact_matrix_group_host\",\"row_bytes\":60,\"broadcast_metadata_bytes\":32,\"all_group_metadata_carries_checked\":true,\"lossless_words_and_trailing_metadata\":%llu,\"ordered_groups\":%llu,\"matrix_admitted_groups\":%llu,\"original_narrow_groups\":%llu,\"register_queue_sources_checked\":%llu,\"raw_carry_mismatches\":0,\"rejected_output_unchanged\":true,\"widths_through8192\":true}\n",(unsigned long long)metadata,(unsigned long long)groups,(unsigned long long)accepted,(unsigned long long)rejected,(unsigned long long)queue);
}
