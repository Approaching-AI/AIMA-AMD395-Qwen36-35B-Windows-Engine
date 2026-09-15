#include "sm121_predicted_group.h"
#include "float_alignment_cases.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
namespace original=qrt_q1_moe_hawkeye;
namespace plan=qrt_sm121_predicted_group;
using Value=original::Value;
bool same(Value a,Value b){return a.significand==b.significand&&a.exponent==b.exponent&&a.negative==b.negative;}
int predicted(float value){uint32_t bits;__builtin_memcpy(&bits,&value,4u);bits&=0x7fffffffu;return bits>=0x7f800000u?512:bits?int(bits>>23u)-127:-133;}
struct Group {
    uint32_t products[16]{};
    original::Value values[17]{};
    qrt_sm121_float_alignment::Group scalar;
    bool eligible=true;
    float approximate=0.0f;
};
int main(){
    size_t groups=0u,packed_accept=0u,packed_reject=0u,float_accept=0u,float_reject=0u;
    size_t prefix_hits=0u,prefix_fallbacks=0u,coefficient_equalities=0u;
    const Value marker{0xa5a5a5a5u,-17,true};
    for(unsigned row=0u;row<8192u;++row){
        std::array<Group,16> inputs{};std::array<float,16> prefix{};
        for(unsigned g=0u;g<16u;++g){auto& input=inputs[g];
            for(unsigned i=0u;i<16u;++i){
                const auto p=qrt_float_alignment_cases::input(row,g,i);
                input.values[i+1u]=original::multiply_bf16(p.left,p.right,-133);
                input.products[i]=qrt_sm121_group16::pack_product(input.values[i+1u]);
                input.eligible&=qrt_sm121_float_alignment::eligible(p.left)&&qrt_sm121_float_alignment::eligible(p.right);
                input.scalar.set(i,p.left,p.right);
                input.approximate+=input.scalar.products[i];
            }
            prefix[g]=input.approximate;
        }
        // Parallel inclusive scan estimates only the incoming exponent. Its
        // rounding/order is deliberately different from the exact recurrence.
        for(unsigned stride=1u;stride<16u;stride*=2u){const auto previous=prefix;
            for(unsigned g=stride;g<16u;++g)prefix[g]=previous[g]+previous[g-stride];}
        Value carry{0u,-133,false};
        for(unsigned g=0u;g<16u;++g){auto& input=inputs[g];input.values[0]=carry;
            const Value reference=original::group_sum<26,-133>(input.values,17u);
            const int guessed=g?predicted(prefix[g-1u]):-133;
            for(int exponent:{int(carry.exponent),guessed,int(carry.exponent)+1,int(carry.exponent)-1,int(carry.exponent)+37,-500,512}){
                const auto packed=plan::prepare_packed(input.products,exponent);Value actual=marker;
                const bool accepted=plan::apply(carry,packed,&actual);
                if(accepted){++packed_accept;assert(same(actual,reference));}
                else{++packed_reject;assert(same(actual,marker));}
                if(exponent==carry.exponent)assert(accepted);
                if(input.eligible){
                    const auto fp=plan::prepare_float(input.scalar,exponent);actual=marker;
                    if(plan::valid(fp)){assert(fp.modulo==packed.modulo&&fp.control==packed.control);++coefficient_equalities;}
                    const bool used=plan::apply(carry,fp,&actual);
                    if(used){++float_accept;assert(same(actual,reference));}
                    else{++float_reject;assert(same(actual,marker));}
                }
            }
            Value next=marker;const auto p=plan::prepare_packed(input.products,guessed);
            if(plan::apply(carry,p,&next))++prefix_hits;
            else{++prefix_fallbacks;next=original::group_sum<26,-133>(input.values,17u);}
            assert(same(next,reference));carry=next;++groups;
            assert(!plan::apply(carry,p,nullptr));
            Value bad=carry;bad.significand=0x1000000u;next=marker;
            assert(!plan::apply(bad,p,&next)&&same(next,marker));
        }
        assert(qrt_float_alignment_cases::output_bits(carry)==qrt_float_alignment_cases::reference(row));
    }
    assert(groups==131072u&&packed_accept>100000u&&packed_reject>100000u&&float_accept>10000u&&float_reject>10000u);
    assert(prefix_hits>10000u&&prefix_fallbacks>0u&&coefficient_equalities>10000u);
    std::printf("{\"kind\":\"predicted_k16_alignment_host\",\"sequences\":8192,\"canonical_groups\":%zu,\"packed_accept\":%zu,\"packed_reject\":%zu,\"float_accept\":%zu,\"float_reject\":%zu,\"identical_float_coefficients\":%zu,\"parallel_prefix_hits\":%zu,\"parallel_prefix_fallbacks\":%zu,\"raw_mismatches\":0,\"rejected_outputs_unchanged\":true,\"plan_bytes\":8,\"inference_acceptance\":false}\n",groups,packed_accept,packed_reject,float_accept,float_reject,coefficient_equalities,prefix_hits,prefix_fallbacks);
}
