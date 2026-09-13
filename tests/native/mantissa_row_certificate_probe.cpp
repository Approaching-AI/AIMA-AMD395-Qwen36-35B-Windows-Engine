#include "../../native/providers/moe_accumulator/sm121_mantissa_row_certificate.h"
#include "../../native/providers/moe_accumulator/sm121_canonical_normalize.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace c = qrt_sm121_mantissa_row_certificate;
namespace h = qrt_q1_moe_hawkeye;
namespace m = qrt_sm121_mantissa_parts;
struct Counts { uint64_t groups=0, accepted=0, mismatches=0; };
float widen(uint16_t x) { return qrt_sm121_native_product::from_bits(uint32_t(x)<<16u); }
unsigned rng(unsigned& x) { x^=x<<13u; x^=x>>17u; x^=x<<5u; return x; }

h::Value inspect(const uint16_t* left,const uint16_t* right,h::Value carry,Counts& counts) {
    const auto lr=c::prepare(left),rr=c::prepare(right);
    h::Value group[17];group[0]=carry;
    for(unsigned i=0;i<16u;++i) group[i+1u]=h::multiply_bf16(left[i],right[i],-133);
    const auto expected=h::group_sum<26,-133>(group,17u);
    ++counts.groups;
    int exponent;
    if(!c::eligible(lr,rr,carry,&exponent)) return expected;
    float partials[4]{};
    for(unsigned i=0;i<16u;++i) {
        const float a[]={widen(m::high(left[i])),widen(m::low(left[i]))};
        const float b[]={widen(m::high(right[i])),widen(m::low(right[i]))};
        for(unsigned x=0;x<2u;++x) for(unsigned y=0;y<2u;++y) {
            volatile float product=a[x]*b[y];
            volatile float sum=partials[2u*x+y]+product;
            partials[2u*x+y]=sum;
        }
    }
    qrt_sm121_group16::AlignedSum sum{};
    if(!c::sum(lr,rr,carry,((left[0]^right[0])&0x8000u)!=0u,partials,&sum)) return expected;
    ++counts.accepted;
    const auto actual=qrt_sm121_canonical::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
    if(actual.significand!=expected.significand || actual.exponent!=expected.exponent || actual.negative!=expected.negative) {
        ++counts.mismatches;
        if(counts.mismatches<=4u)
            std::fprintf(stderr,"ROW_CERT_DIFF carry=%08x actual=%08x expected=%08x alignment=%d\n",
                qrt_sm121_native_product::float_bits(h::value_to_float(carry)),
                qrt_sm121_native_product::float_bits(h::value_to_float(actual)),
                qrt_sm121_native_product::float_bits(h::value_to_float(expected)),exponent);
    }
    return expected;
}
void report(const char* kind,const Counts& counts) {
    std::printf("{\"kind\":\"%s\",\"groups\":%llu,\"coarse_certificate_accepted\":%llu,\"canonical_value_mismatches\":%llu,\"acceptance_fraction\":%.9f,\"cpu_partials_only\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
        kind,static_cast<unsigned long long>(counts.groups),static_cast<unsigned long long>(counts.accepted),
        static_cast<unsigned long long>(counts.mismatches),double(counts.accepted)/double(counts.groups));
}
std::vector<uint16_t> read(const char* path,size_t cells) {
    std::vector<uint16_t> out(cells);
    std::ifstream file(path,std::ios::binary);file.read(reinterpret_cast<char*>(out.data()),cells*2u);
    if(!file || file.peek()!=EOF) throw std::runtime_error("capture size mismatch");
    return out;
}
int main(int argc,char** argv) try {
    Counts total;
    if(argc==2 && std::string(argv[1])=="--selftest") {
        uint16_t edge[16];std::fill(edge,edge+16u,0x3f80u);
        const auto normal=c::prepare(edge);int alignment;
        assert(normal.valid && normal.minimum==127 && normal.maximum==127);
        assert(c::eligible(normal,normal,h::value_from_float(1.0f,-133),&alignment));
        for(uint16_t invalid : {uint16_t(0x7f80u),uint16_t(0x7fc1u),uint16_t(1u),uint16_t(0x0381u)}) {
            edge[15]=invalid;assert(!c::prepare(edge).valid);
            assert(!c::eligible(c::prepare(edge),normal,h::value_from_float(0.0f,-133),&alignment));
        }
        std::fill(edge,edge+16u,0x8000u);
        assert(!c::eligible(c::prepare(edge),normal,h::value_from_float(0.0f,-133),&alignment));
        const c::Range anti_correlated{131,132,true};
        // A row-bound exponent may exceed the true cell exponent. Its extra
        // alignment bit must not silently discard a significant carry bit.
        assert(!c::eligible(anti_correlated,anti_correlated,h::Value{0x800080u,0,false},&alignment));
        assert(c::eligible(anti_correlated,anti_correlated,h::Value{0x800000u,0,false},&alignment));
        const float nonfinite_partials[]={qrt_sm121_native_product::from_bits(0x7f800000u),0.0f,0.0f,0.0f};
        qrt_sm121_group16::AlignedSum rejected{};
        assert(!c::sum(normal,normal,h::value_from_float(0.0f,-133),false,nonfinite_partials,&rejected));
        unsigned seed=0x3957169u;
        for(unsigned trial=0;trial<65536u;++trial) {
            const unsigned spread=1u+trial%16u;
            auto carry=h::value_from_float(trial%3u ? 0.0f :
                qrt_sm121_native_product::from_bits((rng(seed)&0x807fffffu)|((105u+trial%44u)<<23u)),-133);
            for(unsigned step=0;step<16u;++step) {
                uint16_t left[16],right[16];
                for(unsigned i=0;i<16u;++i) {
                    left[i]=uint16_t((rng(seed)&0x807fu)|((115u+rng(seed)%spread)<<7u));
                    right[i]=uint16_t((rng(seed)&0x807fu)|((115u+rng(seed)%spread)<<7u));
                    if(trial%7u==0u && i%2u) {left[i]=left[i-1u];right[i]=right[i-1u]^0x8000u;}
                    if(trial%11u==0u && i%5u==0u) left[i]=0u;
                }
                carry=inspect(left,right,carry,total);
            }
        }
        report("mantissa_row_certificate_selftest",total);
    } else if(argc==4 && std::string(argv[1])=="--q7169") {
        constexpr unsigned tokens=7169u;
        const auto q=read(argv[2],size_t(tokens)*4096u),k=read(argv[3],size_t(tokens)*512u);
        // Deterministic samples cover the causal triangle, every query/KV
        // head, all K16 groups and the full prompt extent. These are capacity
        // diagnostics, not a replacement for the complete tensor/product gate.
        uint64_t dots=0,all_fast=0;
        for(unsigned qi=0;qi<256u;++qi) {
            const unsigned query=qi*(tokens-1u)/255u;
            for(unsigned head=0;head<16u;++head) for(unsigned ki=0;ki<32u;++ki) {
                const unsigned key=ki*query/31u;
                auto carry=h::value_from_float(0.0f,-133);const auto before=total.accepted;
                for(unsigned base=0;base<256u;base+=16u)
                    carry=inspect(q.data()+(size_t(query)*16u+head)*256u+base,
                        k.data()+(size_t(key)*2u+head/8u)*256u+base,carry,total);
                ++dots;all_fast+=total.accepted-before==16u;
            }
        }
        report("mantissa_row_certificate_q7169",total);
        std::printf("{\"sampled_qk_dots\":%llu,\"dots_with_all_16_groups_certified\":%llu,\"query_samples\":256,\"keys_per_query\":32,\"heads\":16,\"model\":\"Qwen3.6-35B-A3B captured layer3 QK\",\"inference_acceptance\":false}\n",
            static_cast<unsigned long long>(dots),static_cast<unsigned long long>(all_fast));
    } else throw std::runtime_error("use --selftest or --q7169 Q K");
    return total.accepted && !total.mismatches ? 0 : 2;
} catch(const std::exception& error) {std::fprintf(stderr,"row_certificate_error=%s\n",error.what());return 3;}
