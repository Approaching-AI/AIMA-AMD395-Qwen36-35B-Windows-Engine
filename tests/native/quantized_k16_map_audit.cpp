#include "checked_quantized_affine_map.h"
#include "../../native/providers/moe_accumulator/sm121_dyadic_carry_scan.h"
#include "float_alignment_cases.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

// Host-only semantic/coverage investigation. The predictor uses ordinary
// FP32 tree sums, never original accumulator traces or GB10 outputs. Actual
// GPU prediction, parallel scan and complete product timing are unqualified.
namespace {
namespace maps = qrt_quantized_affine_map;
namespace modular = qrt_sm121_dyadic_carry_scan;
namespace original = qrt_q1_moe_hawkeye;
namespace cases = qrt_float_alignment_cases;
using Wide = __int128;
enum Reason { accepted, operand, predictor, representation, composition, domain, reasons };
struct Step {
    std::array<original::Value,16> products;
    int product_max=-133, alignment=-133, final_exponent=-133;
    bool identity=true, negative_input=false, negative_sum=false;
    unsigned alignment_shift=0u, final_shift=0u;
    int64_t terms=0;
    maps::Map map{0,0,0};
};
uint32_t bits(float x) { uint32_t u;std::memcpy(&u,&x,4u);return u; }
float from_bf16(uint16_t x) { uint32_t u=uint32_t(x)<<16u;float f;std::memcpy(&f,&u,4u);return f; }
float add_float(float a,float b) { volatile float result=a+b;return result; }
bool finite_normal_or_zero(float x) {
    const uint32_t a=bits(x)&0x7fffffffu;
    return !a || (a>=0x00800000u && a<0x7f800000u);
}
int exponent(float x) {
    const uint32_t a=bits(x)&0x7fffffffu;
    return a?int(a>>23u)-127:-133;
}
bool integer_bits(int64_t value,int unit,uint32_t* output) {
    if(!value){*output=0u;return true;}
    const uint64_t magnitude=value<0?uint64_t(0)-uint64_t(value):uint64_t(value);
    const unsigned width=original::bit_width_u64(magnitude);
    const int e=int(width)-1+unit;
    if(e < -126 || e > 127)return false;
    uint64_t significand;
    if(width>24u){
        const unsigned shift=width-24u;
        if(magnitude&((uint64_t(1)<<shift)-1u))return false;
        significand=magnitude>>shift;
    }else significand=magnitude<<(24u-width);
    *output=(value<0?0x80000000u:0u)|(uint32_t(e+127)<<23u)|(uint32_t(significand)&0x7fffffu);
    return true;
}
bool narrow(Wide value,int64_t* output) {
    if(value<INT64_MIN || value>INT64_MAX)return false;
    *output=int64_t(value);return true;
}
Reason candidate(const uint16_t* left,const uint16_t* right,unsigned count,
    std::vector<uint32_t>* traces,unsigned sabotage=0u) {
    if(!count || count%16u)throw std::runtime_error("invalid dot extent");
    std::vector<Step> steps(count/16u);
    float approximate=0.0f;int unit=1000;
    for(unsigned g=0u;g<steps.size();++g){
        auto& s=steps[g];float products[16];
        s.negative_input=std::signbit(approximate);
        for(unsigned i=0u;i<16u;++i){
            const uint16_t a=left[g*16u+i],b=right[g*16u+i];
            if(!qrt_sm121_float_alignment::eligible(a)||!qrt_sm121_float_alignment::eligible(b))return operand;
            s.products[i]=original::multiply_bf16(a,b,-133);
            s.identity &= s.products[i].significand==0u;
            s.product_max=std::max(s.product_max,int(s.products[i].exponent));
            products[i]=from_bf16(a)*from_bf16(b);
        }
        if(s.identity)continue;
        if(!finite_normal_or_zero(approximate))return predictor;
        s.alignment=std::max(s.product_max,exponent(approximate));
        for(unsigned stride=1u;stride<16u;stride*=2u)
            for(unsigned i=0u;i<16u;i+=2u*stride)
                products[i]=add_float(products[i],products[i+stride]);
        approximate=add_float(approximate,products[0]);
        if(!finite_normal_or_zero(approximate)||approximate==0.0f)return predictor;
        s.final_exponent=exponent(approximate);s.negative_sum=std::signbit(approximate);
        if(sabotage && g==0u){if(sabotage==1u)++s.alignment;else s.negative_sum=!s.negative_sum;}
        unit=std::min(unit,s.alignment-25);
    }
    if(unit==1000)unit=0;
    for(auto& s:steps){
        if(s.identity)continue;
        const int shift=s.alignment-25-unit,final_shift=std::max(0,s.final_exponent-23-unit);
        if(shift<0||shift>60||final_shift>60)return representation;
        s.alignment_shift=unsigned(shift);s.final_shift=unsigned(final_shift);
        int64_t sum=0;
        for(const auto& p:s.products){
            const int drop=s.alignment-p.exponent;
            if(drop<0)return representation;
            const uint32_t term=drop>=32?0u:(p.significand<<2u)>>unsigned(drop);
            sum+=p.negative?-int64_t(term):int64_t(term);
        }
        if(!narrow(Wide(sum)*(Wide(1)<<shift),&s.terms) ||
            !maps::rounding_step(s.alignment_shift,s.final_shift,s.negative_input,s.negative_sum,s.terms,&s.map))
            return representation;
    }
    // This sequential host calculation specifies an associative prefix scan.
    // It evaluates no original carry recurrence and uses no reference trace.
    std::vector<int64_t> prefixes(steps.size()+1u,0);
    maps::Map combined{0,0,0};
    modular::Function modular_combined;
    for(unsigned g=0u;g<steps.size();++g){
        maps::Map next;
        if(!maps::compose(combined,steps[g].map,&next)||!maps::evaluate(next,0,&prefixes[g+1u]))return composition;
        combined=next;
        maps::Map canonical;
        if(!maps::canonicalize(steps[g].map,&canonical))return composition;
        modular_combined=modular::compose(modular_combined,
            {uint64_t(canonical.before),uint64_t(canonical.after),canonical.shift});
        if(modular::evaluate(modular_combined,0u)!=uint64_t(prefixes[g+1u]))
            throw std::runtime_error("signed/modular prefix composition differs");
    }
    std::vector<uint32_t> result(steps.size());
    for(unsigned g=0u;g<steps.size();++g){
        const auto& s=steps[g];const int64_t before=prefixes[g],after=prefixes[g+1u];
        uint32_t before_bits;
        if(!integer_bits(before,unit,&before_bits)||!integer_bits(after,unit,&result[g]))return domain;
        if(s.identity){if(before!=after)return domain;continue;}
        const int carry_exponent=before?int((before_bits&0x7fffffffu)>>23u)-127:-133;
        if(std::max(s.product_max,carry_exponent)!=s.alignment ||
            (before && (before<0)!=s.negative_input))return domain;
        const int64_t grid=int64_t(1)<<s.alignment_shift;
        int64_t total;
        if(!narrow(Wide(before/grid)*grid+s.terms,&total))return domain;
        if(total && (total<0)!=s.negative_sum)return domain;
        if(total){
            const uint64_t magnitude=total<0?uint64_t(0)-uint64_t(total):uint64_t(total);
            const int actual_exponent=int(original::bit_width_u64(magnitude))-1+unit;
            if(actual_exponent < -126 || actual_exponent > 127 ||
                std::max(0,actual_exponent-23-unit)!=int(s.final_shift))return domain;
        }
        const int64_t final_grid=int64_t(1)<<s.final_shift;
        if((total/final_grid)*final_grid!=after)return domain;
    }
    *traces=std::move(result);return accepted;
}
struct Totals {
    uint64_t counts[reasons]{},admitted_groups=0,reference_groups=0;
    void check(const uint16_t* a,const uint16_t* b,unsigned count,unsigned sabotage=0u){
        std::vector<uint32_t> mapped;const auto why=candidate(a,b,count,&mapped,sabotage);
        ++counts[why];original::Value carry{0u,-133,false};
        for(unsigned g=0u;g<count/16u;++g){
            original::Value terms[17];terms[0]=carry;
            for(unsigned i=0u;i<16u;++i)terms[i+1u]=original::multiply_bf16(a[g*16u+i],b[g*16u+i],-133);
            carry=original::group_sum<26,-133>(terms,17u);++reference_groups;
            if(why==accepted){
                ++admitted_groups;
                if(mapped[g]!=cases::output_bits(carry)){
                    std::fprintf(stderr,"DIFF group=%u expected=%08x actual=%08x sabotage=%u\n",g,cases::output_bits(carry),mapped[g],sabotage);
                    throw std::runtime_error("accepted map changed original K16 boundary");
                }
            }
        }
    }
    void report(const char* name){
        std::printf("{\"kind\":\"quantized_k16_map_audit\",\"dataset\":\"%s\",\"accepted_dots\":%llu,\"operand_reject\":%llu,\"predictor_reject\":%llu,\"representation_reject\":%llu,\"composition_reject\":%llu,\"domain_reject\":%llu,\"admitted_original_groups\":%llu,\"reference_groups\":%llu,\"mismatches\":0,\"host_only\":true,\"native_matrix_predictor\":false,\"performance_acceptance\":false,\"inference_acceptance\":false}\n",
            name,(unsigned long long)counts[0],(unsigned long long)counts[1],(unsigned long long)counts[2],
            (unsigned long long)counts[3],(unsigned long long)counts[4],(unsigned long long)counts[5],
            (unsigned long long)admitted_groups,(unsigned long long)reference_groups);
    }
};
std::vector<uint16_t> read(const char* path,size_t words){
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streamoff(words*2u))throw std::runtime_error("capture size");
    std::vector<uint16_t> v(words);f.seekg(0);
    if(!f.read(reinterpret_cast<char*>(v.data()),std::streamsize(words*2u)))throw std::runtime_error("capture read");
    return v;
}
} // namespace
int main(int argc,char** argv)try{
    if(argc==4 && !std::strcmp(argv[1],"--capture")){
        const auto q=read(argv[2],size_t(7169)*16u*256u),k=read(argv[3],size_t(7169)*2u*256u);
        Totals total;
        for(unsigned sample=0u;sample<65536u;++sample){
            const unsigned token=(sample*997u+sample/16u)%7169u,head=sample%16u;
            const unsigned key=cases::random(sample+0x8192395u)%(token+1u);
            total.check(q.data()+(size_t(token)*16u+head)*256u,k.data()+(size_t(key)*2u+head/8u)*256u,256u);
        }
        total.report("captured_q7169_layer3_sampled_65536_dots");return 0;
    }
    if(argc!=1)throw std::runtime_error("use no arguments or --capture Q K");
    Totals total,corrupt;
    for(unsigned row=0u;row<65536u;++row){
        uint16_t a[256],b[256];
        for(unsigned g=0u;g<16u;++g)for(unsigned i=0u;i<16u;++i){
            const auto p=cases::input(row,g,i);a[g*16u+i]=p.left;b[g*16u+i]=p.right;
        }
        total.check(a,b,256u);
        if(row%8u<2u){corrupt.check(a,b,256u,1u);corrupt.check(a,b,256u,2u);}
    }
    total.report("generated_65536_dots");corrupt.report("deliberately_wrong_first_grid_or_sign");
    if(!total.counts[accepted]||!total.counts[domain]||corrupt.counts[accepted])return 2;
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
