#include "../../native/providers/moe_accumulator/sm121_deferred_loss_bound.h"
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace loss=qrt_sm121_signed_loss_bound;
namespace deferred=qrt_sm121_deferred_loss_bound;
namespace base=qrt_sm121_coarse_projection_bound;
namespace scalar=base::scalar;
namespace original=qrt_q1_moe_hawkeye;
uint32_t rng=0x918395u;
uint32_t random_word(){rng^=rng<<13u;rng^=rng>>17u;rng^=rng<<5u;return rng;}
float f32(uint16_t x){return scalar::value(uint32_t(x)<<16u);}
float add(float a,float b){volatile float c=a+b;return c;}
struct Audit {
    size_t checkpoints=0u,old_selected=0u,new_selected=0u,old_final_selected=0u,new_final_selected=0u;
    size_t stable_positive=0u,stable_negative=0u,external=0u,unsupported=0u;
    double legacy_width=0.0,revised_width=0.0;
};
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void dot(const uint16_t* left,const uint16_t* right,unsigned width,unsigned mode,Audit& audit,
    const uint16_t* external=nullptr) {
    bool eligible=true;for(unsigned k=0u;k<width;++k)eligible &= base::eligible(left[k])&&base::eligible(right[k]);
    if(!eligible){
        // The native owner's existing row eligibility gate selects the entire
        // original dot for unsupported operands. Preserve that work here.
        ++audit.unsupported;++audit.old_final_selected;++audit.new_final_selected;
        const float expected=original::accumulate_bf16_hopper_blackwell(0.0f,left,right,width);
        if(external){check(scalar::bf16(expected)==*external,"unsupported original differs from external BF16");++audit.external;}
        return;
    }
    base::State legacy;loss::State directed;deferred::State delayed;original::Value carry{0u,-133,false};
    for(unsigned start=0u;start<width;start+=64u){
        const unsigned length=std::min(64u,width-start);float partial=0.0f,absolute=0.0f;
        for(unsigned group=start;group<start+length;group+=16u){
            original::Value terms[17];terms[0]=carry;double exact=0.0,l1=0.0;
            for(unsigned k=0u;k<16u;++k){
                const double p=double(f32(left[group+k]))*double(f32(right[group+k]));
                exact+=p;l1+=std::abs(p);
                terms[k+1u]=original::multiply_bf16(left[group+k],right[group+k],-133);
            }
            carry=original::group_sum<26,-133>(terms,17u);
            // Host RN products are a diagnostic producer. The two injected
            // arms exercise both directions within the inherited native
            // envelope; they do not execute or characterize native WMMA.
            const double shift=mode==0u?0.0:(mode==1u?0.75:-0.75)*0x1p-19*l1;
            const float signed_value=float(exact+shift),absolute_value=float(l1-shift);
            check(std::abs(double(signed_value)-exact)<=0x1p-19*l1,"signed synthetic native premise");
            check(std::abs(double(absolute_value)-l1)<=0x1p-19*l1,"absolute synthetic native premise");
            partial=add(partial,signed_value);absolute=add(absolute,absolute_value);
        }
        const auto a=loss::summarize(left+start,length),b=loss::summarize(right+start,length);
        check(a.valid&&b.valid,"capture domain");
        unsigned sign=0u;directed=loss::advance(directed,partial,absolute,a,b,&sign);
        legacy=base::advance<4u>(legacy,partial,absolute);
        delayed=deferred::advance(delayed,partial,absolute,loss::counts(a,b));
        const auto envelope=deferred::finalize(delayed,(start+length+63u)/64u);
        check(scalar::bits(envelope.center)==scalar::bits(directed.center),"deferred center changed");
        check(envelope.lower_error>=directed.lower_error && envelope.upper_error>=directed.upper_error,"deferred envelope narrower than original directed bound");
        const float expected=original::value_to_float(carry);
        check(scalar::bits(directed.center)==scalar::bits(legacy.center),"changed diagnostic center");
        check(double(directed.center)-double(expected)<=double(directed.lower_error),"lower undercoverage");
        check(double(expected)-double(directed.center)<=double(directed.upper_error),"upper undercoverage");
        check(directed.lower_error<=legacy.error&&directed.upper_error<=legacy.error,"widened legacy bound");
        const bool old_accept=base::certified(legacy),new_accept=loss::certified(envelope);
        check(double(envelope.center)-double(expected)<=double(envelope.lower_error) && double(expected)-double(envelope.center)<=double(envelope.upper_error),"deferred canonical undercoverage");
        if(new_accept)check(scalar::bf16(expected)==scalar::bf16(envelope.center),"false certificate");
        ++audit.checkpoints;audit.old_selected+=!old_accept;audit.new_selected+=!new_accept;
        audit.stable_positive+=sign==1u;audit.stable_negative+=sign==2u;
        // Dimensionless widths, with zero-input blocks omitted from ratios.
        if(absolute>0.0f){audit.legacy_width+=double(2.0f*legacy.error)/double(absolute);
            audit.revised_width+=(double(envelope.lower_error)+double(envelope.upper_error))/double(absolute);}
        if(start+length==width){audit.old_final_selected+=!old_accept;audit.new_final_selected+=!new_accept;}
    }
    if(external){check(scalar::bf16(original::value_to_float(carry))==*external,"original differs from external BF16");++audit.external;}
}
void print(const char* kind,const Audit& a,unsigned samples,unsigned width,unsigned modes) {
    std::printf("{\"kind\":\"%s\",\"samples\":%u,\"width\":%u,\"native_error_modes\":%u,\"checkpoints\":%zu,\"old_selected_prefixes\":%zu,\"new_selected_prefixes\":%zu,\"old_final_selected\":%zu,\"new_final_selected\":%zu,\"unsupported_dots\":%zu,\"stable_positive_blocks\":%zu,\"stable_negative_blocks\":%zu,\"normalized_width_ratio\":%.17g,\"external_bf16_checks\":%zu,\"undercoverage\":0,\"false_certificates\":0,\"native_matrix_executed\":false,\"native_error_coefficient\":%.17g,\"hardware_error_bound_proven\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
        kind,samples,width,modes,a.checkpoints,a.old_selected,a.new_selected,a.old_final_selected,a.new_final_selected,a.unsupported,a.stable_positive,a.stable_negative,a.revised_width/a.legacy_width,a.external,double(0x1p-19f));
}
void generated(){
    std::array<uint16_t,8192u> left{},right{};
    for(unsigned x=0u;x<65536u;++x){left[0]=uint16_t(x);const auto s=loss::summarize(left.data(),1u);
        check(s.valid==base::eligible(uint16_t(x)),"encoding domain");
        check(s.nonzero==uint64_t(bool(x&0x7fffu))&&s.negative==uint64_t(x>>15u),"encoding metadata");}
    for(unsigned mask=0u;mask<65536u;++mask){
        for(unsigned i=0u;i<16u;++i){left[i]=mask&(1u<<i)?uint16_t(0x3f80u|((i%3u==0u)?0x8000u:0u)):0x8000u;
            right[i]=i%5u?uint16_t(0x3f81u|((i&1u)<<15u)):0u;}
        const auto c=loss::counts(loss::summarize(left.data(),16u),loss::summarize(right.data(),16u));
        unsigned positive=0u,negative=0u;for(unsigned i=0u;i<16u;++i)if((left[i]&0x7fffu)&&(right[i]&0x7fffu))
            ((left[i]^right[i])&0x8000u)?++negative:++positive;
        check(c.positive==positive&&c.negative==negative,"mask sign counts");
    }
    Audit audit;
    constexpr unsigned samples=8192u;
    constexpr unsigned widths[]={16u,64u,80u,272u,512u,1024u,2048u,4096u,8192u};
    for(unsigned sample=0u;sample<samples;++sample){
        const unsigned width=widths[sample%9u];
        const unsigned ae=80u+random_word()%87u,be=80u+random_word()%87u;
        for(unsigned i=0u;i<width;++i){
            left[i]=uint16_t(((ae+random_word()%9u)<<7u)|(random_word()&0x807fu));
            right[i]=uint16_t(((be+random_word()%9u)<<7u)|(random_word()&0x807fu));
            if(sample%7u==0u){left[i]=uint16_t(0x3f81u|((i&1u)<<15u));right[i]=0x3f85u;}
            if(sample%11u==0u){left[i]&=0x7fffu;right[i]=uint16_t(0x3f81u|((sample&1u)<<15u));}
            if(sample%13u==0u){left[i]=uint16_t(0x3fffu|((i%257u<128u)?0x8000u:0u));right[i]=0x3fffu;}
            if(sample%17u==0u){left[i]=uint16_t(((i&1u?80u:174u)<<7u)|(random_word()&0x807fu));right[i]=0x3f80u;}
            if(sample%19u==0u&&i%3u==0u)left[i]=uint16_t((i&1u)<<15u);
            if(sample%23u==0u)left[i]=0u;
        }
        for(unsigned mode=0u;mode<3u;++mode)dot(left.data(),right.data(),width,mode,audit);
    }
    for(float bad:{-1.0f,scalar::infinity(),-scalar::infinity(),std::numeric_limits<float>::quiet_NaN()}){
        check(!loss::certified(loss::advance({},0.0f,bad,{32u,32u})),"bad absolute accepted");
        check(!loss::certified(loss::advance({0.0f,bad,0.0f},0.0f,0.0f,{32u,32u})),"bad entry accepted");
    }
    check(!loss::certified(loss::advance({},0.0f,0.0f,{65u,0u})),"bad count accepted");
    check(!loss::summarize(nullptr).valid&&!loss::summarize(left.data(),65u).valid,"bad span");
    for(unsigned n:{0u,129u,~0u})check(!loss::certified(deferred::finalize({},n)),"bad block count accepted");
    for(float bad:{-1.0f,scalar::infinity(),std::numeric_limits<float>::quiet_NaN()})
        check(!loss::certified(deferred::finalize({0.0f,bad,0.0f},1u)),"bad deferred metadata accepted");
    check(!deferred::width_supported(8208u)&&!deferred::width_supported(15u)&&!deferred::width_supported(0u),"bad width accepted");
    print("deferred_loss_bound_generated",audit,samples,0u,3u);
    std::puts("{\"kind\":\"deferred_loss_generated_widths\",\"widths\":[16,64,80,272,512,1024,2048,4096,8192],\"width_field_zero_means_mixed\":true,\"old_directional_envelopes_contained\":true}");
    std::puts("{\"kind\":\"deferred_loss_metadata\",\"bf16_encodings\":65536,\"support_masks\":65536,\"all_pass\":true}");
}
std::vector<uint16_t> read(const char* path,size_t count){
    std::ifstream f(path,std::ios::binary|std::ios::ate);check(bool(f)&&f.tellg()==std::streamoff(count*2u),"tensor span");
    std::vector<uint16_t> out(count);f.seekg(0);f.read(reinterpret_cast<char*>(out.data()),count*2u);check(bool(f),"tensor read");return out;
}
int main(int argc,char** argv){try{
    if(argc==1){generated();return 0;}
    check(argc==6,"input weights reference-or-dash rows width");
    const unsigned rows=std::stoul(argv[4]),width=std::stoul(argv[5]);
    check(rows&&width&&!(width%16u)&&width<=8192u,"shape");
    const auto input=read(argv[1],size_t(7169u)*width),weight=read(argv[2],size_t(rows)*width);
    const bool external=std::string(argv[3])!="-";
    const auto golden=external?read(argv[3],size_t(8192u)*rows):std::vector<uint16_t>{};
    constexpr unsigned samples=65536u;const uint64_t cells=uint64_t(8192u)*rows;
    Audit audits[3];
    for(unsigned sample=0u;sample<samples;++sample){
        const uint32_t cell=uint32_t((uint64_t(sample)*2654435761ull+1013904223ull)%cells);
        const unsigned token=cell/rows,row=cell%rows,source=token<7169u?token:token-7169u;
        for(unsigned mode=0u;mode<3u;++mode)dot(input.data()+size_t(source)*width,weight.data()+size_t(row)*width,width,mode,audits[mode],external?&golden[cell]:nullptr);
    }
    for(unsigned mode=0u;mode<3u;++mode){std::printf("{\"kind\":\"deferred_loss_capture_mode\",\"mode\":%u}\n",mode);print("deferred_loss_bound_capture",audits[mode],samples,width,1u);}
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
