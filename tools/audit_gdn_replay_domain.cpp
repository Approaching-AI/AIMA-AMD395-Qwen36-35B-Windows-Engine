#include "../native/providers/gdn/sm121_exp2_table.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr unsigned tokens=7169, chunks=113, segments=8;
using Flags=std::array<uint16_t,chunks*32>;
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
uint32_t bits(float x){uint32_t y;std::memcpy(&y,&x,4);return y;}
float from(uint32_t x){float y;std::memcpy(&y,&x,4);return y;}
uint16_t rounded(float x){
    uint32_t u=bits(x);
    if((u&0x7fffffffu)>0x7f800000u)return uint16_t((u|0x00400000u)>>16u);
    return uint16_t((u+0x7fffu+((u>>16u)&1u))>>16u);
}
template<class T>std::vector<T> read(const std::string& root,const char* name,size_t n){
    std::ifstream f(root+"/"+name,std::ios::binary|std::ios::ate);
    require(f && f.tellg()==std::streamoff(n*sizeof(T)),"incorrect artifact size");
    std::vector<T> a(n);f.seekg(0);f.read(reinterpret_cast<char*>(a.data()),n*sizeof(T));
    require(bool(f),"artifact read failed");return a;
}
bool eligible(uint16_t x,unsigned low,unsigned high){
    const unsigned magnitude=x&0x7fffu,exponent=magnitude>>7;
    return !magnitude || (exponent>=low && exponent<=high);
}
// A nonnegative finite BF16 value describes an inclusive superset of its
// original FP32 rounding bin. Multiplication by the nonnegative original
// decay is monotone. Preserve both gradual-underflow and flushed-zero
// possibilities: uncertain cells cannot reject or admit the candidate.
// 0 = all possibilities admitted, 1 = all rejected, 2 = unresolved.
unsigned residual_class(uint16_t value,float decay,unsigned low,unsigned high){
    const unsigned magnitude=value&0x7fffu;
    if(magnitude>=0x7f80u || !std::isfinite(decay) || decay<0.0f || decay>1.0f)return 2;
    const uint32_t center=magnitude<<16u;
    const uint32_t left=center>=0x8000u?center-0x8000u:0u,right=center+0x8000u;
    volatile float product_low=from(left)*decay,product_high=from(right)*decay;
    unsigned a=rounded(product_low)&0x7fffu,b=rounded(product_high)&0x7fffu;
    if((bits(product_low)&0x7f800000u)==0u || (bits(decay)&0x7f800000u)==0u)a=0u;
    require(a<=b,"nonmonotone residual interval");
    const unsigned first=low<<7u,last=(high<<7u)|127u;
    if(!b || (a>=first && b<=last))return 0;
    if((a>0u && b<first) || a>last)return 1;
    return 2;
}
unsigned pop(unsigned x){return unsigned(__builtin_popcount(x));}
std::array<uint16_t,segments*32> fold(const Flags& a){
    std::array<uint16_t,segments*32> out{};
    for(unsigned c=0;c<chunks;++c)for(unsigned h=0;h<32;++h)out[(c/16)*32+h]|=a[c*32+h];
    return out;
}
template<size_t N>unsigned count(const std::array<uint16_t,N>& a){unsigned n=0;for(auto x:a)n+=pop(x);return n;}
int main(int argc,char** argv)try{
    require(argc==2,"requires verified local capture directory");
    require(std::fesetround(FE_TONEAREST)==0,"rounding mode");
    // Independent checks of BF16 bin containment and source round trips.
    unsigned bin_checks=0;
    for(unsigned word=0;word<0x7f80u;++word){
        const uint32_t center=word<<16u,low=center>=0x8000u?center-0x8000u:0u,high=center+0x8000u;
        require(rounded(from(center))==word,"BF16 center round trip");
        for(uint32_t candidate:std::array<uint32_t,7>{low?low-1u:0u,low,low+1u,center,high-1u,high,high+1u}){
            if((rounded(from(candidate))&0x7fffu)==word)require(candidate>=low && candidate<=high,"positive bin containment");
            if((rounded(from(candidate|0x80000000u))&0x7fffu)==word)require(candidate>=low && candidate<=high,"negative bin containment");
            ++bin_checks;
        }
    }
    require(residual_class(0x3f80u,1.0f,95,159)==0,"normal endpoint class");
    require(residual_class(0x3f80u,0.0f,95,159)==0,"zero decay class");
    require(residual_class(0x3f80u,from(80u<<23u),95,159)==1,"tiny normal endpoint class");
    require(residual_class(0x7fc1u,1.0f,95,159)==2,"nonfinite uncertainty");
    require(residual_class(0x0001u,1.0f,95,159)==2,"denormal uncertainty");

    const std::string root=argv[1];
    const auto w=read<uint16_t>(root,"full-w-bf16.bin",size_t(tokens)*4096);
    const auto k=read<uint16_t>(root,"full-k-normalized-bf16.bin",size_t(tokens)*2048);
    const auto h=read<uint16_t>(root,"full-chunk-state-bf16.bin",size_t(chunks)*524288);
    const auto v=read<uint16_t>(root,"full-v-new-bf16.bin",size_t(tokens)*4096);
    const auto g=read<float>(root,"full-g-cumsum-f32.bin",size_t(tokens)*32);
    const auto table=read<unsigned char>(root,"sm121-exp2-negative-f490940d.bin",qrt_sm121_exp2::table_bytes);
    require(qrt_sm121_exp2::valid_layout(table.data(),table.size()),"original exponential table layout");
    std::vector<float> decay(size_t(tokens)*32);
    for(unsigned row=0;row<tokens;++row)for(unsigned head=0;head<32;++head){
        const unsigned last=std::min(tokens,((row/64)+1)*64)-1;
        volatile float difference=g[size_t(last)*32+head]-g[size_t(row)*32+head];
        volatile float argument=difference*1.4426950408889634074f;
        decay[size_t(row)*32+head]=qrt_sm121_exp2::evaluate(table.data(),argument);
        require(std::isfinite(decay[size_t(row)*32+head]) && decay[size_t(row)*32+head]>=0.0f && decay[size_t(row)*32+head]<=1.0f,"captured decay domain");
    }
    std::printf("{\"kind\":\"original_gb10_residual_domain_interval_audit\",\"bin_checks\":%u,\"tokens\":%u,\"domains\":[",bin_checks,tokens);
    unsigned domain_index=0;
    for(auto domain:std::array<std::array<unsigned,2>,2>{{{95,159},{64,190}}}){
        const unsigned low=domain[0],high=domain[1];Flags base{},bad{},unknown{};
        for(size_t i=0;i<w.size();++i)if(!eligible(w[i],low,high)){const size_t row=i/128;base[(row/32/64)*32+row%32]=0xffffu;}
        for(size_t i=0;i<k.size();++i)if(!eligible(k[i],low,high)){const size_t row=i/128,at=(row/16/64)*32+(row%16)*2;base[at]=base[at+1]=0xffffu;}
        for(size_t i=0;i<h.size();++i)if(!eligible(h[i],low,high)){const size_t column=i/128;base[column/128]|=uint16_t(1u<<((column%128)/8));}
        std::array<size_t,3> cells{};
        for(size_t i=0;i<v.size();++i){
            const size_t row=i/4096,head=(i/128)%32,column=i%128;
            const unsigned classification=residual_class(v[i],decay[row*32+head],low,high);++cells[classification];
            const unsigned at=unsigned((row/64)*32+head),mask=1u<<(column/8);
            if(classification==1)bad[at]|=uint16_t(mask);
            if(classification==2)unknown[at]|=uint16_t(mask);
        }
        Flags rejected{},unresolved{};
        for(unsigned i=0;i<base.size();++i){rejected[i]=base[i]|bad[i];unresolved[i]=unknown[i]&~rejected[i];}
        const auto base_segment=fold(base),bad_segment=fold(rejected),uncertain_segment=fold(unknown);
        unsigned full_base=0,full_bad=0,full_unknown=0,total_unknown=0;
        std::array<unsigned,segments> segment_bad{},segment_unknown{};
        for(unsigned s=0;s<segments;++s)for(unsigned head=0;head<32;++head){
            const unsigned at=s*32+head;
            segment_bad[s]+=pop(bad_segment[at]);
            segment_unknown[s]+=pop(uncertain_segment[at]&~bad_segment[at]);
            if(s<7){full_base+=pop(base_segment[at]);full_bad+=pop(bad_segment[at]);full_unknown+=pop(uncertain_segment[at]&~bad_segment[at]);}
            total_unknown+=pop(uncertain_segment[at]&~bad_segment[at]);
        }
        require(full_base==(domain_index?1008u:1126u),"prior operand-only audit differs");
        std::printf("%s{\"domain\":[%u,%u],\"residual_cells\":{\"admitted\":%zu,\"rejected\":%zu,\"unresolved\":%zu},\"base_chunk_ctas\":%u,\"residual_rejected_chunk_ctas\":%u,\"combined_rejected_chunk_ctas\":%u,\"combined_unresolved_chunk_ctas\":%u,\"full1024_segments\":{\"total_ctas\":3584,\"base_rejected_ctas\":%u,\"combined_rejected_ctas\":%u,\"unresolved_ctas\":%u,\"admitted_ctas\":%u},\"all_segments\":{\"total_ctas\":4096,\"rejected_ctas\":%u,\"unresolved_ctas\":%u},\"rejected_ctas_by_segment\":[",domain_index?",":"",low,high,cells[0],cells[1],cells[2],count(base),count(bad),count(rejected),count(unresolved),full_base,full_bad,full_unknown,3584-full_bad-full_unknown,count(bad_segment),total_unknown);
        for(unsigned s=0;s<segments;++s)std::printf("%s%u",s?",":"",segment_bad[s]);
        std::printf("],\"unresolved_ctas_by_segment\":[");for(unsigned s=0;s<segments;++s)std::printf("%s%u",s?",":"",segment_unknown[s]);std::printf("]}");
        ++domain_index;
    }
    std::printf("],\"uses_original_unrounded_residual\":false,\"uncertainty_includes_gradual_and_flushed_subnormals\":true,\"native_receipts_observed\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n");
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
