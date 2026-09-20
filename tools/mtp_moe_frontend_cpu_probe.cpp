#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "sm121_shared_gate.h"
#include "sm121_router_exp.h"
#include "q1_moe_hawkeye_bf16_accumulator.h"
#include "native/providers/gdn/sm121_mtp_moe_math.h"

float widen(uint16_t x) { return qrt_sm121_shared_gate::value(x); }
uint32_t bits(float x) { uint32_t u; std::memcpy(&u,&x,4); return u; }
uint16_t bf16(float x) { const uint32_t u=bits(x); return uint16_t((u+0x7fffu+((u>>16)&1u))>>16); }
template<class T> std::vector<T> read(const std::string& path) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()<=0 || file.tellg()>128u*1024u*1024u || file.tellg()%sizeof(T))
        throw std::runtime_error("file extent: "+path);
    std::vector<T> data(size_t(file.tellg())/sizeof(T)); file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()),data.size()*sizeof(T));
    if(!file) throw std::runtime_error("short read"); return data;
}
struct Check {
    size_t elements=0, mismatches=0;
    uint32_t first_actual=0, first_expected=0;
    size_t first=0;
    void add(uint32_t actual,uint32_t expected) {
        if(actual!=expected) {
            if(!mismatches){first=elements;first_actual=actual;first_expected=expected;}
            ++mismatches;
        }
        ++elements;
    }
};
int main(int argc,char** argv) try {
    if(argc!=6)throw std::runtime_error("directory shared_gate_weight silu sigmoid router_exp");
    const std::string root=argv[1];
    auto input=read<uint16_t>(root+"/input.bin");
    const size_t rows=input.size()/2048;
    if(!rows||rows>128||input.size()!=rows*2048)throw std::runtime_error("input shape");
    auto bf=[&](const std::string& label,size_t width){auto v=read<uint16_t>(root+"/"+label+".bin");
        if(v.size()!=rows*width)throw std::runtime_error("shape: "+label);
        for(auto x:v)if(!std::isfinite(widen(x)))throw std::runtime_error("nonfinite BF16");return v;};
    auto gate=bf("shared-gate",1),gu=bf("shared-gate-up",1024),act=bf("shared-activated",512),
        down=bf("shared-down",2048),shared=bf("shared",2048),rgu=bf("routed-gate-up",8192),
        ract=bf("routed-activated",4096),weighted=bf("routed-weighted",16384),
        routed=bf("expert-part-1",2048),moe=bf("moe-output",2048),router=bf("router",256),
        fusion=bf("fusion",2048),attention=bf("attention-output",2048),residual=bf("final-residual",2048);
    const auto wg=read<uint16_t>(argv[2]),silu=read<uint16_t>(argv[3]),sigmoid=read<uint16_t>(argv[4]);
    const auto fraction=read<uint32_t>(argv[5]),ids=read<uint32_t>(root+"/topk-ids.bin");
    const auto weights=read<float>(root+"/topk-weights.bin");
    if(wg.size()!=2048||silu.size()!=65536+12||sigmoid.size()!=65536||fraction.size()!=(1u<<23)||
       ids.size()!=rows*8||weights.size()!=rows*8)throw std::runtime_error("table/weight shape");
    std::map<std::string,Check> checks;
    for(size_t t=0;t<rows;++t) {
        float lanes[16];
        for(unsigned l=0;l<16;++l)lanes[l]=qrt_sm121_shared_gate::lane_dot(input.data()+t*2048,wg.data(),l);
        for(unsigned offset=8;offset;offset/=2)for(unsigned l=0;l<offset;++l)lanes[l]=lanes[l]+lanes[l+offset];
        checks["shared_gate_lane16"].add(bf16(lanes[0]),gate[t]);
        checks["shared_gate_k16_diagnostic"].add(bf16(qrt_q1_moe_hawkeye::dot_bf16_hopper_blackwell(
            input.data()+t*2048,wg.data(),2048)),gate[t]);
        for(unsigned i=0;i<512;++i)checks["shared_activation"].add(
            qrt_sm121_mtp::moe_activate(gu[t*1024+i],gu[t*1024+512+i],silu.data()+12),act[t*512+i]);
        for(unsigned route=0;route<8;++route)for(unsigned i=0;i<512;++i) {
            const size_t base=t*8192+route*1024;
            checks["routed_activation"].add(qrt_sm121_mtp::moe_activate(rgu[base+i],rgu[base+512+i],silu.data()+12),
                ract[t*4096+route*512+i]);
        }
        for(unsigned i=0;i<2048;++i) {
            checks["shared_gate_product"].add(qrt_sm121_mtp::moe_shared_product(gate[t],down[t*2048+i],sigmoid.data()),shared[t*2048+i]);
            checks["routed_sum"].add(qrt_sm121_mtp::moe_routed_sum(weighted.data()+t*16384,i),routed[t*2048+i]);
            checks["moe_sum"].add(qrt_sm121_mtp::moe_output(shared[t*2048+i],routed[t*2048+i]),moe[t*2048+i]);
            checks["attention_residual"].add(bf16(widen(fusion[t*2048+i])+widen(attention[t*2048+i])),residual[t*2048+i]);
        }
        uint32_t selected_ids[8];float selected_weights[8];
        if(!qrt_sm121_mtp::moe_route(router.data()+t*256,fraction.data(),selected_ids,selected_weights))
            throw std::runtime_error("invalid router operands");
        for(unsigned k=0;k<8;++k) {
            checks["router_ids"].add(selected_ids[k],ids[t*8+k]);
            checks["router_weight_f32"].add(bits(selected_weights[k]),bits(weights[t*8+k]));
        }
    }
    std::cout<<"{\"rows\":"<<rows<<",\"checks\":{";
    bool comma=false;size_t mismatches=0;
    for(const auto& entry:checks) {
        if(comma)std::cout<<',';comma=true;const auto& c=entry.second;
        std::cout<<'"'<<entry.first<<"\":{\"elements\":"<<c.elements<<",\"mismatches\":"<<c.mismatches<<",\"first_difference\":";
        if(c.mismatches)std::cout<<"{\"index\":"<<c.first<<",\"actual_bits\":"<<c.first_actual<<",\"expected_bits\":"<<c.first_expected<<'}';
        else std::cout<<"null";
        std::cout<<'}';if(entry.first!="shared_gate_k16_diagnostic")mismatches+=c.mismatches;
    }
    std::cout<<"},\"inference_acceptance\":false,\"performance_acceptance\":false}\n";
    return mismatches?1:0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
