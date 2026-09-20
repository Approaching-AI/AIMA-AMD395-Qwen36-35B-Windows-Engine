#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "sm121_mtp_query_math.h"

template<class T> std::vector<T> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() % sizeof(T)) throw std::runtime_error("input file");
    const size_t bytes = static_cast<size_t>(file.tellg());
    if (bytes > 128u*1024u*1024u) throw std::runtime_error("file bound");
    std::vector<T> values(bytes / sizeof(T));file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()),bytes);
    if (!file) throw std::runtime_error("short input"); return values;
}
float head_inverse(const float* values, const unsigned char* table) {
    std::array<float,64> lanes{}, next{};
    for (unsigned i=0;i<64u;++i) lanes[i]=qrt_sm121_q1::head_norm_lane_sumsq(values,i,false);
    for (unsigned mask=16u;mask;mask>>=1) {
        for(unsigned i=0;i<64u;++i) next[i]=qrt_sm121_q1::add(lanes[i],lanes[i^mask]);
        lanes=next;
    }
    const float warps[]={lanes[0],lanes[32]};return qrt_sm121_mtp::query_inverse(warps,table);
}
int main(int argc,char**argv) try {
    if(argc!=8) throw std::runtime_error("q_projection weights expected_norm expected_rope positions rsqrt rope");
    const auto input=read_file<uint16_t>(argv[1]),weights=read_file<uint16_t>(argv[2]);
    const auto expected_norm=read_file<uint16_t>(argv[3]),expected_rope=read_file<uint16_t>(argv[4]);
    const auto positions=read_file<uint32_t>(argv[5]);const auto table=read_file<unsigned char>(argv[6]);
    const auto rope=read_file<uint16_t>(argv[7]);const size_t rows=positions.size();
    if(!rows || rows>8192u || input.size()!=rows*8192u || weights.size()!=256u ||
       expected_norm.size()!=rows*4096u || expected_rope.size()!=expected_norm.size() ||
       rope.size()%64u || !qrt_sm121_rsqrt::valid_layout(table.data(),table.size()))
        throw std::runtime_error("shape or table");
    size_t norm_bad=0,rope_bad=0,first=expected_norm.size();
    for(size_t row=0;row<rows;++row) {
        if(positions[row]>=262144u || positions[row]>=rope.size()/64u) throw std::runtime_error("position");
        for(unsigned head=0;head<16u;++head) {
            const size_t source=row*8192u+head*512u,target=row*4096u+head*256u;
            float values[256];uint16_t normalized[256];
            for(unsigned i=0;i<256u;++i)values[i]=qrt_sm121_q1::widen(input[source+i]);
            const float rstd=head_inverse(values,table.data());
            for(unsigned i=0;i<256u;++i) normalized[i]=qrt_sm121_mtp::normalized(values[i],rstd,weights[i]);
            for(unsigned i=0;i<256u;++i) {
                const uint16_t rotated=qrt_sm121_mtp::key_rotated(normalized,i,rope.data()+size_t(positions[row])*64u);
                if(!std::isfinite(qrt_sm121_q1::widen(rotated)) || !std::isfinite(qrt_sm121_q1::widen(expected_rope[target+i])))
                    throw std::runtime_error("nonfinite boundary");
                const bool a=normalized[i]!=expected_norm[target+i],b=rotated!=expected_rope[target+i];
                norm_bad+=a;rope_bad+=b;if((a||b)&&first==expected_norm.size())first=target+i;
            }
        }
    }
    std::cout<<"{\"rows\":"<<rows<<",\"elements_per_boundary\":"<<expected_norm.size()
        <<",\"norm_bf16_mismatches\":"<<norm_bad<<",\"rope_bf16_mismatches\":"<<rope_bad
        <<",\"first_difference\":";
    if(first==expected_norm.size())std::cout<<"null";else std::cout<<first;
    std::cout<<",\"native_inference_acceptance\":false}\n";
    return norm_bad||rope_bad?1:0;
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 2; }
