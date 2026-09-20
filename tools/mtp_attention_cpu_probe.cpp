#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "native/providers/gdn/sm121_q1_math.h"
#include "native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"

template<class T> std::vector<T> read(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() % sizeof(T) || file.tellg() > 300u*1024u*1024u)
        throw std::runtime_error("input size");
    const size_t bytes = static_cast<size_t>(file.tellg());
    std::vector<T> values(bytes / sizeof(T)); file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!file) throw std::runtime_error("short input"); return values;
}
int main(int argc, char** argv) try {
    if (argc != 9) throw std::runtime_error("q k v expected exp2 rcp output active_tokens");
    const auto q = read<uint16_t>(argv[1]), k = read<uint16_t>(argv[2]), v = read<uint16_t>(argv[3]);
    const auto expected = read<uint16_t>(argv[4]);
    const auto exp2 = read<unsigned char>(argv[5]), rcp = read<unsigned char>(argv[6]);
    if (q.size()!=4096u || expected.size()!=4096u || k.size()%512u || k.size()!=v.size() ||
        k.size()/512u>262144u || !qrt_sm121_exp2::valid_layout(exp2.data(),exp2.size()) ||
        !qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size())) throw std::runtime_error("shape or table");
    using namespace qrt_sm121_q1;
    namespace mma=qrt_q1_moe_hawkeye;
    const auto exponential=[&](float delta) {
        return qrt_sm121_exp2::evaluate(exp2.data(),multiply(delta,1.4426950408889634074f));
    };
    size_t used=0; const std::string extent(argv[8]);
    const unsigned long parsed=std::stoul(extent,&used);
    if (used!=extent.size() || !parsed || parsed>k.size()/512u) throw std::runtime_error("active extent");
    const unsigned tokens=static_cast<unsigned>(parsed);
    std::vector<uint16_t> actual(4096u); std::vector<size_t> per_head(16u);
    size_t mismatches=0,first=4096u,nonfinite=0; double max_error=0;
    for (unsigned head=0;head<16u;++head) {
        float accumulator[256]{}; float running_max=-INFINITY,running_sum=1.0f;
        for (unsigned tile=0;tile<tokens;tile+=32u) {
            float scores[32],probability[32]; uint16_t p[32];
            float next_max=running_max;
            for (unsigned item=0;item<32u;++item) {
                const unsigned token=tile+item;
                scores[item]=token<tokens ? multiply(mma::dot_bf16_hopper_blackwell(
                    q.data()+head*256u,k.data()+size_t(token)*512u+(head/8u)*256u,256u),0.0625f) : -INFINITY;
                next_max=std::fmax(next_max,scores[item]);
            }
            const float alpha=exponential(add(running_max,-next_max));
            for (unsigned item=0;item<32u;++item) {
                probability[item]=tile+item<tokens?exponential(add(scores[item],-next_max)):0.0f;
                p[item]=bf16(probability[item]);
            }
            unsigned reduced_bits=0;
            for (unsigned offset:{1u,4u,2u,16u,8u}) {
                reduced_bits|=offset;
                for (unsigned item=0;item<32u;++item)
                    if (!(item&reduced_bits)) probability[item]=add(probability[item],probability[item+offset]);
            }
            for (unsigned channel=0;channel<256u;++channel) {
                float partial=multiply(accumulator[channel],alpha);
                for (unsigned begin=0;begin<32u;begin+=16u) {
                    uint16_t values[16];
                    for (unsigned item=0;item<16u;++item) {
                        const unsigned token=tile+begin+item;
                        values[item]=token<tokens?v[size_t(token)*512u+(head/8u)*256u+channel]:0u;
                    }
                    partial=mma::accumulate_bf16_impl<26,16,-133>(partial,p+begin,values,16u);
                }
                accumulator[channel]=partial;
            }
            running_sum=std::fma(running_sum,alpha,probability[0]);running_max=next_max;
        }
        const float inverse=qrt_sm121_attention_rcp::evaluate(rcp.data(),running_sum);
        for (unsigned channel=0;channel<256u;++channel) {
            const size_t index=head*256u+channel;const float value=multiply(accumulator[channel],inverse);
            nonfinite+=!std::isfinite(value);actual[index]=bf16(value);
            max_error=std::max(max_error,std::abs(double(widen(actual[index]))-widen(expected[index])));
            if (actual[index]!=expected[index]) {++mismatches;++per_head[head];if(first==4096u)first=index;}
        }
    }
    std::ofstream output(argv[7],std::ios::binary);output.write(reinterpret_cast<const char*>(actual.data()),8192u);
    if(!output)throw std::runtime_error("output");
    std::cout<<"{\"tokens\":"<<tokens<<",\"elements\":4096,\"bf16_mismatches\":"<<mismatches
        <<",\"nonfinite\":"<<nonfinite<<",\"maximum_absolute_error\":"<<max_error<<",\"first_difference\":";
    if(mismatches)std::cout<<first;else std::cout<<"null";
    std::cout<<",\"per_head_mismatches\":[";
    for(unsigned i=0;i<16u;++i){if(i)std::cout<<',';std::cout<<per_head[i];}
    std::cout<<"],\"native_inference_acceptance\":false}\n";
    return nonfinite?1:0;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 2;}
