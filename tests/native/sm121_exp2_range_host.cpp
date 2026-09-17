#include "../../native/providers/gdn/sm121_exp2_interpolated.h"
#include "../../native/providers/gdn/sm121_exp2_reduced_delta.h"
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

// Host-only complete-domain audit of a possible table representation. It
// compares the pinned SM121 source with itself after exact integer range
// reduction. No native EXP periodicity or performance is assumed here.
int main(int argc,char** argv)try{
    namespace source=qrt_sm121_exp2_interpolated;
    if(argc!=2)throw std::runtime_error("requires verified interpolated EXP table");
    std::ifstream f(argv[1],std::ios::binary|std::ios::ate);
    if(!f || f.tellg()!=std::streamoff(source::table_bytes))throw std::runtime_error("source span");
    std::vector<unsigned char> table(source::table_bytes);f.seekg(0);
    if(!f.read(reinterpret_cast<char*>(table.data()),std::streamsize(table.size())) ||
        !source::valid_layout(table.data(),table.size()))throw std::runtime_error("source layout");
    uint64_t checked=0u,mismatches=0u;uint32_t maximum=0u;
    for(uint32_t magnitude=0x3f800000u;magnitude<0x42fc0000u;++magnitude){
        const unsigned exponent=(magnitude>>23u)-127u;
        const uint32_t mantissa=0x800000u|(magnitude&0x7fffffu);
        const uint32_t integral=mantissa>>(23u-exponent);
        const uint32_t fraction=(mantissa<<exponent)&0x7fffffu;
        if(!qrt_sm121_exp2_reduced_delta::admitted(magnitude|0x80000000u) ||
            qrt_sm121_exp2_reduced_delta::admitted(magnitude) ||
            qrt_sm121_exp2_reduced_delta::fractional_index(magnitude)!=fraction)
            throw std::runtime_error("reduced domain/index mismatch");
        const uint32_t base=source::decode(table.data(),(0x3f800000u|fraction)-source::begin);
        const uint32_t shift=(integral-1u)<<23u;
        if(base<shift+0x00800000u)throw std::runtime_error("non-normal reduced result");
        const uint32_t candidate=base-shift;
        const uint32_t expected=source::decode(table.data(),magnitude-source::begin);
        ++checked;
        if(candidate!=expected){
            if(mismatches<16u)std::printf("{\"kind\":\"exp2_range_counterexample\",\"input_bits\":%u,\"base_input_bits\":%u,\"integer_part\":%u,\"candidate_bits\":%u,\"expected_bits\":%u}\n",magnitude|0x80000000u,0xbf800000u|fraction,integral,candidate,expected);
            ++mismatches;const uint32_t difference=candidate>expected?candidate-expected:expected-candidate;
            if(difference>maximum)maximum=difference;
        }
    }
    std::printf("{\"kind\":\"sm121_exp2_integer_range_host\",\"magnitude_begin_bits\":1065353216,\"magnitude_end_exclusive_bits\":1123811328,\"checked_inputs\":%llu,\"mismatches\":%llu,\"maximum_bit_distance\":%u,\"native_exp_qualified\":false,\"performance_acceptance\":false,\"inference_acceptance\":false}\n",(unsigned long long)checked,(unsigned long long)mismatches,maximum);
    return mismatches ? 1 : 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
