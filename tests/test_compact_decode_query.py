"""Check actual single-query score addresses with independent scalar K16 groups.

The host substitutes the collective primitive; it does not simulate GPU waves
or qualify native arithmetic. Every actual operand read and output is checked.
"""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class CompactDecodeQueryTests(unittest.TestCase):
    def test_actual_score_body_original_groups_and_compact_bounds(self):
        header = (ROOT / 'native/providers/ck_fmha/blackwell_attention.h').read_text()
        actual = '\n'.join(function(header, signature) for signature in (
            'template<bool RelativeQuery>', '__global__ void blackwell_exact_scores_kernel(',
            '__global__ void blackwell_compact_query_scores_kernel('))
        code = r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include "native/providers/moe_accumulator/sm121_group16_modulo.h"
#define __device__
#define __global__
#define __forceinline__ inline
struct Index {unsigned x=0;};
Index blockIdx,threadIdx;
constexpr unsigned kBlackwellSubgroups=16u,kBlackwellMmaGroup=16u;
constexpr unsigned kQueryHeads=16u,kKvHeads=2u,kHeadDim=256u;
constexpr int kBlackwellZeroExponent=-133;
constexpr float kExactScale=0.0625f;
const uint16_t* expected_query=nullptr;
const uint16_t* expected_key=nullptr;
unsigned groups=0;
uint64_t checked_reads=0;
namespace qrt_sm121_wave16 {
qrt_q1_moe_hawkeye::Value accumulate(qrt_q1_moe_hawkeye::Value carry,uint16_t q,uint16_t k,unsigned lane){
    namespace original=qrt_q1_moe_hawkeye;
    assert(groups<16u && lane<16u);
    const unsigned base=groups*16u;
    assert(q==expected_query[base+lane] && k==expected_key[base+lane]);
    ++groups;++checked_reads;
    original::Value values[17];values[0]=carry;
    for(unsigned i=0;i<16u;++i)values[i+1u]=original::multiply_bf16(expected_query[base+i],expected_key[base+i],-133);
    return original::group_sum<26,-133>(values,17u);
}
}
''' + actual + r'''
uint32_t bits(float x){uint32_t v;std::memcpy(&v,&x,4u);return v;}
uint16_t word(unsigned row,unsigned feature,unsigned family){
    constexpr unsigned exponents[]={0u,1u,64u,95u,117u,127u,141u,159u,190u};
    const unsigned value=row*17u+feature*11u;
    if(family==2u)return uint16_t((value&1u)?0x3f80u:0xbf81u);
    const unsigned exponent=family?exponents[value%9u]:117u+value%18u;
    return uint16_t((value&0x807fu)|(exponent<<7u));
}
int main(){
    uint64_t dots=0,cases=0;
    for(unsigned first:{1u,17u,31u,8191u,8192u,32768u,131072u,262144u,264735u}){
        const unsigned tokens=first+1u;
        for(unsigned family=0;family<3u;++family){
            std::vector<uint16_t> q(4096u);
            for(unsigned f=0;f<4096u;++f)q[f]=word(f/256u,f%256u,family);
            const auto saved=q;
            // Only the actual current row is allocated for the compact input.
            // Large history positions are logical labels, not its offsets.
            std::unique_ptr<uint16_t[]> full(new uint16_t[size_t(tokens)*4096u]);
            std::unique_ptr<uint16_t[]> keys(new uint16_t[size_t(tokens)*512u]);
            std::unique_ptr<float[]> scores(new float[size_t(tokens)*16u+1u]);
            std::memcpy(full.get()+size_t(first)*4096u,q.data(),8192u);
            for(unsigned part=0;part<8u;++part){
                const unsigned key=unsigned(uint64_t(first)*part/7u);
                for(unsigned f=0;f<512u;++f)keys[size_t(key)*512u+f]=word(key+f/256u+1u,f%256u+3u,family);
            }
            for(unsigned head=0;head<16u;++head)for(unsigned part=0;part<8u;++part){
                const unsigned key=unsigned(uint64_t(first)*part/7u);
                const size_t cell=size_t(head)*tokens+key;
                expected_query=q.data()+head*256u;
                expected_key=keys.get()+(size_t(key)*2u+head/8u)*256u;
                const float expected=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(
                    0.0f,expected_query,expected_key,256u)*0.0625f;
                blockIdx.x=unsigned(cell/16u);
                for(unsigned relative=0;relative<2u;++relative){
                    scores[cell]=123.0f;
                    for(unsigned lane=0;lane<16u;++lane){
                        threadIdx.x=unsigned(cell%16u)*16u+lane;groups=0;
                        if(relative)blackwell_compact_query_scores_kernel(q.data(),keys.get(),scores.get(),first,1u,tokens,first);
                        else blackwell_exact_scores_kernel(full.get(),keys.get(),scores.get(),first,1u,tokens);
                        assert(groups==16u && bits(scores[cell])==bits(expected));
                    }
                }
                ++dots;
            }
            // Excess launch lanes must not dereference absent inputs or write.
            const size_t end=size_t(tokens)*16u;scores[end]=123.0f;
            blockIdx.x=unsigned(end/16u);threadIdx.x=unsigned(end%16u)*16u;groups=0;
            blackwell_compact_query_scores_kernel(nullptr,nullptr,scores.get(),first,1u,tokens,first);
            assert(groups==0u && scores[end]==123.0f && q==saved);++cases;
        }
    }
    std::printf("{\"kind\":\"compact_decode_query_host\",\"cases\":%llu,\"original_dots\":%llu,\"checked_lane_group_reads\":%llu,\"actual_score_body\":true,\"collective_replaced_by_scalar_reference\":true,\"native_acceptance\":false}\n",
        (unsigned long long)cases,(unsigned long long)dots,(unsigned long long)checked_reads);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-compact-decode-') as directory:
            executable = str(Path(directory) / 'check')
            subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(ROOT),
                '-x', 'c++', '-', '-o', executable], input=code, text=True, check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=60)


if __name__ == '__main__':
    unittest.main()
