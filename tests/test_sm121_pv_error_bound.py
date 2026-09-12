"""Check native-PV candidate envelopes against the independent wide K16 model."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Sm121PvErrorBoundTests(unittest.TestCase):
    def test_cancellation_rescaling_and_bf16_boundary_admission(self):
        source = r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include "sm121_pv_error_bound.h"
#include "q1_moe_hawkeye_bf16_accumulator.h"
namespace b = qrt_sm121_pv_bound;
uint32_t seed=0x39513171u;
uint32_t rnd() { seed^=seed<<13u; seed^=seed>>17u; seed^=seed<<5u; return seed; }
float f32(uint16_t x) { return b::value(uint32_t(x)<<16u); }
float rounded_add(float a,float c) { volatile float out=a+c; return out; }
float rounded_mul(float a,float c) { volatile float out=a*c; return out; }
int main() {
    unsigned groups=0, admitted=0, changed=0;
    for(unsigned trial=0;trial<512u;++trial) {
        float actual=0.0f,reference=0.0f,error=0.0f;
        for(unsigned step=0;step<512u;++step) {
            if(step%2u==0u) {
                const float alpha=step%64u==0u ? 0.625f : (step%8u==0u ? 0.99609375f : 1.0f);
                error=b::rescale(error,actual,alpha);
                actual=rounded_mul(actual,alpha); reference=rounded_mul(reference,alpha);
            }
            uint16_t p[16],v[16]; float sum_abs=0.0f, partial[16];
            for(unsigned i=0;i<16u;++i) {
                p[i]=uint16_t((rnd()&127u)|((trial%3u==0u ? 80u+rnd()%48u : 122u+rnd()%6u)<<7u));
                v[i]=uint16_t((rnd()&0x807fu)|((trial%3u==0u ? 85u+rnd()%80u : 124u+rnd()%7u)<<7u));
                if(trial%5u==0u && i%2u) { p[i]=p[i-1]; v[i]=v[i-1]^0x8000u; }
                if(trial%7u==0u && i%4u==0u) p[i]=0u;
                partial[i]=rounded_mul(f32(p[i]),f32(v[i]));
                sum_abs=rounded_add(sum_abs,std::fabs(partial[i]));
            }
            error=b::group(error,actual,sum_abs);
            reference=qrt_q1_moe_hawkeye::accumulate_bf16_impl<26,16,-133>(reference,p,v,16u);
            if(trial%2u) {
                for(unsigned i=0;i<16u;++i) actual=rounded_add(actual,partial[15u-i]);
            } else {
                for(unsigned stride=8u;stride;stride>>=1u)
                    for(unsigned i=0;i<stride;++i) partial[i]=rounded_add(partial[i],partial[i+stride]);
                actual=rounded_add(actual,partial[0]);
            }
            assert(std::fabs(double(actual)-reference)<=double(error));
            ++groups;
            if(b::bits(actual)!=b::bits(reference)) ++changed;
            if(step%32u==31u) {
                const float reciprocal=0.00390625f+float(trial%11u)*0.00003125f;
                const float out=rounded_mul(actual,reciprocal), ref=rounded_mul(reference,reciprocal);
                const float bound=b::finish(error,actual,reciprocal);
                assert(std::fabs(double(out)-ref)<=double(bound));
                if(b::same_bf16(out,bound)) { assert(b::bf16(out)==b::bf16(ref)); ++admitted; }
            }
        }
    }
    // Walk every normal BF16 midpoint, including powers of two and both signs.
    for(uint32_t encoded=0x0080u;encoded<0x7f7fu;++encoded) {
        const float lo=f32(uint16_t(encoded)), hi=f32(uint16_t(encoded+1u));
        const float middle=float((double(lo)+hi)*0.5);
        const float width=float((double(hi)-lo)*0.25);
        for(float sign : {-1.0f,1.0f}) {
            assert(!b::same_bf16(sign*middle,width));
            assert(b::same_bf16(sign*lo,width*0.25f));
        }
    }
    assert(!b::same_bf16(b::infinity(),0.0f));
    assert(!b::same_bf16(1.0f,b::infinity()));
    assert(!b::same_bf16(1.0f,-1.0f));
    assert(!b::finite(b::group(0.0f,b::infinity(),1.0f)));
    assert(b::rescale(1.0f,1.0f,0.0f)==0.0f);
    assert(groups==262144u && changed>200000u && admitted>0u);
    std::printf("groups=%u differing_fp32=%u admitted_endpoints=%u false_admissions=0\n",groups,changed,admitted);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-pv-bound-') as tmp:
            exe = str(Path(tmp) / 'pv-bound')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                            '-Wall', '-Wextra', '-Werror', '-ffp-contract=off',
                            '-fsanitize=undefined', '-I', str(ROOT/'native/providers/moe_accumulator'),
                            '-x', 'c++', '-', '-o', exe], input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=20)


if __name__ == '__main__':
    unittest.main()
