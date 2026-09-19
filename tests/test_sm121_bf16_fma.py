"""Check single-round BF16 FMA against an unbounded-integer nearest-value oracle."""
from bisect import bisect_left
from pathlib import Path
import itertools
import json
import random
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

def units(bits):
    """Exact finite BF16 value in units of 2**-133."""
    exponent=(bits>>7)&255
    value=((bits&127)+(128 if exponent else 0))<<max(exponent-1,0)
    return -value if bits&0x8000 else value

# Include the hypothetical next finite value 2**128. Rounding to this value
# means infinity, with the same even significand at the overflow midpoint.
VALUES=[units(x)<<133 for x in range(0x7f80)]+[1<<(128+266)]

def oracle(a,b,c):
    value=units(a)*units(b)+(units(c)<<133)
    if value==0:
        return 0x8000 if units(a)*units(b)==units(c)==0 and (a^b)&c&0x8000 else 0
    sign=0x8000 if value<0 else 0
    value=abs(value);upper=bisect_left(VALUES,value)
    if upper>=len(VALUES):return sign|0x7f80
    if upper==0:return sign
    lower=upper-1;dl=value-VALUES[lower];du=VALUES[upper]-value
    result=lower if dl<du or (dl==du and not lower&1) else upper
    return sign|result

class Bf16FmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.directory=Path(cls.temp.name)
        source=cls.directory/'probe.cpp';cls.exe=cls.directory/'probe'
        source.write_text(r'''
#include "native/providers/gdn/sm121_bf16_fma.h"
#include <iostream>
int main(int argc,char**){
    if(argc==2){
        const float a=2.21875f,b=-0.875123f,c=-0.0000000222f;
        const float result=fmaf(a,b,c);uint32_t bits;std::memcpy(&bits,&result,4);
        const auto expected=qrt_sm121_bf16_fma::widen(uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16));
        return qrt_sm121_bf16_fma::rounded(a,b,c)==expected?0:3;
    }
    uint16_t input[3];
    while(std::cin.read(reinterpret_cast<char*>(input),sizeof(input))){
        const float floating=qrt_sm121_bf16_fma::rounded(qrt_sm121_bf16_fma::widen(input[0]),
            qrt_sm121_bf16_fma::widen(input[1]),qrt_sm121_bf16_fma::widen(input[2]));
        uint32_t bits;std::memcpy(&bits,&floating,4);
        uint16_t output[]={qrt_sm121_bf16_fma::round(input[0],input[1],input[2]),
                           qrt_sm121_bf16_fma::exact(input[0],input[1],input[2]),uint16_t(bits>>16)};
        std::cout.write(reinterpret_cast<const char*>(output),sizeof(output));
    }
    return std::cin.eof()?0:2;
}
''')
        subprocess.run(['c++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror',
                        '-fsanitize=address,undefined','-I',str(ROOT),str(source),'-o',str(cls.exe)],
                       check=True,capture_output=True,timeout=60)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def check_cases(self,cases,expected=None):
        raw=b''.join(struct.pack('<3H',*row) for row in cases)
        result=subprocess.run([str(self.exe)],input=raw,capture_output=True,check=True,timeout=30)
        actual=list(struct.iter_unpack('<3H',result.stdout));self.assertEqual(len(actual),len(cases))
        for i,(row,out) in enumerate(zip(cases,actual)):
            want=oracle(*row) if expected is None else expected[i]
            self.assertEqual(out,(want,want,want),f'operands={[hex(x) for x in row]}')

    def test_every_finite_addend_at_real_rotary_midpoint(self):
        self.check_cases([(0x400e,0xbf60,c) for c in range(65536) if c&0x7f80!=0x7f80])

    def test_captured_original_rotary_endpoint(self):
        fixture=json.loads((ROOT/'tests/fixtures/rope_bf16_midpoint.json').read_text())
        case=(fixture['first_bf16'],fixture['cos_bf16'],fixture['rounded_second_sin_bf16']^0x8000)
        expected=fixture['expected_first_bf16']
        self.assertNotEqual(expected,fixture['prior_double_rounded_first_bf16'])
        self.check_cases([case],[expected])

    def test_random_finite_and_cancellation(self):
        rng=random.Random(91018);cases=[]
        while len(cases)<50000:
            row=tuple(rng.randrange(65536) for _ in range(3))
            if all(x&0x7f80!=0x7f80 for x in row):cases.append(row)
        for a in range(65536):
            if a&0x7f80!=0x7f80:
                cases.append((a,0x3f80,a^0x8000))
        self.check_cases(cases)

    def test_subnormal_overflow_and_midpoint_neighbours(self):
        magnitudes=(0,1,2,63,64,65,126,127,128,129,0x3eff,0x3f00,0x3f7f,0x3f80,
                    0x3f81,0x400e,0x7e80,0x7f00,0x7f7e,0x7f7f)
        values=magnitudes+tuple(x|0x8000 for x in magnitudes)
        self.check_cases(list(itertools.product(values,repeat=3)))

    def test_nonfinite_and_signed_zero(self):
        cases=[(0x7f80,0,0),(0xff80,0x3f80,0x7f80),(0x7fc1,0x3f80,0),
               (0x7f80,0xbf80,0),(0x3f80,0x3f80,0xff80),
               (0x8000,0x3f80,0x8000),(0x8000,0x3f80,0)]
        self.check_cases(cases,[0x7fc0,0x7fc0,0x7fc0,0xff80,0xff80,0x8000,0])

    def test_legacy_fp32_coefficients_keep_fp32_arithmetic(self):
        subprocess.run([str(self.exe),'legacy'],check=True,capture_output=True,timeout=10)
