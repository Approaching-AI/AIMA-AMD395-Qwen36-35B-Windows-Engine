"""The gate reduction must reproduce the original GPU's three regressions."""
from pathlib import Path
import base64
import hashlib
import json
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackedGateReductionTests(unittest.TestCase):
    def test_original_gb10_gate_endpoints_and_carriers(self):
        fixture = json.loads((ROOT / 'tests/fixtures/packed_gate_reduction.json').read_text())
        source = r'''
#include "native/providers/moe_accumulator/sm121_packed_dense.h"
#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
std::array<uint16_t,2048> read(const char* path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=4096)throw std::runtime_error("extent");
    std::array<uint16_t,2048> values;f.seekg(0);
    f.read(reinterpret_cast<char*>(values.data()),4096);
    if(!f)throw std::runtime_error("read");return values;
}
template<class Input> uint16_t dot(const Input* x,const uint16_t* w) {
    float sums[16];
    for(unsigned lane=0;lane<16;++lane)sums[lane]=qrt_sm121_packed_dense::gate_lane_dot(x,w,lane);
    for(unsigned offset=8;offset;offset/=2)for(unsigned lane=0;lane<offset;++lane)
        sums[lane]=qrt_sm121_q1::add(sums[lane],sums[lane+offset]);
    return qrt_sm121_q1::bf16(sums[0]);
}
int main(int argc,char** argv)try {
    if(argc!=4)throw std::runtime_error("input weight original-endpoint");
    const auto x=read(argv[1]),w=read(argv[2]);
    const unsigned expected=std::stoul(argv[3]);
    std::array<float,2048> widened;
    for(unsigned i=0;i<2048;++i)widened[i]=qrt_sm121_q1::widen(x[i]);
    if(dot(x.data(),w.data())!=expected||dot(widened.data(),w.data())!=expected)
        throw std::runtime_error("original GB10 endpoint mismatch");
    std::cout<<"original BF16 and widened-carrier endpoints match\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-gate-reduction-') as tmp:
            directory = Path(tmp)
            cpp, executable = directory / 'gate.cpp', directory / 'gate'
            cpp.write_text(source)
            subprocess.run(['c++', '-std=c++17', '-O2', '-ffp-contract=off',
                '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                '-fno-sanitize-recover=all', '-fno-omit-frame-pointer', '-I', str(ROOT),
                str(cpp), '-o', str(executable)], check=True, capture_output=True,
                text=True, timeout=45)
            self.assertEqual(len(fixture['cases']), 3)
            for case in fixture['cases']:
                with self.subTest(layer=case['layer'], position=case['position']):
                    files = []
                    for kind in ('input', 'weight'):
                        raw = base64.b64decode(case[kind + '_bf16_le_base64'], validate=True)
                        self.assertEqual(len(raw), 4096)
                        self.assertEqual(hashlib.sha256(raw).hexdigest(), case[kind + '_sha256'])
                        path = directory / (kind + '.bin')
                        path.write_bytes(raw)
                        files.append(str(path))
                    result = subprocess.run([str(executable), *files, str(case['expected_bf16'])],
                        check=True, capture_output=True, text=True, timeout=20)
                    self.assertIn('original BF16 and widened-carrier endpoints match', result.stdout)


if __name__ == '__main__':
    unittest.main()
