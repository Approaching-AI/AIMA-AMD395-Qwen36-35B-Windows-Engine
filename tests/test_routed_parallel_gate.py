"""Check the independent descriptor and wave ownership boundaries."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RoutedParallelGateTests(unittest.TestCase):
    def test_descriptors_and_all_matrix_cells(self):
        source = r'''
#include <cassert>
#include <cstdint>
#include <climits>
#include <array>
#include "routed_parallel_gate.h"
using namespace qrt_routed_parallel_gate;
int main() {
    for (int expert=0;expert<256;++expert) {
        assert(expert_from_descriptor(expert)==expert);
        assert(expert_from_descriptor(-expert-1)==expert);
        assert(expert_from_descriptor(-256-expert-1)==expert);
    }
    for(int value:{INT_MIN,-513,256,INT_MAX}) assert(expert_from_descriptor(value)==-1);
    std::array<unsigned,64*64> count{};
    for(unsigned wave=0;wave<8;++wave)for(unsigned lane=0;lane<32;++lane)
        for(unsigned element=0;element<8;++element)for(unsigned fragment=0;fragment<2;++fragment) {
            const auto p=cell(wave,lane,element,fragment);
            assert(p.row<64&&p.column<64);++count[p.row*64+p.column];
        }
    for(auto n:count)assert(n==1);
    for(unsigned rows:{1u,15u,16u,17u,31u,32u,33u,63u,64u}) {
        unsigned writes=0;
        for(unsigned wave=0;wave<8;++wave)for(unsigned lane=0;lane<32;++lane)
            for(unsigned element=0;element<8;++element)for(unsigned fragment=0;fragment<2;++fragment)
                writes+=cell(wave,lane,element,fragment).row<rows;
        assert(writes==rows*64);
    }
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-parallel-gate-') as directory:
            executable = str(Path(directory) / 'test')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                            '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                            '-I', str(ROOT / 'native/providers/triton_moe'),
                            '-x', 'c++', '-', '-o', executable],
                           input=source, text=True, check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=15)
