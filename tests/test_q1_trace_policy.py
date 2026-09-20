"""Diagnostic reads cover both selected states within an explicit file budget."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Q1TracePolicyTests(unittest.TestCase):
    def test_selected_positions_layers_and_complete_state_budget(self):
        source = r'''
#include "q1_trace_policy.h"
#include <cassert>
using qrt_q1_trace::Selection;
using qrt_q1_trace::kUnselected;
int main() {
    Selection disabled;
    assert(!disabled.position(0) && !disabled.position(kUnselected));
    Selection selected{263356,263168,1,0,true};
    assert(selected.position(263356) && selected.position(263168));
    assert(!selected.position(263355) && !selected.position(263357));
    assert(selected.raw_position(263356) && selected.raw_position(263168));
    unsigned layers=0;
    size_t files=0,bytes=0;
    for (unsigned layer=0;layer<40;++layer) if (selected.linear_layer(layer)) {
        ++layers;
        // Before/after states and24 small stages at each actual position.
        for (unsigned position=0;position<2;++position) {
            for (unsigned state=0;state<2;++state) {
                assert(selected.may_write(files,bytes,2u<<20));
                ++files;bytes+=2u<<20;
            }
            for (unsigned stage=0;stage<24;++stage) {
                assert(selected.may_write(files,bytes,128u<<10));
                ++files;bytes+=128u<<10;
            }
        }
    }
    assert(layers==30 && files==1560 && bytes==(420u<<20));
    assert(!selected.linear_layer(40) && !selected.linear_layer(kUnselected));
    assert(!selected.may_write(2048,0,1));
    assert(!selected.may_write(0,(512u<<20),1));
    assert(!selected.may_write(0,SIZE_MAX,1));
    assert(!selected.may_write(0,0,0) && !selected.may_write(0,0,(2u<<20)+1));
    selected.all_linear_layers=false;
    assert(selected.linear_layer(0) && !selected.linear_layer(2));
    assert(selected.byte_limit()==(16u<<20) && selected.file_limit()==64);
    selected.second=kUnselected;selected.count=512;
    assert(selected.position(263867) && !selected.position(263868));
    assert(!selected.position(kUnselected) && !selected.raw_position(kUnselected));
    assert(!selected.raw_position(263867));
    selected.count=513;assert(!selected.position(263356));
    selected.count=0;assert(!selected.position(263356));
    selected.count=1;selected.second=263868;assert(!selected.valid_positions());
    selected.second=263356;assert(!selected.valid_positions());
}
'''
        with tempfile.TemporaryDirectory() as directory:
            exe = str(Path(directory) / 'trace-policy')
            subprocess.run(['c++', '-std=c++17', '-O1', '-fsanitize=address,undefined',
                '-fno-sanitize-recover=all', '-I', str(ROOT/'native/providers'),
                '-x', 'c++', '-', '-o', exe], input=source, text=True,
                capture_output=True, check=True, timeout=30)
            subprocess.run([exe], capture_output=True, check=True, timeout=10)
