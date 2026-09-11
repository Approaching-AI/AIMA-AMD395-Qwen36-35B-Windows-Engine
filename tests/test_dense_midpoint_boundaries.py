"""Run the actual dense selector at the lower boundary of power-of-two cells."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class DenseMidpointTests(unittest.TestCase):
    def test_both_signs_l2_and_absolute_bounds_at_binade_transitions(self):
        provider = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        selector = function(provider, '__device__ bool selected_bf16_projection_hawkeye_candidate(')
        source = r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include "native/providers/moe_accumulator/bf16_midpoint_selector.h"
#define __device__
''' + selector + r'''
int main(){
 unsigned checked=0;float input[3]={0,0,1},weight[2]={0,0},absolute[6]{};
 for(int exponent=-70;exponent<=70;++exponent)for(float sign:{-1.0f,1.0f}){
  const float value=sign*std::ldexp(1.0f,exponent);
  const float lower_distance=std::fabs(value)/512.0f;
  weight[1]=lower_distance*1.5f/1.0e-6f;absolute[5]=weight[1];
  // The current-binade midpoint is twice as far away. The former dense
  // predicate rejected these valid correction candidates in both signs.
  assert(selected_bf16_projection_hawkeye_candidate(value,5,2,0,0,1000,nullptr,input,weight));
  assert(selected_bf16_projection_hawkeye_candidate(value,5,2,0,0,1000,absolute,nullptr,nullptr));
  weight[1]=lower_distance*0.5f/1.0e-6f;absolute[5]=weight[1];
  assert(!selected_bf16_projection_hawkeye_candidate(value,5,2,0,0,1000,nullptr,input,weight));
  assert(!selected_bf16_projection_hawkeye_candidate(value,5,2,0,0,1000,absolute,nullptr,nullptr));
  assert(!selected_bf16_projection_hawkeye_candidate(value,5,2,0,0,0,absolute,nullptr,nullptr));
  assert(selected_bf16_projection_hawkeye_candidate(value,5,2,0,3,0,nullptr,nullptr,nullptr));
  ++checked;
 }
 assert(selected_bf16_projection_hawkeye_candidate(std::numeric_limits<float>::infinity(),5,2,0,0,1000,nullptr,input,weight));
 assert(selected_bf16_projection_hawkeye_candidate(std::numeric_limits<float>::quiet_NaN(),5,2,0,0,1000,absolute,nullptr,nullptr));
 std::printf("dense_binade_boundaries=%u actual_selector=pass\n",checked);
}
'''
        source = '#include <initializer_list>\n' + source
        with tempfile.TemporaryDirectory(prefix='qrt-dense-midpoint-') as tmp:
            exe = str(Path(tmp) / 'selector')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=undefined', '-I', str(ROOT), '-x', 'c++', '-', '-o', exe],
                           input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
