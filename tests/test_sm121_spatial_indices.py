"""Check the actual padded work permutation independently of any GPU arithmetic."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class SpatialIndicesTests(unittest.TestCase):
    def test_complete_partial_tiles_are_bijections_and_invalid_shapes_are_empty(self):
        source = r'''
#include "sm121_spatial_indices.h"
#include <cassert>
#include <cstdio>
#include <vector>
#include <utility>
using namespace qrt_sm121_spatial_indices;
int main() {
 unsigned long long visited=0,padding=0;
 for(auto tile:{std::pair{64u,32u},std::pair{128u,64u},std::pair{256u,128u}}) {
  for(auto shape:{std::pair{1u,1u},std::pair{63u,31u},std::pair{64u,32u},
      std::pair{65u,33u},std::pair{257u,129u},std::pair{93u,4097u},
      std::pair{9216u,17u},std::pair{2048u,127u}}) {
   const unsigned count=shape.first*shape.second;
   std::vector<bool> seen(count,false);unsigned found=0;
   const auto size=extent(shape.first,shape.second,tile.first,tile.second);
   assert(size>=count && size%256==0);
   for(unsigned v=0;v<size;v++) {
    auto cell=index(v,shape.first,shape.second,tile.first,tile.second);
    if(cell==UINT32_MAX){++padding;continue;}
    assert(cell<count && !seen[cell]);seen[cell]=true;++found;++visited;
   }
   assert(found==count && seen.front() && seen.back());
   assert(index(size,shape.first,shape.second,tile.first,tile.second)==UINT32_MAX);
  }
  for(auto shape:{std::pair{0u,1u},std::pair{1u,0u},std::pair{16385u,1u},std::pair{1u,8193u}}) {
   assert(!extent(shape.first,shape.second,tile.first,tile.second));
   assert(index(0,shape.first,shape.second,tile.first,tile.second)==UINT32_MAX);
  }
 }
 for(auto tile:{std::pair{0u,0u},std::pair{64u,64u},std::pair{256u,256u},std::pair{UINT32_MAX,UINT32_MAX}})
  assert(!extent(8192,8192,tile.first,tile.second) && index(0,8192,8192,tile.first,tile.second)==UINT32_MAX);
 assert(extent(16384,8192,256,128)==134217728u);
 std::printf("spatial_permutation_cells=%llu padding=%llu all_shapes_bijective=1\n",visited,padding);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-spatial-indices-') as tmp:
            exe = str(Path(tmp) / 'indices')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=undefined', '-fno-sanitize-recover=all', '-I', str(ROOT / 'native/providers/moe_accumulator'),
                            '-x', 'c++', '-', '-o', exe], input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=10)

if __name__ == '__main__':
    unittest.main()
