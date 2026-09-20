"""Bounds and independent examples for original MTP prompt shifting."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MtpPromptInputsTests(unittest.TestCase):
    def test_chunk_tails_and_invalid_inputs(self):
        source = r'''
#include "mtp_prompt_inputs.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>
int main() {
 using qrt_mtp_prompt_inputs::shift;
 uint32_t prompt[]={10,11,12,13,14}, out[]={99,99,99};
 assert(shift(prompt,5,0,3,true,700,out));
 assert(out[0]==11 && out[1]==12 && out[2]==14);
 assert(shift(prompt,5,3,2,false,700,out)); assert(out[0]==14 && out[1]==700);
 assert(shift(prompt,5,1,1,true,700,out) && out[0]==14);
 assert(shift(prompt,5,4,1,false,700,out) && out[0]==700);
 const auto invalid=[&](size_t n,size_t first,size_t rows,bool discarded,uint32_t sample) {
   std::fill_n(out,3,99u); assert(!shift(prompt,n,first,rows,discarded,sample,out));
   assert(std::all_of(out,out+3,[](auto v){return v==99u;}));
 };
 invalid(5,0,3,false,700); invalid(5,3,2,true,700); invalid(5,3,3,false,700);
 invalid(0,0,1,false,700); invalid(262145,0,1,true,700); invalid(5,0,0,true,700);
 invalid(5,std::numeric_limits<size_t>::max(),3,true,700); invalid(5,3,2,false,248320);
 prompt[2]=248320; invalid(5,0,3,true,700); prompt[2]=12;
 prompt[4]=248320; invalid(5,0,3,true,700); prompt[4]=14;
 assert(!shift(nullptr,5,0,3,true,700,out)); assert(!shift(prompt,5,0,3,true,700,nullptr));
 std::vector<uint32_t> long_prompt(262144,23), chunk(8192);
 assert(shift(long_prompt.data(),long_prompt.size(),253952,8192,false,17,chunk.data()));
 assert(chunk.front()==23 && chunk.back()==17);
 assert(!shift(long_prompt.data(),long_prompt.size(),0,8193,true,17,chunk.data()));
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'test.cpp'
            path.write_text(source)
            exe = directory / 'test'
            build = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                '-I', str(ROOT / 'native/providers'), str(path), '-o', str(exe)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
