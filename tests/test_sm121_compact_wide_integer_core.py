"""Check lossless expansion independently of HIP and attention launches."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CompactWideIntegerCoreTests(unittest.TestCase):
    def test_every_bf16_pattern_and_mixed_rows_expand_to_original_bytes(self):
        source = r'''
#include "native/providers/moe_accumulator/sm121_compact_wide_integer_core.h"
#include <cstdint>
#include <cstring>
namespace full = qrt_sm121_wide_core;
namespace compact = qrt_sm121_compact_wide_core;
uint32_t state=0x753b9821u;
uint32_t random_word() { state^=state<<13u;state^=state>>17u;state^=state<<5u;return state; }
int verify(full::Row row) {
    full::prepare(row);
    struct Guarded { uint32_t before[3];compact::Row row;uint32_t after[3]; } guarded;
    std::memset(&guarded,0xa5,sizeof(guarded));
    guarded.row=compact::pack(row);
    const Guarded immutable=guarded;
    auto expanded=compact::expand(guarded.row);
    if(std::memcmp(&expanded,&row,sizeof(row)))return 1;
    if(std::memcmp(&guarded,&immutable,sizeof(guarded)))return 2;
    for(auto word:guarded.before)if(word!=0xa5a5a5a5u)return 3;
    for(auto word:guarded.after)if(word!=0xa5a5a5a5u)return 4;
    // Reproduce the actual cooperative word mapping, including every metadata
    // dword. Pairs may complete in any order after the load barrier.
    full::Row tiled;std::memset(&tiled,0x5a,sizeof(tiled));
    for(unsigned word=0;word<sizeof(compact::Row)/4u;++word) {
        const unsigned target=compact::expanded_word(word);
        std::memcpy(reinterpret_cast<unsigned char*>(&tiled)+4u*target,
                    reinterpret_cast<const unsigned char*>(&guarded.row)+4u*word,4u);
    }
    for(unsigned pair=0;pair<8u;++pair)compact::expand_pair(tiled,(pair*3u+5u)%8u);
    if(std::memcmp(&tiled,&row,sizeof(row)))return 5;
    return 0;
}
int main() {
    for(unsigned pattern=0;pattern<65536u;++pattern) {
        full::Row uniform{},mixed{};
        for(unsigned i=0;i<16u;++i) {
            uniform.original[i]=uint16_t(pattern);
            mixed.original[i]=uint16_t(pattern^(i*4051u));
        }
        if(int status=verify(uniform))return status;
        if(int status=verify(mixed))return 10+status;
    }
    for(unsigned trial=0;trial<8192u;++trial) {
        full::Row row{};
        for(unsigned i=0;i<16u;++i)row.original[i]=uint16_t(random_word());
        if(int status=verify(row))return 20+status;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            cpp = path / "compact-wide.cpp"
            cpp.write_text(source)
            binary = path / "compact-wide"
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-I", str(ROOT), str(cpp), "-o", str(binary)],
                check=True, timeout=45,
            )
            subprocess.run([str(binary)], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
