"""Check lossless packed metadata and folded products against original integers."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FoldedHalfProductsTests(unittest.TestCase):
    def test_original_representation_products_and_carries(self):
        with tempfile.TemporaryDirectory(prefix="qrt-folded-half-") as directory:
            executable = str(Path(directory) / "check")
            # Compile every actual device template using declaration-only HIP
            # substitutes. This checks names/scopes, never GPU arithmetic.
            hip = Path(directory) / "hip"
            hip.mkdir()
            (hip / "hip_runtime.h").write_text(r'''
#pragma once
#include <cstdint>
#define __device__
#define __global__
#define __shared__ static
#define __forceinline__ inline
#define __launch_bounds__(...)
struct Dim { unsigned x = 0; };
inline Dim blockIdx, blockDim, threadIdx;
inline void __syncthreads() {}
inline void atomicOr(unsigned* p, unsigned value) { *p |= value; }
inline int __clz(unsigned value) { return __builtin_clz(value); }
template<class T> T __shfl_xor(T value, unsigned, unsigned) { return value; }
template<class T> T __shfl(T value, unsigned, unsigned) { return value; }
''')
            header = ROOT / "native/providers/moe_accumulator/sm121_folded_half_projection.h"
            source = '#include "' + str(header) + '"\n#include <chrono>\n'
            source += 'void instantiate(const qrt_sm121_folded_half_projection::Row* p) {\n'
            for staging in (1, 2, 4, 8):
                for audit in ("false", "true"):
                    source += f'qrt_sm121_folded_half_projection::dot<{staging},{audit}>(p,p,16);\n'
            source += '}\nint main() { return 0; }\n'
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-fsyntax-only",
                 "-I", directory, "-x", "c++", "-"],
                input=source, text=True, check=True, timeout=30)
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                 str(ROOT / "tests/native/folded_half_products_host.cpp"), "-o", executable],
                check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
