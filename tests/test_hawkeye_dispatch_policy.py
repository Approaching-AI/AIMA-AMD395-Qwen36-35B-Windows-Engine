"""Exercise admission and timeout boundaries without submitting GPU work."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HawkeyeDispatchPolicyTests(unittest.TestCase):
    def test_dense_work_and_exhausted_time_are_rejected(self):
        source = r'''
#include "hawkeye_dispatch_policy.h"
using namespace qrt_hawkeye_dispatch;
static_assert(admitted(0, 0), "empty candidate set");
static_assert(admitted(131072, 64), "inclusive admission boundary");
static_assert(!admitted(131073, 1), "total work cannot exceed the budget");
static_assert(!admitted(1, 65), "one dense block must also be rejected");
static_assert(!admitted(UINT32_MAX, UINT32_MAX), "no integer wraparound");
static_assert(time_remaining(100.0, 10000.0), "inclusive time boundary");
static_assert(!time_remaining(100.001, 1.0), "slow dispatch stops submission");
static_assert(!time_remaining(1.0, 10000.001), "aggregate deadline");
int main() { return 0; }
'''
        with tempfile.TemporaryDirectory(prefix="qrt-hawkeye-policy-") as tmp:
            exe = str(Path(tmp) / "policy-test")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                 "-Werror", "-I", str(ROOT / "native/providers"), "-x", "c++",
                 "-", "-o", exe],
                input=source, text=True, check=True, timeout=30,
            )
            subprocess.run([exe], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
