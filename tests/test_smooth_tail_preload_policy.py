"""Compile the actual native preload selector without HIP or a model."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SmoothTailPreloadPolicyTests(unittest.TestCase):
    def test_retained_and_dense_profile_admission(self):
        provider = (ROOT / "native/providers/whole_provider.cpp").read_text()
        declaration = "constexpr std::array<size_t, 19u> kSmoothTailMoeProviderTokenCounts"
        actual = declaration + provider.split(declaration, 1)[1].split(
            "bool smooth_tail_moe_provider_index(", 1)[0]
        source = "#include <array>\n#include <cstddef>\n" + actual + r'''
int main() {
    size_t base = 0, dense = 0;
    for (size_t q = 0; q <= 8192; ++q) {
        const bool expected_base = q >= 32 && q <= 4096 && (q & (q - 1)) == 0;
        const bool expected_dense = expected_base || q == 2560 || q == 3072 ||
            q == 3328 || q == 3584 || q == 4608 || q == 5120 || q == 5632 ||
            q == 6144 || q == 6656 || q == 7168 || q == 7680;
        if (smooth_tail_moe_provider_required_for_profile(q, false) != expected_base) return 1;
        if (smooth_tail_moe_provider_required_for_profile(q, true) != expected_dense) return 2;
        base += expected_base;
        dense += expected_dense;
    }
    return base == 8 && dense == 19 ? 0 : 3;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-preload-policy-") as temporary:
            executable = str(Path(temporary) / "policy-test")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                            "-Werror", "-x", "c++", "-", "-o", executable],
                           input=source, text=True, check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=5)
        preload = provider.split("bool preload_capacity_sensitive_providers_before_model_store(", 1)[1]
        preload = preload.split("if (aiter_requested)", 1)[0]
        self.assertLess(preload.index("smooth_tail_moe_provider_required_for_profile("),
                        preload.index("load_smooth_tail_triton_selected_moe_full_provider("))
        self.assertIn("++smooth_tail_moe_provider_count", preload)

    def test_default_package_matches_only_retained_capacities(self):
        for script in ("build-runtime.ps1", "package-runtime.ps1"):
            text = (ROOT / "scripts" / script).read_text()
            values = re.search(r"\$smoothTailTokenCounts = @\(([^)]+)\)", text)[1]
            self.assertEqual([int(v.strip()) for v in values.split(",")],
                             [32, 64, 128, 256, 512, 1024, 2048, 4096])


if __name__ == "__main__":
    unittest.main()
