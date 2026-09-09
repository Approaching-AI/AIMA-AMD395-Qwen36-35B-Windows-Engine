import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FlaStateVariantContractTests(unittest.TestCase):
    def test_state_variant_preserves_bf16_boundaries_and_default(self) -> None:
        path = ROOT / "native/generators/compile_q8192_fla_chunk_gdn.py"
        source = path.read_text()
        tree = ast.parse(source)
        kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_fla_chunk_state_kernel")
        self.assertEqual(kernel.args.args[-1].arg, "IEEE_DOT")
        code = ast.get_source_segment(source, kernel)
        self.assertIn("tl.trans(h1).to(w1.dtype).to(tl.float32)", code)
        self.assertIn("current_v = current_v.to(k.dtype.element_ty)", code)
        self.assertEqual(code.count('input_precision="ieee"'), 4)
        self.assertIn('default="wmma"', source)
        self.assertIn('constexprs={"IEEE_DOT": state_dot == "ieee"}', source)
        self.assertIn('"numerics": {"state_dot": state_dot', source)

    def test_builder_binds_aot_mode_and_keeps_native_runtime_profile_unchanged(self) -> None:
        source = (ROOT / "scripts/baiying_build_fla_gdn.ps1").read_text()
        self.assertIn("[ValidateSet('wmma','ieee')][string]$StateDot = 'wmma'", source)
        self.assertIn("$meta.numerics.state_dot -ne $StateDot", source)
        self.assertIn("--state-dot $StateDot", source)
        profile = (ROOT / "engine/runtime.env").read_text()
        self.assertNotIn("STATE_DOT_IEEE", profile)

    def test_reference_ir_audit_refuses_device_visibility_before_importing_triton(self) -> None:
        script = ROOT / "scripts/audit_fla_reference_state_ir.py"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
                env = dict(os.environ, HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1", CUDA_VISIBLE_DEVICES="-1")
                env[key] = "0"
                result = subprocess.run([sys.executable, str(script), "--source", "not-read.py", "--output-dir", str(output)],
                                        env=env, capture_output=True, text=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"CPU-only compiler requires {key}=-1", result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
