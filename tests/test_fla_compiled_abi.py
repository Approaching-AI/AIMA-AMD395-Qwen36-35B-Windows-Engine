import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FlaCompiledAbiTests(unittest.TestCase):
    def setUp(self) -> None:
        source = ROOT / "native/generators/compile_q8192_fla_chunk_gdn.py"
        tree = ast.parse(source.read_text())
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "compiled_kernel_abi")
        namespace = {"re": re}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        self.validate = namespace["compiled_kernel_abi"]
        self.signature = dict(q="*bf16", k="*bf16", v_new="*bf16", chunk_state="*bf16",
                              g_cumsum="*fp32", output_f32="*fp32", tokens="i32")
        self.arguments = [(x, 8, "global_buffer") for x in (0, 8, 16, 24, 32, 40)]
        self.arguments += [(48, 4, "by_value"), (56, 8, "global_buffer"), (64, 8, "global_buffer")]

    def assembly(self, arguments=None, size=72) -> str:
        arguments = self.arguments if arguments is None else arguments
        return (".amdgpu_metadata\n---\namdhsa.kernels:\n  - .args:\n" +
                "".join(f"      - .offset: {offset}\n        .size: {length}\n        .value_kind: {kind}\n"
                        for offset, length, kind in arguments) +
                f"    .group_segment_fixed_size: 0\n    .kernarg_segment_size: {size}\n")

    def test_actual_nine_slot_abi_is_not_confused_with_seven_source_parameters(self) -> None:
        result = self.validate(self.assembly(), self.signature)
        self.assertEqual(len(self.signature), 7)
        self.assertEqual(result["compiled_abi"][-2:], ["global_scratch", "profile_scratch"])
        self.assertEqual(len(result["compiled_abi"]), 9)
        self.assertEqual(result["compiled_argument_offsets"], [0, 8, 16, 24, 32, 40, 48, 56, 64])
        self.assertEqual(result["kernarg_segment_bytes"], 72)

    def test_missing_slots_or_stale_offsets_cannot_emit_launch_metadata(self) -> None:
        with self.assertRaises(ValueError):
            self.validate(self.assembly(self.arguments[:-2]), self.signature)
        changed = self.arguments[:-1] + [(56, 8, "global_buffer")]
        with self.assertRaises(ValueError):
            self.validate(self.assembly(changed), self.signature)
        with self.assertRaises(ValueError):
            self.validate(self.assembly(size=64), self.signature)
        with self.assertRaises(ValueError):
            self.validate("no AMDGPU metadata", self.signature)


if __name__ == "__main__":
    unittest.main()
