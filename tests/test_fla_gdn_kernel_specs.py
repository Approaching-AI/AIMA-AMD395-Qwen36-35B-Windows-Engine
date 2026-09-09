import ast
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FlaGdnKernelSpecsTests(unittest.TestCase):
    def setUp(self) -> None:
        source = ROOT / "native/generators/compile_q8192_fla_chunk_gdn.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "write_provider_kernel_specs")
        namespace = {"Path": Path, "json": json, "Any": object}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        self.write_specs = namespace["write_provider_kernel_specs"]
        self.names = ("qk_l2norm", "v_beta_copy", "gate_cumsum", "scaled_dot_kkt",
                      "solve_tril_64", "recompute_w_u", "chunk_state", "chunk_output")
        self.records = [dict(name=name, file=name + ".hsaco", symbol="_" + name,
                             threads=128, dynamic_shared_bytes=12352)
                        for name in self.names]

    def test_header_uses_exact_compiler_resources_and_abi_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "specs.inc"
            self.write_specs(list(reversed(self.records)), path)
            text = path.read_text()
            self.assertEqual(text.count("128u, 12352u"), 8)
            positions = [text.index('"' + name + '.hsaco"') for name in self.names]
            self.assertEqual(positions, sorted(positions))

    def test_missing_or_invalid_resource_record_cannot_emit_a_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "specs.inc"
            with self.assertRaises(KeyError):
                self.write_specs(self.records[:-1], path)
            self.assertFalse(path.exists())
            for field, value in (("threads", 0), ("threads", 33),
                                 ("threads", 2048), ("dynamic_shared_bytes", 65537)):
                records = [dict(record) for record in self.records]
                records[0][field] = value
                with self.assertRaises(ValueError):
                    self.write_specs(records, path)
                self.assertFalse(path.exists())

    def test_bf16_product_is_promoted_before_rounding_not_after(self) -> None:
        source = (ROOT / "native/generators/compile_q8192_fla_chunk_gdn.py").read_text()
        self.assertNotIn("b_v * b_beta", source)
        self.assertNotIn("b_k * b_beta", source)
        self.assertIn("b_v.to(tl.float32) * b_beta[:, None].to(tl.float32)", source)
        self.assertEqual(source.count("b_k.to(tl.float32) * b_beta[:, None].to(tl.float32)"), 2)
        self.assertIn("bits + 0x7FFF + ((bits >> 16) & 1)", source)
        self.assertIn("(bits | 0x00400000) & 0xFFFF0000", source)

    def test_wsl_aot_build_cannot_discover_a_gpu_and_has_a_hard_timeout(self) -> None:
        source = (ROOT / "scripts/baiying_build_fla_gdn.ps1").read_text()
        self.assertIn("timeout --kill-after=5 $innerTimeout", source)
        self.assertIn("HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1", source)
        self.assertIn("OMP_NUM_THREADS=2 MAX_JOBS=2", source)

    def test_w_u_use_ieee_f32_after_the_bf16_product_boundaries(self) -> None:
        source = (ROOT / "native/generators/compile_q8192_fla_chunk_gdn.py").read_text()
        tree = ast.parse(source)
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_fla_recompute_w_u_kernel")
        for target_name, right in (("b_u", "b_v_beta"), ("b_w", "b_k_beta_g")):
            with self.subTest(target=target_name):
                dot = next(node.value for node in ast.walk(fn) if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == target_name
                                   for target in node.targets))
                self.assertEqual(ast.unparse(dot.args[0]), "b_a.to(tl.float32)")
                self.assertEqual(ast.unparse(dot.args[1]), right + ".to(tl.float32)")
                self.assertEqual([(kw.arg, ast.literal_eval(kw.value)) for kw in dot.keywords],
                                 [("input_precision", "ieee")])


if __name__ == "__main__":
    unittest.main()
