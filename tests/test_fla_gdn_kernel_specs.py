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


if __name__ == "__main__":
    unittest.main()
