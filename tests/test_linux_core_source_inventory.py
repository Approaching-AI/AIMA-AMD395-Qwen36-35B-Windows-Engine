import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "tools/check_linux_core_q8192.py"
SPEC = importlib.util.spec_from_file_location("linux_core_checker", PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class SourceInventoryTests(unittest.TestCase):
    def setUp(self):
        self.entries = [dict(path="third_party/LICENSE", bytes=10, sha256="a" * 64),
                        dict(path="third_party/benchmarks/source.cpp", bytes=20, sha256="b" * 64)]

    def test_windows_and_posix_order_have_identical_file_identities(self):
        self.assertEqual(CHECKER.source_input_inventory(self.entries),
                         CHECKER.source_input_inventory(list(reversed(self.entries))))

    def test_missing_changed_or_extra_file_still_differs(self):
        expected = CHECKER.source_input_inventory(self.entries)
        for candidate in (self.entries[:1],
                          [dict(self.entries[0], bytes=11), self.entries[1]],
                          [dict(self.entries[0], sha256="c" * 64), self.entries[1]],
                          [dict(self.entries[0], path="third_party/other"), self.entries[1]],
                          self.entries + [dict(path="extra", bytes=1, sha256="d" * 64)]):
            with self.subTest(candidate=candidate):
                self.assertNotEqual(expected, CHECKER.source_input_inventory(candidate))

    def test_duplicate_windows_path_aliases_are_rejected(self):
        for path in ("third_party/LICENSE", "THIRD_PARTY/license", "third_party\\LICENSE"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "duplicate"):
                CHECKER.source_input_inventory(self.entries + [dict(self.entries[0], path=path)])

    def test_invalid_inventory_is_rejected(self):
        for candidate in (None, [], {}, [None], [dict(self.entries[0], bytes=True)],
                          [dict(self.entries[0], bytes=-1)], [dict(self.entries[0], sha256="z" * 64)],
                          [dict(self.entries[0], path="")]):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                CHECKER.source_input_inventory(candidate)


if __name__ == "__main__":
    unittest.main()
