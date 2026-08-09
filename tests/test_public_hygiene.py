from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from public_hygiene import scan_bytes, scan_public_tree  # noqa: E402


class PublicHygieneTests(unittest.TestCase):
    def test_detects_private_deployment_details(self) -> None:
        payload = b"endpoint=" + b"192." + b"168.1.7\npath=/ho" + b"me/alice/model\n"
        self.assertEqual(
            {finding.rule for finding in scan_bytes("sample.txt", payload)},
            {"private-ipv4", "private-home-path"},
        )

    def test_allows_placeholders(self) -> None:
        payload = b'api_key="placeholder"\npassword="<set-at-runtime>"\n'
        self.assertEqual(scan_bytes("sample.env", payload), [])

    def test_current_public_tree_is_clean(self) -> None:
        self.assertEqual(scan_public_tree(ROOT), [])


if __name__ == "__main__":
    unittest.main()
