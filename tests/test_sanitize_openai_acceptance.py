import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sanitize_openai_acceptance import sanitize_report  # noqa: E402


class SanitizeOpenAiAcceptanceTests(unittest.TestCase):
    def test_removes_paths_addresses_pids_and_normalizes_artifact_names(self) -> None:
        source = {
            "passed": True,
            "endpoint": "http://127.0.0.1:8000",
            "pid": 42,
            "checks": [
                {
                    "name": "build_provenance",
                    "details": {
                        "path": r"D:\private\build.json",
                        "artifacts": [
                            {
                                "name": r"D:\private\provider.dll",
                                "sha256": "a" * 64,
                            }
                        ],
                    },
                }
            ],
        }
        clean = sanitize_report(json.dumps(source).encode())
        encoded = json.dumps(clean)
        self.assertNotIn("D:", encoded)
        self.assertNotIn("endpoint", clean)
        self.assertNotIn("pid", clean)
        self.assertEqual(
            clean["checks"][0]["details"]["artifacts"][0]["name"],
            "provider.dll",
        )
        self.assertEqual(
            clean["record_type"], "openai_http_product_acceptance_publication"
        )

    def test_rejects_private_text_in_an_unclassified_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "deployment-local"):
            sanitize_report(json.dumps({"message": r"D:\private\state"}).encode())

    def test_rejects_cgnat_addresses_used_by_private_mesh_networks(self) -> None:
        address = ".".join(("100", "91", "39", "109"))
        with self.assertRaisesRegex(ValueError, "deployment-local"):
            sanitize_report(json.dumps({"message": f"authority {address}"}).encode())


if __name__ == "__main__":
    unittest.main()
