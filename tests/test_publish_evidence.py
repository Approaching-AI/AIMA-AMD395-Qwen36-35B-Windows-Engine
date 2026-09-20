import hashlib
import json
import unittest
from tools.publish_evidence import redact_evidence


class PublishEvidenceTests(unittest.TestCase):
    def test_nested_commands_keys_and_raw_hashes(self):
        first = "/" + "Users/" + "reviewer"
        second = "/" + "home/" + "reference"
        digest = "f" * 64
        raw = (json.dumps({
            "command": [first + "/runtime", second + "/capture.py"],
            "stdout": "output\n" + first + "/evidence.bin\nresult",
            "files": {first + "/evidence.bin": {"sha256": digest, "bytes": 8192}},
            "samples": [{"token": 255, "logit": 17.875}, {"token": 82, "logit": 9.5625}],
            "passed": True,
        }, indent=2) + "\n").encode()
        output = redact_evidence(raw, source_revision="revision", source_path="evidence.json")
        document = json.loads(output)
        self.assertNotIn(first, output.decode())
        self.assertNotIn(second, output.decode())
        self.assertEqual(document["samples"], json.loads(raw)["samples"])
        self.assertEqual(next(iter(document["files"].values())), {"sha256": digest, "bytes": 8192})
        metadata = document["public_evidence_view"]
        self.assertEqual(metadata["original_json_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(sum(metadata["home_path_alias_counts"].values()), 4)
        self.assertFalse(metadata["embedded_artifact_hashes_changed"])
        self.assertEqual(redact_evidence(output, source_revision="another", source_path="other"), output)

    def test_existing_alias_collision_and_invalid_inputs(self):
        private = "/" + "home/" + "reference"
        raw = json.dumps({private + "/x": 1, "<evidence-home-1>/x": 2}).encode()
        with self.assertRaises(ValueError):
            redact_evidence(raw, source_revision="revision", source_path="evidence.json")
        for raw in (b"[]", b"null", b"malformed"):
            with self.assertRaises(ValueError):
                redact_evidence(raw, source_revision="revision", source_path="evidence.json")
        raw = b'{ "sha256": "original", "token": 144 }\n'
        self.assertEqual(redact_evidence(raw, source_revision="revision", source_path="evidence.json"), raw)
        with self.assertRaises(ValueError):
            redact_evidence(raw, source_revision="revision", source_path=private + "/evidence.json")


if __name__ == "__main__":
    unittest.main()
