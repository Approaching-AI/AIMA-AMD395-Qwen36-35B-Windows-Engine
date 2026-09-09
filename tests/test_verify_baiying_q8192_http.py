import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_baiying_q8192_http as subject


class BoundedHttpTests(unittest.TestCase):
    def fixture(self):
        prompt, expected = [32] * 8192, [144] * 32
        oracle = {"status": "pass", "correctness_authority": {"host": "gb10-4t"},
                  "prompt": {"token_count": 8192, "u32le_sha256": subject.token_digest(prompt)},
                  "expected": {"output_token_ids": expected, "continuation_token_count": 32,
                               "first_token_id": 144,
                               "output_token_ids_u32le_sha256": subject.token_digest(expected)}}
        return prompt, expected, oracle

    def events(self):
        chunks = [{"choices": [{"text": "hello", "finish_reason": None}]},
                  {"choices": [{"text": "", "finish_reason": "length"}]},
                  {"choices": [], "usage": {"prompt_tokens": 8192, "completion_tokens": 32,
                                             "total_tokens": 8224}, "qrt_metrics": {"ttft_ms": 123}}]
        return [{"data": json.dumps(chunk), "ms": index} for index, chunk in enumerate(chunks)] + [
            {"data": "[DONE]", "ms": 4}]

    def test_reference_shape_and_digests(self):
        prompt, expected, oracle = self.fixture()
        self.assertEqual(subject.validate_fixture(prompt, oracle), expected)
        prompt[-1] = 33
        with self.assertRaisesRegex(ValueError, "prompt digest"):
            subject.validate_fixture(prompt, oracle)

    def test_short_reference_is_rejected(self):
        prompt, _, oracle = self.fixture()
        oracle["expected"]["output_token_ids"].pop()
        with self.assertRaisesRegex(ValueError, "complete 32"):
            subject.validate_fixture(prompt, oracle)

    def test_wrong_authority_is_rejected(self):
        prompt, _, oracle = self.fixture()
        oracle["correctness_authority"]["host"] = "engine-self-hash"
        with self.assertRaisesRegex(ValueError, "authority"):
            subject.validate_fixture(prompt, oracle)

    def test_invalid_tokens_are_rejected(self):
        for tokens in ([True], [-1], [2**32], [1.0], "32"):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                subject.token_digest(tokens)

    def test_stream_success(self):
        self.assertEqual(subject.check_stream(self.events(), "hello"), {"ttft_ms": 123})

    def test_stream_bad_content_or_shape_is_rejected(self):
        events = self.events()
        for broken in (events[:-1], events + [events[-1]],
                       [events[0], events[2], events[1], events[-1]],
                       [events[0], events[1], events[0], events[2], events[-1]],
                       [{"data": '{"error":"native failure"}'}, events[-1]]):
            with self.subTest(events=broken), self.assertRaises((ValueError, json.JSONDecodeError)):
                subject.check_stream(broken, "hello")
        with self.assertRaisesRegex(ValueError, "text differs"):
            subject.check_stream(events, "different")

    def test_nonstream_requires_exact_ids_not_only_text(self):
        value = {"object": "text_completion", "choices": [{"text": "hello", "token_ids": [144] * 32,
                  "finish_reason": "length"}], "usage": {"prompt_tokens": 8192, "completion_tokens": 32,
                  "total_tokens": 8224}}
        subject.check_completion(value, [144] * 32, "hello")
        broken = copy.deepcopy(value)
        broken["choices"][0]["token_ids"][-1] = 220
        with self.assertRaisesRegex(ValueError, "sequence differs"):
            subject.check_completion(broken, [144] * 32, "hello")

    def test_inventory_checks_files_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "kernel.bin"
            artifact.write_bytes(b"kernel")
            manifest = root / "runtime-manifest.json"
            entry = {"path": artifact.name, "bytes": 6, "sha256": subject.fingerprint(artifact)}
            subject.save_json(manifest, {"artifacts": [entry]})
            self.assertEqual(subject.verify_inventory(manifest), 1)
            artifact.write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                subject.verify_inventory(manifest)
            manifest.write_text(json.dumps({"artifacts": [{**entry, "path": "../escape"}]}))
            with self.assertRaisesRegex(ValueError, "escapes"):
                subject.verify_inventory(manifest)

    def test_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "result.json"
            subject.save_json(target, {"original": True})
            with self.assertRaises(FileExistsError):
                subject.save_json(target, {"original": False})

    @unittest.skipIf(os.name == "nt", "non-Windows negative control")
    def test_execution_is_denied_outside_windows(self):
        with self.assertRaisesRegex(ValueError, "native Windows"):
            subject.require_windows_job()

    def test_command_is_foreground_loopback_with_bounded_queue(self):
        config = {"files": {key: {"path": f"/runtime/{key}"} for key in
                            ("server", "provider", "env", "runtime_manifest")},
                  "model": "/model", "model_id": "test", "port": 18093}
        command = subject.server_command(config, Path("/output"))
        self.assertEqual(command[1], "serve")
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--max-queue-depth") + 1], "1")
        self.assertIn("QRT_QWEN36_HAWKEYE_CORRECTION_MAXIMUM_BLOCKS_PER_LAUNCH=8", command)
        self.assertNotIn("--allow-unauthenticated", command)


if __name__ == "__main__":
    unittest.main()
