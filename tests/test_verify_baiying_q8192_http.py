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

    def test_first_token_observation_requires_real_prompt_and_logit(self):
        _, _, oracle = self.fixture()
        oracle["prompt"]["u32le_fnv1a64"] = "fixture-fnv"
        oracle["expected"].update(first_token_raw_logit=10.375, first_token_raw_logit_tolerance=0.125)
        row = {"type": "qrt_server_first_token_observation", "contract_version": 1, "available": True,
               "prefix_route_used": False, "input_tokens": 8192, "output_tokens": 32,
               "prompt_token_ids_fnv1a64": "fixture-fnv", "output_token_id": 144,
               "source": "qrt_engine_report.baseline_output_head_topk_logits[0]",
               "first_token_raw_logit": 10.375}
        encode = lambda rows: "\n".join(json.dumps(value, separators=(",", ":")) for value in rows)
        self.assertEqual(len(subject.check_first_token_observations(encode([row, row]), oracle)), 2)
        for key, value in (("available", False), ("prefix_route_used", True), ("output_token_id", 220),
                           ("first_token_raw_logit", 10.625), ("first_token_raw_logit", float("nan")),
                           ("first_token_raw_logit", None), ("prompt_token_ids_fnv1a64", "different")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                subject.check_first_token_observations(encode([row, {**row, key: value}]), oracle)
        with self.assertRaises(ValueError):
            subject.check_first_token_observations(encode([row]), oracle)

    def test_timing_contract_separates_total_and_mean(self):
        metrics = {"timing_contract_version": 2, "tpot_samples": 31, "decode_total_ms": 1011.8258,
                   "tpot_ms": 1011.8258 / 31}
        subject.check_timing_contract(metrics)
        for key, value in (("tpot_ms", 1011.8258), ("tpot_samples", 0), ("timing_contract_version", 1),
                           ("tpot_ms", float("inf")), ("decode_total_ms", 0)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                subject.check_timing_contract({**metrics, key: value})

    def test_stream_null_usage_on_regular_chunks_is_not_final_usage(self):
        events = self.events()
        for event in events[:2]:
            chunk = json.loads(event["data"])
            chunk["usage"] = None
            event["data"] = json.dumps(chunk)
        self.assertEqual(subject.check_stream(events, "hello"), {"ttft_ms": 123})
        events[-2]["data"] = json.dumps({"choices": [], "usage": None})
        with self.assertRaisesRegex(ValueError, "finish/usage count"):
            subject.check_stream(events, "hello")

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

    def test_saved_replay_requires_original_input_and_healthy_owned_exit(self):
        prompt, expected, oracle = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {name: root / f"{name}.json" for name in
                     ("prompt", "oracle", "result", "completion", "stream", "service", "guard")}
            subject.save_json(paths["prompt"], prompt)
            subject.save_json(paths["oracle"], oracle)
            completion = {"object": "text_completion", "choices": [{"text": "hello", "token_ids": expected,
                          "finish_reason": "length"}], "usage": {"prompt_tokens": 8192, "completion_tokens": 32,
                          "total_tokens": 8224}, "qrt_metrics": {"ttft_ms": 123}}
            subject.save_json(paths["completion"], completion)
            subject.save_json(paths["stream"], self.events())
            original = {"status": "failed", "error": "usage before finish", "pid": 123,
                        "server_commit": "test-commit", "model": "/model", "server_exit_code": 0,
                        "nonstream_exact_32_tokens": True, "ready_wall_ms": 123,
                        "health_before": {"load": {}},
                        "health_during_sse": {"pid": 123, "ready": True,
                                               "queue": {"started_total": 2, "completed_total": 1,
                                                         "active_requests": 1}},
                        "preflight": {"pass": True, "files": {name: {"sha256": subject.fingerprint(paths[name])}
                                                               for name in ("prompt", "oracle")}}}
            subject.save_json(paths["result"], original)
            subject.save_json(paths["service"], {"status": "stopped", "pid": 123, "repo_commit": "test-commit",
                                               "host": "BAIYING", "model_path": "/model", "provider_dll": "/provider"})
            guard = {"host": "BAIYING", "spec": {"model": "/model", "repo_commit": "test-commit"},
                     "reason": "completed", "host_checks_pass": True, "host_checks": {"same_boot": True}}
            subject.save_json(paths["guard"], guard)
            result = subject.replay_saved_run(root, paths["guard"], paths["prompt"], paths["oracle"])
            self.assertEqual(result["status"], "offline_http_result_validation_pass")
            self.assertEqual(result["original_controller_status"], "failed")
            self.assertFalse(result["full_product_gate_pass"])
            self.assertFalse(result["post_request_queue_sample_available"])
            for field, value in (("reason", "timeout"), ("host_checks_pass", False), ("host", "other-host")):
                paths["guard"].write_text(json.dumps({**guard, field: value}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    subject.replay_saved_run(root, paths["guard"], paths["prompt"], paths["oracle"])
            paths["guard"].write_text(json.dumps(guard))
            prompt[-1] = 33
            paths["prompt"].write_text(json.dumps(prompt))
            with self.assertRaises(ValueError):
                subject.replay_saved_run(root, paths["guard"], paths["prompt"], paths["oracle"])

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
