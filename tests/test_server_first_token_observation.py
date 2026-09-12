"""Compile the real report logger; no provider, model or GPU is involved."""

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ServerFirstTokenObservationTests(unittest.TestCase):
    def test_prefix_observation_requires_restored_result_and_matching_token_extension(self):
        bridge = (ROOT / "native/src/qrt_server_bridge.c").read_text()
        declaration = "static void qrt_server_write_prefix_first_token_observation("
        actual = declaration + bridge.split(declaration, 1)[1].split(
            "static size_t qrt_server_prefix_cache_min_tokens(", 1)[0]
        source = r'''
#include "qrt.h"
#include "qrt_prefix_logit.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
''' + actual + r'''
int main(void) {
    qrt_qwen36_resident_prefix_cache_result_v1_t *r = calloc(1u, sizeof(*r));
    const uint32_t input[] = {32, 33, 34}, output[] = {82};
    if (r == NULL) return 1;
    r->completed = r->state_restored = r->exact_prefix_match = 1u;
    r->output_token_count = 1u; r->prefix_token_count = 2u; r->output_tokens[0] = 82u;
    qrt_prefix_first_logit_store(r->reserved, 82u, 9.4375f);
#define OBSERVE() qrt_server_write_prefix_first_token_observation(stdout, r, input, 3u, output, 1u)
    OBSERVE();
    r->state_restored = 0u; OBSERVE(); r->state_restored = 1u;
    r->completed = 0u; OBSERVE(); r->completed = 1u;
    r->exact_prefix_match = 0u; OBSERVE(); r->exact_prefix_match = 1u;
    r->output_token_count = 0u; OBSERVE(); r->output_token_count = 1u;
    r->output_tokens[0] = 220u; OBSERVE(); r->output_tokens[0] = 82u;
    qrt_prefix_first_logit_store(r->reserved, 220u, 9.4375f); OBSERVE();
    qrt_prefix_first_logit_store(r->reserved, 82u, NAN); OBSERVE();
    qrt_prefix_first_logit_store(r->reserved, 82u, INFINITY); OBSERVE();
    r->reserved[0] = 0u; r->reserved[1] = 0u; OBSERVE();
    qrt_server_write_prefix_first_token_observation(stdout, NULL, input, 3u, output, 1u);
    qrt_server_write_prefix_first_token_observation(stdout, r, input, 3u, NULL, 0u);
    free(r); return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-prefix-observation-") as temporary:
            executable = str(Path(temporary) / "prefix-observation-test")
            subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "native/src"), "-x", "c", "-", "-o", executable],
                           input=source, text=True, check=True, timeout=30)
            completed = subprocess.run([executable], text=True, capture_output=True,
                                       check=True, timeout=5)
        values = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(len(values), 12)
        self.assertTrue(values[0]["available"])
        self.assertEqual(values[0]["output_token_id"], 82)
        self.assertEqual(values[0]["prefix_tokens"], 2)
        self.assertEqual(values[0]["first_token_raw_logit"], 9.4375)
        for value in values[1:]:
            self.assertFalse(value["available"])
            self.assertIsNone(value["first_token_raw_logit"])

    def test_observation_rejects_prefix_stale_and_nonfinite_reports(self):
        bridge = (ROOT / "native/src/qrt_server_bridge.c").read_text()
        declaration = "static void qrt_server_write_first_token_observation("
        actual = declaration + bridge.split(declaration, 1)[1].split(
            "static const char *qrt_server_environment_value(", 1)[0]
        source = r'''
#include "qrt.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
''' + actual + r'''
int main(void) {
    qrt_engine_report_t *report = calloc(1u, sizeof(*report));
    const uint32_t input[] = {32, 33, 34}, output[] = {144};
    if (report == NULL) return 1;
    report->baseline_output_head_token_emitted = 1;
    report->baseline_output_head_sampled_token_id = 144;
    report->baseline_output_head_topk_token_ids[0] = 144;
    report->baseline_output_head_topk_logits[0] = 10.375f;
    qrt_server_write_first_token_observation(stdout, report, input, 3, output, 1, 0, 1);
    qrt_server_write_first_token_observation(stdout, report, input, 3, output, 1, 1, 1);
    qrt_server_write_first_token_observation(stdout, report, input, 3, output, 1, 0, 0);
    report->baseline_output_head_topk_logits[0] = NAN;
    qrt_server_write_first_token_observation(stdout, report, input, 3, output, 1, 0, 1);
    report->baseline_output_head_topk_logits[0] = INFINITY;
    qrt_server_write_first_token_observation(stdout, report, input, 3, output, 1, 0, 1);
    report->baseline_output_head_topk_logits[0] = 10.375f;
    report->baseline_output_head_sampled_token_id = 220;
    qrt_server_write_first_token_observation(stdout, report, input, 3, output, 1, 0, 1);
    report->baseline_output_head_sampled_token_id = 144;
    report->baseline_output_head_topk_token_ids[0] = 220;
    qrt_server_write_first_token_observation(stdout, report, input, 3, output, 1, 0, 1);
    report->baseline_output_head_topk_token_ids[0] = 144;
    report->baseline_output_head_token_emitted = 0;
    qrt_server_write_first_token_observation(stdout, report, input, 3, output, 1, 0, 1);
    report->baseline_output_head_token_emitted = 1;
    qrt_server_write_first_token_observation(stdout, report, input, 3, NULL, 0, 0, 1);
    qrt_server_write_first_token_observation(stdout, NULL, input, 3, output, 1, 0, 1);
    free(report);
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-first-token-observation-") as temporary:
            executable = str(Path(temporary) / "observation-test")
            subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "native/src"), "-x", "c", "-", "-o", executable],
                           input=source, text=True, check=True, timeout=30)
            completed = subprocess.run([executable], text=True, capture_output=True, check=True, timeout=5)
        values = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(len(values), 10)
        self.assertTrue(values[0]["available"])
        self.assertEqual(values[0]["first_token_raw_logit"], 10.375)
        digest = 14695981039346656037
        for byte in struct.pack("<3I", 32, 33, 34):
            digest = ((digest ^ byte) * 1099511628211) & (2**64 - 1)
        self.assertEqual(values[0]["prompt_token_ids_fnv1a64"], f"{digest:016x}")
        for value in values[1:]:
            self.assertFalse(value["available"])
            self.assertIsNone(value["first_token_raw_logit"])
        self.assertIn('strcmp(observation, "1") == 0', bridge)
        self.assertIn("input_token_count == QRT_SERVER_RETAINED_Q8192_TOKENS", bridge)


if __name__ == "__main__":
    unittest.main()
