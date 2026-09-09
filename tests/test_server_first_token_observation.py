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
    def test_observation_rejects_prefix_stale_and_nonfinite_reports(self):
        bridge = (ROOT / "native/src/qrt_server_bridge.c").read_text()
        declaration = "static void qrt_server_write_first_token_observation("
        actual = declaration + bridge.split(declaration, 1)[1].split(
            "static int qrt_server_store_tokens(", 1)[0]
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
