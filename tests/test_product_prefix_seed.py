"""A bad first suffix retry must fail before a later warm hit can hide it."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class ProductPrefixSeedTests(unittest.TestCase):
    def test_retry_tokens_token_bound_logit_and_restoration(self):
        product = (ROOT / "native/src/product_cli.c").read_text()
        options = product.split("typedef struct qrt_product_options_t {", 1)[1].split(
            "} qrt_product_options_t;", 1)[0]
        source = r'''
#include "qrt.h"
#include "qrt_prefix_logit.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct qrt_product_options_t {
''' + options + "} qrt_product_options_t;\n" + "\n".join(
            function(product, signature) for signature in (
                "static uint64_t qrt_product_fnv1a64_bytes(",
                "static void qrt_product_print_tokens(",
                "static int qrt_product_record_prefix_seed(",
            )
        ) + r'''
int main(int argc, char **argv) {
    qrt_product_options_t options = {0};
    qrt_qwen36_resident_prefix_cache_fallback_result_v1_t seed = {0};
    qrt_qwen36_resident_prefix_cache_result_v1_t *retry = &seed.hit_result;
    uint32_t expected[512], actual[512];
    const char *test = argc > 1 ? argv[1] : "valid";
    size_t index;
    for (index = 0; index < 512; ++index) expected[index] = actual[index] = (uint32_t)index + 100;
    options.output_token_capacity = 512;
    options.prefix_token_count = 131072;
    options.expected_output_fnv1a64_set = 1;
    options.expected_output_fnv1a64 = qrt_product_fnv1a64_bytes(expected, sizeof(expected));
    seed.completed = seed.fallback_invoked = seed.retry_invoked = 1;
    seed.full_prefill_token_count = 131072;
    seed.seed_output_token_count = 1;
    seed.seed_output_token = 16;
    seed.seed_elapsed_ns = 1000000;
    seed.retry_elapsed_ns = 2000000;
    retry->completed = retry->provider_invoked = retry->exact_prefix_match = 1;
    retry->copy_on_write_transaction = retry->state_restored = 1;
    retry->prefix_token_count = 131072;
    retry->suffix_token_count = 1024;
    retry->output_token_count = 512;
    retry->input_token_ids_fnv1a64 = 123;
    retry->output_token_ids_fnv1a64 = options.expected_output_fnv1a64;
    qrt_prefix_first_logit_store(retry->reserved, actual[0], 6.28125f);
    if (!strcmp(test, "first_token")) actual[0] += 1;
    if (!strcmp(test, "last_token")) actual[511] += 1;
    if (!strcmp(test, "zero_outputs")) retry->output_token_count = 0;
    if (!strcmp(test, "short_outputs")) retry->output_token_count = 511;
    if (!strcmp(test, "overflow_outputs")) retry->output_token_count = UINT32_MAX;
    if (!strcmp(test, "no_restore")) retry->state_restored = 0;
    if (!strcmp(test, "wrong_prefix")) retry->prefix_token_count -= 32;
    if (!strcmp(test, "wrong_suffix")) retry->suffix_token_count -= 1;
    if (!strcmp(test, "wrong_prompt")) retry->input_token_ids_fnv1a64 ^= 1;
    if (!strcmp(test, "owner_not_output1")) seed.seed_output_token_count = 32;
    if (!strcmp(test, "wrong_owner_extent")) seed.full_prefill_token_count -= 8192;
    if (!strcmp(test, "missing_logit")) retry->reserved[0] = 0;
    if (!strcmp(test, "owner_logit")) qrt_prefix_first_logit_store(retry->reserved, 16, 25.0f);
    if (!strcmp(test, "nan_logit")) retry->reserved[1] = ((uint64_t)actual[0] << 32) | UINT32_C(0x7fc00000);
    if (!strcmp(test, "inf_logit")) retry->reserved[1] = ((uint64_t)actual[0] << 32) | UINT32_C(0x7f800000);
    if (!strcmp(test, "bad_expected_digest")) options.expected_output_fnv1a64 ^= 1;
    if (!strcmp(test, "late_token_self_consistent")) {
        actual[511] += 1;
        retry->output_token_ids_fnv1a64 = qrt_product_fnv1a64_bytes(actual, sizeof(actual));
        options.expected_output_fnv1a64_set = 0;
    }
    if (!strcmp(test, "tokens_only")) options.expected_output_fnv1a64_set = 0;
    return qrt_product_record_prefix_seed(&options, 132096, 123, actual,
        !strcmp(test, "no_expected_tokens") ? NULL : expected, &seed) ? 0 : 6;
}
'''
        failures = (
            "first_token", "last_token", "zero_outputs", "short_outputs",
            "overflow_outputs", "no_restore", "wrong_prefix", "wrong_suffix",
            "wrong_prompt", "owner_not_output1", "wrong_owner_extent",
            "missing_logit", "owner_logit", "nan_logit", "inf_logit",
            "bad_expected_digest", "late_token_self_consistent",
        )
        with tempfile.TemporaryDirectory(prefix="qrt-prefix-seed-") as temporary:
            executable = str(Path(temporary) / "prefix-seed")
            subprocess.run(
                [os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                 "-fsanitize=address,undefined", "-I", str(ROOT / "native/src"),
                 "-x", "c", "-", "-o", executable],
                input=source, text=True, check=True, timeout=30,
            )
            for test in ("valid", "tokens_only", "no_expected_tokens", *failures):
                with self.subTest(test=test):
                    run = subprocess.run([executable, test], capture_output=True, text=True, timeout=5)
                    self.assertEqual(run.returncode, 6 if test in failures else 0, run.stderr)
                    self.assertEqual(run.stderr, "")
                    if test in ("zero_outputs", "short_outputs", "overflow_outputs"):
                        self.assertEqual(run.stdout, "")
                        continue
                    row = json.loads(run.stdout)
                    self.assertEqual(row["type"], "prefix_seed")
                    self.assertEqual(row["status"], "fail" if test in failures else "pass")
                    self.assertEqual(row["retry_output_tokens"], 512)
                    if test == "valid":
                        self.assertEqual(row["owner_output_tokens"], 1)
                        self.assertEqual(row["owner_first_token"], 16)
                        self.assertEqual(row["retry_output_token_ids"], list(range(100, 612)))
                        self.assertEqual(row["retry_first_token_raw_logit"], 6.28125)
                        self.assertEqual(row["owner_wall_ms"], 1.0)
                        self.assertEqual(row["retry_wall_ms"], 2.0)
                    if test in ("missing_logit", "owner_logit", "nan_logit", "inf_logit", "first_token"):
                        self.assertFalse(row["retry_first_token_raw_logit_available"])
                        self.assertIsNone(row["retry_first_token_raw_logit"])
                    if test == "late_token_self_consistent":
                        self.assertTrue(row["retry_contract_pass"])
                        self.assertFalse(row["retry_expected_output_tokens_match"])
                    if test == "no_expected_tokens":
                        self.assertFalse(row["retry_expected_output_tokens_supplied"])


if __name__ == "__main__":
    unittest.main()
