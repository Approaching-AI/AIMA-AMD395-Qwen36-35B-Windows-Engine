"""Exercise the actual product CLI with a deterministic, non-model backend."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]

BACKEND = r'''
struct qrt_engine { unsigned cache, seed_calls, owner_calls, hit_calls; uint32_t first; };
static int test_mode(const char *name) {
    const char *mode = getenv("QRT_TEST_OWNER_CASE");
    return mode && !strcmp(mode, name);
}
qrt_status_t qrt_engine_create(const qrt_engine_config_t *config, qrt_engine_t **out) {
    if (!config || config->context_tokens != 4 || config->batch_size != 1) abort();
    *out = calloc(1, sizeof(**out));
    return *out ? QRT_STATUS_OK : QRT_STATUS_OUT_OF_MEMORY;
}
void qrt_engine_free(qrt_engine_t *engine) {
    fprintf(stderr, "TEST_BACKEND seed=%u owner=%u hit=%u cache=%u preload=%u\n",
        engine->seed_calls, engine->owner_calls, engine->hit_calls, engine->cache,
        test_preload_calls);
    free(engine);
}
const char *qrt_strerror(qrt_status_t status) { (void)status; return "test backend"; }
qrt_status_t qrt_engine_report(const qrt_engine_t *engine, qrt_engine_report_t *report) {
    memset(report, 0, sizeof(*report));
    report->baseline_output_head_token_emitted = 1;
    report->baseline_output_head_sampled_token_id = engine->first;
    report->baseline_output_head_topk_token_ids[0] = engine->first;
    report->baseline_output_head_topk_logits[0] = 5.0f;
    report->last_request_ttft_elapsed_ns = 1000000;
    report->last_request_tpot_elapsed_ns = 3000000;
    report->last_request_tpot_sample_count = 3;
    return QRT_STATUS_OK;
}
static void test_result(qrt_qwen36_resident_prefix_cache_result_v1_t *result,
    const uint32_t *input, size_t count, size_t prefix,
    const uint32_t *output, size_t outputs) {
    memset(result, 0, sizeof(*result));
    result->completed = result->provider_invoked = result->exact_prefix_match = 1;
    result->copy_on_write_transaction = result->state_restored = 1;
    result->prefix_token_count = (uint32_t)prefix;
    result->suffix_token_count = (uint32_t)(count - prefix);
    result->output_token_count = (uint32_t)outputs;
    result->input_token_ids_fnv1a64 = qrt_product_fnv1a64_bytes(input, count * 4);
    result->output_token_ids_fnv1a64 = qrt_product_fnv1a64_bytes(output, outputs * 4);
    qrt_prefix_first_logit_store(result->reserved, output[0], 5.0f);
}
qrt_status_t qrt_engine_request_tokens_prefix_fallback_v1(
    qrt_engine_t *engine, const uint32_t *input, size_t count, size_t prefix,
    uint32_t *output, size_t capacity,
    qrt_qwen36_resident_prefix_cache_fallback_result_v1_t *seed) {
    if (count != 4 || prefix != 2 || capacity != 4 || engine->cache) abort();
    engine->cache = 1; ++engine->seed_calls;
    memset(seed, 0, sizeof(*seed));
    seed->completed = seed->fallback_invoked = seed->retry_invoked = 1;
    seed->seed_output_token_count = 1;
    seed->seed_output_token = 16;
    seed->full_prefill_token_count = 2;
    for (size_t i = 0; i < capacity; ++i) output[i] = (uint32_t)(100 + i);
    test_result(&seed->hit_result, input, count, prefix, output, capacity);
    return QRT_STATUS_OK;
}
qrt_status_t qrt_engine_request_tokens_prefix_stream_v1(
    qrt_engine_t *engine, const uint32_t *input, size_t count, size_t prefix,
    uint32_t *output, size_t capacity,
    qrt_qwen36_resident_prefix_cache_result_v1_t *result,
    qrt_token_stream_callback_v1_t callback, void *user) {
    const int owner = count == 3;
    if (prefix != 2 || input[0] != 1 || input[1] != 2) abort();
    if (owner) {
        /* This remains16 even when the comparison file's first ID differs. */
        if (input[2] != 16 || (capacity != 1 && capacity != 31 && capacity != 511)) abort();
        ++engine->owner_calls;
    } else {
        if (count != 4 || input[2] != 7 || input[3] != 8 || capacity != 4) abort();
        ++engine->hit_calls;
    }
    if (engine->cache != 1) return QRT_STATUS_UNSUPPORTED;
    engine->cache = 2;
    for (size_t i = 0; i < capacity; ++i) output[i] = (uint32_t)((owner ? 201 : 100) + i);
    if (owner && test_mode("late_token")) output[capacity - 1] += 1;
    test_result(result, input, count, prefix, output, capacity);
    engine->first = output[0];
    for (size_t i = 0; i < capacity; ++i) {
        qrt_token_stream_event_v1_t event = {0};
        event.struct_size = sizeof(event); event.abi_version = QRT_TOKEN_STREAM_EVENT_ABI_VERSION;
        event.output_index = (uint32_t)i;
        event.phase = i ? QRT_TOKEN_STREAM_PHASE_DECODE : QRT_TOKEN_STREAM_PHASE_PREFILL;
        event.token_id = output[i]; event.token_step_elapsed_ns = 1000;
        event.request_elapsed_ns = 1000 * (i + 1);
        if (owner) {
            if (test_mode("no_callbacks")) break;
            if (test_mode("short_callbacks") && i == capacity - 1) break;
            if (test_mode("callback_token") && i == capacity - 1) event.token_id += 1;
            if (test_mode("callback_index") && i == capacity - 1) event.output_index += 1;
            if (test_mode("callback_clock") && i == capacity - 1) event.request_elapsed_ns = 1;
            if (test_mode("callback_phase") && i == capacity - 1) event.phase = 99;
            if (test_mode("callback_first_phase") && i == 0) event.phase = QRT_TOKEN_STREAM_PHASE_DECODE;
            if (test_mode("callback_abi") && i == capacity - 1) event.abi_version += 1;
            if (test_mode("callback_size") && i == capacity - 1) event.struct_size = 4;
            if (test_mode("callback_vocab") && i == capacity - 1) event.token_id = QRT_QWEN36_VOCAB_SIZE;
            if (test_mode("callback_null")) { callback(user, NULL); break; }
            if (test_mode("callback_state_null")) { callback(NULL, &event); break; }
        }
        if (!callback(user, &event)) break;
    }
    engine->cache = 1;
    if (owner) {
        if (test_mode("no_restore") || test_mode("false_restore")) engine->cache = 2;
        if (test_mode("no_restore")) result->state_restored = 0;
        if (test_mode("no_cow")) result->copy_on_write_transaction = 0;
        if (test_mode("no_exact_match")) result->exact_prefix_match = 0;
        if (test_mode("not_completed")) result->completed = 0;
        if (test_mode("not_invoked")) result->provider_invoked = 0;
        if (test_mode("wrong_prefix")) result->prefix_token_count = 1;
        if (test_mode("wrong_suffix")) result->suffix_token_count = 2;
        if (test_mode("wrong_prompt_digest")) result->input_token_ids_fnv1a64 ^= 1;
        if (test_mode("wrong_output_digest")) result->output_token_ids_fnv1a64 ^= 1;
        if (test_mode("zero_outputs")) result->output_token_count = 0;
        if (test_mode("short_outputs")) result->output_token_count -= 1;
        if (test_mode("overflow_outputs")) result->output_token_count = UINT32_MAX;
        if (test_mode("no_logit")) result->reserved[0] = 0;
        if (test_mode("wrong_logit_token")) qrt_prefix_first_logit_store(result->reserved, output[0] + 1, 5.0f);
        if (test_mode("nan_logit")) result->reserved[1] = ((uint64_t)output[0] << 32) | UINT32_C(0x7fc00000);
        if (test_mode("input_changed")) ((uint32_t *)input)[0] = 42;
        if (test_mode("suffix_changed")) ((uint32_t *)input)[2] = 42;
        if (test_mode("provider_error")) return QRT_STATUS_UNSUPPORTED;
    }
    return QRT_STATUS_OK;
}
qrt_status_t qrt_engine_request_tokens_prefix_v1(qrt_engine_t *e, const uint32_t *i,
    size_t n, size_t p, uint32_t *o, size_t c, qrt_qwen36_resident_prefix_cache_result_v1_t *r) {
    (void)e; (void)i; (void)n; (void)p; (void)o; (void)c; (void)r; abort();
}
qrt_status_t qrt_engine_request_tokens(qrt_engine_t *e, const uint32_t *i,
    size_t n, uint32_t *o, size_t c, size_t *r) {
    (void)e; (void)i; (void)n; (void)o; (void)c; (void)r; abort();
}
qrt_status_t qrt_engine_request_tokens_stream_v1(qrt_engine_t *e, const uint32_t *i,
    size_t n, uint32_t *o, size_t c, size_t *r, qrt_token_stream_callback_v1_t cb, void *u) {
    (void)e; (void)i; (void)n; (void)o; (void)c; (void)r; (void)cb; (void)u; abort();
}
qrt_status_t qrt_engine_prefix_checkpoint_match_v1(qrt_engine_t *e,
    const uint32_t *i, size_t n, size_t c, size_t *p) {
    (void)e; (void)i; (void)n; (void)c; (void)p; abort();
}
'''


class ProductOwnerContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="qrt-owner-continuation-")
        cls.directory = Path(cls.temporary.name)
        cls.executable = cls.directory / "product"
        product = (ROOT / "native/src/product_cli.c").read_text()
        original = function(product, "static int qrt_product_preload_provider(")
        # Only the Windows DLL-loading boundary is replaced. The complete CLI,
        # parser, validators, timing separation and cleanup execute unchanged.
        preload = r'''
static unsigned test_preload_calls;
static int qrt_product_preload_provider(const char *path, const char *model,
    qrt_product_preload_t *result) {
    (void)path; (void)model; ++test_preload_calls;
    memset(result, 0, sizeof(*result)); result->completed = 1; return 1;
}
'''
        assert product.count(original) == 1
        source = product.replace(original, preload) + BACKEND
        subprocess.run(
            [os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
             "-fsanitize=address,undefined", "-I", str(ROOT / "native/src"),
             "-x", "c", "-", "-o", str(cls.executable)],
            input=source, text=True, check=True, timeout=40,
        )
        (cls.directory / "prompt.json").write_text("[1,2,7,8]")
        (cls.directory / "expected.json").write_text("[100,101,102,103]")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_case(self, mode="valid", expected=None, enabled=True, extra=()):
        owner = self.directory / "owner.json"
        if expected is None:
            expected = [16, *range(201, 232)]
        owner.write_text(json.dumps(expected))
        arguments = [str(self.executable), "run", "--model", "test-only",
                     "--tokens", str(self.directory / "prompt.json"),
                     "--expected-output", str(self.directory / "expected.json"),
                     "--output-tokens", "4", "--prefix-tokens", "2"]
        if enabled:
            arguments += ["--expected-owner-output", str(owner)]
        arguments += list(extra)
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=5,
                                env=dict(os.environ, QRT_TEST_OWNER_CASE=mode))
        self.assertNotIn("Sanitizer", result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        return result, rows

    def test_full_workflow_default_and_owner_counts(self):
        for count in (2, 32, 512):
            with self.subTest(count=count):
                expected = [16, *range(201, 200 + count)]
                result, rows = self.run_case(expected=expected)
                self.assertEqual(result.returncode, 0, result.stderr)
                owner = next(r for r in rows if r["type"] == "prefix_owner_continuation")
                self.assertEqual(owner["status"], "pass")
                self.assertEqual(owner["output_token_ids"], expected)
                self.assertEqual(owner["stream_callback_count"], count - 1)
                self.assertTrue(owner["state_restored"])
                self.assertEqual(owner["owner_first_token"], 16)
                self.assertEqual(owner["input_tokens"], 3)
                self.assertEqual(owner["suffix_tokens"], 1)
                self.assertEqual([r["token_id"] for r in rows if r["type"] == "owner_token"], expected[1:])
                self.assertEqual([r["token_id"] for r in rows if r["type"] == "token"], [100, 101, 102, 103])
                summary = rows[-1]
                self.assertEqual(summary["status"], "pass")
                self.assertTrue(summary["prefix_owner_continuation_requested"])
                self.assertEqual(summary["prefix_owner_continuation_tokens"], count)
                self.assertEqual(summary["stream_callback_count"], 4)
                self.assertEqual(summary["prefix_owner_continuation_ms"], owner["wall_ms"])
                self.assertIn("seed=1 owner=1 hit=1 cache=1 preload=1", result.stderr)
        result, rows = self.run_case(enabled=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(r["type"].startswith("owner_") or r["type"] == "prefix_owner_continuation" for r in rows))
        self.assertFalse(rows[-1]["prefix_owner_continuation_requested"])
        self.assertEqual(rows[-1]["prefix_owner_continuation_tokens"], 0)
        self.assertEqual(rows[-1]["prefix_owner_continuation_ms"], 0)
        self.assertIn("seed=1 owner=0 hit=1 cache=1 preload=1", result.stderr)

    def test_backend_failures_stop_before_timed_hit(self):
        failures = (
            "late_token", "no_callbacks", "short_callbacks", "callback_token",
            "callback_index", "callback_clock", "callback_phase", "callback_first_phase",
            "callback_abi", "callback_size", "callback_null", "callback_state_null",
            "callback_vocab", "no_restore", "no_cow", "no_exact_match",
            "not_completed", "not_invoked", "wrong_prefix", "wrong_suffix",
            "wrong_prompt_digest", "wrong_output_digest", "zero_outputs", "short_outputs",
            "overflow_outputs", "no_logit", "wrong_logit_token", "nan_logit",
            "input_changed", "suffix_changed", "provider_error",
        )
        for mode in failures:
            with self.subTest(mode=mode):
                result, rows = self.run_case(mode)
                self.assertEqual(result.returncode, 6, result.stderr)
                owner = rows[-1]
                self.assertEqual(owner["type"], "prefix_owner_continuation")
                self.assertEqual(owner["status"], "fail")
                self.assertFalse(any(r["type"] in ("token", "summary") for r in rows))
                self.assertIn("seed=1 owner=1 hit=0", result.stderr)
        result, rows = self.run_case("false_restore")
        self.assertEqual(result.returncode, 5, result.stderr)
        self.assertEqual(rows[-1]["type"], "prefix_owner_continuation")
        self.assertIn("seed=1 owner=1 hit=1 cache=2", result.stderr)
        self.assertFalse(any(r["type"] == "summary" for r in rows))

    def test_reference_is_never_an_inference_input(self):
        for index in (0, 31):
            expected = [16, *range(201, 232)]
            expected[index] += 1
            result, rows = self.run_case(expected=expected)
            self.assertEqual(result.returncode, 6, result.stderr)
            self.assertEqual(rows[-1]["output_token_ids"], [16, *range(201, 232)])
            self.assertFalse(rows[-1]["expected_output_tokens_match"])
            self.assertIn("seed=1 owner=1 hit=0", result.stderr)

    def test_invalid_options_and_owner_files_precede_model_load(self):
        for expected in ([], [16], [16] * 513, [16, 248320], [16, -1]):
            with self.subTest(expected_length=len(expected)):
                result, rows = self.run_case(expected=expected)
                self.assertEqual(result.returncode, 3, result.stderr)
                self.assertEqual(rows, [])
                self.assertNotIn("TEST_BACKEND", result.stderr)
        result, rows = self.run_case(extra=("--checkpoint-owner", "unused.json"))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(rows, [])
        self.assertNotIn("TEST_BACKEND", result.stderr)
        result, rows = self.run_case(extra=("--prefix-tokens", "0"))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(rows, [])
        self.assertNotIn("TEST_BACKEND", result.stderr)


if __name__ == "__main__":
    unittest.main()
