/* Host-only backend for checking the probe's control flow and rejection.
 * This fixture performs no model inference and cannot qualify a token gate. */
#include <assert.h>
struct qrt_engine { unsigned cache, owners, cancelled; uint32_t first; };
static int fixture_mode(const char *name) {
    const char *mode = getenv("QRT_PREFIX_PROBE_FIXTURE");
    return mode && !strcmp(mode, name);
}
qrt_status_t qrt_engine_create(const qrt_engine_config_t *c, qrt_engine_t **e) {
    assert(c && c->context_tokens == 17408u && c->batch_size == 1u);
    *e = calloc(1u, sizeof(**e));
    return *e ? QRT_STATUS_OK : QRT_STATUS_OUT_OF_MEMORY;
}
void qrt_engine_free(qrt_engine_t *e) {
    fprintf(stderr, "fixture owners=%u cancelled=%u cache=%u\n", e->owners, e->cancelled, e->cache);
    free(e);
}
const char *qrt_strerror(qrt_status_t s) { (void)s; return "host fixture"; }
qrt_status_t qrt_engine_report(const qrt_engine_t *e, qrt_engine_report_t *r) {
    memset(r, 0, sizeof(*r));
    r->baseline_output_head_token_emitted = 1u;
    r->baseline_output_head_sampled_token_id = r->baseline_output_head_topk_token_ids[0] = e->first;
    r->baseline_output_head_topk_logits[0] = 5.0f;
    r->last_request_ttft_elapsed_ns = 1000u;
    r->last_request_tpot_elapsed_ns = 51100u;
    r->last_request_tpot_sample_count = 511u;
    return QRT_STATUS_OK;
}
static void fixture_result(qrt_qwen36_resident_prefix_cache_result_v1_t *r,
    const uint32_t *input, size_t n, size_t prefix, const uint32_t *output, size_t count) {
    memset(r, 0, sizeof(*r));
    r->completed = r->provider_invoked = r->exact_prefix_match = r->copy_on_write_transaction = r->state_restored = 1u;
    r->prefix_token_count = (uint32_t)prefix; r->suffix_token_count = (uint32_t)(n - prefix);
    r->output_token_count = (uint32_t)count;
    r->input_token_ids_fnv1a64 = qrt_product_fnv1a64_bytes(input, n * sizeof(*input));
    r->output_token_ids_fnv1a64 = qrt_product_fnv1a64_bytes(output, count * sizeof(*output));
    r->ttft_elapsed_ns = 1000u; r->tpot_elapsed_ns = (count - 1u) * 100u; r->tpot_sample_count = (uint32_t)count - 1u;
    qrt_prefix_first_logit_store(r->reserved, output[0], 5.0f);
}
qrt_status_t qrt_engine_request_tokens_prefix_fallback_v1(qrt_engine_t *e, const uint32_t *input,
    size_t n, size_t prefix, uint32_t *output, size_t capacity,
    qrt_qwen36_resident_prefix_cache_fallback_result_v1_t *r) {
    size_t i;
    assert(n == 17408u && prefix == 16384u && capacity == 512u && !e->cache);
    memset(r, 0, sizeof(*r));r->completed = r->fallback_invoked = r->retry_invoked = 1u;
    r->seed_output_token_count = 1u;r->seed_output_token = 16u;r->full_prefill_token_count = 16384u;
    for (i = 0u; i < capacity; ++i) output[i] = 100u + (uint32_t)i;
    fixture_result(&r->hit_result, input, n, prefix, output, capacity);e->cache = 1u;
    return QRT_STATUS_OK;
}
qrt_status_t qrt_engine_request_tokens_prefix_stream_v1(qrt_engine_t *e, const uint32_t *input,
    size_t n, size_t prefix, uint32_t *output, size_t capacity,
    qrt_qwen36_resident_prefix_cache_result_v1_t *r, qrt_token_stream_callback_v1_t callback, void *user) {
    size_t i;int owner = n == prefix + 1u;
    assert(prefix == 16384u && callback && e->cache == 1u);
    assert(owner ? n == 16385u && capacity == 31u && input[prefix] == 16u : n == 17408u && capacity == 512u);
    e->cache = 2u;
    if (owner) ++e->owners;
    for (i = 0u; i < capacity; ++i) output[i] = (owner ? 201u : 100u) + (uint32_t)i;
    if (owner && e->cancelled && fixture_mode("wrong_owner")) ++output[2];
    if (!owner && fixture_mode("wrong_cancel_token")) ++output[0];
    fixture_result(r, input, n, prefix, output, capacity);
    e->first = output[0];
    for (i = 0u; i < capacity; ++i) {
        qrt_token_stream_event_v1_t event;
        memset(&event, 0, sizeof(event));event.struct_size = sizeof(event);event.abi_version = QRT_TOKEN_STREAM_EVENT_ABI_VERSION;
        event.phase = i ? QRT_TOKEN_STREAM_PHASE_DECODE : QRT_TOKEN_STREAM_PHASE_PREFILL;
        event.output_index = (uint32_t)i;event.token_id = output[i];event.token_step_elapsed_ns = i ? 100u : 1000u;
        event.request_elapsed_ns = 1000u + i * 100u;event.provider_decode_elapsed_ns = i * 100u;
        if (!callback(user, &event) && !fixture_mode("ignore_cancel")) {
            ++e->cancelled;e->cache = 1u;r->completed = 0u;r->status = QRT_STATUS_UNSUPPORTED;
            strcpy(r->failure_stage, "fixture_cancelled");
            if (fixture_mode("no_restore")) r->state_restored = 0u;
            if (fixture_mode("wrong_restored_count")) r->restored_committed_token_count = 1u;
            return QRT_STATUS_UNSUPPORTED;
        }
    }
    e->cache = 1u;
    return QRT_STATUS_OK;
}
qrt_status_t qrt_engine_request_tokens_prefix_v1(qrt_engine_t *e, const uint32_t *i,
    size_t n, size_t p, uint32_t *o, size_t c, qrt_qwen36_resident_prefix_cache_result_v1_t *r) {
    (void)e;(void)i;(void)n;(void)p;(void)o;(void)c;(void)r;abort();
}
qrt_status_t qrt_engine_request_tokens(qrt_engine_t *e, const uint32_t *i,
    size_t n, uint32_t *o, size_t c, size_t *r) {
    (void)e;(void)i;(void)n;(void)o;(void)c;(void)r;abort();
}
qrt_status_t qrt_engine_request_tokens_stream_v1(qrt_engine_t *e, const uint32_t *i,
    size_t n, uint32_t *o, size_t c, size_t *r, qrt_token_stream_callback_v1_t cb, void *u) {
    (void)e;(void)i;(void)n;(void)o;(void)c;(void)r;(void)cb;(void)u;abort();
}
qrt_status_t qrt_engine_prefix_checkpoint_match_v1(qrt_engine_t *e,
    const uint32_t *i, size_t n, size_t c, size_t *p) {
    (void)e;(void)i;(void)n;(void)c;(void)p;abort();
}
