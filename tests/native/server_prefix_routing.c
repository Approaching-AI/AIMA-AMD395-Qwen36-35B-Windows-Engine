/* Exercise the real bridge with controlled native checkpoint responses.
 * This checks routing/callback safety only, not model inference. */
#include "qrt_server_bridge.c"
#include <assert.h>

struct qrt_engine { int unused; };
static struct qrt_engine native_engine;
static uint32_t input[8192];
static size_t matched_prefix, query_calls, prefix_calls, cold_calls, callbacks;
static size_t prefix_callbacks, prefix_output_count;
static qrt_status_t query_status, prefix_status;
static int cancel_callback;

void qrt_engine_free(qrt_engine_t *engine) { (void)engine; }

qrt_status_t qrt_engine_report(const qrt_engine_t *engine, qrt_engine_report_t *report) {
    assert(engine == &native_engine);
    memset(report, 0, sizeof(*report));
    report->last_request_ttft_elapsed_ns = 123u;
    return QRT_STATUS_OK;
}

qrt_status_t qrt_engine_prefix_checkpoint_match_v1(
    qrt_engine_t *engine, const uint32_t *tokens, size_t count,
    size_t capacity, size_t *prefix
) {
    assert(engine == &native_engine && tokens == input && count == 294u && capacity == 32u);
    ++query_calls;
    *prefix = matched_prefix;
    return query_status;
}

static int emit(qrt_token_stream_callback_v1_t callback, void *user_data) {
    qrt_token_stream_event_v1_t event;
    memset(&event, 0, sizeof(event));
    event.struct_size = sizeof(event);
    event.token_id = 144u;
    return callback(user_data, &event);
}

qrt_status_t qrt_engine_request_tokens_stream_v1(
    qrt_engine_t *engine, const uint32_t *tokens, size_t count,
    uint32_t *output, size_t capacity, size_t *output_count,
    qrt_token_stream_callback_v1_t callback, void *user_data
) {
    assert(engine == &native_engine && tokens == input && count == 294u && capacity == 32u);
    assert(*output_count == 0u);
    ++cold_calls;
    output[0] = 144u;
    *output_count = 1u;
    return emit(callback, user_data) ? QRT_STATUS_OK : QRT_STATUS_UNSUPPORTED;
}

qrt_status_t qrt_engine_request_tokens_prefix_stream_v1(
    qrt_engine_t *engine, const uint32_t *tokens, size_t count, size_t prefix,
    uint32_t *output, size_t capacity,
    qrt_qwen36_resident_prefix_cache_result_v1_t *result,
    qrt_token_stream_callback_v1_t callback, void *user_data
) {
    size_t i;
    assert(engine == &native_engine && tokens == input && count == 294u && capacity == 32u);
    assert(prefix == matched_prefix && result->output_token_count == 0u);
    ++prefix_calls;
    output[0] = 144u;
    result->output_token_count = (uint32_t)prefix_output_count;
    for (i = 0; i < prefix_callbacks; ++i) {
        if (!emit(callback, user_data)) return QRT_STATUS_UNSUPPORTED;
    }
    return prefix_status;
}

static int QRT_CDECL observe(void *user_data, const qrt_token_stream_event_v1_t *event) {
    assert(user_data == &callbacks && event->token_id == 144u);
    ++callbacks;
    return !cancel_callback;
}

static void reset(void) {
    matched_prefix = query_calls = prefix_calls = cold_calls = callbacks = 0u;
    prefix_callbacks = prefix_output_count = 0u;
    query_status = prefix_status = QRT_STATUS_OK;
    cancel_callback = 0;
    assert(setenv("QRT_SERVER_PREFIX_CACHE", "1", 1) == 0);
    assert(setenv("QRT_SERVER_PREFIX_CACHE_MIN_TOKENS", "256", 1) == 0);
}

static void request(qrt_status_t expected, size_t expected_output) {
    qrt_server_engine_t engine = {&native_engine};
    qrt_server_request_report_v1_t report;
    uint32_t output[32];
    size_t output_count = 999u;
    assert(qrt_server_engine_request_tokens_stream_v1(
        &engine, input, 294u, output, 32u, &output_count,
        observe, &callbacks, &report) == expected);
    assert(output_count == expected_output && report.output_token_count == expected_output);
    assert(report.ttft_ns == 123u && report.abi_version == QRT_SERVER_BRIDGE_ABI_VERSION);
}

int main(void) {
    /* Repeated overlapping requests cannot create a cache in the bridge.
     * There is deliberately no nonstream/out1 seed API stub to link against. */
    reset();
    request(QRT_STATUS_OK, 1u);
    request(QRT_STATUS_OK, 1u);
    assert(query_calls == 2u && prefix_calls == 0u && cold_calls == 2u && callbacks == 2u);

    reset(); matched_prefix = 256u; prefix_callbacks = prefix_output_count = 1u;
    request(QRT_STATUS_OK, 1u);
    assert(prefix_calls == 1u && cold_calls == 0u && callbacks == 1u);

    reset(); matched_prefix = 256u; prefix_status = QRT_STATUS_UNSUPPORTED;
    request(QRT_STATUS_OK, 1u);
    assert(prefix_calls == 1u && cold_calls == 1u && callbacks == 1u);

    reset(); matched_prefix = 256u; prefix_status = QRT_STATUS_UNSUPPORTED; prefix_callbacks = 1u;
    request(QRT_STATUS_UNSUPPORTED, 0u);
    assert(prefix_calls == 1u && cold_calls == 0u && callbacks == 1u);

    reset(); matched_prefix = 256u; prefix_status = QRT_STATUS_UNSUPPORTED; prefix_output_count = 1u;
    request(QRT_STATUS_UNSUPPORTED, 0u);
    assert(prefix_calls == 1u && cold_calls == 0u && callbacks == 0u);

    reset(); matched_prefix = 256u; prefix_callbacks = 1u; cancel_callback = 1;
    request(QRT_STATUS_UNSUPPORTED, 0u);
    assert(prefix_calls == 1u && cold_calls == 0u && callbacks == 1u);

    reset(); matched_prefix = 256u; prefix_status = QRT_STATUS_OUT_OF_MEMORY;
    request(QRT_STATUS_OUT_OF_MEMORY, 0u);
    assert(prefix_calls == 1u && cold_calls == 0u && callbacks == 0u);

    reset(); matched_prefix = 256u; assert(setenv("QRT_SERVER_PREFIX_CACHE", "0", 1) == 0);
    request(QRT_STATUS_OK, 1u);
    assert(query_calls == 0u && prefix_calls == 0u && cold_calls == 1u);

    reset(); matched_prefix = 256u; query_status = QRT_STATUS_UNSUPPORTED;
    request(QRT_STATUS_OK, 1u);
    assert(query_calls == 1u && prefix_calls == 0u && cold_calls == 1u);

    reset(); matched_prefix = 128u;
    request(QRT_STATUS_OK, 1u);
    assert(prefix_calls == 0u && cold_calls == 1u);

    reset(); matched_prefix = 294u;
    request(QRT_STATUS_OK, 1u);
    assert(prefix_calls == 0u && cold_calls == 1u);
    puts("11 bridge checkpoint routing and callback cases passed");
    return 0;
}
