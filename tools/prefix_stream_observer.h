#pragma once
#include "../native/src/qrt.h"
#include <inttypes.h>
#include <stdio.h>

/* Observation only: expected IDs are never supplied to model computation. */
typedef struct qrt_prefix_stream_observer_t {
    uint64_t (*now_ns)(void);
    uint64_t started_ns, call;
    const char *kind;
    const uint32_t *expected;
    size_t expected_count, capacity, count, cancel_index;
    qrt_token_stream_callback_v1_t downstream;
    void *downstream_data;
    uint64_t arrival[QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS];
    uint64_t provider_end[QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS];
    uint64_t request_end[QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS];
    int failed, cancelled;
} qrt_prefix_stream_observer_t;

static int QRT_CDECL qrt_prefix_observe_token(
    void *context, const qrt_token_stream_event_v1_t *event
) {
    qrt_prefix_stream_observer_t *s = (qrt_prefix_stream_observer_t *)context;
    uint64_t now;
    size_t i;
    if (!s || s->failed || s->cancelled) return 0;
    i = s->count;
    if (!s->now_ns || !s->kind || !event ||
        event->struct_size != sizeof(*event) || event->abi_version != QRT_TOKEN_STREAM_EVENT_ABI_VERSION ||
        !s->capacity || s->capacity > QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS || i >= s->capacity ||
        event->output_index != i || event->token_id >= QRT_QWEN36_VOCAB_SIZE ||
        event->phase != (i ? QRT_TOKEN_STREAM_PHASE_DECODE : QRT_TOKEN_STREAM_PHASE_PREFILL) ||
        !event->token_step_elapsed_ns || !event->request_elapsed_ns ||
        (!i && event->provider_decode_elapsed_ns) ||
        (i && (event->request_elapsed_ns <= s->request_end[i - 1u] ||
               event->provider_decode_elapsed_ns <= s->provider_end[i - 1u] ||
               event->provider_decode_elapsed_ns - s->provider_end[i - 1u] != event->token_step_elapsed_ns)) ||
        (s->expected && (i >= s->expected_count || event->token_id != s->expected[i]))) {
        s->failed = 1;
        return 0;
    }
    now = s->now_ns();
    if (now < s->started_ns || (i && now - s->started_ns < s->arrival[i - 1u])) {
        s->failed = 1;
        return 0;
    }
    s->arrival[i] = now - s->started_ns;
    s->provider_end[i] = event->provider_decode_elapsed_ns;
    s->request_end[i] = event->request_elapsed_ns;
    ++s->count;
    printf("{\"type\":\"prefix_stream_observation\",\"call\":%" PRIu64
        ",\"kind\":\"%s\",\"index\":%zu,\"token_id\":%u,\"arrival_ns\":%" PRIu64
        ",\"request_elapsed_ns\":%" PRIu64 ",\"provider_decode_ns\":%" PRIu64
        ",\"step_ns\":%" PRIu64 "}\n", s->call, s->kind, i, event->token_id,
        s->arrival[i], event->request_elapsed_ns, event->provider_decode_elapsed_ns,
        event->token_step_elapsed_ns);
    fflush(stdout);
    if (ferror(stdout) || (s->downstream && !s->downstream(s->downstream_data, event))) {
        s->failed = 1;
        return 0;
    }
    if (i == s->cancel_index) {
        s->cancelled = 1;
        return 0;
    }
    return 1;
}

/* Compare delivery duration within each actual 63-token decode span with its
 * producer duration. A completed-span burst fails even though every callback
 * precedes the request return. Allow 10% clock/scheduling skew plus 5ms.
 * These checks diagnose delivery; they are not product throughput thresholds. */
static int qrt_prefix_observer_live(const qrt_prefix_stream_observer_t *s) {
    size_t first;
    int passed;
    if (!s || s->failed || !s->count || s->count > s->capacity ||
        s->capacity > QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS) return 0;
    passed = 1;
    for (first = 1u; first < s->count; first += QRT_QWEN36_WHOLE_PROVIDER_DECODE_MAX_OUTPUT_TOKENS - 1u) {
        size_t last = first + QRT_QWEN36_WHOLE_PROVIDER_DECODE_MAX_OUTPUT_TOKENS - 2u;
        uint64_t delivered, produced, required;
        int live;
        if (last >= s->count) last = s->count - 1u;
        if (last == first) continue;
        delivered = s->arrival[last] - s->arrival[first];
        produced = s->provider_end[last] - s->provider_end[first];
        required = produced - produced / 10u;
        live = produced && (delivered >= required || required - delivered <= UINT64_C(5000000));
        passed = passed && live;
        printf("{\"type\":\"prefix_stream_span\",\"call\":%" PRIu64
            ",\"kind\":\"%s\",\"first\":%zu,\"last\":%zu,\"delivery_ns\":%" PRIu64
            ",\"production_ns\":%" PRIu64 ",\"live\":%s}\n",
            s->call, s->kind, first, last, delivered, produced, live ? "true" : "false");
    }
    fflush(stdout);
    return passed && !ferror(stdout);
}
