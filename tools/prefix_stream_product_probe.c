/* Real-model delivery/cancellation probe. Reuse the product CLI's original
 * input, loading, output/logit and owner-continuation checks. Only the public
 * prefix calls are wrapped. All added timing is diagnostic, never performance. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "../native/src/qrt.h"
#include "prefix_stream_observer.h"

static qrt_status_t probe_seed(qrt_engine_t *, const uint32_t *, size_t, size_t,
    uint32_t *, size_t, qrt_qwen36_resident_prefix_cache_fallback_result_v1_t *);
static qrt_status_t probe_stream(qrt_engine_t *, const uint32_t *, size_t, size_t,
    uint32_t *, size_t, qrt_qwen36_resident_prefix_cache_result_v1_t *,
    qrt_token_stream_callback_v1_t, void *);
#define qrt_engine_request_tokens_prefix_fallback_v1 probe_seed
#define qrt_engine_request_tokens_prefix_stream_v1 probe_stream
#define main qrt_product_cli_reference_main
#include "../native/src/product_cli.c"
#undef main
#undef qrt_engine_request_tokens_prefix_stream_v1
#undef qrt_engine_request_tokens_prefix_fallback_v1

static const qrt_qwen36_resident_prefix_cache_fallback_result_v1_t *probe_owner;
static uint32_t *probe_expected_suffix, *probe_expected_owner;
static size_t probe_suffix_count, probe_owner_count;
static uint64_t probe_calls;
static unsigned probe_cancel_passes, probe_owner_passes;
static int probe_cancel_sequence_started, probe_failed;

static qrt_status_t probe_seed(qrt_engine_t *engine, const uint32_t *input,
    size_t tokens, size_t prefix, uint32_t *output, size_t capacity,
    qrt_qwen36_resident_prefix_cache_fallback_result_v1_t *result
) {
    const qrt_status_t status = qrt_engine_request_tokens_prefix_fallback_v1(
        engine, input, tokens, prefix, output, capacity, result);
    if (status == QRT_STATUS_OK) probe_owner = result;
    return status;
}

static qrt_status_t probe_actual(qrt_engine_t *engine, const uint32_t *input,
    size_t tokens, size_t prefix, uint32_t *output, size_t capacity,
    qrt_qwen36_resident_prefix_cache_result_v1_t *result,
    qrt_token_stream_callback_v1_t callback, void *user, size_t cancel_index
) {
    qrt_prefix_stream_observer_t observation;
    qrt_status_t status;
    uint64_t wall_ns;
    int passed;
    memset(&observation, 0, sizeof(observation));
    observation.now_ns = qrt_product_now_ns;
    observation.kind = cancel_index != SIZE_MAX ? "cancel" : tokens == prefix + 1u ? "owner" : "suffix";
    observation.call = ++probe_calls;
    observation.capacity = capacity;
    observation.cancel_index = cancel_index;
    observation.downstream = callback;
    observation.downstream_data = user;
    if (tokens != prefix + 1u) {
        observation.expected = probe_expected_suffix;
        observation.expected_count = probe_suffix_count;
    }
    observation.started_ns = qrt_product_now_ns();
    status = qrt_engine_request_tokens_prefix_stream_v1(engine, input, tokens,
        prefix, output, capacity, result, qrt_prefix_observe_token, &observation);
    wall_ns = qrt_product_elapsed_ns(observation.started_ns);
    passed = qrt_prefix_observer_live(&observation) && result->provider_invoked &&
        result->copy_on_write_transaction && result->exact_prefix_match && result->state_restored &&
        result->prefix_token_count == prefix && result->suffix_token_count == tokens - prefix &&
        result->restored_committed_token_count == result->base_committed_token_count;
    if (cancel_index == SIZE_MAX)
        passed = passed && status == QRT_STATUS_OK && result->completed && observation.count == capacity;
    else
        passed = passed && status == QRT_STATUS_UNSUPPORTED && !result->completed &&
            observation.cancelled && observation.count == cancel_index + 1u;
    printf("{\"type\":\"prefix_stream_probe_call\",\"call\":%" PRIu64
        ",\"kind\":\"%s\",\"status\":%d,\"cancel_index\":%zu,\"callbacks\":%zu,"
        "\"state_restored\":%s,\"wall_ns\":%" PRIu64 ",\"return_after_last_callback_ns\":%" PRIu64
        ",\"passed\":%s,\"failure_stage\":", observation.call, observation.kind, (int)status,
        cancel_index, observation.count, result->state_restored ? "true" : "false", wall_ns,
        observation.count && wall_ns >= observation.arrival[observation.count - 1u]
            ? wall_ns - observation.arrival[observation.count - 1u] : UINT64_MAX,
        passed ? "true" : "false");
    qrt_product_print_json_string(result->failure_stage);
    fputs("}\n", stdout);
    fflush(stdout);
    if (!passed || ferror(stdout)) {
        probe_failed = 1;
        return QRT_STATUS_UNSUPPORTED;
    }
    if (cancel_index != SIZE_MAX) ++probe_cancel_passes;
    return status;
}

static qrt_status_t probe_stream(qrt_engine_t *engine, const uint32_t *input,
    size_t tokens, size_t prefix, uint32_t *output, size_t capacity,
    qrt_qwen36_resident_prefix_cache_result_v1_t *result,
    qrt_token_stream_callback_v1_t callback, void *user
) {
    if (!probe_cancel_sequence_started && tokens == prefix + 1024u) {
        static const size_t cancel_indices[] = {0u, 1u, 64u, 511u};
        size_t i;
        qrt_qwen36_resident_prefix_cache_result_v1_t *cancel_result;
        uint32_t cancel_output[QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS];
        probe_cancel_sequence_started = 1;
        if (!probe_owner || prefix != 16384u || capacity != 512u) {
            probe_failed = 1;
            return QRT_STATUS_UNSUPPORTED;
        }
        cancel_result = (qrt_qwen36_resident_prefix_cache_result_v1_t *)calloc(1u, sizeof(*cancel_result));
        if (!cancel_result) {
            probe_failed = 1;
            return QRT_STATUS_OUT_OF_MEMORY;
        }
        for (i = 0u; i < sizeof(cancel_indices) / sizeof(cancel_indices[0]); ++i) {
            uint64_t owner_wall_ns = UINT64_C(0);
            memset(cancel_output, 0, sizeof(cancel_output));
            memset(cancel_result, 0, sizeof(*cancel_result));
            (void)probe_actual(engine, input, tokens, prefix, cancel_output, capacity,
                cancel_result, NULL, NULL, cancel_indices[i]);
            if (probe_failed || !qrt_product_verify_owner_continuation(engine, input, prefix,
                    probe_owner, probe_expected_owner, probe_owner_count, &owner_wall_ns)) {
                probe_failed = 1;
                free(cancel_result);
                return QRT_STATUS_UNSUPPORTED;
            }
            ++probe_owner_passes;
        }
        free(cancel_result);
    }
    return probe_actual(engine, input, tokens, prefix, output, capacity, result, callback, user, SIZE_MAX);
}

int main(int argc, char **argv) {
    qrt_product_options_t options;
    int result;
    if (!qrt_product_parse_options(argc, argv, &options) || options.prefix_token_count != 16384u ||
        options.output_token_capacity != 512u || options.prefix_hit_count != 1u || options.checkpoint_owner_path ||
        !options.expected_output_path || !options.expected_owner_output_path ||
        !qrt_product_parse_token_array(options.expected_output_path, &probe_expected_suffix, &probe_suffix_count) ||
        !qrt_product_parse_token_array(options.expected_owner_output_path, &probe_expected_owner, &probe_owner_count) ||
        probe_suffix_count != 512u || probe_owner_count != 32u) {
        fputs("probe requires prefix16384, suffix1024, output512, one hit and original suffix/owner output files\n", stderr);
        free(probe_expected_owner); free(probe_expected_suffix);
        return 2;
    }
    result = qrt_product_run(&options);
    if (!result && (probe_failed || probe_cancel_passes != 4u || probe_owner_passes != 4u || probe_calls != 10u)) result = 6;
    printf("{\"type\":\"prefix_stream_probe_summary\",\"calls\":%" PRIu64
        ",\"cancel_passes\":%u,\"owner_after_cancel_passes\":%u,\"passed\":%s,"
        "\"diagnostic_timing_only\":true,\"performance_acceptance\":false}\n",
        probe_calls, probe_cancel_passes, probe_owner_passes, result ? "false" : "true");
    fflush(stdout);
    free(probe_expected_owner); free(probe_expected_suffix);
    return ferror(stdout) && !result ? 6 : result;
}
