#ifndef QRT_SERVER_BRIDGE_H
#define QRT_SERVER_BRIDGE_H

#include "qrt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QRT_SERVER_BRIDGE_ABI_VERSION 1u

typedef struct qrt_server_engine qrt_server_engine_t;

typedef struct qrt_server_load_report_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t provider_dll_load_ns;
    uint64_t provider_preload_wall_ns;
    uint64_t provider_preload_reported_ns;
    uint64_t provider_preload_stored_entry_count;
    uint64_t engine_create_ns;
    uint64_t total_load_ns;
    unsigned int provider_preload_completed;
    unsigned int engine_ready;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_server_load_report_v1_t;

typedef struct qrt_server_request_report_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t request_wall_ns;
    uint64_t ttft_ns;
    uint64_t tpot_ns;
    size_t tpot_sample_count;
    size_t output_token_count;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_server_request_report_v1_t;

QRT_API qrt_status_t qrt_server_engine_create_v1(
    const char *model_path,
    const char *provider_dll,
    size_t max_context_tokens,
    qrt_server_engine_t **out_engine,
    qrt_server_load_report_v1_t *out_report
);

QRT_API void qrt_server_engine_free_v1(qrt_server_engine_t *engine);

QRT_API qrt_status_t qrt_server_engine_request_tokens_stream_v1(
    qrt_server_engine_t *engine,
    const uint32_t *input_tokens,
    size_t input_token_count,
    uint32_t *output_tokens,
    size_t output_token_capacity,
    size_t *out_output_token_count,
    qrt_token_stream_callback_v1_t callback,
    void *user_data,
    qrt_server_request_report_v1_t *out_report
);

/*
 * Select the provider's exact prefill boundary for an explicit raw-token,
 * one-token request when the prompt is below the retained q8192 shape.  Other
 * shapes fall through to the ordinary resident request route.
 */
QRT_API qrt_status_t qrt_server_engine_request_tokens_exact_stream_v1(
    qrt_server_engine_t *engine,
    const uint32_t *input_tokens,
    size_t input_token_count,
    uint32_t *output_tokens,
    size_t output_token_capacity,
    size_t *out_output_token_count,
    qrt_token_stream_callback_v1_t callback,
    void *user_data,
    qrt_server_request_report_v1_t *out_report
);

#ifdef __cplusplus
}
#endif

#endif
