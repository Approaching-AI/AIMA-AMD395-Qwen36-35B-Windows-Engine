#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "qrt_server_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define QRT_SERVER_RETAINED_Q8192_TOKENS 8192u
#define QRT_SERVER_QWEN36_ASCII_A_TOKEN_ID 32u
#define QRT_SERVER_QWEN36_ASCII_J_TOKEN_ID 41u

struct qrt_server_engine {
    qrt_engine_t *engine;
    uint32_t *resident_prefix_tokens;
    size_t resident_prefix_token_count;
    size_t resident_prefix_token_capacity;
    uint32_t *previous_request_tokens;
    size_t previous_request_token_count;
    size_t previous_request_token_capacity;
#ifdef _WIN32
    HMODULE provider_module;
#endif
};

static int qrt_server_store_tokens(
    uint32_t **destination,
    size_t *destination_count,
    size_t *destination_capacity,
    const uint32_t *source,
    size_t source_count
) {
    uint32_t *replacement;
    if (destination == NULL || destination_count == NULL ||
        destination_capacity == NULL || source == NULL || source_count == 0u ||
        source_count > SIZE_MAX / sizeof(source[0])) {
        return 0;
    }
    if (*destination_capacity < source_count) {
        replacement = (uint32_t *)realloc(
            *destination,
            source_count * sizeof(source[0])
        );
        if (replacement == NULL) {
            return 0;
        }
        *destination = replacement;
        *destination_capacity = source_count;
    }
    memcpy(*destination, source, source_count * sizeof(source[0]));
    *destination_count = source_count;
    return 1;
}

static size_t qrt_server_common_prefix_tokens(
    const uint32_t *left,
    size_t left_count,
    const uint32_t *right,
    size_t right_count
) {
    size_t count = 0u;
    const size_t limit = left_count < right_count ? left_count : right_count;
    if (left == NULL || right == NULL) {
        return 0u;
    }
    while (count < limit && left[count] == right[count]) {
        ++count;
    }
    return count;
}

static const char *qrt_server_environment_value(
    const char *name,
    char *buffer,
    size_t buffer_capacity
) {
#ifdef _WIN32
    DWORD length;
    if (name == NULL || buffer == NULL || buffer_capacity == 0u ||
        buffer_capacity > (size_t)UINT32_MAX) {
        return NULL;
    }
    length = GetEnvironmentVariableA(
        name,
        buffer,
        (DWORD)buffer_capacity
    );
    return length > 0u && (size_t)length < buffer_capacity ? buffer : NULL;
#else
    (void)buffer;
    (void)buffer_capacity;
    return name != NULL ? getenv(name) : NULL;
#endif
}

static size_t qrt_server_prefix_cache_min_tokens(void) {
    char buffer[64];
    const char *value = qrt_server_environment_value(
        "QRT_SERVER_PREFIX_CACHE_MIN_TOKENS",
        buffer,
        sizeof(buffer)
    );
    char *end = NULL;
    unsigned long long parsed;
    if (value == NULL || value[0] == '\0') {
        return (size_t)QRT_PREFIX_CACHE_BLOCK_TOKENS;
    }
    parsed = strtoull(value, &end, 0);
    if (end == value || (end != NULL && *end != '\0') ||
        parsed > (unsigned long long)SIZE_MAX) {
        return (size_t)QRT_PREFIX_CACHE_BLOCK_TOKENS;
    }
    return (size_t)parsed;
}

static int qrt_server_prefix_cache_enabled(void) {
    char buffer[64];
    const char *value = qrt_server_environment_value(
        "QRT_SERVER_PREFIX_CACHE",
        buffer,
        sizeof(buffer)
    );
    return value == NULL || strcmp(value, "0") != 0;
}

#ifdef _WIN32
static int qrt_server_exact_first_token_enabled(void) {
    char buffer[64];
    const char *value = qrt_server_environment_value(
        "QRT_SERVER_EXACT_FIRST_TOKEN_PREFILL",
        buffer,
        sizeof(buffer)
    );
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static int qrt_server_exact_letter_classifier_enabled(void) {
    char buffer[64];
    const char *value = qrt_server_environment_value(
        "QRT_SERVER_EXACT_LETTER_CLASSIFIER",
        buffer,
        sizeof(buffer)
    );
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static uint64_t qrt_server_fnv1a64_bytes(const void *data, size_t count) {
    const unsigned char *bytes = (const unsigned char *)data;
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t index;
    for (index = 0u; index < count; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
#endif

static int qrt_server_prefix_shape_supported(
    size_t input_token_count,
    size_t prefix_token_count,
    size_t output_token_capacity
) {
    const size_t suffix_token_count =
        input_token_count >= prefix_token_count
            ? input_token_count - prefix_token_count
            : SIZE_MAX;
    return prefix_token_count > 0u &&
        prefix_token_count < input_token_count &&
        suffix_token_count <=
            (size_t)QRT_QWEN36_RESIDENT_PREFIX_CACHE_MAX_SUFFIX_TOKENS &&
        output_token_capacity > 0u &&
        output_token_capacity <=
            (size_t)QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS &&
        suffix_token_count <=
            (size_t)QRT_QWEN36_RESIDENT_PREFIX_CACHE_MAX_TAIL_TOKENS -
                (output_token_capacity - 1u);
}

static uint64_t qrt_server_now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    if (!QueryPerformanceCounter(&counter) ||
        !QueryPerformanceFrequency(&frequency) ||
        frequency.QuadPart <= 0) {
        return GetTickCount64() * UINT64_C(1000000);
    }
    return (uint64_t)(
        ((long double)counter.QuadPart * 1000000000.0L) /
        (long double)frequency.QuadPart
    );
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return UINT64_C(0);
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
        (uint64_t)value.tv_nsec;
#endif
}

static uint64_t qrt_server_elapsed_ns(uint64_t start_ns) {
    const uint64_t end_ns = qrt_server_now_ns();
    return end_ns >= start_ns ? end_ns - start_ns : UINT64_C(0);
}

static void qrt_server_copy_text(char *destination, size_t capacity, const char *source) {
    size_t length;
    if (destination == NULL || capacity == 0u) {
        return;
    }
    if (source == NULL) {
        source = "";
    }
    length = strlen(source);
    if (length >= capacity) {
        length = capacity - 1u;
    }
    memcpy(destination, source, length);
    destination[length] = '\0';
}

static void qrt_server_set_load_failure(
    qrt_server_load_report_v1_t *report,
    const char *stage,
    const char *failure
) {
    if (report == NULL) {
        return;
    }
    qrt_server_copy_text(report->failure_stage, sizeof(report->failure_stage), stage);
    qrt_server_copy_text(report->failure, sizeof(report->failure), failure);
}

qrt_status_t qrt_server_engine_create_v1(
    const char *model_path,
    const char *provider_dll,
    size_t max_context_tokens,
    qrt_server_engine_t **out_engine,
    qrt_server_load_report_v1_t *out_report
) {
    qrt_server_engine_t *server_engine = NULL;
    const uint64_t load_start_ns = qrt_server_now_ns();
#ifdef _WIN32
    uint64_t engine_start_ns;
    qrt_engine_config_t config;
    qrt_status_t status;
#endif

    if (out_report != NULL) {
        memset(out_report, 0, sizeof(*out_report));
        out_report->struct_size = (uint32_t)sizeof(*out_report);
        out_report->abi_version = QRT_SERVER_BRIDGE_ABI_VERSION;
    }
    if (model_path == NULL || model_path[0] == '\0' ||
        provider_dll == NULL || provider_dll[0] == '\0' ||
        max_context_tokens == 0u ||
        max_context_tokens > (size_t)QRT_QWEN36_MAX_POSITION_EMBEDDINGS ||
        out_engine == NULL || out_report == NULL) {
        return QRT_STATUS_INVALID_ARGUMENT;
    }
    *out_engine = NULL;
    server_engine = (qrt_server_engine_t *)calloc(1u, sizeof(*server_engine));
    if (server_engine == NULL) {
        return QRT_STATUS_OUT_OF_MEMORY;
    }

#ifndef _WIN32
    qrt_server_set_load_failure(
        out_report,
        "provider_platform",
        "the native whole-provider service is available only on Windows"
    );
    out_report->total_load_ns = qrt_server_elapsed_ns(load_start_ns);
    free(server_engine);
    return QRT_STATUS_UNSUPPORTED;
#else
    {
        typedef int (QRT_CDECL *qrt_server_preload_fn_t)(
            const char *,
            qrt_qwen36_prefill_descriptor_batch_timing_t *,
            char *,
            size_t,
            char *,
            size_t
        );
        qrt_server_preload_fn_t preload_fn;
        qrt_qwen36_prefill_descriptor_batch_timing_t *timing;
        FARPROC symbol;
        char failure_stage[64] = "";
        char failure[QRT_LOAD_ERROR_CAPACITY] = "";
        uint64_t phase_start_ns = qrt_server_now_ns();
        int preload_ok;

        if (_putenv_s("QRT_QWEN36_WHOLE_PROVIDER_DLL", provider_dll) != 0 ||
            _putenv_s(
                "QRT_QWEN36_WHOLE_PROVIDER_ARBITRARY_CONTEXT",
                "1"
            ) != 0 ||
            _putenv_s(
                "QRT_QWEN36_WHOLE_PROVIDER_DIRECT_REQUEST_ENTRY",
                "1"
            ) != 0 ||
            _putenv_s(
                "QRT_QWEN36_WHOLE_PROVIDER_RESIDENT_SESSION",
                "1"
            ) != 0) {
            qrt_server_set_load_failure(
                out_report,
                "provider_environment",
                "could not configure the whole-provider service environment"
            );
            free(server_engine);
            return QRT_STATUS_IO_ERROR;
        }
        server_engine->provider_module = LoadLibraryA(provider_dll);
        out_report->provider_dll_load_ns = qrt_server_elapsed_ns(phase_start_ns);
        if (server_engine->provider_module == NULL) {
            qrt_server_set_load_failure(
                out_report,
                "provider_dll_load",
                "could not load the whole-provider DLL"
            );
            free(server_engine);
            return QRT_STATUS_IO_ERROR;
        }
        symbol = GetProcAddress(
            server_engine->provider_module,
            "qrt_prefill_descriptor_batch_hip_preload_compact_device_routed_layout_v1"
        );
        if (symbol == NULL) {
            qrt_server_set_load_failure(
                out_report,
                "provider_preload_symbol",
                "whole-provider DLL does not export model preload v1"
            );
            FreeLibrary(server_engine->provider_module);
            free(server_engine);
            return QRT_STATUS_UNSUPPORTED;
        }
        preload_fn = (qrt_server_preload_fn_t)(void *)symbol;
        timing = (qrt_qwen36_prefill_descriptor_batch_timing_t *)calloc(
            1u,
            sizeof(*timing)
        );
        if (timing == NULL) {
            FreeLibrary(server_engine->provider_module);
            free(server_engine);
            return QRT_STATUS_OUT_OF_MEMORY;
        }
        phase_start_ns = qrt_server_now_ns();
        preload_ok = preload_fn(
            model_path,
            timing,
            failure_stage,
            sizeof(failure_stage),
            failure,
            sizeof(failure)
        );
        out_report->provider_preload_wall_ns = qrt_server_elapsed_ns(phase_start_ns);
        out_report->provider_preload_reported_ns =
            timing->compact_device_layout_full_preload_elapsed_ns;
        out_report->provider_preload_stored_entry_count =
            timing->compact_device_layout_full_prepack_stored_entry_count;
        out_report->provider_preload_completed =
            preload_ok != 0 &&
            timing->compact_device_layout_full_prepack_pass != 0u;
        free(timing);
        if (out_report->provider_preload_completed == 0u) {
            qrt_server_set_load_failure(
                out_report,
                failure_stage[0] != '\0' ? failure_stage : "provider_preload",
                failure[0] != '\0' ? failure : "whole-provider preload failed"
            );
            FreeLibrary(server_engine->provider_module);
            free(server_engine);
            return QRT_STATUS_UNSUPPORTED;
        }
    }

    memset(&config, 0, sizeof(config));
    config.model_path = model_path;
    config.context_tokens = max_context_tokens;
    config.batch_size = 1u;
    engine_start_ns = qrt_server_now_ns();
    status = qrt_engine_create(&config, &server_engine->engine);
    out_report->engine_create_ns = qrt_server_elapsed_ns(engine_start_ns);
    out_report->total_load_ns = qrt_server_elapsed_ns(load_start_ns);
    if (status != QRT_STATUS_OK || server_engine->engine == NULL) {
        qrt_server_set_load_failure(
            out_report,
            "engine_create",
            qrt_strerror(status)
        );
        FreeLibrary(server_engine->provider_module);
        free(server_engine);
        return status != QRT_STATUS_OK ? status : QRT_STATUS_UNSUPPORTED;
    }
    out_report->engine_ready = 1u;
    *out_engine = server_engine;
    return QRT_STATUS_OK;
#endif
}

void qrt_server_engine_free_v1(qrt_server_engine_t *engine) {
    if (engine == NULL) {
        return;
    }
    if (engine->engine != NULL) {
        qrt_engine_free(engine->engine);
        engine->engine = NULL;
    }
    free(engine->resident_prefix_tokens);
    engine->resident_prefix_tokens = NULL;
    engine->resident_prefix_token_count = 0u;
    engine->resident_prefix_token_capacity = 0u;
    free(engine->previous_request_tokens);
    engine->previous_request_tokens = NULL;
    engine->previous_request_token_count = 0u;
    engine->previous_request_token_capacity = 0u;
#ifdef _WIN32
    if (engine->provider_module != NULL) {
        FreeLibrary(engine->provider_module);
        engine->provider_module = NULL;
    }
#endif
    free(engine);
}

static qrt_status_t qrt_server_engine_request_tokens_stream_internal(
    qrt_server_engine_t *engine,
    const uint32_t *input_tokens,
    size_t input_token_count,
    uint32_t *output_tokens,
    size_t output_token_capacity,
    size_t *out_output_token_count,
    qrt_token_stream_callback_v1_t callback,
    void *user_data,
    qrt_server_request_report_v1_t *out_report,
    int force_exact_first_token
) {
    qrt_engine_report_t *engine_report;
    qrt_status_t status;
    const uint64_t request_start_ns = qrt_server_now_ns();
    uint64_t prefix_seed_elapsed_ns = UINT64_C(0);
    size_t prefix_hit_token_count = 0u;
    int prefix_seed_required = 0;
    int prefix_route_used = 0;

    /* The exact-first-token provider is Windows-only, but the bridge also
     * compiles as a CPU-safe POSIX ABI smoke target with -Werror. */
    (void)force_exact_first_token;

    if (out_report != NULL) {
        memset(out_report, 0, sizeof(*out_report));
        out_report->struct_size = (uint32_t)sizeof(*out_report);
        out_report->abi_version = QRT_SERVER_BRIDGE_ABI_VERSION;
    }
    if (engine == NULL || engine->engine == NULL || input_tokens == NULL ||
        input_token_count == 0u || output_tokens == NULL ||
        output_token_capacity == 0u || out_output_token_count == NULL ||
        callback == NULL || out_report == NULL) {
        return QRT_STATUS_INVALID_ARGUMENT;
    }
#ifdef _WIN32
    {
        const int exact_letter_classifier =
            qrt_server_exact_letter_classifier_enabled();
    if ((exact_letter_classifier ||
         ((force_exact_first_token ||
           qrt_server_exact_first_token_enabled()) &&
          output_token_capacity == 1u)) &&
        input_token_count <
            (size_t)QRT_SERVER_RETAINED_Q8192_TOKENS &&
        input_token_count <= (size_t)UINT32_MAX) {
        FARPROC symbol = GetProcAddress(
            engine->provider_module,
            "qrt_qwen36_whole_provider_exact_first_token_v1"
        );
        qrt_qwen36_exact_first_token_v1_t exact_first_token_fn;
        qrt_qwen36_exact_first_token_request_v1_t exact_request;
        qrt_qwen36_exact_first_token_result_v1_t exact_result;
        qrt_token_stream_event_v1_t event;
        uint64_t input_digest;
        int exact_ok;

        *out_output_token_count = 0u;
        if (symbol == NULL) {
            qrt_server_copy_text(
                out_report->failure_stage,
                sizeof(out_report->failure_stage),
                "exact_first_token_symbol"
            );
            qrt_server_copy_text(
                out_report->failure,
                sizeof(out_report->failure),
                "configured provider does not export exact first-token v1"
            );
            out_report->request_wall_ns =
                qrt_server_elapsed_ns(request_start_ns);
            return QRT_STATUS_UNSUPPORTED;
        }
        input_digest = qrt_server_fnv1a64_bytes(
            input_tokens,
            input_token_count * sizeof(input_tokens[0])
        );
        if (input_digest == UINT64_C(0)) {
            qrt_server_copy_text(
                out_report->failure_stage,
                sizeof(out_report->failure_stage),
                "exact_first_token_identity"
            );
            qrt_server_copy_text(
                out_report->failure,
                sizeof(out_report->failure),
                "exact first-token input digest cannot be zero"
            );
            out_report->request_wall_ns =
                qrt_server_elapsed_ns(request_start_ns);
            return QRT_STATUS_INVALID_ARGUMENT;
        }
        memset(&exact_request, 0, sizeof(exact_request));
        exact_request.struct_size = (uint32_t)sizeof(exact_request);
        exact_request.abi_version =
            QRT_QWEN36_EXACT_FIRST_TOKEN_ABI_VERSION;
        exact_request.flags = QRT_QWEN36_EXACT_FIRST_TOKEN_FLAG_NONE;
        exact_request.batch_size = 1u;
        exact_request.resident_engine = engine->engine;
        exact_request.input_tokens = input_tokens;
        exact_request.input_token_count = (uint32_t)input_token_count;
        exact_request.expected_input_token_ids_fnv1a64 = input_digest;
        memset(&exact_result, 0, sizeof(exact_result));
        exact_first_token_fn =
            (qrt_qwen36_exact_first_token_v1_t)(void *)symbol;
        exact_ok = exact_first_token_fn(&exact_request, &exact_result);
        if (exact_ok == 0 || exact_result.completed == 0u ||
            exact_result.status != (int32_t)QRT_STATUS_OK ||
            exact_result.batch_size != 1u ||
            exact_result.output_token_id >=
                (uint32_t)QRT_QWEN36_VOCAB_SIZE ||
            exact_result.verifier_input_token_count !=
                (uint32_t)input_token_count ||
            exact_result.input_token_ids_fnv1a64 != input_digest) {
            qrt_server_copy_text(
                out_report->failure_stage,
                sizeof(out_report->failure_stage),
                exact_result.failure_stage[0] != '\0'
                    ? exact_result.failure_stage
                    : "exact_first_token_provider"
            );
            qrt_server_copy_text(
                out_report->failure,
                sizeof(out_report->failure),
                exact_result.failure[0] != '\0'
                    ? exact_result.failure
                    : "exact first-token provider did not return one valid token"
            );
            out_report->request_wall_ns =
                qrt_server_elapsed_ns(request_start_ns);
            return exact_result.status > (int32_t)QRT_STATUS_OK &&
                    exact_result.status <= (int32_t)QRT_STATUS_PARSE_ERROR
                ? (qrt_status_t)exact_result.status
                : QRT_STATUS_UNSUPPORTED;
        }

        if (exact_letter_classifier && output_token_capacity > 1u &&
            (exact_result.output_token_id <
                 (uint32_t)QRT_SERVER_QWEN36_ASCII_A_TOKEN_ID ||
             exact_result.output_token_id >
                 (uint32_t)QRT_SERVER_QWEN36_ASCII_J_TOKEN_ID)) {
            uint32_t *continued_input_tokens;
            size_t generated_token_count = 0u;
            uint64_t continuation_elapsed_ns = UINT64_C(0);
            const uint64_t first_token_elapsed_ns =
                exact_result.total_elapsed_ns;

            if (output_token_capacity > SIZE_MAX - input_token_count ||
                input_token_count + output_token_capacity >
                    SIZE_MAX / sizeof(input_tokens[0])) {
                qrt_server_copy_text(
                    out_report->failure_stage,
                    sizeof(out_report->failure_stage),
                    "exact_letter_classifier_capacity"
                );
                qrt_server_copy_text(
                    out_report->failure,
                    sizeof(out_report->failure),
                    "exact letter continuation token capacity overflowed"
                );
                out_report->request_wall_ns =
                    qrt_server_elapsed_ns(request_start_ns);
                return QRT_STATUS_INVALID_ARGUMENT;
            }
            continued_input_tokens = (uint32_t *)malloc(
                (input_token_count + output_token_capacity) *
                    sizeof(input_tokens[0])
            );
            if (continued_input_tokens == NULL) {
                qrt_server_copy_text(
                    out_report->failure_stage,
                    sizeof(out_report->failure_stage),
                    "exact_letter_classifier_allocation"
                );
                qrt_server_copy_text(
                    out_report->failure,
                    sizeof(out_report->failure),
                    "exact letter continuation token allocation failed"
                );
                out_report->request_wall_ns =
                    qrt_server_elapsed_ns(request_start_ns);
                return QRT_STATUS_OUT_OF_MEMORY;
            }
            memcpy(
                continued_input_tokens,
                input_tokens,
                input_token_count * sizeof(input_tokens[0])
            );
            fprintf(
                stderr,
                "QRT_SERVER_MARK exact_letter_classifier_continue input_tokens=%zu verifier_input_tokens=%u output_token=%u requested_output_tokens=%zu reason=exact_token_not_ascii_A_through_J resident_prefix_mutated=0\n",
                input_token_count,
                exact_result.verifier_input_token_count,
                exact_result.output_token_id,
                output_token_capacity
            );

            for (;;) {
                const uint32_t output_token = exact_result.output_token_id;
                const uint64_t step_elapsed_ns = exact_result.total_elapsed_ns;
                const int is_ascii_letter =
                    output_token >=
                        (uint32_t)QRT_SERVER_QWEN36_ASCII_A_TOKEN_ID &&
                    output_token <=
                        (uint32_t)QRT_SERVER_QWEN36_ASCII_J_TOKEN_ID;
                size_t continued_input_token_count;

                output_tokens[generated_token_count] = output_token;
                memset(&event, 0, sizeof(event));
                event.struct_size = (uint32_t)sizeof(event);
                event.abi_version = QRT_TOKEN_STREAM_EVENT_ABI_VERSION;
                event.phase = generated_token_count == 0u
                    ? QRT_TOKEN_STREAM_PHASE_PREFILL
                    : QRT_TOKEN_STREAM_PHASE_DECODE;
                event.output_index = (uint32_t)generated_token_count;
                event.token_id = output_token;
                event.token_step_elapsed_ns = step_elapsed_ns;
                event.request_elapsed_ns =
                    qrt_server_elapsed_ns(request_start_ns);
                ++generated_token_count;
                *out_output_token_count = generated_token_count;
                if (callback(user_data, &event) == 0) {
                    free(continued_input_tokens);
                    qrt_server_copy_text(
                        out_report->failure_stage,
                        sizeof(out_report->failure_stage),
                        "exact_letter_classifier_callback"
                    );
                    qrt_server_copy_text(
                        out_report->failure,
                        sizeof(out_report->failure),
                        "exact letter continuation callback cancelled emission"
                    );
                    out_report->request_wall_ns =
                        qrt_server_elapsed_ns(request_start_ns);
                    return QRT_STATUS_UNSUPPORTED;
                }
                if (is_ascii_letter ||
                    generated_token_count == output_token_capacity) {
                    break;
                }

                continued_input_tokens[
                    input_token_count + generated_token_count - 1u
                ] = output_token;
                continued_input_token_count =
                    input_token_count + generated_token_count;
                if (continued_input_token_count >=
                        (size_t)QRT_SERVER_RETAINED_Q8192_TOKENS ||
                    continued_input_token_count > (size_t)UINT32_MAX) {
                    free(continued_input_tokens);
                    qrt_server_copy_text(
                        out_report->failure_stage,
                        sizeof(out_report->failure_stage),
                        "exact_letter_classifier_context"
                    );
                    qrt_server_copy_text(
                        out_report->failure,
                        sizeof(out_report->failure),
                        "exact letter continuation reached the exact-prefill context boundary"
                    );
                    out_report->request_wall_ns =
                        qrt_server_elapsed_ns(request_start_ns);
                    return QRT_STATUS_UNSUPPORTED;
                }
                input_digest = qrt_server_fnv1a64_bytes(
                    continued_input_tokens,
                    continued_input_token_count *
                        sizeof(continued_input_tokens[0])
                );
                exact_request.input_tokens = continued_input_tokens;
                exact_request.input_token_count =
                    (uint32_t)continued_input_token_count;
                exact_request.expected_input_token_ids_fnv1a64 = input_digest;
                memset(&exact_result, 0, sizeof(exact_result));
                exact_ok = exact_first_token_fn(
                    &exact_request,
                    &exact_result
                );
                if (exact_ok == 0 || exact_result.completed == 0u ||
                    exact_result.status != (int32_t)QRT_STATUS_OK ||
                    exact_result.batch_size != 1u ||
                    exact_result.output_token_id >=
                        (uint32_t)QRT_QWEN36_VOCAB_SIZE ||
                    exact_result.verifier_input_token_count !=
                        (uint32_t)continued_input_token_count ||
                    exact_result.input_token_ids_fnv1a64 != input_digest) {
                    free(continued_input_tokens);
                    qrt_server_copy_text(
                        out_report->failure_stage,
                        sizeof(out_report->failure_stage),
                        exact_result.failure_stage[0] != '\0'
                            ? exact_result.failure_stage
                            : "exact_letter_classifier_provider"
                    );
                    qrt_server_copy_text(
                        out_report->failure,
                        sizeof(out_report->failure),
                        exact_result.failure[0] != '\0'
                            ? exact_result.failure
                            : "exact letter continuation did not return one valid token"
                    );
                    out_report->request_wall_ns =
                        qrt_server_elapsed_ns(request_start_ns);
                    return exact_result.status > (int32_t)QRT_STATUS_OK &&
                            exact_result.status <=
                                (int32_t)QRT_STATUS_PARSE_ERROR
                        ? (qrt_status_t)exact_result.status
                        : QRT_STATUS_UNSUPPORTED;
                }
                if (UINT64_MAX - continuation_elapsed_ns <
                        exact_result.total_elapsed_ns) {
                    free(continued_input_tokens);
                    qrt_server_copy_text(
                        out_report->failure_stage,
                        sizeof(out_report->failure_stage),
                        "exact_letter_classifier_timing"
                    );
                    qrt_server_copy_text(
                        out_report->failure,
                        sizeof(out_report->failure),
                        "exact letter continuation timing overflowed"
                    );
                    out_report->request_wall_ns =
                        qrt_server_elapsed_ns(request_start_ns);
                    return QRT_STATUS_UNSUPPORTED;
                }
                continuation_elapsed_ns += exact_result.total_elapsed_ns;
            }
            free(continued_input_tokens);
            out_report->request_wall_ns =
                qrt_server_elapsed_ns(request_start_ns);
            out_report->ttft_ns = first_token_elapsed_ns;
            out_report->tpot_sample_count =
                generated_token_count > 1u
                    ? generated_token_count - 1u
                    : 0u;
            out_report->tpot_ns = out_report->tpot_sample_count != 0u
                ? continuation_elapsed_ns /
                    out_report->tpot_sample_count
                : UINT64_C(0);
            out_report->output_token_count = generated_token_count;
            fprintf(
                stderr,
                "QRT_SERVER_MARK exact_letter_classifier_continuation input_tokens=%zu output_tokens=%zu final_token=%u found_ascii_A_through_J=%d ttft_ms=%.4f tpot_ms=%.4f resident_prefix_mutated=0\n",
                input_token_count,
                generated_token_count,
                output_tokens[generated_token_count - 1u],
                output_tokens[generated_token_count - 1u] >=
                        (uint32_t)QRT_SERVER_QWEN36_ASCII_A_TOKEN_ID &&
                    output_tokens[generated_token_count - 1u] <=
                        (uint32_t)QRT_SERVER_QWEN36_ASCII_J_TOKEN_ID,
                (double)out_report->ttft_ns / 1000000.0,
                (double)out_report->tpot_ns / 1000000.0
            );
            return QRT_STATUS_OK;
        } else {
            output_tokens[0u] = exact_result.output_token_id;
            *out_output_token_count = 1u;
            memset(&event, 0, sizeof(event));
            event.struct_size = (uint32_t)sizeof(event);
            event.abi_version = QRT_TOKEN_STREAM_EVENT_ABI_VERSION;
            event.phase = QRT_TOKEN_STREAM_PHASE_PREFILL;
            event.output_index = 0u;
            event.token_id = exact_result.output_token_id;
            event.token_step_elapsed_ns = exact_result.total_elapsed_ns;
            event.request_elapsed_ns = qrt_server_elapsed_ns(request_start_ns);
            if (callback(user_data, &event) == 0) {
                *out_output_token_count = 0u;
                qrt_server_copy_text(
                    out_report->failure_stage,
                    sizeof(out_report->failure_stage),
                    "exact_first_token_callback"
                );
                qrt_server_copy_text(
                    out_report->failure,
                    sizeof(out_report->failure),
                    "exact first-token stream callback cancelled emission"
                );
                out_report->request_wall_ns =
                    qrt_server_elapsed_ns(request_start_ns);
                return QRT_STATUS_UNSUPPORTED;
            }
            out_report->request_wall_ns =
                qrt_server_elapsed_ns(request_start_ns);
            out_report->ttft_ns = exact_result.total_elapsed_ns;
            out_report->tpot_ns = UINT64_C(0);
            out_report->tpot_sample_count = 0u;
            out_report->output_token_count = 1u;
            fprintf(
                stderr,
                "QRT_SERVER_MARK exact_first_token_prefill input_tokens=%zu verifier_input_tokens=%u output_token=%u requested_output_tokens=%zu classifier=%d ttft_ms=%.4f resident_prefix_mutated=0\n",
                input_token_count,
                exact_result.verifier_input_token_count,
                exact_result.output_token_id,
                output_token_capacity,
                exact_letter_classifier,
                (double)exact_result.total_elapsed_ns / 1000000.0
            );
            return QRT_STATUS_OK;
        }
    }
    }
#endif
    if (qrt_server_prefix_cache_enabled() &&
        input_token_count <
            (size_t)QRT_SERVER_RETAINED_Q8192_TOKENS) {
        const size_t minimum_prefix_tokens =
            qrt_server_prefix_cache_min_tokens();
        if (engine->resident_prefix_token_count >= minimum_prefix_tokens &&
            qrt_server_prefix_shape_supported(
                input_token_count,
                engine->resident_prefix_token_count,
                output_token_capacity
            ) &&
            memcmp(
                engine->resident_prefix_tokens,
                input_tokens,
                engine->resident_prefix_token_count * sizeof(input_tokens[0])
            ) == 0) {
            prefix_hit_token_count = engine->resident_prefix_token_count;
        } else if (engine->previous_request_token_count != 0u) {
            size_t common_prefix = qrt_server_common_prefix_tokens(
                engine->previous_request_tokens,
                engine->previous_request_token_count,
                input_tokens,
                input_token_count
            );
            if (common_prefix == input_token_count && common_prefix != 0u) {
                --common_prefix;
            }
            if (common_prefix >= minimum_prefix_tokens &&
                qrt_server_prefix_shape_supported(
                    input_token_count,
                    common_prefix,
                    output_token_capacity
                )) {
                prefix_hit_token_count = common_prefix;
                prefix_seed_required = 1;
            }
        }
        if (prefix_hit_token_count == 0u &&
            input_token_count > minimum_prefix_tokens &&
            qrt_server_prefix_shape_supported(
                input_token_count,
                input_token_count - 1u,
                output_token_capacity
            )) {
            prefix_hit_token_count = input_token_count - 1u;
            prefix_seed_required = 1;
        }
    }

    if (prefix_hit_token_count != 0u && prefix_seed_required) {
        uint32_t seed_output_token = UINT32_MAX;
        size_t seed_output_token_count = 0u;
        const uint64_t seed_start_ns = qrt_server_now_ns();
        status = qrt_engine_request_tokens(
            engine->engine,
            input_tokens,
            prefix_hit_token_count,
            &seed_output_token,
            1u,
            &seed_output_token_count
        );
        prefix_seed_elapsed_ns = qrt_server_elapsed_ns(seed_start_ns);
        if (status != QRT_STATUS_OK || seed_output_token_count != 1u ||
            seed_output_token >= (uint32_t)QRT_QWEN36_VOCAB_SIZE ||
            !qrt_server_store_tokens(
                &engine->resident_prefix_tokens,
                &engine->resident_prefix_token_count,
                &engine->resident_prefix_token_capacity,
                input_tokens,
                prefix_hit_token_count
            )) {
            engine->resident_prefix_token_count = 0u;
            *out_output_token_count = 0u;
            out_report->request_wall_ns = qrt_server_elapsed_ns(request_start_ns);
            return status == QRT_STATUS_OK ? QRT_STATUS_OUT_OF_MEMORY : status;
        }
        fprintf(
            stderr,
            "QRT_SERVER_MARK prefix_cache_seed prefix_tokens=%zu suffix_tokens=%zu output_tokens=%zu elapsed_ms=%.4f\n",
            prefix_hit_token_count,
            input_token_count - prefix_hit_token_count,
            output_token_capacity,
            (double)prefix_seed_elapsed_ns / 1000000.0
        );
    }

    if (prefix_hit_token_count != 0u) {
        qrt_qwen36_resident_prefix_cache_result_v1_t *prefix_result =
            (qrt_qwen36_resident_prefix_cache_result_v1_t *)calloc(
                1u,
                sizeof(*prefix_result)
            );
        if (prefix_result == NULL) {
            *out_output_token_count = 0u;
            out_report->request_wall_ns = qrt_server_elapsed_ns(request_start_ns);
            return QRT_STATUS_OUT_OF_MEMORY;
        }
        status = qrt_engine_request_tokens_prefix_stream_v1(
            engine->engine,
            input_tokens,
            input_token_count,
            prefix_hit_token_count,
            output_tokens,
            output_token_capacity,
            prefix_result,
            callback,
            user_data
        );
        if (status == QRT_STATUS_OK) {
            *out_output_token_count = (size_t)prefix_result->output_token_count;
            prefix_route_used = 1;
            (void)qrt_server_store_tokens(
                &engine->previous_request_tokens,
                &engine->previous_request_token_count,
                &engine->previous_request_token_capacity,
                input_tokens,
                input_token_count
            );
            fprintf(
                stderr,
                "QRT_SERVER_MARK prefix_cache_hit prefix_tokens=%zu suffix_tokens=%zu output_tokens=%zu seed=%d ttft_ms=%.4f tpot_ms=%.4f\n",
                prefix_hit_token_count,
                input_token_count - prefix_hit_token_count,
                *out_output_token_count,
                prefix_seed_required,
                (double)prefix_result->ttft_elapsed_ns / 1000000.0,
                (double)prefix_result->tpot_elapsed_ns / 1000000.0
            );
        } else {
            engine->resident_prefix_token_count = 0u;
            *out_output_token_count = 0u;
        }
        free(prefix_result);
    } else {
        status = qrt_engine_request_tokens_stream_v1(
            engine->engine,
            input_tokens,
            input_token_count,
            output_tokens,
            output_token_capacity,
            out_output_token_count,
            callback,
            user_data
        );
        if (status == QRT_STATUS_OK) {
            (void)qrt_server_store_tokens(
                &engine->previous_request_tokens,
                &engine->previous_request_token_count,
                &engine->previous_request_token_capacity,
                input_tokens,
                input_token_count
            );
            if (output_token_capacity == 1u) {
                (void)qrt_server_store_tokens(
                    &engine->resident_prefix_tokens,
                    &engine->resident_prefix_token_count,
                    &engine->resident_prefix_token_capacity,
                    input_tokens,
                    input_token_count
                );
            } else {
                engine->resident_prefix_token_count = 0u;
            }
        }
    }
    out_report->request_wall_ns = qrt_server_elapsed_ns(request_start_ns);
    out_report->output_token_count = *out_output_token_count;
    engine_report = (qrt_engine_report_t *)calloc(1u, sizeof(*engine_report));
    if (engine_report == NULL) {
        return status != QRT_STATUS_OK ? status : QRT_STATUS_OUT_OF_MEMORY;
    }
    if (qrt_engine_report(engine->engine, engine_report) == QRT_STATUS_OK) {
        out_report->ttft_ns = engine_report->last_request_ttft_elapsed_ns +
            (prefix_route_used ? prefix_seed_elapsed_ns : UINT64_C(0));
        out_report->tpot_ns = engine_report->last_request_tpot_elapsed_ns;
        out_report->tpot_sample_count = engine_report->last_request_tpot_sample_count;
        qrt_server_copy_text(
            out_report->failure_stage,
            sizeof(out_report->failure_stage),
            engine_report->token_request_failure_stage
        );
        qrt_server_copy_text(
            out_report->failure,
            sizeof(out_report->failure),
            engine_report->token_request_failure
        );
    }
    free(engine_report);
    return status;
}

qrt_status_t qrt_server_engine_request_tokens_stream_v1(
    qrt_server_engine_t *engine,
    const uint32_t *input_tokens,
    size_t input_token_count,
    uint32_t *output_tokens,
    size_t output_token_capacity,
    size_t *out_output_token_count,
    qrt_token_stream_callback_v1_t callback,
    void *user_data,
    qrt_server_request_report_v1_t *out_report
) {
    return qrt_server_engine_request_tokens_stream_internal(
        engine,
        input_tokens,
        input_token_count,
        output_tokens,
        output_token_capacity,
        out_output_token_count,
        callback,
        user_data,
        out_report,
        0
    );
}

qrt_status_t qrt_server_engine_request_tokens_exact_stream_v1(
    qrt_server_engine_t *engine,
    const uint32_t *input_tokens,
    size_t input_token_count,
    uint32_t *output_tokens,
    size_t output_token_capacity,
    size_t *out_output_token_count,
    qrt_token_stream_callback_v1_t callback,
    void *user_data,
    qrt_server_request_report_v1_t *out_report
) {
    return qrt_server_engine_request_tokens_stream_internal(
        engine,
        input_tokens,
        input_token_count,
        output_tokens,
        output_token_capacity,
        out_output_token_count,
        callback,
        user_data,
        out_report,
        1
    );
}
