#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "qrt.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define QRT_PRODUCT_MAX_TOKEN_FILE_BYTES (16u * 1024u * 1024u)
#define QRT_PRODUCT_MAX_INPUT_TOKENS 300000u
#define QRT_PRODUCT_MAX_PREFIX_HITS 8u

typedef struct qrt_product_options_t {
    const char *model_path;
    const char *tokens_path;
    const char *expected_output_path;
    const char *env_path;
    const char *provider_dll;
    size_t output_token_capacity;
    size_t prefix_token_count;
    size_t prefix_hit_count;
    uint64_t expected_prompt_fnv1a64;
    uint64_t expected_output_fnv1a64;
    int expected_prompt_fnv1a64_set;
    int expected_output_fnv1a64_set;
    int ignore_eos;
    int prefix_negative_guard;
} qrt_product_options_t;

typedef struct qrt_product_stream_t {
    size_t callback_count;
    uint32_t tokens[QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS];
    uint64_t first_callback_wall_ns;
    uint64_t last_callback_wall_ns;
    uint64_t last_request_elapsed_ns;
    uint64_t request_start_ns;
    size_t request_index;
    int failed;
} qrt_product_stream_t;

typedef struct qrt_product_preload_t {
#ifdef _WIN32
    HMODULE module;
#endif
    uint64_t dll_load_ns;
    uint64_t call_wall_ns;
    uint64_t reported_preload_ns;
    uint64_t stored_entry_count;
    int completed;
} qrt_product_preload_t;

static void qrt_product_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s run --model PATH --tokens FILE --output-tokens N "
        "[--prefix-tokens N] [--prefix-hits N] "
        "[--prefix-negative-guard] [--expected-output FILE] "
        "[--expected-prompt-fnv HEX] [--expected-output-fnv HEX] "
        "[--env-file FILE] [--provider-dll FILE] [--ignore-eos]\n",
        program
    );
    fputs(
        "token files are UTF-8 JSON arrays containing unsigned token IDs\n",
        stderr
    );
}

static uint64_t qrt_product_now_ns(void) {
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

static uint64_t qrt_product_elapsed_ns(uint64_t start_ns) {
    const uint64_t end_ns = qrt_product_now_ns();
    return end_ns >= start_ns ? end_ns - start_ns : UINT64_C(0);
}

static void qrt_product_release_preload(qrt_product_preload_t *preload) {
#ifdef _WIN32
    if (preload != NULL && preload->module != NULL) {
        FreeLibrary(preload->module);
        preload->module = NULL;
    }
#else
    (void)preload;
#endif
}

static int qrt_product_preload_provider(
    const char *provider_dll,
    const char *model_path,
    qrt_product_preload_t *out_preload
) {
    if (provider_dll == NULL || provider_dll[0] == '\0' ||
        model_path == NULL || model_path[0] == '\0' ||
        out_preload == NULL) {
        return 0;
    }
    memset(out_preload, 0, sizeof(*out_preload));
#ifndef _WIN32
    fputs("whole-provider preload is available only on Windows\n", stderr);
    return 0;
#else
    typedef int (QRT_CDECL *qrt_product_preload_fn_t)(
        const char *,
        qrt_qwen36_prefill_descriptor_batch_timing_t *,
        char *,
        size_t,
        char *,
        size_t
    );
    qrt_qwen36_prefill_descriptor_batch_timing_t *timing = NULL;
    qrt_product_preload_fn_t preload_fn;
    FARPROC symbol;
    char failure_stage[64] = "";
    char failure[QRT_LOAD_ERROR_CAPACITY] = "";
    uint64_t start_ns = qrt_product_now_ns();
    int ok;

    out_preload->module = LoadLibraryA(provider_dll);
    out_preload->dll_load_ns = qrt_product_elapsed_ns(start_ns);
    if (out_preload->module == NULL) {
        fprintf(
            stderr,
            "could not load whole-provider DLL for preload: error=%lu\n",
            (unsigned long)GetLastError()
        );
        return 0;
    }
    symbol = GetProcAddress(
        out_preload->module,
        "qrt_prefill_descriptor_batch_hip_preload_compact_device_routed_layout_v1"
    );
    if (symbol == NULL) {
        fputs("whole-provider DLL does not export model preload v1\n", stderr);
        qrt_product_release_preload(out_preload);
        return 0;
    }
    preload_fn = (qrt_product_preload_fn_t)(void *)symbol;
    timing = (qrt_qwen36_prefill_descriptor_batch_timing_t *)calloc(
        1u,
        sizeof(*timing)
    );
    if (timing == NULL) {
        qrt_product_release_preload(out_preload);
        return 0;
    }
    start_ns = qrt_product_now_ns();
    ok = preload_fn(
        model_path,
        timing,
        failure_stage,
        sizeof(failure_stage),
        failure,
        sizeof(failure)
    );
    out_preload->call_wall_ns = qrt_product_elapsed_ns(start_ns);
    out_preload->reported_preload_ns =
        timing->compact_device_layout_full_preload_elapsed_ns;
    out_preload->stored_entry_count =
        timing->compact_device_layout_full_prepack_stored_entry_count;
    out_preload->completed = ok != 0 &&
        timing->compact_device_layout_full_prepack_pass != 0u;
    free(timing);
    if (!out_preload->completed) {
        fprintf(
            stderr,
            "whole-provider preload failed: stage=%s failure=%s\n",
            failure_stage,
            failure
        );
        qrt_product_release_preload(out_preload);
        return 0;
    }
    return 1;
#endif
}

static uint64_t qrt_product_fnv1a64_bytes(
    const void *data,
    size_t bytes
) {
    const unsigned char *cursor = (const unsigned char *)data;
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t index;
    for (index = 0u; index < bytes; ++index) {
        hash ^= (uint64_t)cursor[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int qrt_product_parse_size(
    const char *text,
    size_t minimum,
    size_t maximum,
    size_t *out_value
) {
    char *end = NULL;
    unsigned long long value;
    if (text == NULL || out_value == NULL || text[0] == '\0' ||
        text[0] == '-') {
        return 0;
    }
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < (unsigned long long)minimum ||
        value > (unsigned long long)maximum) {
        return 0;
    }
    *out_value = (size_t)value;
    return 1;
}

static int qrt_product_parse_hex_u64(
    const char *text,
    uint64_t *out_value
) {
    char *end = NULL;
    unsigned long long value;
    if (text == NULL || out_value == NULL || text[0] == '\0' ||
        text[0] == '-') {
        return 0;
    }
    if (text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        text += 2;
    }
    errno = 0;
    value = strtoull(text, &end, 16);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *out_value = (uint64_t)value;
    return 1;
}

static int qrt_product_parse_options(
    int argc,
    char **argv,
    qrt_product_options_t *out_options
) {
    int index;
    qrt_product_options_t options;
    if (out_options == NULL || argc < 2 || strcmp(argv[1], "run") != 0) {
        return 0;
    }
    memset(&options, 0, sizeof(options));
    options.prefix_hit_count = 1u;
    for (index = 2; index < argc; ++index) {
        const char *name = argv[index];
        const char *value = NULL;
        if (strcmp(name, "--ignore-eos") == 0) {
            options.ignore_eos = 1;
            continue;
        }
        if (strcmp(name, "--prefix-negative-guard") == 0) {
            options.prefix_negative_guard = 1;
            continue;
        }
        if (index + 1 >= argc) {
            return 0;
        }
        value = argv[++index];
        if (strcmp(name, "--model") == 0) {
            options.model_path = value;
        } else if (strcmp(name, "--tokens") == 0) {
            options.tokens_path = value;
        } else if (strcmp(name, "--expected-output") == 0) {
            options.expected_output_path = value;
        } else if (strcmp(name, "--env-file") == 0) {
            options.env_path = value;
        } else if (strcmp(name, "--provider-dll") == 0) {
            options.provider_dll = value;
        } else if (strcmp(name, "--output-tokens") == 0) {
            if (!qrt_product_parse_size(
                    value,
                    1u,
                    QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS,
                    &options.output_token_capacity
                )) {
                return 0;
            }
        } else if (strcmp(name, "--prefix-tokens") == 0) {
            if (!qrt_product_parse_size(
                    value,
                    1u,
                    QRT_PRODUCT_MAX_INPUT_TOKENS,
                    &options.prefix_token_count
                )) {
                return 0;
            }
        } else if (strcmp(name, "--prefix-hits") == 0) {
            if (!qrt_product_parse_size(
                    value,
                    1u,
                    QRT_PRODUCT_MAX_PREFIX_HITS,
                    &options.prefix_hit_count
                )) {
                return 0;
            }
        } else if (strcmp(name, "--expected-prompt-fnv") == 0) {
            if (!qrt_product_parse_hex_u64(
                    value,
                    &options.expected_prompt_fnv1a64
                )) {
                return 0;
            }
            options.expected_prompt_fnv1a64_set = 1;
        } else if (strcmp(name, "--expected-output-fnv") == 0) {
            if (!qrt_product_parse_hex_u64(
                    value,
                    &options.expected_output_fnv1a64
                )) {
                return 0;
            }
            options.expected_output_fnv1a64_set = 1;
        } else {
            return 0;
        }
    }
    if (options.model_path == NULL || options.tokens_path == NULL ||
        options.output_token_capacity == 0u ||
        (options.prefix_token_count == 0u &&
         (options.prefix_hit_count != 1u ||
          options.prefix_negative_guard))) {
        return 0;
    }
    *out_options = options;
    return 1;
}

static int qrt_product_read_file(
    const char *path,
    char **out_data,
    size_t *out_bytes
) {
    FILE *file;
    long length;
    char *data;
    size_t bytes_read;
    if (path == NULL || out_data == NULL || out_bytes == NULL) {
        return 0;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "could not open %s: %s\n", path, strerror(errno));
        return 0;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        (unsigned long)length > QRT_PRODUCT_MAX_TOKEN_FILE_BYTES ||
        fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "could not size bounded input file %s\n", path);
        fclose(file);
        return 0;
    }
    data = (char *)malloc((size_t)length + 1u);
    if (data == NULL) {
        fclose(file);
        return 0;
    }
    bytes_read = fread(data, 1u, (size_t)length, file);
    if (bytes_read != (size_t)length || ferror(file)) {
        fprintf(stderr, "could not read %s\n", path);
        free(data);
        fclose(file);
        return 0;
    }
    data[bytes_read] = '\0';
    fclose(file);
    *out_data = data;
    *out_bytes = bytes_read;
    return 1;
}

static void qrt_product_skip_space(const char **cursor) {
    while (**cursor != '\0' && isspace((unsigned char)**cursor)) {
        ++*cursor;
    }
}

static int qrt_product_parse_token_array(
    const char *path,
    uint32_t **out_tokens,
    size_t *out_token_count
) {
    char *data = NULL;
    size_t data_bytes = 0u;
    size_t capacity = 1024u;
    size_t count = 0u;
    uint32_t *tokens = NULL;
    const char *cursor;
    int after_comma = 0;
    if (!qrt_product_read_file(path, &data, &data_bytes)) {
        return 0;
    }
    (void)data_bytes;
    cursor = data;
    if (data_bytes >= 3u &&
        (unsigned char)cursor[0] == 0xefu &&
        (unsigned char)cursor[1] == 0xbbu &&
        (unsigned char)cursor[2] == 0xbfu) {
        cursor += 3;
    }
    qrt_product_skip_space(&cursor);
    if (*cursor++ != '[') {
        fprintf(stderr, "%s is not a JSON token array\n", path);
        free(data);
        return 0;
    }
    tokens = (uint32_t *)malloc(capacity * sizeof(*tokens));
    if (tokens == NULL) {
        free(data);
        return 0;
    }
    for (;;) {
        unsigned long value;
        char *end = NULL;
        qrt_product_skip_space(&cursor);
        if (*cursor == ']') {
            if (after_comma) {
                fprintf(stderr, "trailing comma in token array %s\n", path);
                free(tokens);
                free(data);
                return 0;
            }
            ++cursor;
            break;
        }
        if (!isdigit((unsigned char)*cursor)) {
            fprintf(stderr, "invalid token-array syntax in %s\n", path);
            free(tokens);
            free(data);
            return 0;
        }
        errno = 0;
        value = strtoul(cursor, &end, 10);
        if (errno != 0 || end == cursor ||
            value >= (unsigned long)QRT_QWEN36_VOCAB_SIZE ||
            count >= QRT_PRODUCT_MAX_INPUT_TOKENS) {
            fprintf(stderr, "invalid or out-of-range token in %s\n", path);
            free(tokens);
            free(data);
            return 0;
        }
        if (count == capacity) {
            size_t next_capacity = capacity * 2u;
            uint32_t *next;
            if (next_capacity > QRT_PRODUCT_MAX_INPUT_TOKENS) {
                next_capacity = QRT_PRODUCT_MAX_INPUT_TOKENS;
            }
            next = (uint32_t *)realloc(
                tokens,
                next_capacity * sizeof(*tokens)
            );
            if (next == NULL) {
                free(tokens);
                free(data);
                return 0;
            }
            tokens = next;
            capacity = next_capacity;
        }
        tokens[count++] = (uint32_t)value;
        after_comma = 0;
        cursor = end;
        qrt_product_skip_space(&cursor);
        if (*cursor == ',') {
            ++cursor;
            after_comma = 1;
        } else if (*cursor == ']') {
            continue;
        } else {
            fprintf(stderr, "invalid token-array delimiter in %s\n", path);
            free(tokens);
            free(data);
            return 0;
        }
    }
    qrt_product_skip_space(&cursor);
    if (*cursor != '\0' || count == 0u) {
        fprintf(stderr, "%s has trailing content or no tokens\n", path);
        free(tokens);
        free(data);
        return 0;
    }
    free(data);
    *out_tokens = tokens;
    *out_token_count = count;
    return 1;
}

static char *qrt_product_trim(char *text) {
    char *end;
    while (*text != '\0' && isspace((unsigned char)*text)) {
        ++text;
    }
    end = text + strlen(text);
    while (end > text && isspace((unsigned char)end[-1])) {
        --end;
    }
    *end = '\0';
    return text;
}

static int qrt_product_set_env(const char *name, const char *value) {
#ifdef _WIN32
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}

static int qrt_product_load_env_file(const char *path) {
    char *data = NULL;
    size_t bytes = 0u;
    char *line;
    char *next;
    if (path == NULL) {
        return 1;
    }
    if (!qrt_product_read_file(path, &data, &bytes)) {
        return 0;
    }
    (void)bytes;
    line = data;
    if (bytes >= 3u &&
        (unsigned char)line[0] == 0xefu &&
        (unsigned char)line[1] == 0xbbu &&
        (unsigned char)line[2] == 0xbfu) {
        line += 3;
    }
    while (*line != '\0') {
        char *entry;
        char *equals;
        next = strpbrk(line, "\r\n");
        if (next != NULL) {
            *next = '\0';
        }
        entry = qrt_product_trim(line);
        if (*entry != '\0' && *entry != '#') {
            equals = strchr(entry, '=');
            if (equals == NULL) {
                fprintf(stderr, "invalid environment profile line: %s\n", entry);
                free(data);
                return 0;
            }
            *equals++ = '\0';
            entry = qrt_product_trim(entry);
            equals = qrt_product_trim(equals);
            if (*entry == '\0' || !qrt_product_set_env(entry, equals)) {
                fprintf(stderr, "could not set environment key %s\n", entry);
                free(data);
                return 0;
            }
        }
        if (next == NULL) {
            break;
        }
        line = next + 1;
        if (*line == '\n' && next[0] == '\0') {
            ++line;
        }
    }
    free(data);
    return 1;
}

static void qrt_product_print_json_string(const char *value) {
    const unsigned char *cursor = (const unsigned char *)(value != NULL ? value : "");
    fputc('"', stdout);
    while (*cursor != 0u) {
        switch (*cursor) {
            case '"': fputs("\\\"", stdout); break;
            case '\\': fputs("\\\\", stdout); break;
            case '\b': fputs("\\b", stdout); break;
            case '\f': fputs("\\f", stdout); break;
            case '\n': fputs("\\n", stdout); break;
            case '\r': fputs("\\r", stdout); break;
            case '\t': fputs("\\t", stdout); break;
            default:
                if (*cursor < 0x20u) {
                    fprintf(stdout, "\\u%04x", (unsigned int)*cursor);
                } else {
                    fputc((int)*cursor, stdout);
                }
                break;
        }
        ++cursor;
    }
    fputc('"', stdout);
}

static int QRT_CDECL qrt_product_stream_callback(
    void *user_data,
    const qrt_token_stream_event_v1_t *event
) {
    qrt_product_stream_t *stream = (qrt_product_stream_t *)user_data;
    const uint64_t callback_wall_ns = qrt_product_elapsed_ns(
        stream != NULL ? stream->request_start_ns : UINT64_C(0)
    );
    if (stream == NULL || event == NULL ||
        event->struct_size != (uint32_t)sizeof(*event) ||
        event->abi_version != QRT_TOKEN_STREAM_EVENT_ABI_VERSION ||
        event->output_index != (uint32_t)stream->callback_count ||
        stream->callback_count >= QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS ||
        event->token_id >= QRT_QWEN36_VOCAB_SIZE ||
        (stream->callback_count != 0u &&
         event->request_elapsed_ns <= stream->last_request_elapsed_ns)) {
        if (stream != NULL) {
            stream->failed = 1;
        }
        return 0;
    }
    if (stream->callback_count == 0u) {
        stream->first_callback_wall_ns = callback_wall_ns;
    }
    stream->tokens[stream->callback_count++] = event->token_id;
    stream->last_callback_wall_ns = callback_wall_ns;
    stream->last_request_elapsed_ns = event->request_elapsed_ns;
    fputs("{\"type\":\"token\",\"request_index\":", stdout);
    fprintf(
        stdout,
        "%zu,\"index\":%u,\"phase\":\"%s\",\"token_id\":%u,"
        "\"step_ms\":%.6f,\"request_elapsed_ms\":%.6f}\n",
        stream->request_index,
        event->output_index,
        event->phase == QRT_TOKEN_STREAM_PHASE_PREFILL ? "prefill" : "decode",
        event->token_id,
        (double)event->token_step_elapsed_ns / 1000000.0,
        (double)event->request_elapsed_ns / 1000000.0
    );
    fflush(stdout);
    return ferror(stdout) ? 0 : 1;
}

static void qrt_product_print_tokens(
    const uint32_t *tokens,
    size_t count
) {
    size_t index;
    fputc('[', stdout);
    for (index = 0u; index < count; ++index) {
        if (index != 0u) {
            fputc(',', stdout);
        }
        fprintf(stdout, "%u", tokens[index]);
    }
    fputc(']', stdout);
}

static void qrt_product_print_elapsed_ms_array(
    const uint64_t *values,
    size_t count
) {
    size_t index;
    fputc('[', stdout);
    for (index = 0u; index < count; ++index) {
        if (index != 0u) {
            fputc(',', stdout);
        }
        fprintf(stdout, "%.6f", (double)values[index] / 1000000.0);
    }
    fputc(']', stdout);
}

static void qrt_product_print_fnv1a64_array(
    const uint64_t *values,
    size_t count
) {
    size_t index;
    fputc('[', stdout);
    for (index = 0u; index < count; ++index) {
        if (index != 0u) {
            fputc(',', stdout);
        }
        fprintf(stdout, "\"%016" PRIx64 "\"", values[index]);
    }
    fputc(']', stdout);
}

static int qrt_product_run(const qrt_product_options_t *options) {
    uint32_t *input_tokens = NULL;
    size_t input_token_count = 0u;
    uint32_t *expected_output_tokens = NULL;
    size_t expected_output_token_count = 0u;
    uint32_t output_tokens[QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS];
    uint32_t guard_output_tokens[QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS];
    size_t output_token_count = 0u;
    qrt_engine_config_t config;
    qrt_engine_t *engine = NULL;
    qrt_engine_report_t *report = NULL;
    qrt_qwen36_resident_prefix_cache_result_v1_t *prefix_result = NULL;
    qrt_product_stream_t stream;
    qrt_product_preload_t preload;
    qrt_status_t status;
    uint64_t load_start_ns;
    uint64_t load_wall_ns;
    uint64_t engine_create_start_ns;
    uint64_t engine_create_wall_ns;
    uint64_t seed_wall_ns = UINT64_C(0);
    uint64_t request_wall_ns;
    uint64_t prompt_digest;
    uint64_t output_digest;
    uint64_t ttft_ns = UINT64_C(0);
    uint64_t tpot_ns = UINT64_C(0);
    size_t tpot_samples = 0u;
    uint64_t request_wall_ns_total = UINT64_C(0);
    uint64_t caller_ttft_ns_total = UINT64_C(0);
    uint64_t caller_last_callback_ns_total = UINT64_C(0);
    uint64_t provider_ttft_ns_total = UINT64_C(0);
    uint64_t provider_tpot_ns_total = UINT64_C(0);
    size_t provider_tpot_samples_total = 0u;
    size_t total_stream_callback_count = 0u;
    uint64_t prefix_hit_request_wall_ns[QRT_PRODUCT_MAX_PREFIX_HITS] = {0};
    uint64_t prefix_hit_caller_ttft_ns[QRT_PRODUCT_MAX_PREFIX_HITS] = {0};
    uint64_t prefix_hit_provider_ttft_ns[QRT_PRODUCT_MAX_PREFIX_HITS] = {0};
    uint64_t prefix_hit_suffix_ns[QRT_PRODUCT_MAX_PREFIX_HITS] = {0};
    uint64_t prefix_hit_tpot_ns[QRT_PRODUCT_MAX_PREFIX_HITS] = {0};
    uint64_t prefix_hit_total_ns[QRT_PRODUCT_MAX_PREFIX_HITS] = {0};
    uint64_t prefix_hit_input_fnv1a64[QRT_PRODUCT_MAX_PREFIX_HITS] = {0};
    uint64_t prefix_hit_suffix_fnv1a64[QRT_PRODUCT_MAX_PREFIX_HITS] = {0};
    uint64_t prefix_hit_teacher_fnv1a64[QRT_PRODUCT_MAX_PREFIX_HITS] = {0};
    size_t request_count;
    size_t request_index;
    size_t prefix_contract_pass_count = 0u;
    size_t prefix_output_match_count = 0u;
    size_t prefix_stream_match_count = 0u;
    size_t prefix_state_restored_count = 0u;
    int prefix_negative_guard_pass = 1;
    int prefix_negative_guard_status = (int)QRT_STATUS_OK;
    unsigned int prefix_negative_guard_provider_invoked = 0u;
    int token_match = 1;
    int digest_match = 1;
    int prompt_match = 1;
    int stream_match = 1;
    int callbacks_before_return;
    int state_restored;
    int first_token_report_available = 0;
    int first_token_report_token_emitted = 0;
    int first_token_report_matches_output = 0;
    int first_token_raw_logit_available = 0;
    uint32_t first_token_report_token_id = 0u;
    uint32_t first_token_report_top1_token_id = 0u;
    float first_token_raw_logit = 0.0f;
    int final_pass;
    int exit_code = 0;
    size_t index;

    memset(&preload, 0, sizeof(preload));

    if (!qrt_product_load_env_file(options->env_path)) {
        return 3;
    }
    if (options->provider_dll != NULL &&
        !qrt_product_set_env(
            "QRT_QWEN36_WHOLE_PROVIDER_DLL",
            options->provider_dll
        )) {
        fputs("could not set whole-provider DLL path\n", stderr);
        return 3;
    }
    if (options->ignore_eos &&
        !qrt_product_set_env(
            "QRT_QWEN36_RESIDENT_PREFIX_CACHE_IGNORE_EOS",
            "1"
        )) {
        fputs("could not set ignore-EOS policy\n", stderr);
        return 3;
    }
    if (!qrt_product_parse_token_array(
            options->tokens_path,
            &input_tokens,
            &input_token_count
        )) {
        return 3;
    }
    if (options->prefix_token_count >= input_token_count) {
        fputs("prefix token count must be smaller than input token count\n", stderr);
        free(input_tokens);
        return 3;
    }
    if (options->expected_output_path != NULL &&
        !qrt_product_parse_token_array(
            options->expected_output_path,
            &expected_output_tokens,
            &expected_output_token_count
        )) {
        free(input_tokens);
        return 3;
    }
    if (expected_output_tokens != NULL &&
        expected_output_token_count != options->output_token_capacity) {
        fputs("expected output length does not match --output-tokens\n", stderr);
        free(expected_output_tokens);
        free(input_tokens);
        return 3;
    }

    prompt_digest = qrt_product_fnv1a64_bytes(
        input_tokens,
        input_token_count * sizeof(input_tokens[0])
    );
    prompt_match = !options->expected_prompt_fnv1a64_set ||
        prompt_digest == options->expected_prompt_fnv1a64;
    if (!prompt_match) {
        fprintf(
            stderr,
            "prompt digest mismatch: expected %016" PRIx64
            ", got %016" PRIx64 "\n",
            options->expected_prompt_fnv1a64,
            prompt_digest
        );
        free(expected_output_tokens);
        free(input_tokens);
        return 6;
    }

    memset(&config, 0, sizeof(config));
    config.model_path = options->model_path;
    config.context_tokens = input_token_count;
    config.batch_size = 1u;
    load_start_ns = qrt_product_now_ns();
    if (!qrt_product_preload_provider(
            getenv("QRT_QWEN36_WHOLE_PROVIDER_DLL"),
            options->model_path,
            &preload
        )) {
        free(expected_output_tokens);
        free(input_tokens);
        return 4;
    }
    engine_create_start_ns = qrt_product_now_ns();
    status = qrt_engine_create(&config, &engine);
    engine_create_wall_ns = qrt_product_elapsed_ns(engine_create_start_ns);
    load_wall_ns = qrt_product_elapsed_ns(load_start_ns);
    if (status != QRT_STATUS_OK || engine == NULL) {
        fprintf(
            stderr,
            "qrt_engine_create failed: %s\n",
            qrt_strerror(status)
        );
        qrt_product_release_preload(&preload);
        free(expected_output_tokens);
        free(input_tokens);
        return 4;
    }

    memset(output_tokens, 0, sizeof(output_tokens));
    memset(guard_output_tokens, 0, sizeof(guard_output_tokens));
    memset(&stream, 0, sizeof(stream));
    prefix_result = (qrt_qwen36_resident_prefix_cache_result_v1_t *)calloc(
        1u,
        sizeof(*prefix_result)
    );
    report = (qrt_engine_report_t *)calloc(1u, sizeof(*report));
    if (prefix_result == NULL || report == NULL) {
        exit_code = 4;
        goto cleanup;
    }

    if (options->prefix_token_count != 0u) {
        uint32_t seed_output = UINT32_MAX;
        size_t seed_output_count = 0u;
        const uint64_t seed_start_ns = qrt_product_now_ns();
        status = qrt_engine_request_tokens(
            engine,
            input_tokens,
            options->prefix_token_count,
            &seed_output,
            1u,
            &seed_output_count
        );
        seed_wall_ns = qrt_product_elapsed_ns(seed_start_ns);
        if (status != QRT_STATUS_OK || seed_output_count != 1u) {
            fprintf(
                stderr,
                "resident prefix seed failed: %s count=%zu\n",
                qrt_strerror(status),
                seed_output_count
            );
            exit_code = 5;
            goto cleanup;
        }
    }

    request_count = options->prefix_token_count != 0u
        ? options->prefix_hit_count
        : 1u;
    for (request_index = 0u; request_index < request_count; ++request_index) {
        int request_stream_match = 1;
        int request_token_match = 1;
        int request_digest_match;
        int request_callbacks_before_return;
        int request_state_restored;
        uint64_t request_provider_ttft_ns = UINT64_C(0);
        uint64_t request_provider_tpot_ns = UINT64_C(0);
        size_t request_provider_tpot_samples = 0u;

        memset(output_tokens, 0, sizeof(output_tokens));
        memset(prefix_result, 0, sizeof(*prefix_result));
        memset(&stream, 0, sizeof(stream));
        stream.request_index = request_index;
        stream.request_start_ns = qrt_product_now_ns();
        if (options->prefix_token_count != 0u) {
            status = qrt_engine_request_tokens_prefix_stream_v1(
                engine,
                input_tokens,
                input_token_count,
                options->prefix_token_count,
                output_tokens,
                options->output_token_capacity,
                prefix_result,
                qrt_product_stream_callback,
                &stream
            );
            output_token_count = prefix_result->output_token_count;
            request_provider_ttft_ns = prefix_result->ttft_elapsed_ns;
            request_provider_tpot_ns = prefix_result->tpot_elapsed_ns;
            request_provider_tpot_samples =
                (size_t)prefix_result->tpot_sample_count;
        } else {
            status = qrt_engine_request_tokens_stream_v1(
                engine,
                input_tokens,
                input_token_count,
                output_tokens,
                options->output_token_capacity,
                &output_token_count,
                qrt_product_stream_callback,
                &stream
            );
        }
        request_wall_ns = qrt_product_elapsed_ns(stream.request_start_ns);
        if (qrt_engine_report(engine, report) == QRT_STATUS_OK &&
            options->prefix_token_count == 0u) {
            request_provider_ttft_ns = report->last_request_ttft_elapsed_ns;
            request_provider_tpot_ns = report->last_request_tpot_elapsed_ns;
            request_provider_tpot_samples =
                report->last_request_tpot_sample_count;
            first_token_report_available = 1;
            first_token_report_token_emitted =
                report->baseline_output_head_token_emitted != 0;
            first_token_report_token_id =
                report->baseline_output_head_sampled_token_id;
            first_token_report_top1_token_id =
                report->baseline_output_head_topk_token_ids[0];
            first_token_raw_logit =
                report->baseline_output_head_topk_logits[0];
            first_token_report_matches_output =
                first_token_report_token_emitted &&
                output_token_count != 0u &&
                first_token_report_token_id == output_tokens[0] &&
                first_token_report_top1_token_id == output_tokens[0];
            first_token_raw_logit_available =
                first_token_report_matches_output &&
                isfinite((double)first_token_raw_logit);
        }
        if (status != QRT_STATUS_OK || stream.failed ||
            output_token_count != options->output_token_capacity ||
            stream.callback_count != output_token_count) {
            fprintf(
                stderr,
                "streaming request %zu failed: %s output=%zu callbacks=%zu "
                "callback_failed=%d stage=%s failure=%s\n",
                request_index,
                qrt_strerror(status),
                output_token_count,
                stream.callback_count,
                stream.failed,
                report != NULL ? report->token_request_failure_stage : "",
                report != NULL ? report->token_request_failure : ""
            );
            exit_code = 5;
            goto cleanup;
        }
        for (index = 0u; index < output_token_count; ++index) {
            if (output_tokens[index] != stream.tokens[index]) {
                request_stream_match = 0;
            }
            if (expected_output_tokens != NULL &&
                output_tokens[index] != expected_output_tokens[index]) {
                request_token_match = 0;
            }
        }
        output_digest = qrt_product_fnv1a64_bytes(
            output_tokens,
            output_token_count * sizeof(output_tokens[0])
        );
        request_digest_match = !options->expected_output_fnv1a64_set ||
            output_digest == options->expected_output_fnv1a64;
        request_callbacks_before_return =
            stream.last_callback_wall_ns <= request_wall_ns;
        request_state_restored = options->prefix_token_count == 0u ||
            prefix_result->state_restored != 0u;

        stream_match = stream_match && request_stream_match;
        token_match = token_match && request_token_match;
        digest_match = digest_match && request_digest_match;
        callbacks_before_return = request_index == 0u
            ? request_callbacks_before_return
            : callbacks_before_return && request_callbacks_before_return;
        state_restored = request_index == 0u
            ? request_state_restored
            : state_restored && request_state_restored;
        request_wall_ns_total += request_wall_ns;
        caller_ttft_ns_total += stream.first_callback_wall_ns;
        caller_last_callback_ns_total += stream.last_callback_wall_ns;
        provider_ttft_ns_total += request_provider_ttft_ns;
        provider_tpot_ns_total += request_provider_tpot_ns;
        provider_tpot_samples_total += request_provider_tpot_samples;
        total_stream_callback_count += stream.callback_count;

        if (options->prefix_token_count != 0u) {
            const int prefix_contract_pass =
                prefix_result->completed != 0u &&
                prefix_result->provider_invoked != 0u &&
                prefix_result->exact_prefix_match != 0u &&
                prefix_result->copy_on_write_transaction != 0u &&
                prefix_result->state_restored != 0u &&
                prefix_result->prefix_token_count ==
                    (uint32_t)options->prefix_token_count &&
                prefix_result->suffix_token_count ==
                    (uint32_t)(input_token_count -
                        options->prefix_token_count) &&
                prefix_result->output_token_count ==
                    (uint32_t)options->output_token_capacity &&
                prefix_result->input_token_ids_fnv1a64 == prompt_digest &&
                prefix_result->output_token_ids_fnv1a64 == output_digest;
            prefix_hit_request_wall_ns[request_index] = request_wall_ns;
            prefix_hit_caller_ttft_ns[request_index] =
                stream.first_callback_wall_ns;
            prefix_hit_provider_ttft_ns[request_index] =
                request_provider_ttft_ns;
            prefix_hit_suffix_ns[request_index] =
                prefix_result->suffix_elapsed_ns;
            prefix_hit_tpot_ns[request_index] =
                request_provider_tpot_samples != 0u
                    ? request_provider_tpot_ns /
                        request_provider_tpot_samples
                    : UINT64_C(0);
            prefix_hit_total_ns[request_index] =
                prefix_result->total_elapsed_ns;
            prefix_hit_input_fnv1a64[request_index] =
                prefix_result->input_token_ids_fnv1a64;
            prefix_hit_suffix_fnv1a64[request_index] =
                prefix_result->suffix_token_ids_fnv1a64;
            prefix_hit_teacher_fnv1a64[request_index] =
                prefix_result->teacher_forced_prediction_ids_fnv1a64;
            prefix_contract_pass_count += prefix_contract_pass ? 1u : 0u;
            prefix_output_match_count +=
                request_token_match && request_digest_match ? 1u : 0u;
            prefix_stream_match_count +=
                request_stream_match && request_callbacks_before_return
                    ? 1u
                    : 0u;
            prefix_state_restored_count += request_state_restored ? 1u : 0u;
        }
    }
    request_wall_ns = request_wall_ns_total / request_count;
    ttft_ns = provider_ttft_ns_total / request_count;
    tpot_ns = provider_tpot_ns_total;
    tpot_samples = provider_tpot_samples_total;
    stream.first_callback_wall_ns = caller_ttft_ns_total / request_count;
    stream.last_callback_wall_ns =
        caller_last_callback_ns_total / request_count;
    stream.callback_count = total_stream_callback_count;

    if (options->prefix_token_count != 0u &&
        options->prefix_negative_guard) {
        const uint32_t saved_token = input_tokens[0];
        input_tokens[0] = saved_token + 1u < QRT_QWEN36_VOCAB_SIZE
            ? saved_token + 1u
            : 0u;
        memset(prefix_result, 0, sizeof(*prefix_result));
        status = qrt_engine_request_tokens_prefix_v1(
            engine,
            input_tokens,
            input_token_count,
            options->prefix_token_count,
            guard_output_tokens,
            options->output_token_capacity,
            prefix_result
        );
        input_tokens[0] = saved_token;
        prefix_negative_guard_status = (int)status;
        prefix_negative_guard_provider_invoked =
            prefix_result->provider_invoked;
        prefix_negative_guard_pass =
            status != QRT_STATUS_OK &&
            prefix_result->completed == 0u &&
            prefix_result->provider_invoked == 0u &&
            prefix_result->exact_prefix_match == 0u;
    }
    final_pass = stream_match && token_match && digest_match && prompt_match &&
        callbacks_before_return && state_restored &&
        prefix_negative_guard_pass &&
        (options->prefix_token_count == 0u ||
         (prefix_contract_pass_count == request_count &&
          prefix_output_match_count == request_count &&
          prefix_stream_match_count == request_count &&
          prefix_state_restored_count == request_count));

    fprintf(
        stdout,
        "{\"type\":\"summary\",\"status\":\"%s\",",
        final_pass ? "pass" : "fail"
    );
    fputs("\"host\":", stdout);
    qrt_product_print_json_string(getenv("COMPUTERNAME"));
    fputs(",\"model_path\":", stdout);
    qrt_product_print_json_string(options->model_path);
    fputs(",\"provider_dll\":", stdout);
    qrt_product_print_json_string(getenv("QRT_QWEN36_WHOLE_PROVIDER_DLL"));
    fprintf(
        stdout,
        ",\"first_token_raw_logit_contract_version\":1,"
        "\"first_token_report_available\":%s,"
        "\"first_token_report_token_emitted\":%s,"
        "\"first_token_report_matches_output\":%s,"
        "\"first_token_raw_logit_available\":%s,"
        "\"first_token_report_token_id\":",
        first_token_report_available ? "true" : "false",
        first_token_report_token_emitted ? "true" : "false",
        first_token_report_matches_output ? "true" : "false",
        first_token_raw_logit_available ? "true" : "false"
    );
    if (first_token_report_available) {
        fprintf(stdout, "%u", first_token_report_token_id);
    } else {
        fputs("null", stdout);
    }
    fputs(",\"first_token_report_top1_token_id\":", stdout);
    if (first_token_report_available) {
        fprintf(stdout, "%u", first_token_report_top1_token_id);
    } else {
        fputs("null", stdout);
    }
    fputs(",\"first_token_raw_logit\":", stdout);
    if (first_token_raw_logit_available) {
        fprintf(stdout, "%.9g", (double)first_token_raw_logit);
    } else {
        fputs("null", stdout);
    }
    fputs(",\"first_token_raw_logit_source\":", stdout);
    if (first_token_report_available) {
        qrt_product_print_json_string(
            "qrt_engine_report.baseline_output_head_topk_logits[0]"
        );
    } else {
        fputs("null", stdout);
    }
    fprintf(
        stdout,
        ",\"batch_size\":1,\"input_tokens\":%zu,\"prefix_tokens\":%zu,"
        "\"suffix_tokens\":%zu,\"output_tokens\":%zu,"
        "\"prefix_hit_count\":%zu,\"prefix_contract_pass_count\":%zu,"
        "\"prefix_output_match_count\":%zu,"
        "\"prefix_stream_match_count\":%zu,"
        "\"prefix_state_restored_count\":%zu,"
        "\"prefix_negative_guard_requested\":%s,"
        "\"prefix_negative_guard_pass\":%s,"
        "\"prefix_negative_guard_status\":%d,"
        "\"prefix_negative_guard_provider_invoked\":%u,"
        "\"prompt_token_ids_fnv1a64\":\"%016" PRIx64 "\","
        "\"output_token_ids_fnv1a64\":\"%016" PRIx64 "\","
        "\"engine_load_ms\":%.6f,\"provider_preload_ms\":%.6f,"
        "\"provider_preload_reported_ms\":%.6f,"
        "\"provider_preload_dll_load_ms\":%.6f,"
        "\"provider_preload_stored_entries\":%" PRIu64 ","
        "\"engine_create_ms\":%.6f,\"prefix_seed_ms\":%.6f,"
        "\"request_wall_ms\":%.6f,\"ttft_ms\":%.6f,"
        "\"provider_ttft_ms\":%.6f,"
        "\"prefill_tokens_per_second\":%.6f,"
        "\"tpot_ms\":%.6f,\"decode_tokens_per_second\":%.6f,"
        "\"stream_callback_count\":%zu,"
        "\"first_callback_wall_ms\":%.6f,"
        "\"last_callback_wall_ms\":%.6f,"
        "\"callbacks_before_return\":%s,"
        "\"stream_matches_output\":%s,"
        "\"expected_prompt_match\":%s,"
        "\"expected_output_tokens_match\":%s,"
        "\"expected_output_digest_match\":%s,"
        "\"state_restored\":%s,\"output_token_ids\":",
        input_token_count,
        options->prefix_token_count,
        input_token_count - options->prefix_token_count,
        output_token_count,
        request_count,
        prefix_contract_pass_count,
        prefix_output_match_count,
        prefix_stream_match_count,
        prefix_state_restored_count,
        options->prefix_negative_guard ? "true" : "false",
        prefix_negative_guard_pass ? "true" : "false",
        prefix_negative_guard_status,
        prefix_negative_guard_provider_invoked,
        prompt_digest,
        output_digest,
        (double)load_wall_ns / 1000000.0,
        (double)preload.call_wall_ns / 1000000.0,
        (double)preload.reported_preload_ns / 1000000.0,
        (double)preload.dll_load_ns / 1000000.0,
        preload.stored_entry_count,
        (double)engine_create_wall_ns / 1000000.0,
        (double)seed_wall_ns / 1000000.0,
        (double)request_wall_ns / 1000000.0,
        (double)stream.first_callback_wall_ns / 1000000.0,
        (double)ttft_ns / 1000000.0,
        stream.first_callback_wall_ns != 0u ?
            (double)input_token_count *
                1000000000.0 /
                (double)stream.first_callback_wall_ns : 0.0,
        tpot_samples != 0u ?
            (double)tpot_ns / 1000000.0 / (double)tpot_samples : 0.0,
        tpot_ns != 0u ? (double)tpot_samples * 1000000000.0 /
            (double)tpot_ns : 0.0,
        total_stream_callback_count,
        (double)stream.first_callback_wall_ns / 1000000.0,
        (double)stream.last_callback_wall_ns / 1000000.0,
        callbacks_before_return ? "true" : "false",
        stream_match ? "true" : "false",
        prompt_match ? "true" : "false",
        token_match ? "true" : "false",
        digest_match ? "true" : "false",
        state_restored ? "true" : "false"
    );
    qrt_product_print_tokens(output_tokens, output_token_count);
    fputs(",\"prefix_hit_request_wall_ms\":", stdout);
    qrt_product_print_elapsed_ms_array(
        prefix_hit_request_wall_ns,
        options->prefix_token_count != 0u ? request_count : 0u
    );
    fputs(",\"prefix_hit_first_callback_wall_ms\":", stdout);
    qrt_product_print_elapsed_ms_array(
        prefix_hit_caller_ttft_ns,
        options->prefix_token_count != 0u ? request_count : 0u
    );
    fputs(",\"prefix_hit_provider_ttft_ms\":", stdout);
    qrt_product_print_elapsed_ms_array(
        prefix_hit_provider_ttft_ns,
        options->prefix_token_count != 0u ? request_count : 0u
    );
    fputs(",\"prefix_hit_suffix_ms\":", stdout);
    qrt_product_print_elapsed_ms_array(
        prefix_hit_suffix_ns,
        options->prefix_token_count != 0u ? request_count : 0u
    );
    fputs(",\"prefix_hit_tpot_ms\":", stdout);
    qrt_product_print_elapsed_ms_array(
        prefix_hit_tpot_ns,
        options->prefix_token_count != 0u ? request_count : 0u
    );
    fputs(",\"prefix_hit_total_ms\":", stdout);
    qrt_product_print_elapsed_ms_array(
        prefix_hit_total_ns,
        options->prefix_token_count != 0u ? request_count : 0u
    );
    fputs(",\"prefix_hit_input_token_ids_fnv1a64\":", stdout);
    qrt_product_print_fnv1a64_array(
        prefix_hit_input_fnv1a64,
        options->prefix_token_count != 0u ? request_count : 0u
    );
    fputs(",\"prefix_hit_suffix_token_ids_fnv1a64\":", stdout);
    qrt_product_print_fnv1a64_array(
        prefix_hit_suffix_fnv1a64,
        options->prefix_token_count != 0u ? request_count : 0u
    );
    fputs(",\"prefix_hit_teacher_prediction_ids_fnv1a64\":", stdout);
    qrt_product_print_fnv1a64_array(
        prefix_hit_teacher_fnv1a64,
        options->prefix_token_count != 0u ? request_count : 0u
    );
    fputs("}\n", stdout);
    fflush(stdout);

    if (!final_pass) {
        exit_code = 6;
    }

cleanup:
    if (engine != NULL) {
        qrt_engine_free(engine);
    }
    qrt_product_release_preload(&preload);
    free(report);
    free(prefix_result);
    free(expected_output_tokens);
    free(input_tokens);
    return exit_code;
}

int main(int argc, char **argv) {
    qrt_product_options_t options;
    if (!qrt_product_parse_options(argc, argv, &options)) {
        qrt_product_usage(argc > 0 ? argv[0] : "qrt-product");
        return 2;
    }
    return qrt_product_run(&options);
}
