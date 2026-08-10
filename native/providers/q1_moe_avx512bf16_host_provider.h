#ifndef QRT_Q1_MOE_AVX512BF16_HOST_PROVIDER_H
#define QRT_Q1_MOE_AVX512BF16_HOST_PROVIDER_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#if defined(QRT_Q1_MOE_AVX512BF16_HOST_PROVIDER_BUILD_DLL)
#define QRT_Q1_MOE_AVX512BF16_HOST_API __declspec(dllexport)
#elif defined(QRT_Q1_MOE_AVX512BF16_HOST_PROVIDER_USE_DLL)
#define QRT_Q1_MOE_AVX512BF16_HOST_API __declspec(dllimport)
#else
#define QRT_Q1_MOE_AVX512BF16_HOST_API
#endif
#define QRT_Q1_MOE_AVX512BF16_HOST_CALL __cdecl
#else
#define QRT_Q1_MOE_AVX512BF16_HOST_API
#define QRT_Q1_MOE_AVX512BF16_HOST_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define QRT_Q1_MOE_AVX512BF16_HOST_ABI_VERSION UINT32_C(4)
#define QRT_Q1_MOE_AVX512BF16_HOST_HIDDEN UINT32_C(2048)
#define QRT_Q1_MOE_AVX512BF16_HOST_INTERMEDIATE UINT32_C(512)
#define QRT_Q1_MOE_AVX512BF16_HOST_ROUTES UINT32_C(8)
#define QRT_Q1_MOE_AVX512BF16_HOST_EXPERT_COUNT UINT32_C(256)
#define QRT_Q1_MOE_AVX512BF16_HOST_WORKERS UINT32_C(16)
#define QRT_Q1_MOE_AVX512BF16_HOST_GATE_UP_BYTES_PER_EXPERT \
    UINT64_C(4194304)
#define QRT_Q1_MOE_AVX512BF16_HOST_DOWN_BYTES_PER_EXPERT UINT64_C(2097152)
#define QRT_Q1_MOE_AVX512BF16_HOST_BYTES_PER_EXPERT_PAIR UINT64_C(6291456)

typedef struct qrt_q1_moe_avx512bf16_host_provider_t
    qrt_q1_moe_avx512bf16_host_provider_t;

typedef enum qrt_q1_moe_avx512bf16_host_status_t {
    QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR = -1,
    QRT_Q1_MOE_AVX512BF16_HOST_STATUS_FALLBACK_CACHE_MISS = 0,
    QRT_Q1_MOE_AVX512BF16_HOST_STATUS_SUCCESS = 1
} qrt_q1_moe_avx512bf16_host_status_t;

typedef enum qrt_q1_moe_avx512bf16_host_error_t {
    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_NONE = 0,
    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_INVALID_ARGUMENT = 1,
    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_UNSUPPORTED_CPU = 2,
    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_OUT_OF_MEMORY = 3,
    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_WIN32_PATH = 4,
    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_WIN32_OPEN = 5,
    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_WIN32_MAPPING = 6,
    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_SOURCE_BOUNDS = 7,
    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_CACHE_BUSY = 8,
    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_NONFINITE = 9,
    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_INTERNAL = 10,
    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_RESIDENT_COPY = 11,
    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_RESIDENT_ALLOCATION = 12
} qrt_q1_moe_avx512bf16_host_error_t;

typedef enum qrt_q1_moe_avx512bf16_host_arithmetic_mode_t {
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32 = 0,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_HAWKEYE_AMPERE_LOVELACE = 1,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_HAWKEYE_HOPPER = 2,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_HAWKEYE_AMPERE_LOVELACE_LEGACY_ENDPOINTS = 3,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_VLLM_ENDPOINTS = 4,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_HAWKEYE_AMPERE_LOVELACE_FINAL_BF16 = 5,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_HAWKEYE_AMPERE_LOVELACE_ACTIVATION_FINAL_BF16 = 6,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_HAWKEYE_AMPERE_LOVELACE_WEIGHTED_SEQUENTIAL = 7,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_FACTOR_GROUP8_WIDTH26 = 8,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_FACTOR_GROUP16_WIDTH25 = 9,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_FACTOR_GROUP4_WIDTH25 = 10,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_FACTOR_GROUP8_WIDTH24 = 11,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_FINAL_BF16 = 12,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_ACTIVATION_FINAL_BF16 = 13,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_WEIGHTED_SEQUENTIAL = 14,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_ACTIVATION_BF16 = 15,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_WEIGHTED_BF16 = 16,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_ACTIVATION_WEIGHTED_BF16 = 17,
    QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_WEIGHTED_FINAL_BF16 = 18
} qrt_q1_moe_avx512bf16_host_arithmetic_mode_t;

/* A synchronous device-to-host copy.  Return exactly zero on success. */
typedef int32_t (QRT_Q1_MOE_AVX512BF16_HOST_CALL
    *qrt_q1_moe_avx512bf16_host_device_copy_to_host_fn)(
        void *context,
        void *host_destination,
        uint64_t device_source_address,
        uint64_t byte_count
    );

/* Synchronous host allocation/free callbacks.  Return exactly zero on
 * success.  The allocator must honor the requested alignment. */
typedef int32_t (QRT_Q1_MOE_AVX512BF16_HOST_CALL
    *qrt_q1_moe_avx512bf16_host_allocate_fn)(
        void *allocator_context,
        uint64_t byte_count,
        uint64_t alignment,
        void **out_host_pointer
    );

typedef int32_t (QRT_Q1_MOE_AVX512BF16_HOST_CALL
    *qrt_q1_moe_avx512bf16_host_free_fn)(
        void *allocator_context,
        void *host_pointer,
        uint64_t byte_count,
        uint64_t alignment
    );

typedef struct qrt_q1_moe_avx512bf16_host_support_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t cpuid_osxsave;
    uint32_t cpuid_avx;
    uint32_t cpuid_avx512f;
    uint32_t cpuid_avx512bf16;
    uint32_t xcr0_zmm_state;
    uint32_t supported;
} qrt_q1_moe_avx512bf16_host_support_t;

typedef struct qrt_q1_moe_avx512bf16_host_config_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t worker_count;
    uint32_t cache_entry_capacity;
    uint64_t cache_byte_capacity;
    uint32_t prefetch_virtual_memory;
    uint32_t touch_mapped_pages;
} qrt_q1_moe_avx512bf16_host_config_t;

typedef struct qrt_q1_moe_avx512bf16_host_prewarm_request_t {
    uint32_t struct_size;
    uint32_t layer_index;
    const char *model_dir_utf8;
    const char *gate_up_shard_utf8;
    const char *down_shard_utf8;
    uint64_t gate_up_tensor_absolute_begin;
    uint64_t down_tensor_absolute_begin;
    uint64_t gate_up_bytes_per_expert;
    uint64_t down_bytes_per_expert;
    const uint32_t *expert_ids_by_priority;
    uint32_t expert_id_count;
} qrt_q1_moe_avx512bf16_host_prewarm_request_t;

typedef struct qrt_q1_moe_avx512bf16_host_prewarm_result_t {
    uint32_t struct_size;
    int32_t status;
    uint32_t error_code;
    uint32_t win32_error;
    uint32_t requested_count;
    uint32_t mapped_count;
    uint32_t already_present_count;
    uint32_t failed_count;
    uint32_t eviction_count;
    uint32_t cache_entry_count;
    uint64_t cache_bytes;
    uint64_t cache_peak_bytes;
    uint64_t mapped_bytes;
    uint64_t prefetch_requested_bytes;
    uint64_t touched_bytes;
    uint64_t elapsed_ns;
} qrt_q1_moe_avx512bf16_host_prewarm_result_t;

typedef struct qrt_q1_moe_avx512bf16_host_resident_prewarm_request_t {
    uint32_t struct_size;
    uint32_t layer_index;
    uint64_t source_generation;
    uint64_t gate_up_device_base;
    uint64_t down_device_base;
    uint64_t gate_up_bytes_per_expert;
    uint64_t down_bytes_per_expert;
    qrt_q1_moe_avx512bf16_host_device_copy_to_host_fn copy_to_host;
    void *copy_context;
    /* Supply both callbacks for resident product use.  A null pair selects the
     * CRT-aligned fallback retained for native tests only. */
    qrt_q1_moe_avx512bf16_host_allocate_fn host_allocate;
    qrt_q1_moe_avx512bf16_host_free_fn host_free;
    void *allocator_context;
    const uint32_t *expert_ids_by_priority;
    uint32_t expert_id_count;
} qrt_q1_moe_avx512bf16_host_resident_prewarm_request_t;

typedef struct qrt_q1_moe_avx512bf16_host_resident_prewarm_result_t {
    uint32_t struct_size;
    int32_t status;
    uint32_t error_code;
    int32_t callback_error;
    int32_t allocator_error;
    uint32_t requested_count;
    uint32_t copied_count;
    uint32_t already_present_count;
    uint32_t failed_count;
    uint32_t eviction_count;
    uint32_t cache_entry_count;
    uint64_t cache_bytes;
    uint64_t cache_peak_bytes;
    uint64_t copy_call_count;
    uint64_t copied_bytes;
    uint64_t copy_elapsed_ns;
    uint64_t allocation_call_count;
    uint64_t allocated_bytes;
    uint64_t allocation_elapsed_ns;
    uint64_t elapsed_ns;
} qrt_q1_moe_avx512bf16_host_resident_prewarm_result_t;

typedef struct qrt_q1_moe_avx512bf16_host_run_request_t {
    uint32_t struct_size;
    uint32_t layer_index;
    uint32_t arithmetic_mode;
    const uint16_t *input_bf16;
    const uint32_t *expert_ids;
    const float *route_weights_f32;
    float *output_f32;
} qrt_q1_moe_avx512bf16_host_run_request_t;

typedef struct qrt_q1_moe_avx512bf16_host_run_result_t {
    uint32_t struct_size;
    int32_t status;
    uint32_t error_code;
    uint32_t win32_error;
    uint32_t all_routes_hit;
    uint32_t arithmetic_mode;
    uint32_t route_cache_hits;
    uint32_t route_cache_misses;
    uint32_t cache_entry_count;
    uint64_t cache_bytes;
    uint64_t cache_peak_bytes;
    uint64_t weight_bytes_consumed;
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t lookup_elapsed_ns;
    uint64_t gate_up_elapsed_ns;
    uint64_t down_combine_elapsed_ns;
    uint64_t total_elapsed_ns;
    uint64_t output_fnv1a64;
} qrt_q1_moe_avx512bf16_host_run_result_t;

typedef struct qrt_q1_moe_avx512bf16_host_stats_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t worker_count;
    uint32_t cache_entry_capacity;
    uint64_t cache_byte_capacity;
    uint32_t cache_entry_count;
    uint32_t cache_peak_entry_count;
    uint64_t cache_bytes;
    uint64_t cache_peak_bytes;
    uint64_t cache_clock;
    uint64_t prewarm_call_count;
    uint64_t prewarm_success_count;
    uint64_t prewarm_error_count;
    uint64_t prewarm_requested_entry_count;
    uint64_t prewarm_mapped_entry_count;
    uint64_t prewarm_already_present_count;
    uint64_t prewarm_failed_entry_count;
    uint64_t prewarm_eviction_count;
    uint64_t prewarm_mapped_bytes;
    uint64_t prewarm_prefetch_requested_bytes;
    uint64_t prewarm_touched_bytes;
    uint64_t prewarm_elapsed_ns;
    uint64_t resident_prewarm_call_count;
    uint64_t resident_prewarm_success_count;
    uint64_t resident_prewarm_error_count;
    uint64_t resident_prewarm_requested_entry_count;
    uint64_t resident_prewarm_copied_entry_count;
    uint64_t resident_prewarm_already_present_count;
    uint64_t resident_prewarm_failed_entry_count;
    uint64_t resident_prewarm_eviction_count;
    uint64_t resident_copy_call_count;
    uint64_t resident_copied_bytes;
    uint64_t resident_copy_elapsed_ns;
    uint64_t resident_prewarm_elapsed_ns;
    uint64_t run_call_count;
    uint64_t run_success_count;
    uint64_t run_fallback_count;
    uint64_t run_error_count;
    uint64_t run_route_lookup_count;
    uint64_t run_route_hit_count;
    uint64_t run_route_miss_count;
    uint64_t run_weight_bytes_consumed;
    uint64_t run_input_bytes;
    uint64_t run_output_bytes;
    uint64_t run_lookup_elapsed_ns;
    uint64_t run_gate_up_elapsed_ns;
    uint64_t run_down_combine_elapsed_ns;
    uint64_t run_total_elapsed_ns;
    uint32_t last_error_code;
    uint32_t last_win32_error;
    int32_t last_resident_callback_error;
} qrt_q1_moe_avx512bf16_host_stats_t;

QRT_Q1_MOE_AVX512BF16_HOST_API int32_t
QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_query_support(
    qrt_q1_moe_avx512bf16_host_support_t *out_support
);

QRT_Q1_MOE_AVX512BF16_HOST_API int32_t
QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_create(
    const qrt_q1_moe_avx512bf16_host_config_t *config,
    qrt_q1_moe_avx512bf16_host_provider_t **out_provider
);

QRT_Q1_MOE_AVX512BF16_HOST_API int32_t
QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_prewarm(
    qrt_q1_moe_avx512bf16_host_provider_t *provider,
    const qrt_q1_moe_avx512bf16_host_prewarm_request_t *request,
    qrt_q1_moe_avx512bf16_host_prewarm_result_t *out_result
);

QRT_Q1_MOE_AVX512BF16_HOST_API int32_t
QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_prewarm_resident(
    qrt_q1_moe_avx512bf16_host_provider_t *provider,
    const qrt_q1_moe_avx512bf16_host_resident_prewarm_request_t *request,
    qrt_q1_moe_avx512bf16_host_resident_prewarm_result_t *out_result
);

QRT_Q1_MOE_AVX512BF16_HOST_API int32_t
QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_run(
    qrt_q1_moe_avx512bf16_host_provider_t *provider,
    const qrt_q1_moe_avx512bf16_host_run_request_t *request,
    qrt_q1_moe_avx512bf16_host_run_result_t *out_result
);

QRT_Q1_MOE_AVX512BF16_HOST_API int32_t
QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_get_stats(
    qrt_q1_moe_avx512bf16_host_provider_t *provider,
    qrt_q1_moe_avx512bf16_host_stats_t *out_stats
);

/* The returned thread-local snapshot remains valid until this thread calls
 * last_error again. */
QRT_Q1_MOE_AVX512BF16_HOST_API const char *
QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_last_error(
    qrt_q1_moe_avx512bf16_host_provider_t *provider
);

/*
 * Waits for a prewarm or run already executing on provider, then destroys it.
 * The caller must prevent any API call from starting or remaining queued once
 * release begins.  The raw provider handle is invalid after a successful call.
 */
QRT_Q1_MOE_AVX512BF16_HOST_API int32_t
QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_release(
    qrt_q1_moe_avx512bf16_host_provider_t *provider,
    qrt_q1_moe_avx512bf16_host_stats_t *out_final_stats
);

#ifdef __cplusplus
}
#endif

#endif
