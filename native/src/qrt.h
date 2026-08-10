#ifndef QRT_H
#define QRT_H

#include <stddef.h>
#include <stdint.h>

#ifdef QRT_STATIC
#define QRT_API
#elif defined(_WIN32)
#define QRT_API __declspec(dllexport)
#else
#define QRT_API
#endif

#if defined(_WIN32)
#define QRT_CDECL __cdecl
#else
#define QRT_CDECL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum qrt_status_t {
    QRT_STATUS_OK = 0,
    QRT_STATUS_INVALID_ARGUMENT = 1,
    QRT_STATUS_OUT_OF_MEMORY = 2,
    QRT_STATUS_NOT_IMPLEMENTED = 3,
    QRT_STATUS_UNSUPPORTED = 4,
    QRT_STATUS_IO_ERROR = 5,
    QRT_STATUS_PARSE_ERROR = 6
} qrt_status_t;

typedef struct qrt_target_contract_t {
    const char *repo;
    const char *model;
    const char *precision;
    const char *primary_runtime_language;
    const char *tooling_language;
    const char *target_os;
    const char *target_device;
    const char *test_host;
    const char *dependency_policy;
    const char *status;
} qrt_target_contract_t;

typedef struct qrt_engine qrt_engine_t;

typedef struct qrt_engine_config_t {
    const char *model_path;
    size_t context_tokens;
    unsigned int batch_size;
} qrt_engine_config_t;

#define QRT_TOKEN_STREAM_EVENT_ABI_VERSION 1u
#define QRT_TOKEN_STREAM_PHASE_PREFILL 0u
#define QRT_TOKEN_STREAM_PHASE_DECODE 1u

/*
 * request_elapsed_ns is measured from the public streaming request entry and
 * is therefore the caller-visible emission clock.  Index zero is the prefill
 * token and reports TTFT in token_step_elapsed_ns; later indices report the
 * owning decode step and retain the provider-relative cumulative clock for
 * diagnostics.
 */
typedef struct qrt_token_stream_event_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t phase;
    uint32_t output_index;
    uint32_t token_id;
    uint32_t reserved0;
    uint64_t token_step_elapsed_ns;
    uint64_t request_elapsed_ns;
    uint64_t provider_decode_elapsed_ns;
} qrt_token_stream_event_v1_t;

typedef int (QRT_CDECL *qrt_token_stream_callback_v1_t)(
    void *user_data,
    const qrt_token_stream_event_v1_t *event
);

#define QRT_QWEN36_HIDDEN_SIZE 2048u
#define QRT_QWEN36_LAYER_COUNT 40u
#define QRT_QWEN36_ROUTED_OUTPUT_ULP_BUCKET_COUNT 6u
#define QRT_QWEN36_ATTENTION_HEADS 16u
#define QRT_QWEN36_KV_HEADS 2u
#define QRT_QWEN36_HEAD_DIM 256u
#define QRT_QWEN36_VOCAB_SIZE 248320u
#define QRT_QWEN36_EXPERT_COUNT 256u
#define QRT_QWEN36_EXPERTS_PER_TOKEN 8u
#define QRT_QWEN36_EXPERT_INTERMEDIATE_SIZE 512u
#define QRT_QWEN36_MAX_POSITION_EMBEDDINGS 262144u
#define QRT_QWEN36_LINEAR_ATTENTION_LAYERS 30u
#define QRT_QWEN36_FULL_ATTENTION_LAYERS 10u
#define QRT_QWEN36_TOKEN_EMBEDDING_BYTES (QRT_QWEN36_HIDDEN_SIZE * sizeof(uint16_t))
#define QRT_QWEN36_BF16_QMATVEC_WEIGHT_ELEMENTS \
    (QRT_QWEN36_HIDDEN_SIZE * QRT_QWEN36_EXPERT_INTERMEDIATE_SIZE)
#define QRT_QWEN36_BF16_QMATVEC_EXPECTED_FNV1A64 UINT64_C(0x08b9542376446c89)
#define QRT_LOAD_MAX_SHARDS 128u
#define QRT_LOAD_ERROR_CAPACITY 192u
#define QRT_MICRO_HIDDEN_SIZE 8u
#define QRT_MICRO_EXPERT_COUNT 2u
#define QRT_MICRO_EXPERT_INTERMEDIATE_SIZE 4u
#define QRT_MICRO_VOCAB_SIZE 8u
#define QRT_MICRO_PROMPT_TOKENS 2u
#define QRT_MICRO_EXPECTED_TRACE_FNV1A64 UINT64_C(0xbcfd6dfeb5eafe89)
#define QRT_PREFIX_CACHE_SELFTEST_PREFIX_TOKENS 65536u
#define QRT_PREFIX_CACHE_SELFTEST_SUFFIX_TOKENS 1024u
#define QRT_PREFIX_CACHE_BLOCK_TOKENS 256u
#define QRT_PREFIX_CACHE_TARGET_OUTPUT_TOKENS 512u
#define QRT_PREFIX_CACHE_EXECUTION_SUFFIX_TOKENS 16u
#define QRT_PREFIX_CACHE_EXECUTION_OUTPUT_TOKENS 2u
#define QRT_PREFIX_CACHE_FULL_KV_WINDOW_TOKENS 4u
#define QRT_PREFIX_CACHE_FULL_KV_COMPRESSED_BLOCK_TOKENS \
    QRT_PREFIX_CACHE_BLOCK_TOKENS
#define QRT_PREFIX_CACHE_FULL_KV_COMPRESSED_PAYLOAD_SAMPLES 4u
#define QRT_QWEN36_TENSOR_NAME_CAPACITY 160u

#ifndef QRT_QWEN36_PREFIX_FULL_KV_COMPRESSED_VALUE_PAYLOAD_DEFINED
#define QRT_QWEN36_PREFIX_FULL_KV_COMPRESSED_VALUE_PAYLOAD_DEFINED
typedef struct qrt_qwen36_prefix_full_kv_compressed_value_payload_t {
    size_t block_start_token;
    size_t block_token_count;
    uint64_t key_digest_fnv1a64;
    uint64_t value_digest_fnv1a64;
    uint64_t summary_digest_fnv1a64;
    float key_samples[QRT_PREFIX_CACHE_FULL_KV_COMPRESSED_PAYLOAD_SAMPLES];
    float value_samples[QRT_PREFIX_CACHE_FULL_KV_COMPRESSED_PAYLOAD_SAMPLES];
} qrt_qwen36_prefix_full_kv_compressed_value_payload_t;
#endif

typedef struct qrt_qwen36_decode_cache_payload_view_t {
    size_t layer_count;
    size_t token_count;
    size_t prompt_token_capacity;
    size_t linear_ring_elements;
    size_t linear_core_state_elements;
    size_t full_kv_history_token_elements;
    size_t full_kv_window_elements;
    size_t full_kv_compressed_block_count;
    size_t full_kv_compressed_block_tokens;
    size_t full_kv_compressed_covered_tokens;
    size_t full_kv_compressed_value_payload_stride_bytes;
    size_t linear_layer_count;
    size_t full_attention_layer_count;
    unsigned int prefix_full_kv_compressed_enabled;
    unsigned int prefix_full_kv_compressed_value_payload_enabled;
    uint64_t payload_view_digest_fnv1a64;
    const float *linear_qkv_rings[QRT_QWEN36_LAYER_COUNT];
    const float *linear_core_states[QRT_QWEN36_LAYER_COUNT];
    const float *full_kv_histories[QRT_QWEN36_LAYER_COUNT];
    const float *prefix_full_kv_windows[QRT_QWEN36_LAYER_COUNT];
    const uint64_t
        *prefix_full_kv_compressed_block_digests[QRT_QWEN36_LAYER_COUNT];
    const qrt_qwen36_prefix_full_kv_compressed_value_payload_t
        *prefix_full_kv_compressed_value_payloads[QRT_QWEN36_LAYER_COUNT];
} qrt_qwen36_decode_cache_payload_view_t;

typedef struct qrt_bf16_qmatvec_result_t {
    size_t hidden_size;
    size_t expert_intermediate_size;
    size_t weight_elements;
    size_t payload_bytes;
    uint64_t expected_output_fnv1a64;
    uint64_t output_fnv1a64;
    float max_abs_diff;
    int correctness_pass;
} qrt_bf16_qmatvec_result_t;

typedef struct qrt_micro_result_t {
    size_t hidden_size;
    size_t expert_count;
    size_t expert_intermediate_size;
    size_t vocab_size;
    size_t prompt_tokens;
    size_t kv_tokens;
    int selected_expert;
    int generated_token;
    int streaming_callbacks;
    int exact_prefix_hit_tokens;
    int shared_prefix_hit_tokens;
    int unrelated_prefix_hit_tokens;
    int contamination_guard_pass;
    uint64_t expected_trace_fnv1a64;
    uint64_t trace_fnv1a64;
    int rmsnorm_pass;
    int rope_pass;
    int attention_kv_pass;
    int linear_attention_pass;
    int moe_pass;
    int lm_head_pass;
    int prefix_cache_pass;
    int streaming_pass;
    int correctness_pass;
} qrt_micro_result_t;

typedef struct qrt_prefix_cache_selftest_result_t {
    size_t cached_prefix_tokens;
    size_t suffix_tokens;
    size_t exact_request_tokens;
    size_t shared_request_tokens;
    size_t unrelated_request_tokens;
    size_t block_tokens;
    size_t owner_table_entries;
    size_t exact_prefix_hit_tokens;
    size_t shared_prefix_hit_tokens;
    size_t unrelated_prefix_hit_tokens;
    size_t shared_owner_alias_entries;
    uint64_t route_fingerprint_fnv1a64;
    uint64_t prefix_identity_fnv1a64;
    uint64_t exact_request_fnv1a64;
    uint64_t shared_request_fnv1a64;
    uint64_t unrelated_request_fnv1a64;
    uint64_t owner_table_fnv1a64;
    uint64_t suffix_handoff_fnv1a64;
    uint64_t result_digest_fnv1a64;
    uint64_t forbidden_copy_bytes;
    int exact_prefix_pass;
    int shared_prefix_pass;
    int contamination_guard_pass;
    int owner_table_pass;
    int suffix_handoff_pass;
    int no_forbidden_copy_pass;
    int correctness_pass;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_prefix_cache_selftest_result_t;

typedef struct qrt_prefix_cache_full_model_result_t {
    size_t context_tokens;
    unsigned int batch_size;
    int model_loaded;
    int engine_ready;
    char engine_status[64];
    size_t manifest_tensor_count;
    size_t manifest_shard_count;
    uint64_t manifest_total_bytes;
    uint64_t manifest_data_bytes;
    int manifest_shape_pass;
    int manifest_ready;
    uint64_t engine_create_elapsed_ns;
    uint64_t manifest_elapsed_ns;
    size_t prompt_tokens;
    size_t cached_prefix_tokens;
    size_t suffix_tokens;
    size_t requested_output_tokens;
    size_t output_tokens_emitted;
    size_t block_tokens;
    size_t owner_table_entries;
    size_t exact_prefix_hit_tokens;
    size_t shared_prefix_hit_tokens;
    size_t unrelated_prefix_hit_tokens;
    size_t shared_owner_alias_entries;
    uint64_t route_fingerprint_fnv1a64;
    uint64_t model_manifest_fnv1a64;
    uint64_t prefix_identity_fnv1a64;
    uint64_t cache_key_fnv1a64;
    uint64_t suffix_handoff_fnv1a64;
    uint64_t result_digest_fnv1a64;
    uint64_t forbidden_copy_bytes;
    uint64_t copy_on_write_bytes;
    uint64_t prefix_seed_elapsed_ns;
    uint64_t prefix_lookup_elapsed_ns;
    uint64_t suffix_handoff_elapsed_ns;
    uint64_t ttft_elapsed_ns;
    uint64_t tpot_elapsed_ns;
    int exact_prefix_pass;
    int shared_prefix_pass;
    int contamination_guard_pass;
    int owner_table_pass;
    int suffix_handoff_pass;
    int no_forbidden_copy_pass;
    int model_load_pass;
    int output_correctness_evaluated;
    int output_correctness_pass;
    int timing_fields_present;
    int full_model_contract_pass;
    int inference_success_claimed;
    int product_performance_accepted;
    char timing_scope[96];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_prefix_cache_full_model_result_t;

typedef struct qrt_prefix_cache_request_execution_result_t {
    size_t context_tokens;
    unsigned int batch_size;
    int model_loaded;
    int engine_ready;
    char engine_status[64];
    size_t manifest_tensor_count;
    size_t manifest_shard_count;
    uint64_t manifest_total_bytes;
    uint64_t manifest_data_bytes;
    int manifest_shape_pass;
    int manifest_ready;
    uint64_t engine_create_elapsed_ns;
    uint64_t manifest_elapsed_ns;
    size_t prompt_tokens;
    size_t cached_prefix_tokens;
    size_t suffix_tokens;
    size_t requested_output_tokens;
    size_t executed_input_tokens;
    size_t output_token_capacity;
    size_t output_tokens_emitted;
    uint32_t output_tokens[QRT_PREFIX_CACHE_EXECUTION_OUTPUT_TOKENS];
    size_t block_tokens;
    size_t owner_table_entries;
    size_t exact_prefix_hit_tokens;
    size_t shared_prefix_hit_tokens;
    size_t unrelated_prefix_hit_tokens;
    size_t shared_owner_alias_entries;
    uint64_t route_fingerprint_fnv1a64;
    uint64_t model_manifest_fnv1a64;
    uint64_t prefix_identity_fnv1a64;
    uint64_t cache_key_fnv1a64;
    uint64_t suffix_handoff_fnv1a64;
    uint64_t request_tokens_fnv1a64;
    uint64_t output_tokens_fnv1a64;
    uint64_t block_integration_digest_fnv1a64;
    uint64_t hidden_handoff_digest_fnv1a64;
    uint64_t request_handoff_digest_fnv1a64;
    uint64_t request_handoff_request_digest_fnv1a64;
    uint64_t output_head_input_hidden_fnv1a64;
    uint64_t output_head_logits_fnv1a64;
    uint64_t output_head_sampler_fnv1a64;
    uint64_t result_digest_fnv1a64;
    uint64_t forbidden_copy_bytes;
    uint64_t copy_on_write_bytes;
    uint64_t prefix_seed_elapsed_ns;
    uint64_t prefix_lookup_elapsed_ns;
    uint64_t suffix_handoff_elapsed_ns;
    uint64_t handoff_setup_elapsed_ns;
    uint64_t request_execution_elapsed_ns;
    uint64_t ttft_elapsed_ns;
    uint64_t tpot_elapsed_ns;
    size_t tpot_sample_count;
    int exact_prefix_pass;
    int shared_prefix_pass;
    int contamination_guard_pass;
    int owner_table_pass;
    int suffix_handoff_pass;
    int no_forbidden_copy_pass;
    int model_load_pass;
    int handoff_setup_pass;
    int request_handoff_attached;
    int request_handoff_consumed;
    int request_execution_attempted;
    int request_execution_status_code;
    size_t report_request_count;
    size_t report_token_request_count;
    size_t report_request_handoff_last_input_token_count;
    size_t report_request_handoff_last_output_token_capacity;
    uint32_t report_request_handoff_last_input_token;
    uint32_t report_output_head_top1_token_id;
    int report_output_head_token_emitted;
    int output_correctness_evaluated;
    int output_correctness_pass;
    int timing_fields_present;
    int prefix_state_injection_supported;
    int full_prefix_state_semantics_evaluated;
    int report_prefix_tensor_linear_state_seeded;
    int report_prefix_tensor_linear_state_consumed;
    int report_prefix_tensor_full_kv_window_state_requested;
    int report_prefix_tensor_full_kv_compressed_state_requested;
    int report_prefix_tensor_full_kv_state_seeded;
    int report_prefix_tensor_full_kv_state_consumed;
    int report_prefix_tensor_full_kv_compressed_state_seeded;
    int report_prefix_tensor_full_kv_compressed_state_consumed;
    size_t report_prefix_tensor_linear_seed_layer_count;
    size_t report_prefix_tensor_linear_seed_value_count;
    size_t report_prefix_tensor_full_kv_seed_layer_count;
    size_t report_prefix_tensor_full_kv_seed_value_count;
    size_t report_prefix_tensor_full_kv_window_token_count;
    size_t report_prefix_tensor_full_kv_cached_prefix_tokens;
    size_t report_prefix_tensor_full_kv_absolute_start_position;
    size_t report_prefix_tensor_full_kv_absolute_last_position;
    size_t report_prefix_tensor_full_kv_compressed_block_count;
    size_t report_prefix_tensor_full_kv_compressed_block_tokens;
    size_t report_prefix_tensor_full_kv_compressed_covered_tokens;
    size_t report_prefix_tensor_full_kv_compressed_tail_block_index;
    size_t report_prefix_tensor_full_kv_compressed_tail_offset_tokens;
    int report_prefix_tensor_full_kv_compressed_backend_span_handoff_evaluated;
    size_t report_prefix_tensor_full_kv_compressed_backend_span_layer_count;
    size_t report_prefix_tensor_full_kv_compressed_backend_span_eval_count;
    size_t report_prefix_tensor_full_kv_compressed_backend_span_block_reference_count;
    size_t report_prefix_tensor_full_kv_compressed_backend_span_token_reference_count;
    int report_prefix_tensor_full_kv_compressed_value_payload_present;
    size_t report_prefix_tensor_full_kv_compressed_value_payload_layer_count;
    size_t report_prefix_tensor_full_kv_compressed_value_payload_block_count;
    size_t report_prefix_tensor_full_kv_compressed_value_payload_sample_count;
    uint64_t report_prefix_tensor_linear_seed_digest_fnv1a64;
    uint64_t report_prefix_tensor_linear_core_input_digest_fnv1a64;
    uint64_t report_prefix_tensor_linear_core_output_digest_fnv1a64;
    uint64_t report_prefix_tensor_full_kv_seed_digest_fnv1a64;
    uint64_t report_prefix_tensor_full_kv_core_input_digest_fnv1a64;
    uint64_t report_prefix_tensor_full_kv_core_output_digest_fnv1a64;
    uint64_t report_prefix_tensor_full_kv_compressed_seed_digest_fnv1a64;
    uint64_t report_prefix_tensor_full_kv_compressed_position_digest_fnv1a64;
    uint64_t report_prefix_tensor_full_kv_compressed_tail_digest_fnv1a64;
    uint64_t report_prefix_tensor_full_kv_compressed_core_input_digest_fnv1a64;
    uint64_t report_prefix_tensor_full_kv_compressed_core_output_digest_fnv1a64;
    uint64_t report_prefix_tensor_full_kv_compressed_backend_span_input_digest_fnv1a64;
    uint64_t report_prefix_tensor_full_kv_compressed_backend_span_output_digest_fnv1a64;
    uint64_t report_prefix_tensor_full_kv_compressed_value_payload_digest_fnv1a64;
    int request_execution_pass;
    int inference_success_claimed;
    int product_performance_accepted;
    char timing_scope[96];
    char output_correctness_scope[96];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_prefix_cache_request_execution_result_t;

typedef struct qrt_prefix_cache_state_attach_request_t {
    size_t cached_prefix_tokens;
    size_t suffix_tokens;
    size_t requested_output_tokens;
    size_t executed_input_tokens;
    size_t output_token_capacity;
    size_t block_tokens;
    size_t owner_table_entries;
    size_t shared_owner_alias_entries;
    uint64_t model_manifest_fnv1a64;
    uint64_t prefix_identity_fnv1a64;
    uint64_t cache_key_fnv1a64;
    uint64_t suffix_handoff_fnv1a64;
    uint64_t request_tokens_fnv1a64;
    int use_copy_on_write_fallback;
    int request_tensor_linear_state;
    int request_tensor_full_kv_state;
    int request_tensor_full_kv_window_state;
    int request_tensor_full_kv_compressed_state;
} qrt_prefix_cache_state_attach_request_t;

typedef struct qrt_prefix_cache_state_attach_result_t {
    size_t context_tokens;
    size_t cached_prefix_tokens;
    size_t suffix_tokens;
    size_t requested_output_tokens;
    size_t executed_input_tokens;
    size_t output_token_capacity;
    size_t block_tokens;
    size_t owner_table_entries;
    size_t shared_owner_alias_entries;
    size_t request_absolute_start_position;
    size_t request_absolute_last_input_position;
    size_t first_output_position;
    size_t last_output_position;
    size_t prefix_state_full_attention_token_capacity;
    size_t mutable_request_token_capacity;
    uint64_t model_manifest_fnv1a64;
    uint64_t prefix_identity_fnv1a64;
    uint64_t cache_key_fnv1a64;
    uint64_t suffix_handoff_fnv1a64;
    uint64_t request_tokens_fnv1a64;
    uint64_t linear_attention_state_bytes;
    uint64_t full_attention_kv_bytes;
    uint64_t total_prefix_state_bytes;
    uint64_t mutable_linear_attention_state_bytes;
    uint64_t mutable_full_attention_kv_bytes;
    uint64_t copy_on_write_bytes;
    uint64_t forbidden_copy_bytes;
    uint64_t state_digest_fnv1a64;
    uint64_t position_digest_fnv1a64;
    uint64_t result_digest_fnv1a64;
    size_t report_state_attach_count;
    size_t report_state_consumed_request_count;
    size_t report_last_request_input_token_count;
    size_t report_last_request_output_token_capacity;
    size_t report_last_request_absolute_start_position;
    size_t report_last_request_absolute_last_input_position;
    size_t report_last_request_first_output_position;
    size_t report_last_request_last_output_position;
    uint64_t report_last_request_digest_fnv1a64;
    int prefix_state_attached;
    int copy_on_write_fallback_used;
    int tensor_linear_state_requested;
    int tensor_full_kv_state_requested;
    int tensor_full_kv_window_state_requested;
    int tensor_full_kv_compressed_state_requested;
    int state_accounting_pass;
    int position_accounting_pass;
    int no_forbidden_copy_pass;
    int request_path_consumed;
    char state_mode[64];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_prefix_cache_state_attach_result_t;

typedef struct qrt_prefix_cache_state_injection_result_t {
    qrt_prefix_cache_request_execution_result_t request;
    qrt_prefix_cache_state_attach_result_t state_attach;
    uint64_t result_digest_fnv1a64;
    int prefix_state_injection_supported;
    int full_prefix_state_semantics_evaluated;
    int state_injection_pass;
    int inference_success_claimed;
    int product_performance_accepted;
    char timing_scope[96];
    char output_correctness_scope[96];
    char state_scope[96];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_prefix_cache_state_injection_result_t;

typedef struct qrt_prefix_cache_tensor_state_result_t {
    qrt_prefix_cache_request_execution_result_t request;
    qrt_prefix_cache_state_attach_result_t state_attach;
    uint64_t result_digest_fnv1a64;
    int linear_tensor_state_requested;
    int linear_tensor_state_seeded;
    int linear_tensor_state_consumed;
    int linear_tensor_state_correctness_pass;
    int full_kv_tensor_state_requested;
    int full_kv_tensor_state_seeded;
    int full_kv_tensor_state_consumed;
    int full_kv_tensor_state_correctness_pass;
    int tensor_state_classification_pass;
    int tensor_backed_prefix_state_correctness_pass;
    int inference_success_claimed;
    int product_performance_accepted;
    char first_missing_tensor_boundary[96];
    char timing_scope[96];
    char output_correctness_scope[96];
    char state_scope[96];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_prefix_cache_tensor_state_result_t;

typedef struct qrt_prefix_cache_full_kv_tensor_state_result_t {
    qrt_prefix_cache_request_execution_result_t request;
    qrt_prefix_cache_state_attach_result_t state_attach;
    uint64_t result_digest_fnv1a64;
    int linear_tensor_state_requested;
    int linear_tensor_state_seeded;
    int linear_tensor_state_consumed;
    int linear_tensor_state_correctness_pass;
    int full_kv_tensor_state_requested;
    int full_kv_window_state_requested;
    int full_kv_tensor_state_seeded;
    int full_kv_tensor_state_consumed;
    int full_kv_tensor_state_correctness_pass;
    int full_kv_window_representation_evaluated;
    int full_kv_complete_span_evaluated;
    int tensor_backed_prefix_state_correctness_pass;
    size_t full_kv_window_token_count;
    size_t full_kv_seed_layer_count;
    size_t full_kv_seed_value_count;
    size_t full_kv_cached_prefix_tokens;
    size_t full_kv_absolute_start_position;
    size_t full_kv_absolute_last_position;
    int inference_success_claimed;
    int product_performance_accepted;
    char next_full_kv_boundary[96];
    char timing_scope[96];
    char output_correctness_scope[96];
    char state_scope[128];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_prefix_cache_full_kv_tensor_state_result_t;

typedef struct qrt_prefix_cache_full_kv_compressed_state_result_t {
    qrt_prefix_cache_request_execution_result_t request;
    qrt_prefix_cache_state_attach_result_t state_attach;
    uint64_t result_digest_fnv1a64;
    int linear_tensor_state_correctness_pass;
    int full_kv_window_state_correctness_pass;
    int full_kv_compressed_state_requested;
    int full_kv_compressed_state_seeded;
    int full_kv_compressed_state_consumed;
    int full_kv_compressed_position_map_pass;
    int full_kv_compressed_tail_window_equivalence_pass;
    int full_kv_compressed_equivalence_pass;
    int full_kv_compressed_backend_span_handoff_evaluated;
    int full_kv_compressed_backend_span_classification_pass;
    int full_kv_compressed_backend_span_value_payload_present;
    int full_kv_compressed_value_payload_summary_pass;
    int full_kv_complete_value_span_evaluated;
    int tensor_backed_prefix_state_correctness_pass;
    size_t full_kv_compressed_block_count;
    size_t full_kv_compressed_block_tokens;
    size_t full_kv_compressed_covered_tokens;
    size_t full_kv_compressed_tail_block_index;
    size_t full_kv_compressed_tail_offset_tokens;
    size_t full_kv_compressed_backend_span_layer_count;
    size_t full_kv_compressed_backend_span_eval_count;
    size_t full_kv_compressed_backend_span_block_reference_count;
    size_t full_kv_compressed_backend_span_token_reference_count;
    size_t full_kv_compressed_value_payload_layer_count;
    size_t full_kv_compressed_value_payload_block_count;
    size_t full_kv_compressed_value_payload_sample_count;
    size_t full_kv_window_token_count;
    size_t full_kv_seed_layer_count;
    size_t full_kv_seed_value_count;
    int inference_success_claimed;
    int product_performance_accepted;
    uint64_t full_kv_compressed_value_payload_digest_fnv1a64;
    char first_missing_backend_span_boundary[96];
    char next_full_kv_boundary[96];
    char timing_scope[96];
    char output_correctness_scope[96];
    char state_scope[128];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_prefix_cache_full_kv_compressed_state_result_t;

typedef enum qrt_qwen36_baseline_attention_kind_t {
    QRT_QWEN36_BASELINE_ATTENTION_LINEAR = 0,
    QRT_QWEN36_BASELINE_ATTENTION_FULL = 1
} qrt_qwen36_baseline_attention_kind_t;

typedef enum qrt_qwen36_tensor_kind_t {
    QRT_QWEN36_TENSOR_TOKEN_EMBEDDING = 0,
    QRT_QWEN36_TENSOR_INPUT_NORM = 1,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_QKV = 2,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_Z = 3,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_A = 4,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_B = 5,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_CONV = 6,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_A_LOG = 7,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_DT_BIAS = 8,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_NORM = 9,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_OUT_PROJ = 10,
    QRT_QWEN36_TENSOR_FULL_ATTN_Q = 11,
    QRT_QWEN36_TENSOR_FULL_ATTN_K = 12,
    QRT_QWEN36_TENSOR_FULL_ATTN_V = 13,
    QRT_QWEN36_TENSOR_FULL_ATTN_O = 14,
    QRT_QWEN36_TENSOR_FULL_ATTN_Q_NORM = 15,
    QRT_QWEN36_TENSOR_FULL_ATTN_K_NORM = 16,
    QRT_QWEN36_TENSOR_POST_ATTENTION_NORM = 17,
    QRT_QWEN36_TENSOR_MOE_ROUTER = 18,
    QRT_QWEN36_TENSOR_MOE_EXPERT_GATE_UP = 19,
    QRT_QWEN36_TENSOR_MOE_EXPERT_DOWN = 20,
    QRT_QWEN36_TENSOR_MOE_SHARED_GATE = 21,
    QRT_QWEN36_TENSOR_MOE_SHARED_GATE_PROJ = 22,
    QRT_QWEN36_TENSOR_MOE_SHARED_UP_PROJ = 23,
    QRT_QWEN36_TENSOR_MOE_SHARED_DOWN = 24,
    QRT_QWEN36_TENSOR_FINAL_NORM = 25,
    QRT_QWEN36_TENSOR_LM_HEAD = 26
} qrt_qwen36_tensor_kind_t;

typedef enum qrt_qwen36_execution_mode_t {
    QRT_QWEN36_EXECUTION_PREFILL = 1,
    QRT_QWEN36_EXECUTION_DECODE_ONE = 2
} qrt_qwen36_execution_mode_t;

typedef struct qrt_qwen36_baseline_layer_descriptor_t {
    unsigned int layer_index;
    qrt_qwen36_baseline_attention_kind_t attention_kind;
    unsigned int hidden_size;
    unsigned int attention_heads;
    unsigned int kv_heads;
    unsigned int head_dim;
    unsigned int expert_count;
    unsigned int experts_per_token;
    unsigned int expert_intermediate_size;
    unsigned int has_moe;
    unsigned int has_shared_expert;
    unsigned int module_count;
} qrt_qwen36_baseline_layer_descriptor_t;

typedef struct qrt_qwen36_baseline_plan_t {
    unsigned int model_specific;
    unsigned int batch_size;
    unsigned int descriptor_table_driven;
    unsigned int baseline_complete_engine_step;
    unsigned int prefill_entrypoint_planned;
    unsigned int decode_one_entrypoint_planned;
    unsigned int layer_count;
    unsigned int linear_attention_layer_count;
    unsigned int full_attention_layer_count;
    unsigned int hidden_size;
    unsigned int vocab_size;
    unsigned int expert_count;
    unsigned int experts_per_token;
    unsigned int expert_intermediate_size;
    const qrt_qwen36_baseline_layer_descriptor_t *layers;
} qrt_qwen36_baseline_plan_t;

typedef struct qrt_qwen36_baseline_check_t {
    unsigned int model_specific;
    unsigned int batch_size;
    unsigned int descriptor_table_driven;
    unsigned int prefill_entrypoint_planned;
    unsigned int decode_one_entrypoint_planned;
    unsigned int layer_count;
    unsigned int linear_attention_layer_count;
    unsigned int full_attention_layer_count;
    unsigned int first_full_attention_layer;
    unsigned int last_full_attention_layer;
    unsigned int layer_descriptor_count_matches;
    unsigned int layer_schedule_pass;
    unsigned int tensor_name_builder_pass;
    uint64_t descriptor_fnv1a64;
    char sample_token_embedding[QRT_QWEN36_TENSOR_NAME_CAPACITY];
    char sample_linear_qkv[QRT_QWEN36_TENSOR_NAME_CAPACITY];
    char sample_full_q[QRT_QWEN36_TENSOR_NAME_CAPACITY];
    char sample_moe_router[QRT_QWEN36_TENSOR_NAME_CAPACITY];
    char sample_final_norm[QRT_QWEN36_TENSOR_NAME_CAPACITY];
    char sample_lm_head[QRT_QWEN36_TENSOR_NAME_CAPACITY];
    int correctness_pass;
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_qwen36_baseline_check_t;

typedef struct qrt_qwen36_baseline_execution_result_t {
    qrt_qwen36_execution_mode_t mode;
    size_t input_token_count;
    size_t output_token_count;
    unsigned int baseline_plan_attached;
    unsigned int descriptor_table_driven;
    unsigned int module_dispatch_started;
    unsigned int layer_count;
    unsigned int linear_attention_layer_count;
    unsigned int full_attention_layer_count;
    unsigned int first_missing_layer;
    uint64_t descriptor_fnv1a64;
    unsigned int token_embedding_materialized;
    size_t input_embeddings_materialized_count;
    uint64_t input_embeddings_fnv1a64;
    uint64_t last_token_embedding_fnv1a64;
    uint64_t token_embedding_bytes_read;
    uint64_t token_embedding_elapsed_ns;
    unsigned int layer0_input_norm_applied;
    size_t layer0_input_norm_materialized_count;
    uint64_t layer0_input_norm_fnv1a64;
    uint64_t last_layer0_input_norm_fnv1a64;
    uint64_t layer0_input_norm_weight_fnv1a64;
    uint64_t layer0_input_norm_weight_bytes_read;
    uint64_t layer0_input_norm_elapsed_ns;
    unsigned int layer0_qkv_projection_applied;
    size_t layer0_qkv_projection_materialized_count;
    uint64_t layer0_qkv_projection_fnv1a64;
    uint64_t last_layer0_qkv_projection_fnv1a64;
    uint64_t layer0_qkv_projection_weight_fnv1a64;
    uint64_t layer0_qkv_projection_weight_bytes_read;
    uint64_t layer0_qkv_projection_elapsed_ns;
    unsigned int layer0_zab_projection_applied;
    size_t layer0_zab_projection_materialized_count;
    uint64_t layer0_zab_projection_fnv1a64;
    uint64_t last_layer0_zab_projection_fnv1a64;
    uint64_t layer0_z_projection_weight_fnv1a64;
    uint64_t layer0_a_projection_weight_fnv1a64;
    uint64_t layer0_b_projection_weight_fnv1a64;
    uint64_t layer0_zab_projection_weight_bytes_read;
    uint64_t layer0_zab_projection_elapsed_ns;
    unsigned int layer0_conv_qkv_applied;
    size_t layer0_conv_qkv_materialized_count;
    uint64_t layer0_conv_qkv_fnv1a64;
    uint64_t last_layer0_conv_qkv_fnv1a64;
    uint64_t layer0_conv_qkv_weight_fnv1a64;
    uint64_t layer0_conv_qkv_weight_bytes_read;
    uint64_t layer0_conv_qkv_elapsed_ns;
    unsigned int layer0_postconv_qkv_applied;
    size_t layer0_postconv_qkv_materialized_count;
    uint64_t layer0_postconv_q_scaled_fnv1a64;
    uint64_t last_layer0_postconv_q_scaled_fnv1a64;
    uint64_t layer0_postconv_k_norm_fnv1a64;
    uint64_t last_layer0_postconv_k_norm_fnv1a64;
    uint64_t layer0_postconv_value_fnv1a64;
    uint64_t last_layer0_postconv_value_fnv1a64;
    uint64_t layer0_postconv_qkv_elapsed_ns;
    unsigned int layer0_gate_rows_applied;
    size_t layer0_gate_rows_materialized_count;
    uint64_t layer0_gate_g_fnv1a64;
    uint64_t last_layer0_gate_g_fnv1a64;
    uint64_t layer0_gate_beta_fnv1a64;
    uint64_t last_layer0_gate_beta_fnv1a64;
    uint64_t layer0_a_log_fnv1a64;
    uint64_t layer0_dt_bias_fnv1a64;
    uint64_t layer0_gate_weight_bytes_read;
    uint64_t layer0_gate_rows_elapsed_ns;
    unsigned int layer0_core_rows_applied;
    size_t layer0_core_rows_materialized_count;
    uint64_t layer0_core_rows_fnv1a64;
    uint64_t last_layer0_core_rows_fnv1a64;
    uint64_t layer0_core_final_state_fnv1a64;
    uint64_t layer0_core_rows_elapsed_ns;
    unsigned int layer0_gated_rmsnorm_applied;
    size_t layer0_gated_rmsnorm_materialized_count;
    uint64_t layer0_gated_rmsnorm_fnv1a64;
    uint64_t last_layer0_gated_rmsnorm_fnv1a64;
    uint64_t layer0_linear_norm_weight_fnv1a64;
    uint64_t layer0_linear_norm_weight_bytes_read;
    uint64_t layer0_gated_rmsnorm_elapsed_ns;
    unsigned int layer0_out_projection_applied;
    size_t layer0_out_projection_materialized_count;
    uint64_t layer0_out_projection_fnv1a64;
    uint64_t last_layer0_out_projection_fnv1a64;
    uint64_t layer0_out_projection_weight_fnv1a64;
    uint64_t layer0_out_projection_weight_bytes_read;
    uint64_t layer0_out_projection_elapsed_ns;
    unsigned int layer0_residual_hidden_applied;
    size_t layer0_residual_hidden_materialized_count;
    uint64_t layer0_residual_hidden_fnv1a64;
    uint64_t last_layer0_residual_hidden_fnv1a64;
    uint64_t layer0_residual_hidden_elapsed_ns;
    unsigned int layer0_post_attention_rmsnorm_applied;
    size_t layer0_post_attention_rmsnorm_materialized_count;
    uint64_t layer0_post_attention_rmsnorm_fnv1a64;
    uint64_t last_layer0_post_attention_rmsnorm_fnv1a64;
    uint64_t layer0_post_attention_norm_weight_fnv1a64;
    uint64_t layer0_post_attention_norm_weight_bytes_read;
    uint64_t layer0_post_attention_rmsnorm_elapsed_ns;
    unsigned int layer0_moe_router_applied;
    size_t layer0_moe_router_materialized_count;
    uint64_t layer0_moe_router_logits_fnv1a64;
    uint64_t last_layer0_moe_router_logits_fnv1a64;
    uint64_t layer0_moe_router_topk_ids_fnv1a64;
    uint64_t last_layer0_moe_router_topk_ids_fnv1a64;
    uint64_t layer0_moe_router_topk_weights_fnv1a64;
    uint64_t last_layer0_moe_router_topk_weights_fnv1a64;
    uint64_t layer0_moe_router_weight_fnv1a64;
    uint64_t layer0_moe_router_weight_bytes_read;
    uint64_t layer0_moe_router_elapsed_ns;
    unsigned int layer0_moe_expert_applied;
    size_t layer0_moe_expert_materialized_count;
    size_t layer0_moe_expert_selected_count;
    uint64_t layer0_moe_expert_routed_fnv1a64;
    uint64_t last_layer0_moe_expert_routed_fnv1a64;
    uint64_t layer0_moe_expert_selected_ids_fnv1a64;
    uint64_t layer0_moe_expert_gate_up_weight_fnv1a64;
    uint64_t layer0_moe_expert_down_weight_fnv1a64;
    uint64_t layer0_moe_expert_weight_bytes_read;
    uint64_t layer0_moe_expert_elapsed_ns;
    unsigned int layer0_moe_shared_expert_applied;
    size_t layer0_moe_shared_expert_materialized_count;
    uint64_t layer0_moe_shared_expert_fnv1a64;
    uint64_t last_layer0_moe_shared_expert_fnv1a64;
    uint64_t layer0_moe_combined_fnv1a64;
    uint64_t last_layer0_moe_combined_fnv1a64;
    uint64_t layer0_moe_shared_gate_fnv1a64;
    uint64_t last_layer0_moe_shared_gate_fnv1a64;
    uint64_t layer0_moe_shared_gate_weight_fnv1a64;
    uint64_t layer0_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t layer0_moe_shared_up_proj_weight_fnv1a64;
    uint64_t layer0_moe_shared_down_weight_fnv1a64;
    uint64_t layer0_moe_shared_expert_weight_bytes_read;
    uint64_t layer0_moe_shared_expert_elapsed_ns;
    unsigned int layer0_output_residual_applied;
    size_t layer0_output_residual_materialized_count;
    uint64_t layer0_output_residual_fnv1a64;
    uint64_t last_layer0_output_residual_fnv1a64;
    uint64_t layer0_output_residual_elapsed_ns;
    unsigned int layer1_input_norm_applied;
    size_t layer1_input_norm_materialized_count;
    uint64_t layer1_input_norm_fnv1a64;
    uint64_t last_layer1_input_norm_fnv1a64;
    uint64_t layer1_input_norm_weight_fnv1a64;
    uint64_t layer1_input_norm_weight_bytes_read;
    uint64_t layer1_input_norm_elapsed_ns;
    unsigned int layer1_qkv_projection_applied;
    size_t layer1_qkv_projection_materialized_count;
    uint64_t layer1_qkv_projection_fnv1a64;
    uint64_t last_layer1_qkv_projection_fnv1a64;
    uint64_t layer1_qkv_projection_weight_fnv1a64;
    uint64_t layer1_qkv_projection_weight_bytes_read;
    uint64_t layer1_qkv_projection_elapsed_ns;
    unsigned int layer1_zab_projection_applied;
    size_t layer1_zab_projection_materialized_count;
    uint64_t layer1_zab_projection_fnv1a64;
    uint64_t last_layer1_zab_projection_fnv1a64;
    uint64_t layer1_z_projection_weight_fnv1a64;
    uint64_t layer1_a_projection_weight_fnv1a64;
    uint64_t layer1_b_projection_weight_fnv1a64;
    uint64_t layer1_zab_projection_weight_bytes_read;
    uint64_t layer1_zab_projection_elapsed_ns;
    unsigned int layer1_conv_qkv_applied;
    size_t layer1_conv_qkv_materialized_count;
    uint64_t layer1_conv_qkv_fnv1a64;
    uint64_t last_layer1_conv_qkv_fnv1a64;
    uint64_t layer1_conv_qkv_weight_fnv1a64;
    uint64_t layer1_conv_qkv_weight_bytes_read;
    uint64_t layer1_conv_qkv_elapsed_ns;
    unsigned int layer1_postconv_qkv_applied;
    size_t layer1_postconv_qkv_materialized_count;
    uint64_t layer1_postconv_q_scaled_fnv1a64;
    uint64_t last_layer1_postconv_q_scaled_fnv1a64;
    uint64_t layer1_postconv_k_norm_fnv1a64;
    uint64_t last_layer1_postconv_k_norm_fnv1a64;
    uint64_t layer1_postconv_value_fnv1a64;
    uint64_t last_layer1_postconv_value_fnv1a64;
    uint64_t layer1_postconv_qkv_elapsed_ns;
    unsigned int layer1_gate_rows_applied;
    size_t layer1_gate_rows_materialized_count;
    uint64_t layer1_gate_g_fnv1a64;
    uint64_t last_layer1_gate_g_fnv1a64;
    uint64_t layer1_gate_beta_fnv1a64;
    uint64_t last_layer1_gate_beta_fnv1a64;
    uint64_t layer1_a_log_fnv1a64;
    uint64_t layer1_dt_bias_fnv1a64;
    uint64_t layer1_gate_weight_bytes_read;
    uint64_t layer1_gate_rows_elapsed_ns;
    unsigned int layer1_core_rows_applied;
    size_t layer1_core_rows_materialized_count;
    uint64_t layer1_core_rows_fnv1a64;
    uint64_t last_layer1_core_rows_fnv1a64;
    uint64_t layer1_core_final_state_fnv1a64;
    uint64_t layer1_core_rows_elapsed_ns;
    unsigned int layer1_gated_rmsnorm_applied;
    size_t layer1_gated_rmsnorm_materialized_count;
    uint64_t layer1_gated_rmsnorm_fnv1a64;
    uint64_t last_layer1_gated_rmsnorm_fnv1a64;
    uint64_t layer1_linear_norm_weight_fnv1a64;
    uint64_t layer1_linear_norm_weight_bytes_read;
    uint64_t layer1_gated_rmsnorm_elapsed_ns;
    unsigned int layer1_out_projection_applied;
    size_t layer1_out_projection_materialized_count;
    uint64_t layer1_out_projection_fnv1a64;
    uint64_t last_layer1_out_projection_fnv1a64;
    uint64_t layer1_out_projection_weight_fnv1a64;
    uint64_t layer1_out_projection_weight_bytes_read;
    uint64_t layer1_out_projection_elapsed_ns;
    unsigned int layer1_residual_hidden_applied;
    size_t layer1_residual_hidden_materialized_count;
    uint64_t layer1_residual_hidden_fnv1a64;
    uint64_t last_layer1_residual_hidden_fnv1a64;
    uint64_t layer1_residual_hidden_elapsed_ns;
    unsigned int layer1_post_attention_rmsnorm_applied;
    size_t layer1_post_attention_rmsnorm_materialized_count;
    uint64_t layer1_post_attention_rmsnorm_fnv1a64;
    uint64_t last_layer1_post_attention_rmsnorm_fnv1a64;
    uint64_t layer1_post_attention_norm_weight_fnv1a64;
    uint64_t layer1_post_attention_norm_weight_bytes_read;
    uint64_t layer1_post_attention_rmsnorm_elapsed_ns;
    unsigned int layer1_moe_router_applied;
    size_t layer1_moe_router_materialized_count;
    uint64_t layer1_moe_router_logits_fnv1a64;
    uint64_t last_layer1_moe_router_logits_fnv1a64;
    uint64_t layer1_moe_router_topk_ids_fnv1a64;
    uint64_t last_layer1_moe_router_topk_ids_fnv1a64;
    uint64_t layer1_moe_router_topk_weights_fnv1a64;
    uint64_t last_layer1_moe_router_topk_weights_fnv1a64;
    uint64_t layer1_moe_router_weight_fnv1a64;
    uint64_t layer1_moe_router_weight_bytes_read;
    uint64_t layer1_moe_router_elapsed_ns;
    unsigned int layer1_moe_expert_applied;
    size_t layer1_moe_expert_materialized_count;
    size_t layer1_moe_expert_selected_count;
    uint64_t layer1_moe_expert_routed_fnv1a64;
    uint64_t last_layer1_moe_expert_routed_fnv1a64;
    uint64_t layer1_moe_expert_selected_ids_fnv1a64;
    uint64_t layer1_moe_expert_gate_up_weight_fnv1a64;
    uint64_t layer1_moe_expert_down_weight_fnv1a64;
    uint64_t layer1_moe_expert_weight_bytes_read;
    uint64_t layer1_moe_expert_elapsed_ns;
    unsigned int layer1_moe_shared_expert_applied;
    size_t layer1_moe_shared_expert_materialized_count;
    uint64_t layer1_moe_shared_expert_fnv1a64;
    uint64_t last_layer1_moe_shared_expert_fnv1a64;
    uint64_t layer1_moe_combined_fnv1a64;
    uint64_t last_layer1_moe_combined_fnv1a64;
    uint64_t layer1_moe_shared_gate_fnv1a64;
    uint64_t last_layer1_moe_shared_gate_fnv1a64;
    uint64_t layer1_moe_shared_gate_weight_fnv1a64;
    uint64_t layer1_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t layer1_moe_shared_up_proj_weight_fnv1a64;
    uint64_t layer1_moe_shared_down_weight_fnv1a64;
    uint64_t layer1_moe_shared_expert_weight_bytes_read;
    uint64_t layer1_moe_shared_expert_elapsed_ns;
    unsigned int layer1_output_residual_applied;
    size_t layer1_output_residual_materialized_count;
    uint64_t layer1_output_residual_fnv1a64;
    uint64_t last_layer1_output_residual_fnv1a64;
    uint64_t layer1_output_residual_elapsed_ns;
    unsigned int layer2_input_norm_applied;
    size_t layer2_input_norm_materialized_count;
    uint64_t layer2_input_norm_fnv1a64;
    uint64_t last_layer2_input_norm_fnv1a64;
    uint64_t layer2_input_norm_weight_fnv1a64;
    uint64_t layer2_input_norm_weight_bytes_read;
    uint64_t layer2_input_norm_elapsed_ns;
    unsigned int layer2_qkv_projection_applied;
    size_t layer2_qkv_projection_materialized_count;
    uint64_t layer2_qkv_projection_fnv1a64;
    uint64_t last_layer2_qkv_projection_fnv1a64;
    uint64_t layer2_qkv_projection_weight_fnv1a64;
    uint64_t layer2_qkv_projection_weight_bytes_read;
    uint64_t layer2_qkv_projection_elapsed_ns;
    unsigned int layer2_zab_projection_applied;
    size_t layer2_zab_projection_materialized_count;
    uint64_t layer2_zab_projection_fnv1a64;
    uint64_t last_layer2_zab_projection_fnv1a64;
    uint64_t layer2_z_projection_weight_fnv1a64;
    uint64_t layer2_a_projection_weight_fnv1a64;
    uint64_t layer2_b_projection_weight_fnv1a64;
    uint64_t layer2_zab_projection_weight_bytes_read;
    uint64_t layer2_zab_projection_elapsed_ns;
    unsigned int layer2_conv_qkv_applied;
    size_t layer2_conv_qkv_materialized_count;
    uint64_t layer2_conv_qkv_fnv1a64;
    uint64_t last_layer2_conv_qkv_fnv1a64;
    uint64_t layer2_conv_qkv_weight_fnv1a64;
    uint64_t layer2_conv_qkv_weight_bytes_read;
    uint64_t layer2_conv_qkv_elapsed_ns;
    unsigned int layer2_postconv_qkv_applied;
    size_t layer2_postconv_qkv_materialized_count;
    uint64_t layer2_postconv_q_scaled_fnv1a64;
    uint64_t last_layer2_postconv_q_scaled_fnv1a64;
    uint64_t layer2_postconv_k_norm_fnv1a64;
    uint64_t last_layer2_postconv_k_norm_fnv1a64;
    uint64_t layer2_postconv_value_fnv1a64;
    uint64_t last_layer2_postconv_value_fnv1a64;
    uint64_t layer2_postconv_qkv_elapsed_ns;
    unsigned int layer2_gate_rows_applied;
    size_t layer2_gate_rows_materialized_count;
    uint64_t layer2_gate_g_fnv1a64;
    uint64_t last_layer2_gate_g_fnv1a64;
    uint64_t layer2_gate_beta_fnv1a64;
    uint64_t last_layer2_gate_beta_fnv1a64;
    uint64_t layer2_a_log_fnv1a64;
    uint64_t layer2_dt_bias_fnv1a64;
    uint64_t layer2_gate_weight_bytes_read;
    uint64_t layer2_gate_rows_elapsed_ns;
    unsigned int layer2_core_rows_applied;
    size_t layer2_core_rows_materialized_count;
    uint64_t layer2_core_rows_fnv1a64;
    uint64_t last_layer2_core_rows_fnv1a64;
    uint64_t layer2_core_final_state_fnv1a64;
    uint64_t layer2_core_rows_elapsed_ns;
    unsigned int layer2_gated_rmsnorm_applied;
    size_t layer2_gated_rmsnorm_materialized_count;
    uint64_t layer2_gated_rmsnorm_fnv1a64;
    uint64_t last_layer2_gated_rmsnorm_fnv1a64;
    uint64_t layer2_linear_norm_weight_fnv1a64;
    uint64_t layer2_linear_norm_weight_bytes_read;
    uint64_t layer2_gated_rmsnorm_elapsed_ns;
    unsigned int layer2_out_projection_applied;
    size_t layer2_out_projection_materialized_count;
    uint64_t layer2_out_projection_fnv1a64;
    uint64_t last_layer2_out_projection_fnv1a64;
    uint64_t layer2_out_projection_weight_fnv1a64;
    uint64_t layer2_out_projection_weight_bytes_read;
    uint64_t layer2_out_projection_elapsed_ns;
    unsigned int layer2_residual_hidden_applied;
    size_t layer2_residual_hidden_materialized_count;
    uint64_t layer2_residual_hidden_fnv1a64;
    uint64_t last_layer2_residual_hidden_fnv1a64;
    uint64_t layer2_residual_hidden_elapsed_ns;
    char first_missing_module[64];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    int inference_success_claimed;
} qrt_qwen36_baseline_execution_result_t;

typedef struct qrt_qwen36_baseline_surface_check_t {
    unsigned int baseline_plan_attached;
    unsigned int layer_count;
    unsigned int linear_attention_layer_count;
    unsigned int full_attention_layer_count;
    unsigned int prefill_entrypoint_planned;
    unsigned int decode_one_entrypoint_planned;
    uint64_t descriptor_fnv1a64;
    size_t prefill_call_count;
    size_t decode_one_call_count;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    int correctness_pass;
} qrt_qwen36_baseline_surface_check_t;

#define QRT_QWEN36_LINEAR_ATTENTION_TABLE_LAYER_CAPACITY \
    QRT_QWEN36_LINEAR_ATTENTION_LAYERS
#define QRT_QWEN36_LINEAR_ATTENTION_TABLE_MODULE_COUNT 10u

typedef struct qrt_qwen36_linear_attention_table_layer_result_t {
    unsigned int layer_index;
    unsigned int linear_ordinal;
    unsigned int descriptor_alignment_pass;
    unsigned int tensor_name_alignment_pass;
    unsigned int tensors_materialized_count;
    uint64_t tensor_name_fnv1a64;
    uint64_t weight_fnv1a64;
    uint64_t weight_bytes_read;
    uint64_t input_norm_fnv1a64;
    uint64_t qkv_projection_fnv1a64;
    uint64_t zab_projection_fnv1a64;
    uint64_t conv_qkv_fnv1a64;
    uint64_t postconv_q_scaled_fnv1a64;
    uint64_t postconv_k_norm_fnv1a64;
    uint64_t postconv_value_fnv1a64;
    uint64_t gate_g_fnv1a64;
    uint64_t gate_beta_fnv1a64;
    uint64_t core_rows_fnv1a64;
    uint64_t core_final_state_fnv1a64;
    uint64_t gated_rmsnorm_fnv1a64;
    uint64_t out_projection_fnv1a64;
    uint64_t residual_hidden_fnv1a64;
    uint64_t elapsed_ns;
    char first_failure_stage[64];
    int correctness_pass;
} qrt_qwen36_linear_attention_table_layer_result_t;

typedef struct qrt_qwen36_linear_attention_table_check_t {
    unsigned int baseline_plan_attached;
    unsigned int layer_count;
    unsigned int linear_attention_layer_count;
    unsigned int checked_linear_layer_count;
    unsigned int skipped_full_attention_layer_count;
    unsigned int verified_module_count;
    uint64_t descriptor_fnv1a64;
    uint64_t tensor_name_table_fnv1a64;
    uint64_t digest_fnv1a64;
    uint64_t total_weight_bytes_read;
    uint64_t total_elapsed_ns;
    qrt_qwen36_linear_attention_table_layer_result_t
        layers[QRT_QWEN36_LINEAR_ATTENTION_TABLE_LAYER_CAPACITY];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    int correctness_pass;
    int inference_success_claimed;
} qrt_qwen36_linear_attention_table_check_t;

#define QRT_QWEN36_FULL_ATTENTION_TABLE_LAYER_CAPACITY \
    QRT_QWEN36_FULL_ATTENTION_LAYERS
#define QRT_QWEN36_FULL_ATTENTION_TABLE_MODULE_COUNT 7u

typedef struct qrt_qwen36_full_attention_table_layer_result_t {
    unsigned int layer_index;
    unsigned int full_ordinal;
    unsigned int descriptor_alignment_pass;
    unsigned int tensor_name_alignment_pass;
    unsigned int tensors_materialized_count;
    uint64_t tensor_name_fnv1a64;
    uint64_t weight_fnv1a64;
    uint64_t weight_bytes_read;
    uint64_t input_norm_weight_fnv1a64;
    uint64_t q_weight_fnv1a64;
    uint64_t k_weight_fnv1a64;
    uint64_t v_weight_fnv1a64;
    uint64_t o_weight_fnv1a64;
    uint64_t q_norm_weight_fnv1a64;
    uint64_t k_norm_weight_fnv1a64;
    uint64_t elapsed_ns;
    char first_failure_stage[64];
    int correctness_pass;
} qrt_qwen36_full_attention_table_layer_result_t;

typedef struct qrt_qwen36_full_attention_table_check_t {
    unsigned int baseline_plan_attached;
    unsigned int layer_count;
    unsigned int full_attention_layer_count;
    unsigned int checked_full_attention_layer_count;
    unsigned int skipped_linear_attention_layer_count;
    unsigned int verified_module_count;
    uint64_t descriptor_fnv1a64;
    uint64_t tensor_name_table_fnv1a64;
    uint64_t digest_fnv1a64;
    uint64_t total_weight_bytes_read;
    uint64_t total_elapsed_ns;
    qrt_qwen36_full_attention_table_layer_result_t
        layers[QRT_QWEN36_FULL_ATTENTION_TABLE_LAYER_CAPACITY];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    int correctness_pass;
    int inference_success_claimed;
} qrt_qwen36_full_attention_table_check_t;

#define QRT_QWEN36_FULL_ATTENTION_COMPUTE_LAYER_CAPACITY \
    QRT_QWEN36_FULL_ATTENTION_LAYERS
#define QRT_QWEN36_FULL_ATTENTION_COMPUTE_MODULE_COUNT 6u

typedef struct qrt_qwen36_full_attention_compute_layer_result_t {
    unsigned int layer_index;
    unsigned int full_ordinal;
    unsigned int descriptor_alignment_pass;
    unsigned int tensor_name_alignment_pass;
    unsigned int tensors_materialized_count;
    unsigned int compute_module_count;
    uint64_t tensor_name_fnv1a64;
    uint64_t weight_fnv1a64;
    uint64_t weight_bytes_read;
    uint64_t input_norm_fnv1a64;
    uint64_t qkv_projection_fnv1a64;
    uint64_t qk_norm_fnv1a64;
    uint64_t rope_attention_fnv1a64;
    uint64_t out_projection_fnv1a64;
    uint64_t residual_hidden_fnv1a64;
    uint64_t elapsed_ns;
    char first_failure_stage[64];
    int correctness_pass;
} qrt_qwen36_full_attention_compute_layer_result_t;

typedef struct qrt_qwen36_full_attention_compute_check_t {
    unsigned int baseline_plan_attached;
    unsigned int layer_count;
    unsigned int full_attention_layer_count;
    unsigned int checked_full_attention_layer_count;
    unsigned int skipped_linear_attention_layer_count;
    unsigned int verified_tensor_slot_count;
    unsigned int verified_compute_module_count;
    uint64_t descriptor_fnv1a64;
    uint64_t tensor_name_table_fnv1a64;
    uint64_t digest_fnv1a64;
    uint64_t total_weight_bytes_read;
    uint64_t total_elapsed_ns;
    qrt_qwen36_full_attention_compute_layer_result_t
        layers[QRT_QWEN36_FULL_ATTENTION_COMPUTE_LAYER_CAPACITY];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    int correctness_pass;
    int inference_success_claimed;
} qrt_qwen36_full_attention_compute_check_t;

#define QRT_QWEN36_MOE_BLOCK_TABLE_LAYER_CAPACITY QRT_QWEN36_LAYER_COUNT
#define QRT_QWEN36_MOE_BLOCK_TABLE_TENSOR_SLOT_COUNT 8u
#define QRT_QWEN36_MOE_BLOCK_TABLE_EXPERT_SAMPLE_COUNT 3u

typedef struct qrt_qwen36_moe_block_table_layer_result_t {
    unsigned int layer_index;
    unsigned int descriptor_alignment_pass;
    unsigned int tensor_name_alignment_pass;
    unsigned int tensor_location_alignment_pass;
    unsigned int tensors_located_count;
    unsigned int tensors_materialized_count;
    unsigned int expert_sample_count;
    unsigned int expert_sample_ids[QRT_QWEN36_MOE_BLOCK_TABLE_EXPERT_SAMPLE_COUNT];
    uint64_t tensor_name_fnv1a64;
    uint64_t tensor_location_fnv1a64;
    uint64_t weight_fnv1a64;
    uint64_t weight_bytes_read;
    uint64_t post_attention_norm_weight_fnv1a64;
    uint64_t router_weight_fnv1a64;
    uint64_t expert_gate_up_sample_fnv1a64;
    uint64_t expert_down_sample_fnv1a64;
    uint64_t shared_gate_weight_fnv1a64;
    uint64_t shared_gate_proj_weight_fnv1a64;
    uint64_t shared_up_proj_weight_fnv1a64;
    uint64_t shared_down_weight_fnv1a64;
    uint64_t elapsed_ns;
    char first_failure_stage[64];
    int correctness_pass;
} qrt_qwen36_moe_block_table_layer_result_t;

typedef struct qrt_qwen36_moe_block_table_check_t {
    unsigned int baseline_plan_attached;
    unsigned int layer_count;
    unsigned int checked_layer_count;
    unsigned int verified_tensor_slot_count;
    unsigned int verified_expert_sample_count;
    uint64_t descriptor_fnv1a64;
    uint64_t tensor_name_table_fnv1a64;
    uint64_t tensor_location_table_fnv1a64;
    uint64_t digest_fnv1a64;
    uint64_t total_weight_bytes_read;
    uint64_t total_elapsed_ns;
    qrt_qwen36_moe_block_table_layer_result_t
        layers[QRT_QWEN36_MOE_BLOCK_TABLE_LAYER_CAPACITY];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    int correctness_pass;
    int inference_success_claimed;
} qrt_qwen36_moe_block_table_check_t;

#define QRT_QWEN36_MOE_BLOCK_COMPUTE_LAYER_CAPACITY QRT_QWEN36_LAYER_COUNT
#define QRT_QWEN36_MOE_BLOCK_COMPUTE_TENSOR_SLOT_COUNT \
    QRT_QWEN36_MOE_BLOCK_TABLE_TENSOR_SLOT_COUNT
#define QRT_QWEN36_MOE_BLOCK_COMPUTE_MODULE_COUNT 5u

typedef struct qrt_qwen36_moe_block_compute_layer_result_t {
    unsigned int layer_index;
    unsigned int descriptor_alignment_pass;
    unsigned int tensor_name_alignment_pass;
    unsigned int tensors_materialized_count;
    unsigned int compute_module_count;
    unsigned int selected_expert_count;
    unsigned int selected_expert_ids[QRT_QWEN36_EXPERTS_PER_TOKEN];
    uint64_t tensor_name_fnv1a64;
    uint64_t weight_fnv1a64;
    uint64_t weight_bytes_read;
    uint64_t post_attention_norm_weight_fnv1a64;
    uint64_t router_weight_fnv1a64;
    uint64_t selected_expert_ids_fnv1a64;
    uint64_t selected_expert_gate_up_weight_fnv1a64;
    uint64_t selected_expert_down_weight_fnv1a64;
    uint64_t shared_gate_weight_fnv1a64;
    uint64_t shared_gate_proj_weight_fnv1a64;
    uint64_t shared_up_proj_weight_fnv1a64;
    uint64_t shared_down_weight_fnv1a64;
    uint64_t post_attention_rmsnorm_fnv1a64;
    uint64_t router_logits_fnv1a64;
    uint64_t router_topk_ids_fnv1a64;
    uint64_t router_topk_weights_fnv1a64;
    uint64_t routed_expert_output_fnv1a64;
    uint64_t shared_expert_output_fnv1a64;
    uint64_t shared_gate_logit_fnv1a64;
    uint64_t combined_moe_fnv1a64;
    uint64_t output_residual_fnv1a64;
    uint64_t elapsed_ns;
    char first_failure_stage[64];
    int correctness_pass;
} qrt_qwen36_moe_block_compute_layer_result_t;

typedef struct qrt_qwen36_moe_block_compute_check_t {
    unsigned int baseline_plan_attached;
    unsigned int layer_count;
    unsigned int checked_layer_count;
    unsigned int verified_tensor_slot_count;
    unsigned int verified_compute_module_count;
    uint64_t descriptor_fnv1a64;
    uint64_t tensor_name_table_fnv1a64;
    uint64_t digest_fnv1a64;
    uint64_t total_weight_bytes_read;
    uint64_t total_elapsed_ns;
    qrt_qwen36_moe_block_compute_layer_result_t
        layers[QRT_QWEN36_MOE_BLOCK_COMPUTE_LAYER_CAPACITY];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    int correctness_pass;
    int inference_success_claimed;
} qrt_qwen36_moe_block_compute_check_t;

#define QRT_QWEN36_OUTPUT_HEAD_TENSOR_SLOT_COUNT 2u
#define QRT_QWEN36_OUTPUT_HEAD_COMPUTE_MODULE_COUNT 3u
#define QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK 5u
#define QRT_QWEN36_PRODUCT_Q8192_LAYER0_SOURCE_STAGE_COUNT 26u
#define QRT_QWEN36_PRODUCT_Q8192_LAYER1_STAGE_COUNT 28u
#define QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY QRT_QWEN36_LAYER_COUNT
#define QRT_QWEN36_TRACE_SAMPLE_COUNT 8u
#define QRT_QWEN36_LAYER0_LINEAR_QKV_TRACE_VALUE_COUNT 8192u
#define QRT_QWEN36_LAYER0_LINEAR_Z_TRACE_VALUE_COUNT 4096u
#define QRT_QWEN36_LAYER0_LINEAR_A_TRACE_VALUE_COUNT 32u
#define QRT_QWEN36_LAYER0_LINEAR_B_TRACE_VALUE_COUNT 32u
#define QRT_QWEN36_LAYER0_LINEAR_CONV_TRACE_VALUE_COUNT 8192u
#define QRT_QWEN36_LAYER0_LINEAR_POSTCONV_Q_TRACE_VALUE_COUNT 2048u
#define QRT_QWEN36_LAYER0_LINEAR_POSTCONV_K_TRACE_VALUE_COUNT 2048u
#define QRT_QWEN36_LAYER0_LINEAR_POSTCONV_VALUE_TRACE_VALUE_COUNT 4096u
#define QRT_QWEN36_LAYER0_LINEAR_GATE_TRACE_VALUE_COUNT 32u
#define QRT_QWEN36_LAYER0_LINEAR_CORE_TRACE_VALUE_COUNT 4096u
#define QRT_QWEN36_LAYER0_LINEAR_OUT_INPUT_TRACE_VALUE_COUNT 4096u
#define QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT 4u
#define QRT_QWEN36_LAYER0_LINEAR_QKV_BLOCK_VARIANT_COUNT 10u
#define QRT_QWEN36_LAYER0_LINEAR_QKV_BLOCK_VARIANT_VALUE_COUNT \
    (QRT_QWEN36_LAYER0_LINEAR_QKV_BLOCK_VARIANT_COUNT * \
     QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT)
#define QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_TRACE_TOKEN_COUNT 17u
#define QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_TRACE_ROW_COUNT \
    QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT
#define QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_TRACE_VALUE_COUNT \
    (QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_TRACE_ROW_COUNT * \
     QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_TRACE_TOKEN_COUNT)
#define QRT_QWEN36_LAYER0_PREFILL_QKV_MISMATCH_TRACE_ROW_COUNT 88u
#define QRT_QWEN36_LAYER0_PREFILL_QKV_MISMATCH_TRACE_VALUE_COUNT \
    (QRT_QWEN36_LAYER0_PREFILL_QKV_MISMATCH_TRACE_ROW_COUNT * \
     QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_TRACE_TOKEN_COUNT)
#define QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_ROW_HASH_COUNT \
    QRT_QWEN36_LAYER0_LINEAR_QKV_TRACE_VALUE_COUNT
#define QRT_QWEN36_LAYER0_PREFILL_CONV_SILU_PROVIDER_ROW_HASH_COUNT \
    QRT_QWEN36_LAYER0_LINEAR_QKV_TRACE_VALUE_COUNT
#define QRT_QWEN36_LAYER0_PREFILL_CONV_SILU_MISMATCH_TRACE_ROW_COUNT 159u
#define QRT_QWEN36_LAYER0_PREFILL_CONV_SILU_MISMATCH_TRACE_VALUE_COUNT \
    (QRT_QWEN36_LAYER0_PREFILL_CONV_SILU_MISMATCH_TRACE_ROW_COUNT * \
     QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_TRACE_TOKEN_COUNT)
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_ROW_HASH_COUNT 32u
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_TRACE_TOKEN_COUNT \
    QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_TRACE_TOKEN_COUNT
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_VECTOR_DIM 128u
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_K_EXPANDED_TRACE_ROW_COUNT 32u
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_VALUE_TRACE_ROW_COUNT 21u
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_G_TRACE_ROW_COUNT 31u
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_BETA_TRACE_ROW_COUNT 32u
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_K_EXPANDED_TRACE_VALUE_COUNT \
    (QRT_QWEN36_LAYER0_PREFILL_CHUNK_K_EXPANDED_TRACE_ROW_COUNT * \
     QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_TRACE_TOKEN_COUNT * \
     QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_VECTOR_DIM)
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_VALUE_TRACE_VALUE_COUNT \
    (QRT_QWEN36_LAYER0_PREFILL_CHUNK_VALUE_TRACE_ROW_COUNT * \
     QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_TRACE_TOKEN_COUNT * \
     QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_VECTOR_DIM)
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_G_TRACE_VALUE_COUNT \
    (QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_G_TRACE_ROW_COUNT * \
     QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_TRACE_TOKEN_COUNT)
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_BETA_TRACE_VALUE_COUNT \
    (QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_BETA_TRACE_ROW_COUNT * \
     QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_TRACE_TOKEN_COUNT)
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_INPUT_TRACE_VALUE_COUNT \
    QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_G_TRACE_VALUE_COUNT
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_STATE_ROW_HASH_COUNT \
    QRT_QWEN36_LAYER0_LINEAR_CORE_TRACE_VALUE_COUNT
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_STATE_TRACE_ROW_COUNT 32u
#define QRT_QWEN36_LAYER0_PREFILL_CHUNK_STATE_TRACE_VALUE_COUNT \
    (QRT_QWEN36_LAYER0_PREFILL_CHUNK_STATE_TRACE_ROW_COUNT * \
     QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_VECTOR_DIM)
#define QRT_QWEN36_LAYER0_LINEAR_CONV_SOURCE_TRACE_ROW_COUNT 16u
#define QRT_QWEN36_LAYER0_LINEAR_CONV_SOURCE_TRACE_TAP_COUNT 4u
#define QRT_QWEN36_LAYER0_LINEAR_CONV_SOURCE_TRACE_VALUE_COUNT \
    (QRT_QWEN36_LAYER0_LINEAR_CONV_SOURCE_TRACE_ROW_COUNT * \
     QRT_QWEN36_LAYER0_LINEAR_CONV_SOURCE_TRACE_TAP_COUNT)

typedef struct qrt_qwen36_output_head_compute_check_t {
    unsigned int baseline_plan_attached;
    unsigned int layer_count;
    unsigned int descriptor_alignment_pass;
    unsigned int tensor_name_alignment_pass;
    unsigned int tensors_materialized_count;
    unsigned int compute_module_count;
    unsigned int sampled_token_id;
    unsigned int topk_token_ids[QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK];
    float topk_logits[QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK];
    uint64_t descriptor_fnv1a64;
    uint64_t tensor_name_fnv1a64;
    uint64_t weight_fnv1a64;
    uint64_t weight_bytes_read;
    uint64_t final_norm_weight_fnv1a64;
    uint64_t lm_head_weight_fnv1a64;
    uint64_t input_hidden_fnv1a64;
    uint64_t final_norm_output_fnv1a64;
    uint64_t logits_fnv1a64;
    uint64_t sampler_fnv1a64;
    uint64_t elapsed_ns;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    int correctness_pass;
    int inference_success_claimed;
} qrt_qwen36_output_head_compute_check_t;

#define QRT_QWEN36_BASELINE_BLOCK_INTEGRATION_LAYER_BLOCK_COUNT \
    QRT_QWEN36_LAYER_COUNT
#define QRT_QWEN36_BASELINE_BLOCK_INTEGRATION_OUTPUT_BLOCK_COUNT 1u
#define QRT_QWEN36_BASELINE_BLOCK_INTEGRATION_MODULE_CLASS_COUNT 24u

typedef struct qrt_qwen36_baseline_block_integration_check_t {
    unsigned int baseline_plan_attached;
    unsigned int layer_count;
    unsigned int linear_attention_layer_count;
    unsigned int full_attention_layer_count;
    unsigned int integrated_layer_block_count;
    unsigned int integrated_output_head_block_count;
    unsigned int integrated_linear_attention_layer_count;
    unsigned int integrated_full_attention_layer_count;
    unsigned int integrated_moe_layer_count;
    unsigned int integrated_module_class_count;
    unsigned int linear_attention_contract_closed;
    unsigned int full_attention_contract_closed;
    unsigned int moe_contract_closed;
    unsigned int output_head_contract_closed;
    uint64_t descriptor_fnv1a64;
    uint64_t linear_attention_digest_fnv1a64;
    uint64_t full_attention_digest_fnv1a64;
    uint64_t moe_digest_fnv1a64;
    uint64_t output_head_digest_fnv1a64;
    uint64_t integration_digest_fnv1a64;
    uint64_t total_weight_bytes_read;
    uint64_t total_elapsed_ns;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    int correctness_pass;
    int inference_success_claimed;
} qrt_qwen36_baseline_block_integration_check_t;

#define QRT_QWEN36_BASELINE_HIDDEN_HANDOFF_LAYER_CAPACITY \
    QRT_QWEN36_LAYER_COUNT
#define QRT_QWEN36_BASELINE_HIDDEN_HANDOFF_OUTPUT_BLOCK_COUNT 1u

typedef struct qrt_qwen36_baseline_hidden_handoff_check_t {
    unsigned int baseline_plan_attached;
    unsigned int layer_count;
    unsigned int linear_attention_layer_count;
    unsigned int full_attention_layer_count;
    unsigned int handoff_layer_count;
    unsigned int output_head_handoff_count;
    unsigned int hidden_size;
    unsigned int buffer_swap_count;
    uint64_t descriptor_fnv1a64;
    uint64_t block_integration_digest_fnv1a64;
    uint64_t initial_hidden_fnv1a64;
    uint64_t layer_handoff_fnv1a64
        [QRT_QWEN36_BASELINE_HIDDEN_HANDOFF_LAYER_CAPACITY];
    uint64_t final_hidden_fnv1a64;
    uint64_t output_head_input_fnv1a64;
    uint64_t handoff_digest_fnv1a64;
    uint64_t total_elapsed_ns;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    int correctness_pass;
    int inference_success_claimed;
} qrt_qwen36_baseline_hidden_handoff_check_t;

typedef struct qrt_qwen36_baseline_request_handoff_check_t {
    unsigned int baseline_plan_attached;
    unsigned int request_handoff_attached;
    unsigned int layer_count;
    unsigned int handoff_layer_count;
    unsigned int output_head_handoff_count;
    unsigned int hidden_size;
    uint64_t descriptor_fnv1a64;
    uint64_t block_integration_digest_fnv1a64;
    uint64_t hidden_handoff_digest_fnv1a64;
    uint64_t final_hidden_fnv1a64;
    uint64_t output_head_input_fnv1a64;
    uint64_t request_handoff_digest_fnv1a64;
    uint64_t total_elapsed_ns;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    int correctness_pass;
    int inference_success_claimed;
} qrt_qwen36_baseline_request_handoff_check_t;

typedef struct qrt_load_manifest_result_t {
    size_t config_bytes;
    size_t index_bytes;
    size_t tensor_count;
    size_t shard_count;
    size_t shard_headers_read;
    size_t shard_header_tensor_count;
    uint64_t metadata_total_size_bytes;
    uint64_t total_shard_file_bytes;
    uint64_t total_safetensors_header_bytes;
    uint64_t total_safetensors_data_bytes;
    size_t hidden_size;
    size_t layer_count;
    size_t attention_heads;
    size_t kv_heads;
    size_t head_dim;
    size_t vocab_size;
    size_t expert_count;
    size_t experts_per_token;
    size_t expert_intermediate_size;
    size_t max_position_embeddings;
    size_t linear_attention_layers;
    size_t full_attention_layers;
    uint64_t total_elapsed_ns;
    uint64_t config_read_ns;
    uint64_t config_parse_ns;
    uint64_t index_read_ns;
    uint64_t index_parse_ns;
    uint64_t shard_stat_ns;
    uint64_t shard_header_ns;
    int config_present;
    int index_present;
    int config_shape_pass;
    int index_pass;
    int shard_headers_pass;
    int used_memory_mapping;
    int manifest_ready;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_load_manifest_result_t;

typedef struct qrt_prepack_accounting_result_t {
    size_t total_tensor_count;
    size_t required_runtime_tensor_count;
    size_t deferred_tensor_count;
    size_t language_tensor_count;
    size_t visual_tensor_count;
    size_t mtp_tensor_count;
    size_t unknown_tensor_count;
    size_t required_prepack_candidate_tensor_count;
    uint64_t total_tensor_bytes;
    uint64_t required_runtime_bytes;
    uint64_t deferred_bytes;
    uint64_t language_bytes;
    uint64_t visual_bytes;
    uint64_t mtp_bytes;
    uint64_t unknown_bytes;
    uint64_t required_prepack_candidate_bytes;
    uint64_t required_prepack_completed_bytes;
    uint64_t deferred_materialization_bytes;
    uint64_t first_request_materialized_bytes;
    uint64_t accounting_elapsed_ns;
    int accounting_ready;
    int byte_accounting_pass;
    int prepack_completed;
    int first_request_materialization_reported;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_prepack_accounting_result_t;

typedef struct qrt_selected_moe_prepack_result_t {
    int prepack_requested;
    int prepack_completed;
    int byte_accounting_pass;
    size_t layer_count;
    size_t expert_count;
    size_t candidate_entry_count;
    size_t loaded_entry_count;
    size_t skipped_existing_entry_count;
    uint64_t candidate_bytes;
    uint64_t loaded_bytes;
    uint64_t skipped_existing_bytes;
    uint64_t read_bytes;
    uint64_t elapsed_ns;
    uint64_t weight_digest_fnv1a64;
    uint64_t store_failure_count;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
} qrt_selected_moe_prepack_result_t;

#define QRT_QWEN36_PREFILL_DESCRIPTOR_BATCH_TIMING_BUCKET_COUNT 17u

typedef struct qrt_qwen36_prefill_descriptor_batch_timing_t {
    uint64_t descriptor_layer_stack_wall_clock_ns;
    uint64_t repeated_segments_wall_clock_ns;
    uint64_t full_attention_layers_wall_clock_ns;
    uint64_t post_stack_wall_clock_ns;
    uint64_t final_output_residual_materialization_wall_clock_ns;
    uint64_t slowest_repeated_segment_wall_clock_ns;
    uint64_t slowest_full_attention_layer_wall_clock_ns;
    unsigned int repeated_segment_call_count;
    unsigned int full_attention_layer_call_count;
    unsigned int slowest_repeated_segment_first_layer;
    unsigned int slowest_repeated_segment_last_layer;
    unsigned int slowest_full_attention_layer;
    size_t slowest_repeated_segment_target_token_count;
    size_t slowest_full_attention_target_token_count;
    uint64_t layer_stack_wall_clock_bucket_elapsed_ns
        [QRT_QWEN36_PREFILL_DESCRIPTOR_BATCH_TIMING_BUCKET_COUNT];
    unsigned int layer_stack_wall_clock_bucket_call_count
        [QRT_QWEN36_PREFILL_DESCRIPTOR_BATCH_TIMING_BUCKET_COUNT];
    uint64_t layer_stack_host_materialized_output_bytes;
    uint64_t layer_stack_host_materialized_stage_count;
    uint64_t layer_stack_host_materialized_output_bytes_by_bucket
        [QRT_QWEN36_PREFILL_DESCRIPTOR_BATCH_TIMING_BUCKET_COUNT];
    unsigned int layer_stack_host_materialized_stage_count_by_bucket
        [QRT_QWEN36_PREFILL_DESCRIPTOR_BATCH_TIMING_BUCKET_COUNT];
    uint64_t layer_stack_resident_d2d_linear_core_state_lookup_count;
    uint64_t layer_stack_resident_d2d_linear_core_state_hit_count;
    uint64_t layer_stack_resident_d2d_linear_core_state_miss_count;
    uint64_t layer_stack_resident_d2d_linear_core_state_store_count;
    uint64_t layer_stack_resident_d2d_linear_core_state_store_bytes;
    uint64_t layer_stack_resident_d2d_linear_core_state_h2d_saved_bytes;
    uint64_t full_attention_resident_core_call_count;
    uint64_t full_attention_resident_core_weight_load_ns;
    uint64_t full_attention_resident_core_device_alloc_ns;
    uint64_t full_attention_resident_core_h2d_ns;
    uint64_t full_attention_resident_core_kernel_ns;
    uint64_t full_attention_resident_core_d2h_ns;
    uint64_t full_attention_resident_core_free_ns;
    uint64_t full_attention_resident_core_weight_read_bytes;
    uint64_t full_attention_resident_core_device_alloc_bytes;
    uint64_t full_attention_resident_core_h2d_bytes;
    uint64_t full_attention_resident_core_d2h_bytes;
    uint64_t full_attention_resident_core_score_scratch_call_count;
    uint64_t full_attention_resident_core_score_scratch_bytes;
    uint64_t routed_expert_profiled_gate_up_avg_ns_sum;
    uint64_t routed_expert_profiled_down_avg_ns_sum;
    unsigned int routed_expert_profiled_gate_up_phase_count;
    unsigned int routed_expert_profiled_down_phase_count;
    uint64_t compact_device_layout_full_prepack_elapsed_ns;
    uint64_t compact_device_layout_full_prepack_layer_count;
    uint64_t compact_device_layout_full_prepack_requested_entry_count;
    uint64_t compact_device_layout_full_prepack_stored_entry_count;
    uint64_t compact_device_layout_full_prepack_store_failure_count;
    uint64_t compact_device_layout_full_prepack_stored_bytes;
    unsigned int compact_device_layout_full_prepack_requested;
    unsigned int compact_device_layout_full_prepack_pass;
    uint64_t descriptor_pre_stack_wall_clock_ns;
    uint64_t descriptor_layer1_cache_wall_clock_ns;
    uint64_t descriptor_layer1_frontier_attach_wall_clock_ns;
    uint64_t descriptor_segment_plan_wall_clock_ns;
    uint64_t descriptor_export_frontier_context_rebuild_wall_clock_ns;
    uint64_t descriptor_export_continuation_callback_wall_clock_ns;
    uint64_t descriptor_export_result_copy_wall_clock_ns;
    uint64_t descriptor_export_total_wall_clock_ns;
    uint64_t descriptor_c_continuation_execute_wall_clock_ns;
    uint64_t descriptor_c_continuation_report_store_wall_clock_ns;
    uint64_t descriptor_c_continuation_total_wall_clock_ns;
    uint64_t compact_device_layout_full_preload_elapsed_ns;
    uint64_t direct_early_entry_start_elapsed_ns;
    uint64_t direct_early_entry_pre_layer_stack_wall_clock_ns;
    uint64_t direct_early_entry_layer_stack_wall_clock_ns;
    uint64_t direct_early_entry_post_layer_stack_wall_clock_ns;
    uint64_t provider_post_direct_to_reported_wall_clock_ns;
} qrt_qwen36_prefill_descriptor_batch_timing_t;

typedef struct qrt_engine_report_t {
    size_t context_tokens;
    unsigned int batch_size;
    int manifest_loaded;
    int ready;
    int inference_implemented;
    size_t request_count;
    size_t generate_call_count;
    size_t token_request_count;
    size_t load_manifest_call_count;
    size_t prepack_accounting_call_count;
    uint64_t create_elapsed_ns;
    uint64_t request_elapsed_ns;
    uint64_t first_request_elapsed_ns;
    uint64_t last_request_elapsed_ns;
    uint64_t last_request_ttft_elapsed_ns;
    uint64_t last_request_tpot_elapsed_ns;
    size_t last_request_tpot_sample_count;
    size_t last_request_output_token_count;
    size_t last_request_descriptor_output_prefill_sample_count;
    size_t last_request_descriptor_output_decode_token_count;
    unsigned int last_request_descriptor_output_autoregressive_decode;
    int prefix_cache_state_attached;
    int prefix_cache_copy_on_write_fallback_used;
    size_t prefix_cache_state_attach_count;
    size_t prefix_cache_state_consumed_request_count;
    size_t prefix_cache_cached_prefix_tokens;
    size_t prefix_cache_suffix_tokens;
    size_t prefix_cache_requested_output_tokens;
    size_t prefix_cache_executed_input_tokens;
    size_t prefix_cache_output_token_capacity;
    size_t prefix_cache_block_tokens;
    size_t prefix_cache_owner_table_entries;
    size_t prefix_cache_shared_owner_alias_entries;
    size_t prefix_cache_request_absolute_start_position;
    size_t prefix_cache_request_absolute_last_input_position;
    size_t prefix_cache_first_output_position;
    size_t prefix_cache_last_output_position;
    size_t prefix_cache_last_request_input_token_count;
    size_t prefix_cache_last_request_output_token_capacity;
    size_t prefix_cache_last_request_absolute_start_position;
    size_t prefix_cache_last_request_absolute_last_input_position;
    size_t prefix_cache_last_request_first_output_position;
    size_t prefix_cache_last_request_last_output_position;
    uint64_t prefix_cache_linear_attention_state_bytes;
    uint64_t prefix_cache_full_attention_kv_bytes;
    uint64_t prefix_cache_total_prefix_state_bytes;
    uint64_t prefix_cache_mutable_linear_attention_state_bytes;
    uint64_t prefix_cache_mutable_full_attention_kv_bytes;
    uint64_t prefix_cache_copy_on_write_bytes;
    uint64_t prefix_cache_forbidden_copy_bytes;
    uint64_t prefix_cache_state_digest_fnv1a64;
    uint64_t prefix_cache_position_digest_fnv1a64;
    uint64_t prefix_cache_last_request_digest_fnv1a64;
    int prefix_cache_tensor_linear_state_requested;
    int prefix_cache_tensor_full_kv_state_requested;
    int prefix_cache_tensor_full_kv_window_state_requested;
    int prefix_cache_tensor_full_kv_compressed_state_requested;
    int prefix_cache_tensor_linear_state_seeded;
    int prefix_cache_tensor_linear_state_consumed;
    int prefix_cache_tensor_full_kv_state_seeded;
    int prefix_cache_tensor_full_kv_state_consumed;
    int prefix_cache_tensor_full_kv_compressed_state_seeded;
    int prefix_cache_tensor_full_kv_compressed_state_consumed;
    size_t prefix_cache_tensor_linear_seed_layer_count;
    size_t prefix_cache_tensor_linear_seed_value_count;
    size_t prefix_cache_tensor_full_kv_seed_layer_count;
    size_t prefix_cache_tensor_full_kv_seed_value_count;
    size_t prefix_cache_tensor_full_kv_window_token_count;
    size_t prefix_cache_tensor_full_kv_cached_prefix_tokens;
    size_t prefix_cache_tensor_full_kv_absolute_start_position;
    size_t prefix_cache_tensor_full_kv_absolute_last_position;
    size_t prefix_cache_tensor_full_kv_compressed_block_count;
    size_t prefix_cache_tensor_full_kv_compressed_block_tokens;
    size_t prefix_cache_tensor_full_kv_compressed_covered_tokens;
    size_t prefix_cache_tensor_full_kv_compressed_tail_block_index;
    size_t prefix_cache_tensor_full_kv_compressed_tail_offset_tokens;
    int prefix_cache_tensor_full_kv_compressed_backend_span_handoff_evaluated;
    size_t prefix_cache_tensor_full_kv_compressed_backend_span_layer_count;
    size_t prefix_cache_tensor_full_kv_compressed_backend_span_eval_count;
    size_t prefix_cache_tensor_full_kv_compressed_backend_span_block_reference_count;
    size_t prefix_cache_tensor_full_kv_compressed_backend_span_token_reference_count;
    int prefix_cache_tensor_full_kv_compressed_value_payload_present;
    size_t prefix_cache_tensor_full_kv_compressed_value_payload_layer_count;
    size_t prefix_cache_tensor_full_kv_compressed_value_payload_block_count;
    size_t prefix_cache_tensor_full_kv_compressed_value_payload_sample_count;
    uint64_t prefix_cache_tensor_linear_seed_digest_fnv1a64;
    uint64_t prefix_cache_tensor_linear_core_input_digest_fnv1a64;
    uint64_t prefix_cache_tensor_linear_core_output_digest_fnv1a64;
    uint64_t prefix_cache_tensor_full_kv_seed_digest_fnv1a64;
    uint64_t prefix_cache_tensor_full_kv_core_input_digest_fnv1a64;
    uint64_t prefix_cache_tensor_full_kv_core_output_digest_fnv1a64;
    uint64_t prefix_cache_tensor_full_kv_compressed_seed_digest_fnv1a64;
    uint64_t prefix_cache_tensor_full_kv_compressed_position_digest_fnv1a64;
    uint64_t prefix_cache_tensor_full_kv_compressed_tail_digest_fnv1a64;
    uint64_t prefix_cache_tensor_full_kv_compressed_core_input_digest_fnv1a64;
    uint64_t prefix_cache_tensor_full_kv_compressed_core_output_digest_fnv1a64;
    uint64_t prefix_cache_tensor_full_kv_compressed_backend_span_input_digest_fnv1a64;
    uint64_t prefix_cache_tensor_full_kv_compressed_backend_span_output_digest_fnv1a64;
    uint64_t prefix_cache_tensor_full_kv_compressed_value_payload_digest_fnv1a64;
    uint64_t prepack_accounting_elapsed_ns;
    int resident_weight_cache_enabled;
    size_t resident_weight_cache_entry_count;
    size_t resident_weight_cache_index_capacity;
    uint64_t resident_weight_cache_bytes;
    uint64_t resident_weight_cache_hit_count;
    uint64_t resident_weight_cache_miss_count;
    uint64_t resident_weight_cache_hit_bytes;
    uint64_t resident_weight_cache_miss_bytes;
    uint64_t resident_weight_cache_store_failure_count;
    uint64_t resident_weight_cache_index_lookup_count;
    uint64_t resident_weight_cache_index_probe_count;
    uint64_t resident_weight_cache_index_fallback_count;
    size_t selected_moe_weight_prepack_candidate_entry_count;
    uint64_t selected_moe_weight_prepack_candidate_bytes;
    size_t selected_moe_weight_request_materialized_entry_count;
    uint64_t selected_moe_weight_request_materialized_bytes;
    size_t selected_moe_weight_prepack_remaining_entry_count;
    uint64_t selected_moe_weight_prepack_remaining_bytes;
    size_t tensor_location_cache_entry_count;
    uint64_t tensor_location_cache_hit_count;
    uint64_t tensor_location_cache_miss_count;
    uint64_t tensor_location_cache_store_failure_count;
    int tensor_shard_read_cache_enabled;
    size_t tensor_shard_read_cache_entry_count;
    uint64_t tensor_shard_read_cache_hit_count;
    uint64_t tensor_shard_read_cache_miss_count;
    uint64_t tensor_shard_read_cache_read_bytes;
    uint64_t tensor_shard_read_cache_opened_file_bytes;
    uint64_t tensor_shard_read_cache_fallback_count;
    uint64_t tensor_shard_read_cache_store_failure_count;
    int selected_expert_weight_byte_hash_enabled;
    int baseline_plan_attached;
    unsigned int baseline_layer_count;
    unsigned int baseline_linear_attention_layer_count;
    unsigned int baseline_full_attention_layer_count;
    unsigned int baseline_prefill_entrypoint_planned;
    unsigned int baseline_decode_one_entrypoint_planned;
    uint64_t baseline_descriptor_fnv1a64;
    size_t baseline_prefill_call_count;
    size_t baseline_decode_one_call_count;
    char baseline_failure_stage[64];
    char baseline_failure[QRT_LOAD_ERROR_CAPACITY];
    uint64_t token_embedding_elapsed_ns;
    uint64_t layer0_input_norm_elapsed_ns;
    uint64_t layer0_qkv_projection_elapsed_ns;
    uint64_t layer0_zab_projection_elapsed_ns;
    uint64_t layer0_conv_qkv_elapsed_ns;
    uint64_t layer0_postconv_qkv_elapsed_ns;
    uint64_t layer0_gate_rows_elapsed_ns;
    uint64_t layer0_core_rows_elapsed_ns;
    uint64_t layer0_gated_rmsnorm_elapsed_ns;
    uint64_t layer0_out_projection_elapsed_ns;
    uint64_t layer0_residual_hidden_elapsed_ns;
    uint64_t layer0_post_attention_rmsnorm_elapsed_ns;
    uint64_t layer0_moe_router_elapsed_ns;
    uint64_t layer0_moe_expert_elapsed_ns;
    uint64_t layer0_moe_shared_expert_elapsed_ns;
    uint64_t layer0_output_residual_elapsed_ns;
    uint64_t layer1_input_norm_elapsed_ns;
    uint64_t layer1_qkv_projection_elapsed_ns;
    uint64_t layer1_zab_projection_elapsed_ns;
    uint64_t layer1_conv_qkv_elapsed_ns;
    uint64_t layer1_postconv_qkv_elapsed_ns;
    uint64_t layer1_gate_rows_elapsed_ns;
    uint64_t layer1_core_rows_elapsed_ns;
    uint64_t layer1_gated_rmsnorm_elapsed_ns;
    uint64_t layer1_out_projection_elapsed_ns;
    uint64_t layer1_residual_hidden_elapsed_ns;
    uint64_t layer1_post_attention_rmsnorm_elapsed_ns;
    uint64_t layer1_moe_router_elapsed_ns;
    uint64_t layer1_moe_expert_elapsed_ns;
    uint64_t layer1_moe_shared_expert_elapsed_ns;
    uint64_t layer1_output_residual_elapsed_ns;
    uint64_t layer2_input_norm_elapsed_ns;
    uint64_t layer2_qkv_projection_elapsed_ns;
    uint64_t layer2_zab_projection_elapsed_ns;
    uint64_t layer2_conv_qkv_elapsed_ns;
    uint64_t layer2_postconv_qkv_elapsed_ns;
    uint64_t layer2_gate_rows_elapsed_ns;
    uint64_t layer2_core_rows_elapsed_ns;
    uint64_t layer2_gated_rmsnorm_elapsed_ns;
    uint64_t layer2_out_projection_elapsed_ns;
    uint64_t layer2_residual_hidden_elapsed_ns;
    uint64_t layer2_post_attention_rmsnorm_elapsed_ns;
    uint64_t layer2_moe_router_elapsed_ns;
    uint64_t layer2_moe_expert_elapsed_ns;
    uint64_t layer2_moe_shared_expert_elapsed_ns;
    uint64_t layer2_output_residual_elapsed_ns;
    uint64_t layer3_input_norm_elapsed_ns;
    uint64_t layer3_qkv_projection_elapsed_ns;
    uint64_t layer3_qk_norm_elapsed_ns;
    uint64_t layer3_rope_attention_elapsed_ns;
    uint64_t layer3_output_projection_elapsed_ns;
    uint64_t layer3_residual_hidden_elapsed_ns;
    uint64_t layer3_post_attention_rmsnorm_elapsed_ns;
    uint64_t layer3_moe_router_elapsed_ns;
    uint64_t layer3_moe_expert_elapsed_ns;
    uint64_t layer3_moe_shared_expert_elapsed_ns;
    uint64_t layer3_output_residual_elapsed_ns;
    uint64_t layer4_input_norm_elapsed_ns;
    uint64_t layer4_qkv_projection_elapsed_ns;
    uint64_t layer4_zab_projection_elapsed_ns;
    uint64_t layer4_conv_qkv_elapsed_ns;
    uint64_t layer4_postconv_qkv_elapsed_ns;
    uint64_t layer4_gate_rows_elapsed_ns;
    uint64_t layer4_core_rows_elapsed_ns;
    uint64_t layer4_gated_rmsnorm_elapsed_ns;
    uint64_t layer4_out_projection_elapsed_ns;
    uint64_t layer4_residual_hidden_elapsed_ns;
    uint64_t layer4_post_attention_rmsnorm_elapsed_ns;
    uint64_t layer4_moe_router_elapsed_ns;
    uint64_t layer4_moe_expert_elapsed_ns;
    uint64_t layer4_moe_shared_expert_elapsed_ns;
    uint64_t layer4_output_residual_elapsed_ns;
    uint64_t layer5_input_norm_elapsed_ns;
    uint64_t layer5_qkv_projection_elapsed_ns;
    uint64_t layer5_zab_projection_elapsed_ns;
    uint64_t layer5_conv_qkv_elapsed_ns;
    uint64_t layer5_postconv_qkv_elapsed_ns;
    uint64_t layer5_gate_rows_elapsed_ns;
    uint64_t layer5_core_rows_elapsed_ns;
    uint64_t layer5_gated_rmsnorm_elapsed_ns;
    uint64_t layer5_out_projection_elapsed_ns;
    uint64_t layer5_residual_hidden_elapsed_ns;
    uint64_t layer5_post_attention_rmsnorm_elapsed_ns;
    uint64_t layer5_moe_router_elapsed_ns;
    uint64_t layer5_moe_expert_elapsed_ns;
    uint64_t layer5_moe_shared_expert_elapsed_ns;
    uint64_t layer5_output_residual_elapsed_ns;
    uint64_t layer6_input_norm_elapsed_ns;
    uint64_t layer6_qkv_projection_elapsed_ns;
    uint64_t layer6_zab_projection_elapsed_ns;
    uint64_t layer6_conv_qkv_elapsed_ns;
    uint64_t layer6_postconv_qkv_elapsed_ns;
    uint64_t layer6_gate_rows_elapsed_ns;
    uint64_t layer6_core_rows_elapsed_ns;
    uint64_t layer6_gated_rmsnorm_elapsed_ns;
    uint64_t layer6_out_projection_elapsed_ns;
    uint64_t layer6_residual_hidden_elapsed_ns;
    uint64_t layer6_post_attention_rmsnorm_elapsed_ns;
    uint64_t layer6_moe_router_elapsed_ns;
    uint64_t layer6_moe_expert_elapsed_ns;
    uint64_t layer6_moe_shared_expert_elapsed_ns;
    uint64_t layer6_output_residual_elapsed_ns;
    uint64_t layer7_input_norm_elapsed_ns;
    uint64_t layer7_qkv_projection_elapsed_ns;
    uint64_t layer7_qk_norm_elapsed_ns;
    uint64_t layer7_rope_elapsed_ns;
    uint64_t layer7_attention_elapsed_ns;
    uint64_t layer7_output_projection_elapsed_ns;
    uint64_t layer7_residual_hidden_elapsed_ns;
    uint64_t layer7_post_attention_rmsnorm_elapsed_ns;
    uint64_t layer7_moe_router_elapsed_ns;
    uint64_t layer7_moe_expert_elapsed_ns;
    uint64_t layer7_moe_shared_expert_elapsed_ns;
    uint64_t layer7_output_residual_elapsed_ns;
    uint64_t layer8_input_norm_elapsed_ns;
    uint64_t layer8_qkv_projection_elapsed_ns;
    uint64_t layer8_zab_projection_elapsed_ns;
    uint64_t layer8_conv_qkv_elapsed_ns;
    uint64_t layer8_postconv_qkv_elapsed_ns;
    uint64_t layer8_gate_rows_elapsed_ns;
    uint64_t layer8_core_rows_elapsed_ns;
    uint64_t layer8_gated_rmsnorm_elapsed_ns;
    uint64_t layer8_out_projection_elapsed_ns;
    uint64_t layer8_residual_hidden_elapsed_ns;
    uint64_t layer8_post_attention_rmsnorm_elapsed_ns;
    uint64_t layer8_moe_router_elapsed_ns;
    uint64_t layer8_moe_expert_elapsed_ns;
    uint64_t layer8_moe_shared_expert_elapsed_ns;
    uint64_t layer8_output_residual_elapsed_ns;
    uint64_t layer8_handoff_elapsed_ns;
    uint64_t layer9_loop_body_elapsed_ns;
    uint64_t layer9_input_norm_elapsed_ns;
    uint64_t layer9_qkv_projection_elapsed_ns;
    uint64_t layer9_zab_projection_elapsed_ns;
    uint64_t layer9_conv_qkv_elapsed_ns;
    uint64_t layer9_postconv_qkv_elapsed_ns;
    uint64_t layer9_gate_rows_elapsed_ns;
    uint64_t layer9_core_rows_elapsed_ns;
    uint64_t layer9_gated_rmsnorm_elapsed_ns;
    uint64_t layer9_out_projection_elapsed_ns;
    uint64_t layer9_residual_hidden_elapsed_ns;
    uint64_t layer9_post_attention_rmsnorm_elapsed_ns;
    uint64_t layer9_moe_router_elapsed_ns;
    uint64_t layer9_moe_expert_elapsed_ns;
    uint64_t layer9_moe_shared_expert_elapsed_ns;
    uint64_t layer9_output_residual_elapsed_ns;
    uint64_t layer10_input_norm_elapsed_ns;
    uint64_t layer10_qkv_projection_elapsed_ns;
    uint64_t layer10_zab_projection_elapsed_ns;
    uint64_t layer10_conv_qkv_elapsed_ns;
    uint64_t layer10_postconv_qkv_elapsed_ns;
    uint64_t layer10_gate_rows_elapsed_ns;
    uint64_t layer10_core_rows_elapsed_ns;
    uint64_t layer10_gated_rmsnorm_elapsed_ns;
    uint64_t layer10_out_projection_elapsed_ns;
    uint64_t layer10_residual_hidden_elapsed_ns;
    uint64_t layer10_post_attention_rmsnorm_elapsed_ns;
    uint64_t qwen36_layer_descriptor_helper_elapsed_ns;
    uint64_t qwen36_descriptor_layer_batch_elapsed_ns;
    uint64_t qwen36_descriptor_layer_loop_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_loop_body_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_input_norm_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_qkv_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_qk_norm_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_rope_attention_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_projection_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_residual_hidden_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_post_attention_rmsnorm_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_router_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_residual_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_handoff_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_loop_body_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_input_norm_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_post_attention_rmsnorm_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_handoff_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_loop_body_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_input_norm_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_post_attention_rmsnorm_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_handoff_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_loop_body_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_input_norm_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_post_attention_rmsnorm_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_handoff_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_loop_body_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_input_norm_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_post_attention_rmsnorm_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_handoff_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_loop_body_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_input_norm_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_post_attention_rmsnorm_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_elapsed_ns;
    uint64_t qwen36_descriptor_layer_body_elapsed_ns;
    uint64_t input_embeddings_fnv1a64;
    uint64_t last_token_embedding_fnv1a64;
    uint64_t token_embedding_bytes_read;
    uint64_t layer0_input_norm_fnv1a64;
    uint64_t last_layer0_input_norm_fnv1a64;
    uint64_t layer0_input_norm_weight_fnv1a64;
    uint64_t layer0_input_norm_weight_bytes_read;
    uint64_t layer0_qkv_projection_fnv1a64;
    uint64_t last_layer0_qkv_projection_fnv1a64;
    uint64_t layer0_qkv_projection_weight_fnv1a64;
    uint64_t layer0_qkv_projection_weight_bytes_read;
    uint64_t layer0_zab_projection_fnv1a64;
    uint64_t last_layer0_zab_projection_fnv1a64;
    uint64_t layer0_z_projection_weight_fnv1a64;
    uint64_t layer0_a_projection_weight_fnv1a64;
    uint64_t layer0_b_projection_weight_fnv1a64;
    uint64_t layer0_zab_projection_weight_bytes_read;
    uint64_t layer0_conv_qkv_fnv1a64;
    uint64_t last_layer0_conv_qkv_fnv1a64;
    uint64_t layer0_conv_qkv_weight_fnv1a64;
    uint64_t layer0_conv_qkv_weight_bytes_read;
    uint64_t layer0_postconv_q_scaled_fnv1a64;
    uint64_t last_layer0_postconv_q_scaled_fnv1a64;
    uint64_t layer0_postconv_k_norm_fnv1a64;
    uint64_t last_layer0_postconv_k_norm_fnv1a64;
    uint64_t layer0_postconv_value_fnv1a64;
    uint64_t last_layer0_postconv_value_fnv1a64;
    uint64_t layer0_gate_g_fnv1a64;
    uint64_t last_layer0_gate_g_fnv1a64;
    uint64_t layer0_gate_beta_fnv1a64;
    uint64_t last_layer0_gate_beta_fnv1a64;
    uint64_t layer0_a_log_fnv1a64;
    uint64_t layer0_dt_bias_fnv1a64;
    uint64_t layer0_gate_weight_bytes_read;
    uint64_t layer0_core_rows_fnv1a64;
    uint64_t last_layer0_core_rows_fnv1a64;
    uint64_t layer0_core_final_state_fnv1a64;
    uint64_t layer0_gated_rmsnorm_fnv1a64;
    uint64_t last_layer0_gated_rmsnorm_fnv1a64;
    uint64_t layer0_linear_norm_weight_fnv1a64;
    uint64_t layer0_linear_norm_weight_bytes_read;
    uint64_t layer0_out_projection_fnv1a64;
    uint64_t last_layer0_out_projection_fnv1a64;
    uint64_t layer0_out_projection_weight_fnv1a64;
    uint64_t layer0_out_projection_weight_bytes_read;
    uint64_t layer0_residual_hidden_fnv1a64;
    uint64_t last_layer0_residual_hidden_fnv1a64;
    uint64_t layer0_post_attention_rmsnorm_fnv1a64;
    uint64_t last_layer0_post_attention_rmsnorm_fnv1a64;
    uint64_t layer0_post_attention_norm_weight_fnv1a64;
    uint64_t layer0_post_attention_norm_weight_bytes_read;
    uint64_t layer0_moe_router_logits_fnv1a64;
    uint64_t last_layer0_moe_router_logits_fnv1a64;
    uint64_t layer0_moe_router_topk_ids_fnv1a64;
    uint64_t last_layer0_moe_router_topk_ids_fnv1a64;
    uint64_t layer0_moe_router_topk_weights_fnv1a64;
    uint64_t last_layer0_moe_router_topk_weights_fnv1a64;
    uint64_t layer0_moe_router_weight_fnv1a64;
    uint64_t layer0_moe_router_weight_bytes_read;
    uint64_t layer0_moe_expert_routed_fnv1a64;
    uint64_t last_layer0_moe_expert_routed_fnv1a64;
    uint64_t layer0_moe_expert_selected_ids_fnv1a64;
    uint64_t layer0_moe_expert_gate_up_weight_fnv1a64;
    uint64_t layer0_moe_expert_down_weight_fnv1a64;
    uint64_t layer0_moe_expert_weight_bytes_read;
    uint64_t layer0_moe_shared_expert_fnv1a64;
    uint64_t last_layer0_moe_shared_expert_fnv1a64;
    uint64_t layer0_moe_combined_fnv1a64;
    uint64_t last_layer0_moe_combined_fnv1a64;
    uint64_t layer0_moe_shared_gate_fnv1a64;
    uint64_t last_layer0_moe_shared_gate_fnv1a64;
    uint64_t layer0_moe_shared_gate_weight_fnv1a64;
    uint64_t layer0_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t layer0_moe_shared_up_proj_weight_fnv1a64;
    uint64_t layer0_moe_shared_down_weight_fnv1a64;
    uint64_t layer0_moe_shared_expert_weight_bytes_read;
    uint64_t layer0_output_residual_fnv1a64;
    uint64_t last_layer0_output_residual_fnv1a64;
    uint64_t layer1_input_norm_fnv1a64;
    uint64_t last_layer1_input_norm_fnv1a64;
    uint64_t layer1_input_norm_weight_fnv1a64;
    uint64_t layer1_input_norm_weight_bytes_read;
    uint64_t layer1_qkv_projection_fnv1a64;
    uint64_t last_layer1_qkv_projection_fnv1a64;
    uint64_t layer1_qkv_projection_weight_fnv1a64;
    uint64_t layer1_qkv_projection_weight_bytes_read;
    uint64_t layer1_zab_projection_fnv1a64;
    uint64_t last_layer1_zab_projection_fnv1a64;
    uint64_t layer1_z_projection_weight_fnv1a64;
    uint64_t layer1_a_projection_weight_fnv1a64;
    uint64_t layer1_b_projection_weight_fnv1a64;
    uint64_t layer1_zab_projection_weight_bytes_read;
    uint64_t layer1_conv_qkv_fnv1a64;
    uint64_t last_layer1_conv_qkv_fnv1a64;
    uint64_t layer1_conv_qkv_weight_fnv1a64;
    uint64_t layer1_conv_qkv_weight_bytes_read;
    uint64_t layer1_postconv_q_scaled_fnv1a64;
    uint64_t last_layer1_postconv_q_scaled_fnv1a64;
    uint64_t layer1_postconv_k_norm_fnv1a64;
    uint64_t last_layer1_postconv_k_norm_fnv1a64;
    uint64_t layer1_postconv_value_fnv1a64;
    uint64_t last_layer1_postconv_value_fnv1a64;
    uint64_t layer1_gate_g_fnv1a64;
    uint64_t last_layer1_gate_g_fnv1a64;
    uint64_t layer1_gate_beta_fnv1a64;
    uint64_t last_layer1_gate_beta_fnv1a64;
    uint64_t layer1_a_log_fnv1a64;
    uint64_t layer1_dt_bias_fnv1a64;
    uint64_t layer1_gate_weight_bytes_read;
    uint64_t layer1_core_rows_fnv1a64;
    uint64_t last_layer1_core_rows_fnv1a64;
    uint64_t layer1_core_final_state_fnv1a64;
    uint64_t layer1_gated_rmsnorm_fnv1a64;
    uint64_t last_layer1_gated_rmsnorm_fnv1a64;
    uint64_t layer1_linear_norm_weight_fnv1a64;
    uint64_t layer1_linear_norm_weight_bytes_read;
    uint64_t layer1_out_projection_fnv1a64;
    uint64_t last_layer1_out_projection_fnv1a64;
    uint64_t layer1_out_projection_weight_fnv1a64;
    uint64_t layer1_out_projection_weight_bytes_read;
    uint64_t layer1_residual_hidden_fnv1a64;
    uint64_t last_layer1_residual_hidden_fnv1a64;
    uint64_t layer1_post_attention_rmsnorm_fnv1a64;
    uint64_t last_layer1_post_attention_rmsnorm_fnv1a64;
    uint64_t layer1_post_attention_norm_weight_fnv1a64;
    uint64_t layer1_post_attention_norm_weight_bytes_read;
    uint64_t layer1_moe_router_logits_fnv1a64;
    uint64_t last_layer1_moe_router_logits_fnv1a64;
    uint64_t layer1_moe_router_topk_ids_fnv1a64;
    uint64_t last_layer1_moe_router_topk_ids_fnv1a64;
    uint64_t layer1_moe_router_topk_weights_fnv1a64;
    uint64_t last_layer1_moe_router_topk_weights_fnv1a64;
    uint64_t layer1_moe_router_weight_fnv1a64;
    uint64_t layer1_moe_router_weight_bytes_read;
    uint64_t layer1_moe_expert_routed_fnv1a64;
    uint64_t last_layer1_moe_expert_routed_fnv1a64;
    uint64_t layer1_moe_expert_selected_ids_fnv1a64;
    uint64_t layer1_moe_expert_gate_up_weight_fnv1a64;
    uint64_t layer1_moe_expert_down_weight_fnv1a64;
    uint64_t layer1_moe_expert_weight_bytes_read;
    uint64_t layer1_moe_shared_expert_fnv1a64;
    uint64_t last_layer1_moe_shared_expert_fnv1a64;
    uint64_t layer1_moe_combined_fnv1a64;
    uint64_t last_layer1_moe_combined_fnv1a64;
    uint64_t layer1_moe_shared_gate_fnv1a64;
    uint64_t last_layer1_moe_shared_gate_fnv1a64;
    uint64_t layer1_moe_shared_gate_weight_fnv1a64;
    uint64_t layer1_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t layer1_moe_shared_up_proj_weight_fnv1a64;
    uint64_t layer1_moe_shared_down_weight_fnv1a64;
    uint64_t layer1_moe_shared_expert_weight_bytes_read;
    uint64_t layer1_output_residual_fnv1a64;
    uint64_t last_layer1_output_residual_fnv1a64;
    uint64_t layer2_input_norm_fnv1a64;
    uint64_t last_layer2_input_norm_fnv1a64;
    uint64_t layer2_input_norm_weight_fnv1a64;
    uint64_t layer2_input_norm_weight_bytes_read;
    uint64_t layer2_qkv_projection_fnv1a64;
    uint64_t last_layer2_qkv_projection_fnv1a64;
    uint64_t layer2_qkv_projection_weight_fnv1a64;
    uint64_t layer2_qkv_projection_weight_bytes_read;
    uint64_t layer2_zab_projection_fnv1a64;
    uint64_t last_layer2_zab_projection_fnv1a64;
    uint64_t layer2_z_projection_weight_fnv1a64;
    uint64_t layer2_a_projection_weight_fnv1a64;
    uint64_t layer2_b_projection_weight_fnv1a64;
    uint64_t layer2_zab_projection_weight_bytes_read;
    uint64_t layer2_conv_qkv_fnv1a64;
    uint64_t last_layer2_conv_qkv_fnv1a64;
    uint64_t layer2_conv_qkv_weight_fnv1a64;
    uint64_t layer2_conv_qkv_weight_bytes_read;
    uint64_t layer2_postconv_q_scaled_fnv1a64;
    uint64_t last_layer2_postconv_q_scaled_fnv1a64;
    uint64_t layer2_postconv_k_norm_fnv1a64;
    uint64_t last_layer2_postconv_k_norm_fnv1a64;
    uint64_t layer2_postconv_value_fnv1a64;
    uint64_t last_layer2_postconv_value_fnv1a64;
    uint64_t layer2_gate_g_fnv1a64;
    uint64_t last_layer2_gate_g_fnv1a64;
    uint64_t layer2_gate_beta_fnv1a64;
    uint64_t last_layer2_gate_beta_fnv1a64;
    uint64_t layer2_a_log_fnv1a64;
    uint64_t layer2_dt_bias_fnv1a64;
    uint64_t layer2_gate_weight_bytes_read;
    uint64_t layer2_core_rows_fnv1a64;
    uint64_t last_layer2_core_rows_fnv1a64;
    uint64_t layer2_core_final_state_fnv1a64;
    uint64_t layer2_gated_rmsnorm_fnv1a64;
    uint64_t last_layer2_gated_rmsnorm_fnv1a64;
    uint64_t layer2_linear_norm_weight_fnv1a64;
    uint64_t layer2_linear_norm_weight_bytes_read;
    uint64_t layer2_out_projection_fnv1a64;
    uint64_t last_layer2_out_projection_fnv1a64;
    uint64_t layer2_out_projection_weight_fnv1a64;
    uint64_t layer2_out_projection_weight_bytes_read;
    uint64_t layer2_residual_hidden_fnv1a64;
    uint64_t last_layer2_residual_hidden_fnv1a64;
    uint64_t layer2_post_attention_rmsnorm_fnv1a64;
    uint64_t last_layer2_post_attention_rmsnorm_fnv1a64;
    uint64_t layer2_post_attention_norm_weight_fnv1a64;
    uint64_t layer2_post_attention_norm_weight_bytes_read;
    uint64_t layer2_moe_router_logits_fnv1a64;
    uint64_t last_layer2_moe_router_logits_fnv1a64;
    uint64_t layer2_moe_router_topk_ids_fnv1a64;
    uint64_t last_layer2_moe_router_topk_ids_fnv1a64;
    uint64_t layer2_moe_router_topk_weights_fnv1a64;
    uint64_t last_layer2_moe_router_topk_weights_fnv1a64;
    uint64_t layer2_moe_router_weight_fnv1a64;
    uint64_t layer2_moe_router_weight_bytes_read;
    uint64_t layer2_moe_expert_routed_fnv1a64;
    uint64_t last_layer2_moe_expert_routed_fnv1a64;
    uint64_t layer2_moe_expert_selected_ids_fnv1a64;
    uint64_t layer2_moe_expert_gate_up_weight_fnv1a64;
    uint64_t layer2_moe_expert_down_weight_fnv1a64;
    uint64_t layer2_moe_expert_weight_bytes_read;
    uint64_t layer2_moe_shared_expert_fnv1a64;
    uint64_t last_layer2_moe_shared_expert_fnv1a64;
    uint64_t layer2_moe_combined_fnv1a64;
    uint64_t last_layer2_moe_combined_fnv1a64;
    uint64_t layer2_moe_shared_gate_fnv1a64;
    uint64_t last_layer2_moe_shared_gate_fnv1a64;
    uint64_t layer2_moe_shared_gate_weight_fnv1a64;
    uint64_t layer2_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t layer2_moe_shared_up_proj_weight_fnv1a64;
    uint64_t layer2_moe_shared_down_weight_fnv1a64;
    uint64_t layer2_moe_shared_expert_weight_bytes_read;
    uint64_t layer2_output_residual_fnv1a64;
    uint64_t last_layer2_output_residual_fnv1a64;
    uint64_t layer3_input_norm_fnv1a64;
    uint64_t last_layer3_input_norm_fnv1a64;
    uint64_t layer3_input_norm_weight_fnv1a64;
    uint64_t layer3_input_norm_weight_bytes_read;
    uint64_t layer3_qkv_projection_fnv1a64;
    uint64_t last_layer3_qkv_projection_fnv1a64;
    uint64_t layer3_qkv_projection_weight_fnv1a64;
    uint64_t layer3_qkv_projection_weight_bytes_read;
    uint64_t layer3_qk_norm_fnv1a64;
    uint64_t last_layer3_qk_norm_fnv1a64;
    uint64_t layer3_q_norm_weight_fnv1a64;
    uint64_t layer3_k_norm_weight_fnv1a64;
    uint64_t layer3_qk_norm_weight_bytes_read;
    uint64_t layer3_rope_attention_fnv1a64;
    uint64_t last_layer3_rope_attention_fnv1a64;
    uint64_t layer3_output_projection_fnv1a64;
    uint64_t last_layer3_output_projection_fnv1a64;
    uint64_t layer3_output_projection_weight_fnv1a64;
    uint64_t layer3_output_projection_weight_bytes_read;
    uint64_t layer3_residual_hidden_fnv1a64;
    uint64_t last_layer3_residual_hidden_fnv1a64;
    uint64_t layer3_post_attention_rmsnorm_fnv1a64;
    uint64_t last_layer3_post_attention_rmsnorm_fnv1a64;
    uint64_t layer3_post_attention_norm_weight_fnv1a64;
    uint64_t layer3_post_attention_norm_weight_bytes_read;
    uint64_t layer3_moe_router_logits_fnv1a64;
    uint64_t last_layer3_moe_router_logits_fnv1a64;
    uint64_t layer3_moe_router_topk_ids_fnv1a64;
    uint64_t last_layer3_moe_router_topk_ids_fnv1a64;
    uint64_t layer3_moe_router_topk_weights_fnv1a64;
    uint64_t last_layer3_moe_router_topk_weights_fnv1a64;
    uint64_t layer3_moe_router_weight_fnv1a64;
    uint64_t layer3_moe_router_weight_bytes_read;
    uint64_t layer3_moe_expert_routed_fnv1a64;
    uint64_t last_layer3_moe_expert_routed_fnv1a64;
    uint64_t layer3_moe_expert_selected_ids_fnv1a64;
    uint64_t layer3_moe_expert_gate_up_weight_fnv1a64;
    uint64_t layer3_moe_expert_down_weight_fnv1a64;
    uint64_t layer3_moe_expert_weight_bytes_read;
    uint64_t layer3_moe_shared_expert_fnv1a64;
    uint64_t last_layer3_moe_shared_expert_fnv1a64;
    uint64_t layer3_moe_combined_fnv1a64;
    uint64_t last_layer3_moe_combined_fnv1a64;
    uint64_t layer3_moe_shared_gate_fnv1a64;
    uint64_t last_layer3_moe_shared_gate_fnv1a64;
    uint64_t layer3_moe_shared_gate_weight_fnv1a64;
    uint64_t layer3_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t layer3_moe_shared_up_proj_weight_fnv1a64;
    uint64_t layer3_moe_shared_down_weight_fnv1a64;
    uint64_t layer3_moe_shared_expert_weight_bytes_read;
    uint64_t layer3_output_residual_fnv1a64;
    uint64_t last_layer3_output_residual_fnv1a64;
    uint64_t layer4_input_norm_fnv1a64;
    uint64_t last_layer4_input_norm_fnv1a64;
    uint64_t layer4_input_norm_weight_fnv1a64;
    uint64_t layer4_input_norm_weight_bytes_read;
    uint64_t layer4_qkv_projection_fnv1a64;
    uint64_t last_layer4_qkv_projection_fnv1a64;
    uint64_t layer4_qkv_projection_weight_fnv1a64;
    uint64_t layer4_qkv_projection_weight_bytes_read;
    uint64_t layer4_zab_projection_fnv1a64;
    uint64_t last_layer4_zab_projection_fnv1a64;
    uint64_t layer4_z_projection_weight_fnv1a64;
    uint64_t layer4_a_projection_weight_fnv1a64;
    uint64_t layer4_b_projection_weight_fnv1a64;
    uint64_t layer4_zab_projection_weight_bytes_read;
    uint64_t layer4_conv_qkv_fnv1a64;
    uint64_t last_layer4_conv_qkv_fnv1a64;
    uint64_t layer4_conv_qkv_weight_fnv1a64;
    uint64_t layer4_conv_qkv_weight_bytes_read;
    uint64_t layer4_postconv_q_scaled_fnv1a64;
    uint64_t last_layer4_postconv_q_scaled_fnv1a64;
    uint64_t layer4_postconv_k_norm_fnv1a64;
    uint64_t last_layer4_postconv_k_norm_fnv1a64;
    uint64_t layer4_postconv_value_fnv1a64;
    uint64_t last_layer4_postconv_value_fnv1a64;
    uint64_t layer4_gate_g_fnv1a64;
    uint64_t last_layer4_gate_g_fnv1a64;
    uint64_t layer4_gate_beta_fnv1a64;
    uint64_t last_layer4_gate_beta_fnv1a64;
    uint64_t layer4_a_log_fnv1a64;
    uint64_t layer4_dt_bias_fnv1a64;
    uint64_t layer4_gate_weight_bytes_read;
    uint64_t layer4_core_rows_fnv1a64;
    uint64_t last_layer4_core_rows_fnv1a64;
    uint64_t layer4_core_final_state_fnv1a64;
    uint64_t layer4_gated_rmsnorm_fnv1a64;
    uint64_t last_layer4_gated_rmsnorm_fnv1a64;
    uint64_t layer4_linear_norm_weight_fnv1a64;
    uint64_t layer4_linear_norm_weight_bytes_read;
    uint64_t layer4_out_projection_fnv1a64;
    uint64_t last_layer4_out_projection_fnv1a64;
    uint64_t layer4_out_projection_weight_fnv1a64;
    uint64_t layer4_out_projection_weight_bytes_read;
    uint64_t layer4_residual_hidden_fnv1a64;
    uint64_t last_layer4_residual_hidden_fnv1a64;
    uint64_t layer4_post_attention_rmsnorm_fnv1a64;
    uint64_t last_layer4_post_attention_rmsnorm_fnv1a64;
    uint64_t layer4_post_attention_norm_weight_fnv1a64;
    uint64_t layer4_post_attention_norm_weight_bytes_read;
    uint64_t layer4_moe_router_logits_fnv1a64;
    uint64_t last_layer4_moe_router_logits_fnv1a64;
    uint64_t layer4_moe_router_topk_ids_fnv1a64;
    uint64_t last_layer4_moe_router_topk_ids_fnv1a64;
    uint64_t layer4_moe_router_topk_weights_fnv1a64;
    uint64_t last_layer4_moe_router_topk_weights_fnv1a64;
    uint64_t layer4_moe_router_weight_fnv1a64;
    uint64_t layer4_moe_router_weight_bytes_read;
    uint64_t layer4_moe_expert_routed_fnv1a64;
    uint64_t last_layer4_moe_expert_routed_fnv1a64;
    uint64_t layer4_moe_expert_selected_ids_fnv1a64;
    uint64_t layer4_moe_expert_gate_up_weight_fnv1a64;
    uint64_t layer4_moe_expert_down_weight_fnv1a64;
    uint64_t layer4_moe_expert_weight_bytes_read;
    uint64_t layer4_moe_shared_expert_fnv1a64;
    uint64_t last_layer4_moe_shared_expert_fnv1a64;
    uint64_t layer4_moe_combined_fnv1a64;
    uint64_t last_layer4_moe_combined_fnv1a64;
    uint64_t layer4_moe_shared_gate_fnv1a64;
    uint64_t last_layer4_moe_shared_gate_fnv1a64;
    uint64_t layer4_moe_shared_gate_weight_fnv1a64;
    uint64_t layer4_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t layer4_moe_shared_up_proj_weight_fnv1a64;
    uint64_t layer4_moe_shared_down_weight_fnv1a64;
    uint64_t layer4_moe_shared_expert_weight_bytes_read;
    uint64_t layer4_output_residual_fnv1a64;
    uint64_t last_layer4_output_residual_fnv1a64;
    uint64_t layer5_input_norm_fnv1a64;
    uint64_t last_layer5_input_norm_fnv1a64;
    uint64_t layer5_input_norm_weight_fnv1a64;
    uint64_t layer5_input_norm_weight_bytes_read;
    uint64_t layer5_qkv_projection_fnv1a64;
    uint64_t last_layer5_qkv_projection_fnv1a64;
    uint64_t layer5_qkv_projection_weight_fnv1a64;
    uint64_t layer5_qkv_projection_weight_bytes_read;
    uint64_t layer5_zab_projection_fnv1a64;
    uint64_t last_layer5_zab_projection_fnv1a64;
    uint64_t layer5_z_projection_weight_fnv1a64;
    uint64_t layer5_a_projection_weight_fnv1a64;
    uint64_t layer5_b_projection_weight_fnv1a64;
    uint64_t layer5_zab_projection_weight_bytes_read;
    uint64_t layer5_conv_qkv_fnv1a64;
    uint64_t last_layer5_conv_qkv_fnv1a64;
    uint64_t layer5_conv_qkv_weight_fnv1a64;
    uint64_t layer5_conv_qkv_weight_bytes_read;
    uint64_t layer5_postconv_q_scaled_fnv1a64;
    uint64_t last_layer5_postconv_q_scaled_fnv1a64;
    uint64_t layer5_postconv_k_norm_fnv1a64;
    uint64_t last_layer5_postconv_k_norm_fnv1a64;
    uint64_t layer5_postconv_value_fnv1a64;
    uint64_t last_layer5_postconv_value_fnv1a64;
    uint64_t layer5_gate_g_fnv1a64;
    uint64_t last_layer5_gate_g_fnv1a64;
    uint64_t layer5_gate_beta_fnv1a64;
    uint64_t last_layer5_gate_beta_fnv1a64;
    uint64_t layer5_a_log_fnv1a64;
    uint64_t layer5_dt_bias_fnv1a64;
    uint64_t layer5_gate_weight_bytes_read;
    uint64_t layer5_core_rows_fnv1a64;
    uint64_t last_layer5_core_rows_fnv1a64;
    uint64_t layer5_core_final_state_fnv1a64;
    uint64_t layer5_gated_rmsnorm_fnv1a64;
    uint64_t last_layer5_gated_rmsnorm_fnv1a64;
    uint64_t layer5_linear_norm_weight_fnv1a64;
    uint64_t layer5_linear_norm_weight_bytes_read;
    uint64_t layer5_out_projection_fnv1a64;
    uint64_t last_layer5_out_projection_fnv1a64;
    uint64_t layer5_out_projection_weight_fnv1a64;
    uint64_t layer5_out_projection_weight_bytes_read;
    uint64_t layer5_residual_hidden_fnv1a64;
    uint64_t last_layer5_residual_hidden_fnv1a64;
    uint64_t layer5_post_attention_rmsnorm_fnv1a64;
    uint64_t last_layer5_post_attention_rmsnorm_fnv1a64;
    uint64_t layer5_post_attention_norm_weight_fnv1a64;
    uint64_t layer5_post_attention_norm_weight_bytes_read;
    uint64_t layer5_moe_router_logits_fnv1a64;
    uint64_t last_layer5_moe_router_logits_fnv1a64;
    uint64_t layer5_moe_router_topk_ids_fnv1a64;
    uint64_t last_layer5_moe_router_topk_ids_fnv1a64;
    uint64_t layer5_moe_router_topk_weights_fnv1a64;
    uint64_t last_layer5_moe_router_topk_weights_fnv1a64;
    uint64_t layer5_moe_router_weight_fnv1a64;
    uint64_t layer5_moe_router_weight_bytes_read;
    uint64_t layer5_moe_expert_routed_fnv1a64;
    uint64_t last_layer5_moe_expert_routed_fnv1a64;
    uint64_t layer5_moe_expert_selected_ids_fnv1a64;
    uint64_t layer5_moe_expert_gate_up_weight_fnv1a64;
    uint64_t layer5_moe_expert_down_weight_fnv1a64;
    uint64_t layer5_moe_expert_weight_bytes_read;
    uint64_t layer5_moe_shared_expert_fnv1a64;
    uint64_t last_layer5_moe_shared_expert_fnv1a64;
    uint64_t layer5_moe_combined_fnv1a64;
    uint64_t last_layer5_moe_combined_fnv1a64;
    uint64_t layer5_moe_shared_gate_fnv1a64;
    uint64_t last_layer5_moe_shared_gate_fnv1a64;
    uint64_t layer5_moe_shared_gate_weight_fnv1a64;
    uint64_t layer5_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t layer5_moe_shared_up_proj_weight_fnv1a64;
    uint64_t layer5_moe_shared_down_weight_fnv1a64;
    uint64_t layer5_moe_shared_expert_weight_bytes_read;
    uint64_t layer5_output_residual_fnv1a64;
    uint64_t last_layer5_output_residual_fnv1a64;
    uint64_t layer6_input_norm_fnv1a64;
    uint64_t last_layer6_input_norm_fnv1a64;
    uint64_t layer6_input_norm_weight_fnv1a64;
    uint64_t layer6_input_norm_weight_bytes_read;
    uint64_t layer6_qkv_projection_fnv1a64;
    uint64_t last_layer6_qkv_projection_fnv1a64;
    uint64_t layer6_qkv_projection_weight_fnv1a64;
    uint64_t layer6_qkv_projection_weight_bytes_read;
    uint64_t layer6_zab_projection_fnv1a64;
    uint64_t last_layer6_zab_projection_fnv1a64;
    uint64_t layer6_z_projection_weight_fnv1a64;
    uint64_t layer6_a_projection_weight_fnv1a64;
    uint64_t layer6_b_projection_weight_fnv1a64;
    uint64_t layer6_zab_projection_weight_bytes_read;
    uint64_t layer6_conv_qkv_fnv1a64;
    uint64_t last_layer6_conv_qkv_fnv1a64;
    uint64_t layer6_conv_qkv_weight_fnv1a64;
    uint64_t layer6_conv_qkv_weight_bytes_read;
    uint64_t layer6_postconv_q_scaled_fnv1a64;
    uint64_t last_layer6_postconv_q_scaled_fnv1a64;
    uint64_t layer6_postconv_k_norm_fnv1a64;
    uint64_t last_layer6_postconv_k_norm_fnv1a64;
    uint64_t layer6_postconv_value_fnv1a64;
    uint64_t last_layer6_postconv_value_fnv1a64;
    uint64_t layer6_gate_g_fnv1a64;
    uint64_t last_layer6_gate_g_fnv1a64;
    uint64_t layer6_gate_beta_fnv1a64;
    uint64_t last_layer6_gate_beta_fnv1a64;
    uint64_t layer6_a_log_fnv1a64;
    uint64_t layer6_dt_bias_fnv1a64;
    uint64_t layer6_gate_weight_bytes_read;
    uint64_t layer6_core_rows_fnv1a64;
    uint64_t last_layer6_core_rows_fnv1a64;
    uint64_t layer6_core_final_state_fnv1a64;
    uint64_t layer6_gated_rmsnorm_fnv1a64;
    uint64_t last_layer6_gated_rmsnorm_fnv1a64;
    uint64_t layer6_linear_norm_weight_fnv1a64;
    uint64_t layer6_linear_norm_weight_bytes_read;
    uint64_t layer6_out_projection_fnv1a64;
    uint64_t last_layer6_out_projection_fnv1a64;
    uint64_t layer6_out_projection_weight_fnv1a64;
    uint64_t layer6_out_projection_weight_bytes_read;
    uint64_t layer6_residual_hidden_fnv1a64;
    uint64_t last_layer6_residual_hidden_fnv1a64;
    uint64_t layer6_post_attention_rmsnorm_fnv1a64;
    uint64_t last_layer6_post_attention_rmsnorm_fnv1a64;
    uint64_t layer6_post_attention_norm_weight_fnv1a64;
    uint64_t layer6_post_attention_norm_weight_bytes_read;
    uint64_t layer6_moe_router_logits_fnv1a64;
    uint64_t last_layer6_moe_router_logits_fnv1a64;
    uint64_t layer6_moe_router_topk_ids_fnv1a64;
    uint64_t last_layer6_moe_router_topk_ids_fnv1a64;
    uint64_t layer6_moe_router_topk_weights_fnv1a64;
    uint64_t last_layer6_moe_router_topk_weights_fnv1a64;
    uint64_t layer6_moe_router_weight_fnv1a64;
    uint64_t layer6_moe_router_weight_bytes_read;
    uint64_t layer6_moe_expert_routed_fnv1a64;
    uint64_t last_layer6_moe_expert_routed_fnv1a64;
    uint64_t layer6_moe_expert_selected_ids_fnv1a64;
    uint64_t layer6_moe_expert_gate_up_weight_fnv1a64;
    uint64_t layer6_moe_expert_down_weight_fnv1a64;
    uint64_t layer6_moe_expert_weight_bytes_read;
    uint64_t layer6_moe_shared_expert_fnv1a64;
    uint64_t last_layer6_moe_shared_expert_fnv1a64;
    uint64_t layer6_moe_combined_fnv1a64;
    uint64_t last_layer6_moe_combined_fnv1a64;
    uint64_t layer6_moe_shared_gate_fnv1a64;
    uint64_t last_layer6_moe_shared_gate_fnv1a64;
    uint64_t layer6_moe_shared_gate_weight_fnv1a64;
    uint64_t layer6_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t layer6_moe_shared_up_proj_weight_fnv1a64;
    uint64_t layer6_moe_shared_down_weight_fnv1a64;
    uint64_t layer6_moe_shared_expert_weight_bytes_read;
    uint64_t layer6_output_residual_fnv1a64;
    uint64_t last_layer6_output_residual_fnv1a64;
    uint64_t layer7_input_norm_fnv1a64;
    uint64_t last_layer7_input_norm_fnv1a64;
    uint64_t layer7_input_norm_weight_fnv1a64;
    uint64_t layer7_input_norm_weight_bytes_read;
    uint64_t layer7_qkv_projection_fnv1a64;
    uint64_t last_layer7_qkv_projection_fnv1a64;
    uint64_t layer7_qkv_projection_weight_fnv1a64;
    uint64_t layer7_qkv_projection_weight_bytes_read;
    uint64_t layer7_qk_norm_fnv1a64;
    uint64_t last_layer7_qk_norm_fnv1a64;
    uint64_t layer7_q_norm_weight_fnv1a64;
    uint64_t layer7_k_norm_weight_fnv1a64;
    uint64_t layer7_qk_norm_weight_bytes_read;
    uint64_t layer7_rope_fnv1a64;
    uint64_t last_layer7_rope_fnv1a64;
    uint64_t layer7_attention_fnv1a64;
    uint64_t last_layer7_attention_fnv1a64;
    uint64_t layer7_output_projection_fnv1a64;
    uint64_t last_layer7_output_projection_fnv1a64;
    uint64_t layer7_output_projection_weight_fnv1a64;
    uint64_t layer7_output_projection_weight_bytes_read;
    uint64_t layer7_residual_hidden_fnv1a64;
    uint64_t last_layer7_residual_hidden_fnv1a64;
    uint64_t layer7_post_attention_rmsnorm_fnv1a64;
    uint64_t last_layer7_post_attention_rmsnorm_fnv1a64;
    uint64_t layer7_post_attention_norm_weight_fnv1a64;
    uint64_t layer7_post_attention_norm_weight_bytes_read;
    uint64_t layer7_moe_router_logits_fnv1a64;
    uint64_t last_layer7_moe_router_logits_fnv1a64;
    uint64_t layer7_moe_router_topk_ids_fnv1a64;
    uint64_t last_layer7_moe_router_topk_ids_fnv1a64;
    uint64_t layer7_moe_router_topk_weights_fnv1a64;
    uint64_t last_layer7_moe_router_topk_weights_fnv1a64;
    uint64_t layer7_moe_router_weight_fnv1a64;
    uint64_t layer7_moe_router_weight_bytes_read;
    uint64_t layer7_moe_expert_routed_fnv1a64;
    uint64_t last_layer7_moe_expert_routed_fnv1a64;
    uint64_t layer7_moe_expert_selected_ids_fnv1a64;
    uint64_t layer7_moe_expert_gate_up_weight_fnv1a64;
    uint64_t layer7_moe_expert_down_weight_fnv1a64;
    uint64_t layer7_moe_expert_weight_bytes_read;
    uint64_t layer7_moe_shared_expert_fnv1a64;
    uint64_t last_layer7_moe_shared_expert_fnv1a64;
    uint64_t layer7_moe_combined_fnv1a64;
    uint64_t last_layer7_moe_combined_fnv1a64;
    uint64_t layer7_moe_shared_gate_fnv1a64;
    uint64_t last_layer7_moe_shared_gate_fnv1a64;
    uint64_t layer7_moe_shared_gate_weight_fnv1a64;
    uint64_t layer7_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t layer7_moe_shared_up_proj_weight_fnv1a64;
    uint64_t layer7_moe_shared_down_weight_fnv1a64;
    uint64_t layer7_moe_shared_expert_weight_bytes_read;
    uint64_t layer7_output_residual_fnv1a64;
    uint64_t last_layer7_output_residual_fnv1a64;
    uint64_t layer8_input_norm_fnv1a64;
    uint64_t last_layer8_input_norm_fnv1a64;
    uint64_t layer8_input_norm_weight_fnv1a64;
    uint64_t layer8_input_norm_weight_bytes_read;
    uint64_t layer8_qkv_projection_fnv1a64;
    uint64_t last_layer8_qkv_projection_fnv1a64;
    uint64_t layer8_qkv_projection_weight_fnv1a64;
    uint64_t layer8_qkv_projection_weight_bytes_read;
    uint64_t layer8_zab_projection_fnv1a64;
    uint64_t last_layer8_zab_projection_fnv1a64;
    uint64_t layer8_z_projection_weight_fnv1a64;
    uint64_t layer8_a_projection_weight_fnv1a64;
    uint64_t layer8_b_projection_weight_fnv1a64;
    uint64_t layer8_zab_projection_weight_bytes_read;
    uint64_t layer8_conv_qkv_fnv1a64;
    uint64_t last_layer8_conv_qkv_fnv1a64;
    uint64_t layer8_conv_qkv_weight_fnv1a64;
    uint64_t layer8_conv_qkv_weight_bytes_read;
    uint64_t layer8_postconv_q_scaled_fnv1a64;
    uint64_t last_layer8_postconv_q_scaled_fnv1a64;
    uint64_t layer8_postconv_k_norm_fnv1a64;
    uint64_t last_layer8_postconv_k_norm_fnv1a64;
    uint64_t layer8_postconv_value_fnv1a64;
    uint64_t last_layer8_postconv_value_fnv1a64;
    uint64_t layer8_gate_g_fnv1a64;
    uint64_t last_layer8_gate_g_fnv1a64;
    uint64_t layer8_gate_beta_fnv1a64;
    uint64_t last_layer8_gate_beta_fnv1a64;
    uint64_t layer8_a_log_fnv1a64;
    uint64_t layer8_dt_bias_fnv1a64;
    uint64_t layer8_gate_weight_bytes_read;
    uint64_t layer8_core_rows_fnv1a64;
    uint64_t last_layer8_core_rows_fnv1a64;
    uint64_t layer8_core_final_state_fnv1a64;
    uint64_t layer8_gated_rmsnorm_fnv1a64;
    uint64_t last_layer8_gated_rmsnorm_fnv1a64;
    uint64_t layer8_linear_norm_weight_fnv1a64;
    uint64_t layer8_linear_norm_weight_bytes_read;
    uint64_t layer8_out_projection_fnv1a64;
    uint64_t last_layer8_out_projection_fnv1a64;
    uint64_t layer8_out_projection_weight_fnv1a64;
    uint64_t layer8_out_projection_weight_bytes_read;
    uint64_t layer8_residual_hidden_fnv1a64;
    uint64_t last_layer8_residual_hidden_fnv1a64;
    uint64_t layer8_post_attention_rmsnorm_fnv1a64;
    uint64_t last_layer8_post_attention_rmsnorm_fnv1a64;
    uint64_t layer8_post_attention_norm_weight_fnv1a64;
    uint64_t layer8_post_attention_norm_weight_bytes_read;
    uint64_t layer8_moe_router_logits_fnv1a64;
    uint64_t last_layer8_moe_router_logits_fnv1a64;
    uint64_t layer8_moe_router_topk_ids_fnv1a64;
    uint64_t last_layer8_moe_router_topk_ids_fnv1a64;
    uint64_t layer8_moe_router_topk_weights_fnv1a64;
    uint64_t last_layer8_moe_router_topk_weights_fnv1a64;
    uint64_t layer8_moe_router_weight_fnv1a64;
    uint64_t layer8_moe_router_weight_bytes_read;
    uint64_t layer8_moe_expert_routed_fnv1a64;
    uint64_t last_layer8_moe_expert_routed_fnv1a64;
    uint64_t layer8_moe_expert_selected_ids_fnv1a64;
    uint64_t layer8_moe_expert_gate_up_weight_fnv1a64;
    uint64_t layer8_moe_expert_down_weight_fnv1a64;
    uint64_t layer8_moe_expert_weight_bytes_read;
    uint64_t layer8_moe_shared_expert_fnv1a64;
    uint64_t last_layer8_moe_shared_expert_fnv1a64;
    uint64_t layer8_moe_combined_fnv1a64;
    uint64_t last_layer8_moe_combined_fnv1a64;
    uint64_t layer8_moe_shared_gate_fnv1a64;
    uint64_t last_layer8_moe_shared_gate_fnv1a64;
    uint64_t layer8_moe_shared_gate_weight_fnv1a64;
    uint64_t layer8_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t layer8_moe_shared_up_proj_weight_fnv1a64;
    uint64_t layer8_moe_shared_down_weight_fnv1a64;
    uint64_t layer8_moe_shared_expert_weight_bytes_read;
    uint64_t layer8_output_residual_fnv1a64;
    uint64_t last_layer8_output_residual_fnv1a64;
    uint64_t layer8_handoff_digest_fnv1a64;
    uint64_t layer8_handoff_source_hidden_fnv1a64;
    uint64_t layer8_handoff_next_input_fnv1a64;
    uint64_t layer8_handoff_output_hidden_fnv1a64;
    uint64_t layer9_loop_body_digest_fnv1a64;
    uint64_t layer9_loop_body_input_hidden_fnv1a64;
    uint64_t layer9_loop_body_output_hidden_fnv1a64;
    uint64_t layer9_input_norm_digest_fnv1a64;
    uint64_t layer9_input_norm_input_hidden_fnv1a64;
    uint64_t layer9_input_norm_fnv1a64;
    uint64_t last_layer9_input_norm_fnv1a64;
    uint64_t layer9_input_norm_weight_fnv1a64;
    uint64_t layer9_input_norm_weight_bytes_read;
    uint64_t layer9_input_norm_output_hidden_fnv1a64;
    uint64_t layer9_qkv_projection_fnv1a64;
    uint64_t last_layer9_qkv_projection_fnv1a64;
    uint64_t layer9_qkv_projection_weight_fnv1a64;
    uint64_t layer9_qkv_projection_weight_bytes_read;
    uint64_t layer9_zab_projection_fnv1a64;
    uint64_t last_layer9_zab_projection_fnv1a64;
    uint64_t layer9_z_projection_weight_fnv1a64;
    uint64_t layer9_a_projection_weight_fnv1a64;
    uint64_t layer9_b_projection_weight_fnv1a64;
    uint64_t layer9_zab_projection_weight_bytes_read;
    uint64_t layer9_conv_qkv_fnv1a64;
    uint64_t last_layer9_conv_qkv_fnv1a64;
    uint64_t layer9_conv_qkv_weight_fnv1a64;
    uint64_t layer9_conv_qkv_weight_bytes_read;
    uint64_t layer9_postconv_q_scaled_fnv1a64;
    uint64_t last_layer9_postconv_q_scaled_fnv1a64;
    uint64_t layer9_postconv_k_norm_fnv1a64;
    uint64_t last_layer9_postconv_k_norm_fnv1a64;
    uint64_t layer9_postconv_value_fnv1a64;
    uint64_t last_layer9_postconv_value_fnv1a64;
    uint64_t layer9_gate_g_fnv1a64;
    uint64_t last_layer9_gate_g_fnv1a64;
    uint64_t layer9_gate_beta_fnv1a64;
    uint64_t last_layer9_gate_beta_fnv1a64;
    uint64_t layer9_a_log_fnv1a64;
    uint64_t layer9_dt_bias_fnv1a64;
    uint64_t layer9_gate_weight_bytes_read;
    uint64_t layer9_core_rows_fnv1a64;
    uint64_t last_layer9_core_rows_fnv1a64;
    uint64_t layer9_core_final_state_fnv1a64;
    uint64_t layer9_gated_rmsnorm_fnv1a64;
    uint64_t last_layer9_gated_rmsnorm_fnv1a64;
    uint64_t layer9_linear_norm_weight_fnv1a64;
    uint64_t layer9_linear_norm_weight_bytes_read;
    uint64_t layer9_out_projection_fnv1a64;
    uint64_t last_layer9_out_projection_fnv1a64;
    uint64_t layer9_out_projection_weight_fnv1a64;
    uint64_t layer9_out_projection_weight_bytes_read;
    uint64_t layer9_residual_hidden_fnv1a64;
    uint64_t last_layer9_residual_hidden_fnv1a64;
    uint64_t layer9_post_attention_rmsnorm_fnv1a64;
    uint64_t last_layer9_post_attention_rmsnorm_fnv1a64;
    uint64_t layer9_post_attention_norm_weight_fnv1a64;
    uint64_t layer9_post_attention_norm_weight_bytes_read;
    uint64_t layer9_moe_router_logits_fnv1a64;
    uint64_t last_layer9_moe_router_logits_fnv1a64;
    uint64_t layer9_moe_router_topk_ids_fnv1a64;
    uint64_t last_layer9_moe_router_topk_ids_fnv1a64;
    uint64_t layer9_moe_router_topk_weights_fnv1a64;
    uint64_t last_layer9_moe_router_topk_weights_fnv1a64;
    uint64_t layer9_moe_router_weight_fnv1a64;
    uint64_t layer9_moe_router_weight_bytes_read;
    uint64_t layer9_moe_expert_routed_fnv1a64;
    uint64_t last_layer9_moe_expert_routed_fnv1a64;
    uint64_t layer9_moe_expert_selected_ids_fnv1a64;
    uint64_t layer9_moe_expert_gate_up_weight_fnv1a64;
    uint64_t layer9_moe_expert_down_weight_fnv1a64;
    uint64_t layer9_moe_expert_weight_bytes_read;
    uint64_t layer9_moe_shared_expert_fnv1a64;
    uint64_t last_layer9_moe_shared_expert_fnv1a64;
    uint64_t layer9_moe_combined_fnv1a64;
    uint64_t last_layer9_moe_combined_fnv1a64;
    uint64_t layer9_moe_shared_gate_fnv1a64;
    uint64_t last_layer9_moe_shared_gate_fnv1a64;
    uint64_t layer9_moe_shared_gate_weight_fnv1a64;
    uint64_t layer9_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t layer9_moe_shared_up_proj_weight_fnv1a64;
    uint64_t layer9_moe_shared_down_weight_fnv1a64;
    uint64_t layer9_moe_shared_expert_weight_bytes_read;
    uint64_t layer9_output_residual_fnv1a64;
    uint64_t last_layer9_output_residual_fnv1a64;
    uint64_t layer10_input_norm_fnv1a64;
    uint64_t last_layer10_input_norm_fnv1a64;
    uint64_t layer10_input_norm_weight_fnv1a64;
    uint64_t layer10_input_norm_weight_bytes_read;
    uint64_t layer10_qkv_projection_fnv1a64;
    uint64_t last_layer10_qkv_projection_fnv1a64;
    uint64_t layer10_qkv_projection_weight_fnv1a64;
    uint64_t layer10_qkv_projection_weight_bytes_read;
    uint64_t layer10_zab_projection_fnv1a64;
    uint64_t last_layer10_zab_projection_fnv1a64;
    uint64_t layer10_z_projection_weight_fnv1a64;
    uint64_t layer10_a_projection_weight_fnv1a64;
    uint64_t layer10_b_projection_weight_fnv1a64;
    uint64_t layer10_zab_projection_weight_bytes_read;
    uint64_t layer10_conv_qkv_fnv1a64;
    uint64_t last_layer10_conv_qkv_fnv1a64;
    uint64_t layer10_conv_qkv_weight_fnv1a64;
    uint64_t layer10_conv_qkv_weight_bytes_read;
    uint64_t layer10_postconv_q_scaled_fnv1a64;
    uint64_t last_layer10_postconv_q_scaled_fnv1a64;
    uint64_t layer10_postconv_k_norm_fnv1a64;
    uint64_t last_layer10_postconv_k_norm_fnv1a64;
    uint64_t layer10_postconv_value_fnv1a64;
    uint64_t last_layer10_postconv_value_fnv1a64;
    uint64_t layer10_gate_g_fnv1a64;
    uint64_t last_layer10_gate_g_fnv1a64;
    uint64_t layer10_gate_beta_fnv1a64;
    uint64_t last_layer10_gate_beta_fnv1a64;
    uint64_t layer10_a_log_fnv1a64;
    uint64_t layer10_dt_bias_fnv1a64;
    uint64_t layer10_gate_weight_bytes_read;
    uint64_t layer10_core_rows_fnv1a64;
    uint64_t last_layer10_core_rows_fnv1a64;
    uint64_t layer10_core_final_state_fnv1a64;
    uint64_t layer10_gated_rmsnorm_fnv1a64;
    uint64_t last_layer10_gated_rmsnorm_fnv1a64;
    uint64_t layer10_linear_norm_weight_fnv1a64;
    uint64_t layer10_linear_norm_weight_bytes_read;
    uint64_t layer10_out_projection_fnv1a64;
    uint64_t last_layer10_out_projection_fnv1a64;
    uint64_t layer10_out_projection_weight_fnv1a64;
    uint64_t layer10_out_projection_weight_bytes_read;
    uint64_t layer10_residual_hidden_fnv1a64;
    uint64_t last_layer10_residual_hidden_fnv1a64;
    uint64_t layer10_post_attention_rmsnorm_fnv1a64;
    uint64_t last_layer10_post_attention_rmsnorm_fnv1a64;
    uint64_t layer10_post_attention_norm_weight_fnv1a64;
    uint64_t layer10_post_attention_norm_weight_bytes_read;
    uint64_t qwen36_layer_descriptor_helper_fnv1a64;
    uint64_t qwen36_descriptor_layer_batch_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_batch_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_batch_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_batch_input_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_batch_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_loop_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_loop_source_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_loop_next_input_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_loop_body_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_loop_body_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_loop_body_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_input_norm_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_input_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_input_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_qkv_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_qkv_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_qkv_projection_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_qkv_projection_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_qkv_projection_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_qkv_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_qkv_projection_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_qk_norm_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_qk_norm_qkv_projection_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_qk_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_qk_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_q_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_k_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_qk_norm_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_qk_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_rope_attention_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_rope_attention_qk_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_rope_attention_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_rope_attention_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_rope_attention_context_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_rope_attention_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_projection_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_projection_rope_attention_context_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_projection_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_output_projection_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_projection_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_projection_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_projection_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_residual_hidden_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_residual_hidden_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_residual_hidden_output_projection_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_residual_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_residual_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_residual_hidden_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_post_attention_rmsnorm_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_post_attention_rmsnorm_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_post_attention_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_post_attention_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_post_attention_rmsnorm_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_router_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_router_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_router_logits_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_moe_router_logits_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_moe_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_moe_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_router_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_router_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_router_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_moe_expert_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_moe_expert_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_moe_expert_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_selected_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_gate_up_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_down_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_moe_shared_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_moe_shared_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_gate_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_moe_shared_gate_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_gate_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_up_proj_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_down_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_residual_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_residual_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_residual_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_output_residual_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_last_output_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_output_residual_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_handoff_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_handoff_source_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_handoff_next_input_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_handoff_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_loop_body_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_loop_body_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_loop_body_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_input_norm_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_input_norm_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_input_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_input_norm_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_input_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_qkv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_last_qkv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_qkv_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_qkv_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_zab_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_conv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_gate_g_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_core_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_core_final_state_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_gated_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_linear_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_out_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_out_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_last_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_post_attention_rmsnorm_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_post_attention_rmsnorm_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_post_attention_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_post_attention_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_post_attention_rmsnorm_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_logits_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_moe_router_logits_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_moe_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_moe_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_moe_expert_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_moe_expert_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_moe_expert_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_selected_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_gate_up_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_down_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_moe_shared_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_moe_shared_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_gate_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_moe_shared_gate_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_gate_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_up_proj_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_down_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_output_residual_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_last_output_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_handoff_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_handoff_source_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_handoff_next_input_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_handoff_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_loop_body_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_loop_body_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_loop_body_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_input_norm_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_input_norm_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_input_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_input_norm_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_input_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_qkv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_last_qkv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_qkv_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_qkv_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_zab_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_conv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_gate_g_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_core_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_core_final_state_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_gated_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_linear_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_out_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_out_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_last_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_post_attention_rmsnorm_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_post_attention_rmsnorm_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_post_attention_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_post_attention_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_post_attention_rmsnorm_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_logits_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_moe_router_logits_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_moe_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_moe_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_moe_expert_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_moe_expert_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_moe_expert_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_selected_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_gate_up_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_down_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_moe_shared_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_moe_shared_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_moe_shared_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_gate_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_moe_shared_gate_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_gate_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_up_proj_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_down_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_output_residual_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_last_output_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_handoff_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_handoff_source_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_handoff_next_input_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_handoff_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_loop_body_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_loop_body_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_loop_body_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_input_norm_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_input_norm_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_last_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_input_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_input_norm_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_input_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_qkv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_last_qkv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_qkv_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_qkv_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_zab_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_conv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_gate_g_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_core_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_core_final_state_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_gated_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_linear_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_out_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_out_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_last_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_post_attention_rmsnorm_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_post_attention_rmsnorm_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_last_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_post_attention_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_post_attention_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_post_attention_rmsnorm_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_logits_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_last_moe_router_logits_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_last_moe_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_last_moe_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_last_moe_expert_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_last_moe_expert_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_last_moe_expert_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_selected_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_gate_up_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_down_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_last_moe_shared_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_last_moe_shared_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_last_moe_shared_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_gate_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_last_moe_shared_gate_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_gate_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_up_proj_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_down_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_handoff_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_handoff_source_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_handoff_next_input_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_handoff_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_loop_body_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_loop_body_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_loop_body_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_input_norm_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_input_norm_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_input_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_input_norm_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_input_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_qkv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_last_qkv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_qkv_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_qkv_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_qk_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_last_qk_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_q_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_k_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_qk_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_rope_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_last_rope_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_context_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_out_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_last_out_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_out_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_out_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_last_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_post_attention_rmsnorm_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_post_attention_rmsnorm_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_post_attention_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_post_attention_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_post_attention_rmsnorm_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_logits_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_moe_router_logits_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_moe_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_moe_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_moe_expert_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_moe_expert_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_moe_expert_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_selected_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_gate_up_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_down_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_moe_shared_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_moe_shared_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_moe_shared_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_gate_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_last_moe_shared_gate_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_gate_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_up_proj_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_down_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_moe_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_handoff_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_handoff_source_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_handoff_next_input_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_handoff_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_loop_body_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_loop_body_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_loop_body_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_input_norm_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_input_norm_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_input_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_input_norm_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_input_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_input_norm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_qkv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_last_qkv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_qkv_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_qkv_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_zab_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_conv_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_gate_g_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_core_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_core_final_state_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_gated_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_linear_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_out_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_out_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_last_residual_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_post_attention_rmsnorm_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_post_attention_rmsnorm_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_post_attention_norm_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_post_attention_norm_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_post_attention_rmsnorm_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_logits_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_moe_router_logits_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_moe_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_moe_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_moe_expert_router_topk_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_moe_expert_router_topk_weights_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_moe_expert_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_selected_ids_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_gate_up_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_down_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_post_attention_rmsnorm_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_moe_shared_routed_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_moe_shared_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_moe_shared_combined_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_gate_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_last_moe_shared_gate_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_gate_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_gate_proj_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_up_proj_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_down_weight_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_weight_bytes_read;
    uint64_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_output_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_digest_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_input_hidden_fnv1a64;
    uint64_t qwen36_descriptor_layer_body_output_hidden_fnv1a64;
    size_t last_input_token_count;
    size_t input_embeddings_materialized_count;
    size_t layer0_input_norm_materialized_count;
    size_t layer0_qkv_projection_materialized_count;
    size_t layer0_zab_projection_materialized_count;
    size_t layer0_conv_qkv_materialized_count;
    size_t layer0_postconv_qkv_materialized_count;
    size_t layer0_gate_rows_materialized_count;
    size_t layer0_core_rows_materialized_count;
    size_t layer0_gated_rmsnorm_materialized_count;
    size_t layer0_out_projection_materialized_count;
    size_t layer0_residual_hidden_materialized_count;
    size_t layer0_post_attention_rmsnorm_materialized_count;
    size_t layer0_moe_router_materialized_count;
    size_t layer0_moe_expert_materialized_count;
    size_t layer0_moe_expert_selected_count;
    size_t layer0_moe_shared_expert_materialized_count;
    size_t layer0_output_residual_materialized_count;
    size_t layer1_input_norm_materialized_count;
    size_t layer1_qkv_projection_materialized_count;
    size_t layer1_zab_projection_materialized_count;
    size_t layer1_conv_qkv_materialized_count;
    size_t layer1_postconv_qkv_materialized_count;
    size_t layer1_gate_rows_materialized_count;
    size_t layer1_core_rows_materialized_count;
    size_t layer1_gated_rmsnorm_materialized_count;
    size_t layer1_out_projection_materialized_count;
    size_t layer1_residual_hidden_materialized_count;
    size_t layer1_post_attention_rmsnorm_materialized_count;
    size_t layer1_moe_router_materialized_count;
    size_t layer1_moe_expert_materialized_count;
    size_t layer1_moe_expert_selected_count;
    size_t layer1_moe_shared_expert_materialized_count;
    size_t layer1_output_residual_materialized_count;
    size_t layer2_input_norm_materialized_count;
    size_t layer2_qkv_projection_materialized_count;
    size_t layer2_zab_projection_materialized_count;
    size_t layer2_conv_qkv_materialized_count;
    size_t layer2_postconv_qkv_materialized_count;
    size_t layer2_gate_rows_materialized_count;
    size_t layer2_core_rows_materialized_count;
    size_t layer2_gated_rmsnorm_materialized_count;
    size_t layer2_out_projection_materialized_count;
    size_t layer2_residual_hidden_materialized_count;
    size_t layer2_post_attention_rmsnorm_materialized_count;
    size_t layer2_moe_router_materialized_count;
    size_t layer2_moe_expert_materialized_count;
    size_t layer2_moe_expert_selected_count;
    size_t layer2_moe_shared_expert_materialized_count;
    size_t layer2_output_residual_materialized_count;
    size_t layer3_input_norm_materialized_count;
    size_t layer3_qkv_projection_materialized_count;
    size_t layer3_qk_norm_materialized_count;
    size_t layer3_rope_attention_materialized_count;
    size_t layer3_output_projection_materialized_count;
    size_t layer3_residual_hidden_materialized_count;
    size_t layer3_post_attention_rmsnorm_materialized_count;
    size_t layer3_moe_router_materialized_count;
    size_t layer3_moe_expert_materialized_count;
    size_t layer3_moe_expert_selected_count;
    size_t layer3_moe_shared_expert_materialized_count;
    size_t layer3_output_residual_materialized_count;
    size_t layer4_input_norm_materialized_count;
    size_t layer4_qkv_projection_materialized_count;
    size_t layer4_zab_projection_materialized_count;
    size_t layer4_conv_qkv_materialized_count;
    size_t layer4_postconv_qkv_materialized_count;
    size_t layer4_gate_rows_materialized_count;
    size_t layer4_core_rows_materialized_count;
    size_t layer4_gated_rmsnorm_materialized_count;
    size_t layer4_out_projection_materialized_count;
    size_t layer4_residual_hidden_materialized_count;
    size_t layer4_post_attention_rmsnorm_materialized_count;
    size_t layer4_moe_router_materialized_count;
    size_t layer4_moe_expert_materialized_count;
    size_t layer4_moe_expert_selected_count;
    size_t layer4_moe_shared_expert_materialized_count;
    size_t layer4_output_residual_materialized_count;
    size_t layer5_input_norm_materialized_count;
    size_t layer5_qkv_projection_materialized_count;
    size_t layer5_zab_projection_materialized_count;
    size_t layer5_conv_qkv_materialized_count;
    size_t layer5_postconv_qkv_materialized_count;
    size_t layer5_gate_rows_materialized_count;
    size_t layer5_core_rows_materialized_count;
    size_t layer5_gated_rmsnorm_materialized_count;
    size_t layer5_out_projection_materialized_count;
    size_t layer5_residual_hidden_materialized_count;
    size_t layer5_post_attention_rmsnorm_materialized_count;
    size_t layer5_moe_router_materialized_count;
    size_t layer5_moe_expert_materialized_count;
    size_t layer5_moe_expert_selected_count;
    size_t layer5_moe_shared_expert_materialized_count;
    size_t layer5_output_residual_materialized_count;
    size_t layer6_input_norm_materialized_count;
    size_t layer6_qkv_projection_materialized_count;
    size_t layer6_zab_projection_materialized_count;
    size_t layer6_conv_qkv_materialized_count;
    size_t layer6_postconv_qkv_materialized_count;
    size_t layer6_gate_rows_materialized_count;
    size_t layer6_core_rows_materialized_count;
    size_t layer6_gated_rmsnorm_materialized_count;
    size_t layer6_out_projection_materialized_count;
    size_t layer6_residual_hidden_materialized_count;
    size_t layer6_post_attention_rmsnorm_materialized_count;
    size_t layer6_moe_router_materialized_count;
    size_t layer6_moe_expert_materialized_count;
    size_t layer6_moe_expert_selected_count;
    size_t layer6_moe_shared_expert_materialized_count;
    size_t layer6_output_residual_materialized_count;
    size_t layer7_input_norm_materialized_count;
    size_t layer7_qkv_projection_materialized_count;
    size_t layer7_qk_norm_materialized_count;
    size_t layer7_rope_materialized_count;
    size_t layer7_attention_materialized_count;
    size_t layer7_output_projection_materialized_count;
    size_t layer7_residual_hidden_materialized_count;
    size_t layer7_post_attention_rmsnorm_materialized_count;
    size_t layer7_moe_router_materialized_count;
    size_t layer7_moe_expert_materialized_count;
    size_t layer7_moe_expert_selected_count;
    size_t layer7_moe_shared_expert_materialized_count;
    size_t layer7_output_residual_materialized_count;
    size_t layer8_input_norm_materialized_count;
    size_t layer8_qkv_projection_materialized_count;
    size_t layer8_zab_projection_materialized_count;
    size_t layer8_conv_qkv_materialized_count;
    size_t layer8_postconv_qkv_materialized_count;
    size_t layer8_gate_rows_materialized_count;
    size_t layer8_core_rows_materialized_count;
    size_t layer8_gated_rmsnorm_materialized_count;
    size_t layer8_out_projection_materialized_count;
    size_t layer8_residual_hidden_materialized_count;
    size_t layer8_post_attention_rmsnorm_materialized_count;
    size_t layer8_moe_router_materialized_count;
    size_t layer8_moe_expert_materialized_count;
    size_t layer8_moe_expert_selected_count;
    size_t layer8_moe_shared_expert_materialized_count;
    size_t layer8_output_residual_materialized_count;
    size_t layer8_handoff_token_count;
    size_t layer9_loop_body_token_count;
    size_t layer9_input_norm_materialized_count;
    size_t layer9_qkv_projection_materialized_count;
    size_t layer9_zab_projection_materialized_count;
    size_t layer9_conv_qkv_materialized_count;
    size_t layer9_postconv_qkv_materialized_count;
    size_t layer9_gate_rows_materialized_count;
    size_t layer9_core_rows_materialized_count;
    size_t layer9_gated_rmsnorm_materialized_count;
    size_t layer9_out_projection_materialized_count;
    size_t layer9_residual_hidden_materialized_count;
    size_t layer9_post_attention_rmsnorm_materialized_count;
    size_t layer9_moe_router_materialized_count;
    size_t layer9_moe_expert_materialized_count;
    size_t layer9_moe_expert_selected_count;
    size_t layer9_moe_shared_expert_materialized_count;
    size_t layer9_output_residual_materialized_count;
    size_t layer10_input_norm_materialized_count;
    size_t layer10_qkv_projection_materialized_count;
    size_t layer10_zab_projection_materialized_count;
    size_t layer10_conv_qkv_materialized_count;
    size_t layer10_postconv_qkv_materialized_count;
    size_t layer10_gate_rows_materialized_count;
    size_t layer10_core_rows_materialized_count;
    size_t layer10_gated_rmsnorm_materialized_count;
    size_t layer10_out_projection_materialized_count;
    size_t layer10_residual_hidden_materialized_count;
    size_t layer10_post_attention_rmsnorm_materialized_count;
    size_t qwen36_layer_descriptor_helper_layer_count;
    size_t qwen36_descriptor_layer_batch_layer_count;
    size_t qwen36_descriptor_layer_batch_token_count;
    size_t qwen36_descriptor_layer_loop_token_count;
    size_t qwen36_descriptor_layer_body_repeated_loop_body_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_input_norm_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_input_norm_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_qkv_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_qkv_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_qk_norm_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_qk_norm_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_rope_attention_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_rope_attention_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_output_projection_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_output_projection_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_residual_hidden_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_residual_hidden_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_post_attention_rmsnorm_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_post_attention_rmsnorm_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_moe_router_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_moe_router_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_selected_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_output_residual_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_output_residual_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_handoff_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_loop_body_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_input_norm_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_input_norm_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_post_attention_rmsnorm_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_post_attention_rmsnorm_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_selected_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_handoff_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_loop_body_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_input_norm_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_input_norm_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_post_attention_rmsnorm_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_post_attention_rmsnorm_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_selected_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_handoff_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_loop_body_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_input_norm_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_input_norm_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_post_attention_rmsnorm_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_post_attention_rmsnorm_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_selected_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_handoff_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_loop_body_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_input_norm_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_input_norm_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_post_attention_rmsnorm_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_post_attention_rmsnorm_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_selected_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_handoff_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_loop_body_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_input_norm_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_input_norm_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_post_attention_rmsnorm_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_post_attention_rmsnorm_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_materialized_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_selected_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_token_count;
    size_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_materialized_count;
    size_t qwen36_descriptor_layer_body_token_count;
    size_t baseline_request_handoff_last_input_token_count;
    size_t baseline_request_handoff_last_output_token_capacity;
    uint32_t last_input_token;
    uint32_t baseline_request_handoff_last_input_token;
    uint32_t layer8_handoff_source_layer_index;
    uint32_t layer8_handoff_next_layer_index;
    uint32_t layer9_loop_body_layer_index;
    uint32_t layer9_loop_body_attention_kind;
    uint32_t qwen36_descriptor_layer_batch_first_layer_index;
    uint32_t qwen36_descriptor_layer_loop_source_layer_index;
    uint32_t qwen36_descriptor_layer_loop_next_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_loop_body_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_loop_body_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_handoff_source_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_handoff_next_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_loop_body_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_loop_body_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_input_norm_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_input_norm_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_attention_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_post_attention_rmsnorm_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_post_attention_rmsnorm_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_handoff_source_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_handoff_next_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_loop_body_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_loop_body_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_input_norm_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_input_norm_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_post_attention_rmsnorm_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_post_attention_rmsnorm_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_handoff_source_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_handoff_next_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_loop_body_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_loop_body_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_input_norm_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_input_norm_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_post_attention_rmsnorm_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_post_attention_rmsnorm_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_handoff_source_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_handoff_next_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_loop_body_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_loop_body_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_input_norm_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_input_norm_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_post_attention_rmsnorm_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_post_attention_rmsnorm_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_handoff_source_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_handoff_next_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_loop_body_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_loop_body_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_input_norm_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_input_norm_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_post_attention_rmsnorm_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_post_attention_rmsnorm_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_attention_kind;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_layer_index;
    uint32_t qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_attention_kind;
    uint32_t qwen36_descriptor_layer_body_layer_index;
    int token_embedding_materialized;
    int layer0_input_norm_applied;
    int layer0_qkv_projection_applied;
    int layer0_zab_projection_applied;
    int layer0_conv_qkv_applied;
    int layer0_postconv_qkv_applied;
    int layer0_gate_rows_applied;
    int layer0_core_rows_applied;
    int layer0_gated_rmsnorm_applied;
    int layer0_out_projection_applied;
    int layer0_residual_hidden_applied;
    int layer0_post_attention_rmsnorm_applied;
    int layer0_moe_router_applied;
    int layer0_moe_expert_applied;
    int layer0_moe_shared_expert_applied;
    int layer0_output_residual_applied;
    int layer1_input_norm_applied;
    int layer1_qkv_projection_applied;
    int layer1_zab_projection_applied;
    int layer1_conv_qkv_applied;
    int layer1_postconv_qkv_applied;
    int layer1_gate_rows_applied;
    int layer1_core_rows_applied;
    int layer1_gated_rmsnorm_applied;
    int layer1_out_projection_applied;
    int layer1_residual_hidden_applied;
    int layer1_post_attention_rmsnorm_applied;
    int layer1_moe_router_applied;
    int layer1_moe_expert_applied;
    int layer1_moe_shared_expert_applied;
    int layer1_output_residual_applied;
    int layer2_input_norm_applied;
    int layer2_qkv_projection_applied;
    int layer2_zab_projection_applied;
    int layer2_conv_qkv_applied;
    int layer2_postconv_qkv_applied;
    int layer2_gate_rows_applied;
    int layer2_core_rows_applied;
    int layer2_gated_rmsnorm_applied;
    int layer2_out_projection_applied;
    int layer2_residual_hidden_applied;
    int layer2_post_attention_rmsnorm_applied;
    int layer2_moe_router_applied;
    int layer2_moe_expert_applied;
    int layer2_moe_shared_expert_applied;
    int layer2_output_residual_applied;
    int layer3_input_norm_applied;
    int layer3_qkv_projection_applied;
    int layer3_qk_norm_applied;
    int layer3_rope_attention_applied;
    int layer3_output_projection_applied;
    int layer3_residual_hidden_applied;
    int layer3_post_attention_rmsnorm_applied;
    int layer3_moe_router_applied;
    int layer3_moe_expert_applied;
    int layer3_moe_shared_expert_applied;
    int layer3_output_residual_applied;
    int layer4_input_norm_applied;
    int layer4_qkv_projection_applied;
    int layer4_zab_projection_applied;
    int layer4_conv_qkv_applied;
    int layer4_postconv_qkv_applied;
    int layer4_gate_rows_applied;
    int layer4_core_rows_applied;
    int layer4_gated_rmsnorm_applied;
    int layer4_out_projection_applied;
    int layer4_residual_hidden_applied;
    int layer4_post_attention_rmsnorm_applied;
    int layer4_moe_router_applied;
    int layer4_moe_expert_applied;
    int layer4_moe_shared_expert_applied;
    int layer4_output_residual_applied;
    int layer5_input_norm_applied;
    int layer5_qkv_projection_applied;
    int layer5_zab_projection_applied;
    int layer5_conv_qkv_applied;
    int layer5_postconv_qkv_applied;
    int layer5_gate_rows_applied;
    int layer5_core_rows_applied;
    int layer5_gated_rmsnorm_applied;
    int layer5_out_projection_applied;
    int layer5_residual_hidden_applied;
    int layer5_post_attention_rmsnorm_applied;
    int layer5_moe_router_applied;
    int layer5_moe_expert_applied;
    int layer5_moe_shared_expert_applied;
    int layer5_output_residual_applied;
    int layer6_input_norm_applied;
    int layer6_qkv_projection_applied;
    int layer6_zab_projection_applied;
    int layer6_conv_qkv_applied;
    int layer6_postconv_qkv_applied;
    int layer6_gate_rows_applied;
    int layer6_core_rows_applied;
    int layer6_gated_rmsnorm_applied;
    int layer6_out_projection_applied;
    int layer6_residual_hidden_applied;
    int layer6_post_attention_rmsnorm_applied;
    int layer6_moe_router_applied;
    int layer6_moe_expert_applied;
    int layer6_moe_shared_expert_applied;
    int layer6_output_residual_applied;
    int layer7_input_norm_applied;
    int layer7_qkv_projection_applied;
    int layer7_qk_norm_applied;
    int layer7_rope_applied;
    int layer7_attention_applied;
    int layer7_output_projection_applied;
    int layer7_residual_hidden_applied;
    int layer7_post_attention_rmsnorm_applied;
    int layer7_moe_router_applied;
    int layer7_moe_expert_applied;
    int layer7_moe_shared_expert_applied;
    int layer7_output_residual_applied;
    int layer8_input_norm_applied;
    int layer8_qkv_projection_applied;
    int layer8_zab_projection_applied;
    int layer8_conv_qkv_applied;
    int layer8_postconv_qkv_applied;
    int layer8_gate_rows_applied;
    int layer8_core_rows_applied;
    int layer8_gated_rmsnorm_applied;
    int layer8_out_projection_applied;
    int layer8_residual_hidden_applied;
    int layer8_post_attention_rmsnorm_applied;
    int layer8_moe_router_applied;
    int layer8_moe_expert_applied;
    int layer8_moe_shared_expert_applied;
    int layer8_output_residual_applied;
    int layer8_handoff_applied;
    int layer9_loop_body_applied;
    int layer9_input_norm_applied;
    int layer9_qkv_projection_applied;
    int layer9_zab_projection_applied;
    int layer9_conv_qkv_applied;
    int layer9_postconv_qkv_applied;
    int layer9_gate_rows_applied;
    int layer9_core_rows_applied;
    int layer9_gated_rmsnorm_applied;
    int layer9_out_projection_applied;
    int layer9_residual_hidden_applied;
    int layer9_post_attention_rmsnorm_applied;
    int layer9_moe_router_applied;
    int layer9_moe_expert_applied;
    int layer9_moe_shared_expert_applied;
    int layer9_output_residual_applied;
    int layer10_input_norm_applied;
    int layer10_qkv_projection_applied;
    int layer10_zab_projection_applied;
    int layer10_conv_qkv_applied;
    int layer10_postconv_qkv_applied;
    int layer10_gate_rows_applied;
    int layer10_core_rows_applied;
    int layer10_gated_rmsnorm_applied;
    int layer10_out_projection_applied;
    int layer10_residual_hidden_applied;
    int layer10_post_attention_rmsnorm_applied;
    int qwen36_layer_descriptor_helper_applied;
    int qwen36_descriptor_layer_batch_started;
    int qwen36_descriptor_layer_loop_applied;
    int qwen36_descriptor_layer_body_repeated_loop_body_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_input_norm_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_qkv_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_qk_norm_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_rope_attention_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_output_projection_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_residual_hidden_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_post_attention_rmsnorm_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_moe_router_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_moe_expert_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_moe_shared_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_output_residual_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_handoff_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_loop_body_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_input_norm_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_attention_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_post_attention_rmsnorm_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_moe_router_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_moe_expert_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_moe_shared_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_handoff_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_loop_body_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_input_norm_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_attention_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_post_attention_rmsnorm_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_router_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_expert_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_moe_shared_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_handoff_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_loop_body_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_input_norm_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_attention_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_post_attention_rmsnorm_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_router_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_expert_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_moe_shared_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_handoff_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_loop_body_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_input_norm_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_attention_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_post_attention_rmsnorm_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_router_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_expert_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_moe_shared_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_handoff_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_loop_body_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_input_norm_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_attention_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_post_attention_rmsnorm_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_router_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_expert_applied;
    int qwen36_descriptor_layer_body_repeated_full_attention_next_output_residual_output_residual_output_residual_output_residual_moe_shared_applied;
    int qwen36_descriptor_layer_body_applied;
    int baseline_request_handoff_attached;
    unsigned int baseline_request_handoff_layer_count;
    unsigned int baseline_request_handoff_output_head_count;
    unsigned int baseline_request_handoff_hidden_size;
    uint64_t baseline_request_handoff_descriptor_fnv1a64;
    uint64_t baseline_request_handoff_block_integration_digest_fnv1a64;
    uint64_t baseline_request_handoff_hidden_handoff_digest_fnv1a64;
    uint64_t baseline_request_handoff_final_hidden_fnv1a64;
    uint64_t baseline_request_handoff_output_head_input_fnv1a64;
    uint64_t baseline_request_handoff_digest_fnv1a64;
    uint64_t baseline_request_handoff_request_digest_fnv1a64;
    uint64_t baseline_block_handoff_input_hidden_fnv1a64;
    uint64_t baseline_block_handoff_output_hidden_fnv1a64;
    uint64_t baseline_block_handoff_digest_fnv1a64;
    unsigned int baseline_block_handoff_layer_count;
    uint64_t baseline_plain_block_input_hidden_fnv1a64;
    uint64_t baseline_plain_block_output_hidden_fnv1a64;
    uint64_t baseline_plain_block_digest_fnv1a64;
    uint64_t baseline_plain_block_weight_bytes_read;
    uint64_t baseline_plain_block_elapsed_ns;
    uint64_t baseline_plain_block_weight_materialize_elapsed_ns;
    uint64_t baseline_plain_block_input_norm_elapsed_ns;
    uint64_t baseline_plain_block_linear_qkv_elapsed_ns;
    uint64_t baseline_plain_block_linear_zab_elapsed_ns;
    uint64_t baseline_plain_block_linear_conv_elapsed_ns;
    uint64_t baseline_plain_block_linear_postconv_elapsed_ns;
    uint64_t baseline_plain_block_linear_gate_elapsed_ns;
    uint64_t baseline_plain_block_linear_core_elapsed_ns;
    uint64_t baseline_plain_block_linear_gated_elapsed_ns;
    uint64_t baseline_plain_block_linear_out_elapsed_ns;
    uint64_t baseline_plain_block_linear_residual_elapsed_ns;
    int baseline_plain_block_linear_projection_parallel_enabled;
    unsigned int baseline_plain_block_linear_projection_parallel_worker_count;
    uint64_t baseline_plain_block_linear_projection_parallel_dispatch_count;
    uint64_t baseline_plain_block_linear_projection_parallel_task_count;
    uint64_t baseline_plain_block_linear_projection_parallel_serial_task_count;
    uint64_t baseline_plain_block_full_qkv_elapsed_ns;
    uint64_t baseline_plain_block_full_qk_norm_elapsed_ns;
    uint64_t baseline_plain_block_full_rope_attention_elapsed_ns;
    uint64_t baseline_plain_block_full_out_elapsed_ns;
    uint64_t baseline_plain_block_full_residual_elapsed_ns;
    uint64_t baseline_plain_block_post_attention_norm_elapsed_ns;
    uint64_t baseline_plain_block_moe_router_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_zero_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_load_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_load_prepare_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_load_tasks_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_load_finalize_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_compute_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_compute_gate_up_activation_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_compute_down_projection_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_hash_elapsed_ns;
    uint64_t baseline_plain_block_moe_shared_elapsed_ns;
    uint64_t baseline_plain_block_moe_output_residual_elapsed_ns;
    unsigned int baseline_plain_block_layer_count;
    unsigned int baseline_plain_block_module_output_count;
    unsigned int baseline_plain_block_input_norm_layer_count;
    unsigned int baseline_plain_block_attention_module_output_count;
    unsigned int baseline_plain_block_linear_attention_layer_count;
    unsigned int baseline_plain_block_full_attention_layer_count;
    unsigned int baseline_plain_block_post_attention_norm_layer_count;
    unsigned int baseline_plain_block_moe_layer_count;
    unsigned int baseline_plain_block_moe_module_output_count;
    size_t baseline_plain_block_trace_token_position;
    unsigned int baseline_plain_block_trace_layer_count;
    uint64_t baseline_plain_block_trace_input_hidden_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_input_norm_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_attention_residual_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_qkv_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_zab_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_conv_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_qkv_ring_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_conv_output_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_postconv_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_gate_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_core_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_core_state_input_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_core_state_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_core_state_output_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_gated_norm_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_out_projection_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_linear_residual_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_post_attention_norm_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint64_t baseline_plain_block_trace_output_hidden_fnv1a64
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    uint32_t baseline_plain_block_trace_router_top1_expert_id
        [QRT_QWEN36_PLAIN_BLOCK_TRACE_LAYER_CAPACITY];
    float baseline_plain_block_trace_layer0_input_norm_sample
        [QRT_QWEN36_TRACE_SAMPLE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_sample
        [QRT_QWEN36_TRACE_SAMPLE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_values
        [QRT_QWEN36_LAYER0_LINEAR_QKV_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_linear_qkv_variant_indices
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_current
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_forward_rne
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_reverse_rne
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_forward_fma_rne
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_reverse_fma_rne
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_double_rne
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_forward_trunc
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_forward_fma_trunc
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_reverse_fma_trunc
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_double_trunc
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_forward_sum
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_reverse_sum
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_forward_fma_sum
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_reverse_fma_sum
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_variant_double_sum
        [QRT_QWEN36_LAYER0_LINEAR_QKV_VARIANT_TRACE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_linear_qkv_block_variant_sizes
        [QRT_QWEN36_LAYER0_LINEAR_QKV_BLOCK_VARIANT_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_block_variant_rne
        [QRT_QWEN36_LAYER0_LINEAR_QKV_BLOCK_VARIANT_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_block_variant_trunc
        [QRT_QWEN36_LAYER0_LINEAR_QKV_BLOCK_VARIANT_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_block_variant_fma_rne
        [QRT_QWEN36_LAYER0_LINEAR_QKV_BLOCK_VARIANT_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_qkv_block_variant_fma_trunc
        [QRT_QWEN36_LAYER0_LINEAR_QKV_BLOCK_VARIANT_VALUE_COUNT];
    size_t baseline_plain_block_trace_layer0_input_norm_numel;
    float baseline_plain_block_trace_layer0_input_norm_min;
    float baseline_plain_block_trace_layer0_input_norm_max;
    float baseline_plain_block_trace_layer0_input_norm_mean;
    float baseline_plain_block_trace_layer0_input_norm_rms;
    float baseline_plain_block_trace_layer0_input_norm_max_abs;
    size_t baseline_plain_block_trace_layer0_linear_qkv_numel;
    float baseline_plain_block_trace_layer0_linear_qkv_min;
    float baseline_plain_block_trace_layer0_linear_qkv_max;
    float baseline_plain_block_trace_layer0_linear_qkv_mean;
    float baseline_plain_block_trace_layer0_linear_qkv_rms;
    float baseline_plain_block_trace_layer0_linear_qkv_max_abs;
    float baseline_plain_block_trace_layer0_linear_z_sample
        [QRT_QWEN36_TRACE_SAMPLE_COUNT];
    float baseline_plain_block_trace_layer0_linear_z_values
        [QRT_QWEN36_LAYER0_LINEAR_Z_TRACE_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_a_values
        [QRT_QWEN36_LAYER0_LINEAR_A_TRACE_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_b_values
        [QRT_QWEN36_LAYER0_LINEAR_B_TRACE_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_conv_values
        [QRT_QWEN36_LAYER0_LINEAR_CONV_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_linear_conv_source_row_indices
        [QRT_QWEN36_LAYER0_LINEAR_CONV_SOURCE_TRACE_ROW_COUNT];
    uint32_t baseline_plain_block_trace_layer0_linear_conv_source_token_indices
        [QRT_QWEN36_LAYER0_LINEAR_CONV_SOURCE_TRACE_TAP_COUNT];
    float baseline_plain_block_trace_layer0_linear_conv_source_qkv_values
        [QRT_QWEN36_LAYER0_LINEAR_CONV_SOURCE_TRACE_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_conv_source_weight_values
        [QRT_QWEN36_LAYER0_LINEAR_CONV_SOURCE_TRACE_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_postconv_q_scaled_values
        [QRT_QWEN36_LAYER0_LINEAR_POSTCONV_Q_TRACE_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_postconv_k_norm_values
        [QRT_QWEN36_LAYER0_LINEAR_POSTCONV_K_TRACE_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_postconv_value_values
        [QRT_QWEN36_LAYER0_LINEAR_POSTCONV_VALUE_TRACE_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_gate_g_values
        [QRT_QWEN36_LAYER0_LINEAR_GATE_TRACE_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_gate_beta_values
        [QRT_QWEN36_LAYER0_LINEAR_GATE_TRACE_VALUE_COUNT];
    size_t baseline_plain_block_trace_layer0_prefill_provider_token_count;
    uint64_t baseline_plain_block_trace_layer0_prefill_provider_input_norm_fnv1a64;
    uint64_t baseline_plain_block_trace_layer0_prefill_provider_qkv_fnv1a64;
    uint32_t baseline_plain_block_trace_layer0_prefill_provider_qkv_row_indices
        [QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_TRACE_ROW_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_provider_qkv_token_indices
        [QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_TRACE_TOKEN_COUNT];
    float baseline_plain_block_trace_layer0_prefill_provider_qkv_values
        [QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_provider_qkv_mismatch_row_indices
        [QRT_QWEN36_LAYER0_PREFILL_QKV_MISMATCH_TRACE_ROW_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_provider_qkv_mismatch_token_indices
        [QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_TRACE_TOKEN_COUNT];
    float baseline_plain_block_trace_layer0_prefill_provider_qkv_mismatch_values
        [QRT_QWEN36_LAYER0_PREFILL_QKV_MISMATCH_TRACE_VALUE_COUNT];
    uint64_t baseline_plain_block_trace_layer0_prefill_provider_qkv_row_fnv1a64
        [QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_ROW_HASH_COUNT];
    uint64_t baseline_plain_block_trace_layer0_prefill_provider_conv_silu_fnv1a64;
    uint64_t baseline_plain_block_trace_layer0_prefill_provider_conv_silu_row_fnv1a64
        [QRT_QWEN36_LAYER0_PREFILL_CONV_SILU_PROVIDER_ROW_HASH_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_provider_conv_silu_mismatch_row_indices
        [QRT_QWEN36_LAYER0_PREFILL_CONV_SILU_MISMATCH_TRACE_ROW_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_provider_conv_silu_mismatch_token_indices
        [QRT_QWEN36_LAYER0_PREFILL_QKV_PROVIDER_TRACE_TOKEN_COUNT];
    float baseline_plain_block_trace_layer0_prefill_provider_conv_silu_mismatch_values
        [QRT_QWEN36_LAYER0_PREFILL_CONV_SILU_MISMATCH_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_provider_conv_silu_mismatch_value_bits
        [QRT_QWEN36_LAYER0_PREFILL_CONV_SILU_MISMATCH_TRACE_VALUE_COUNT];
    uint64_t baseline_plain_block_trace_layer0_prefill_provider_a_projection_fnv1a64;
    uint64_t baseline_plain_block_trace_layer0_prefill_provider_b_projection_fnv1a64;
    size_t baseline_plain_block_trace_layer0_prefill_chunk_token_count;
    uint64_t baseline_plain_block_trace_layer0_prefill_chunk_k_expanded_fnv1a64;
    uint64_t baseline_plain_block_trace_layer0_prefill_chunk_value_fnv1a64;
    uint64_t baseline_plain_block_trace_layer0_prefill_chunk_gate_g_fnv1a64;
    uint64_t baseline_plain_block_trace_layer0_prefill_chunk_gate_beta_fnv1a64;
    uint64_t baseline_plain_block_trace_layer0_prefill_chunk_k_expanded_row_fnv1a64
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_ROW_HASH_COUNT];
    uint64_t baseline_plain_block_trace_layer0_prefill_chunk_value_row_fnv1a64
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_ROW_HASH_COUNT];
    uint64_t baseline_plain_block_trace_layer0_prefill_chunk_gate_g_row_fnv1a64
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_ROW_HASH_COUNT];
    uint64_t baseline_plain_block_trace_layer0_prefill_chunk_gate_beta_row_fnv1a64
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_ROW_HASH_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_selected_token_indices
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_INPUT_TRACE_TOKEN_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_k_expanded_selected_row_indices
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_K_EXPANDED_TRACE_ROW_COUNT];
    float baseline_plain_block_trace_layer0_prefill_chunk_k_expanded_selected_values
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_K_EXPANDED_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_k_expanded_selected_value_bits
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_K_EXPANDED_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_value_selected_row_indices
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_VALUE_TRACE_ROW_COUNT];
    float baseline_plain_block_trace_layer0_prefill_chunk_value_selected_values
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_VALUE_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_value_selected_value_bits
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_VALUE_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_gate_g_selected_row_indices
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_G_TRACE_ROW_COUNT];
    float baseline_plain_block_trace_layer0_prefill_chunk_gate_g_selected_values
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_G_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_gate_g_selected_value_bits
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_G_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_gate_beta_selected_row_indices
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_BETA_TRACE_ROW_COUNT];
    float baseline_plain_block_trace_layer0_prefill_chunk_gate_beta_selected_values
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_BETA_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_gate_beta_selected_value_bits
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_BETA_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_gate_input_selected_row_indices
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_G_TRACE_ROW_COUNT];
    float baseline_plain_block_trace_layer0_prefill_chunk_gate_a_selected_values
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_INPUT_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_gate_a_selected_value_bits
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_INPUT_TRACE_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_prefill_chunk_gate_a_plus_dt_selected_values
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_INPUT_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_gate_a_plus_dt_selected_value_bits
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_GATE_INPUT_TRACE_VALUE_COUNT];
    uint64_t baseline_plain_block_trace_layer0_prefill_chunk_state_fnv1a64;
    uint64_t baseline_plain_block_trace_layer0_prefill_chunk_state_row_fnv1a64
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_STATE_ROW_HASH_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_state_selected_row_indices
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_STATE_TRACE_ROW_COUNT];
    float baseline_plain_block_trace_layer0_prefill_chunk_state_selected_values
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_STATE_TRACE_VALUE_COUNT];
    uint32_t baseline_plain_block_trace_layer0_prefill_chunk_state_selected_value_bits
        [QRT_QWEN36_LAYER0_PREFILL_CHUNK_STATE_TRACE_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_core_sample
        [QRT_QWEN36_TRACE_SAMPLE_COUNT];
    float baseline_plain_block_trace_layer0_linear_core_values
        [QRT_QWEN36_LAYER0_LINEAR_CORE_TRACE_VALUE_COUNT];
    size_t baseline_plain_block_trace_layer0_linear_z_numel;
    float baseline_plain_block_trace_layer0_linear_z_min;
    float baseline_plain_block_trace_layer0_linear_z_max;
    float baseline_plain_block_trace_layer0_linear_z_mean;
    float baseline_plain_block_trace_layer0_linear_z_rms;
    float baseline_plain_block_trace_layer0_linear_z_max_abs;
    size_t baseline_plain_block_trace_layer0_linear_core_numel;
    float baseline_plain_block_trace_layer0_linear_core_min;
    float baseline_plain_block_trace_layer0_linear_core_max;
    float baseline_plain_block_trace_layer0_linear_core_mean;
    float baseline_plain_block_trace_layer0_linear_core_rms;
    float baseline_plain_block_trace_layer0_linear_core_max_abs;
    float baseline_plain_block_trace_layer0_linear_out_input_sample
        [QRT_QWEN36_TRACE_SAMPLE_COUNT];
    float baseline_plain_block_trace_layer0_linear_out_input_values
        [QRT_QWEN36_LAYER0_LINEAR_OUT_INPUT_TRACE_VALUE_COUNT];
    float baseline_plain_block_trace_layer0_linear_out_output_sample
        [QRT_QWEN36_TRACE_SAMPLE_COUNT];
    size_t baseline_plain_block_trace_layer0_linear_out_input_numel;
    float baseline_plain_block_trace_layer0_linear_out_input_min;
    float baseline_plain_block_trace_layer0_linear_out_input_max;
    float baseline_plain_block_trace_layer0_linear_out_input_mean;
    float baseline_plain_block_trace_layer0_linear_out_input_rms;
    float baseline_plain_block_trace_layer0_linear_out_input_max_abs;
    size_t baseline_plain_block_trace_layer0_linear_out_output_numel;
    float baseline_plain_block_trace_layer0_linear_out_output_min;
    float baseline_plain_block_trace_layer0_linear_out_output_max;
    float baseline_plain_block_trace_layer0_linear_out_output_mean;
    float baseline_plain_block_trace_layer0_linear_out_output_rms;
    float baseline_plain_block_trace_layer0_linear_out_output_max_abs;
    size_t baseline_prompt_context_token_count;
    size_t baseline_prompt_state_position;
    uint32_t baseline_prompt_last_input_token;
    unsigned int baseline_prompt_state_summary_mode;
    unsigned int baseline_prompt_state_sequence_token_loop;
    unsigned int baseline_prompt_state_full_sequence_prefill;
    unsigned int baseline_prompt_state_linear_attention_state_carried;
    unsigned int baseline_prompt_state_full_attention_kv_carried;
    size_t baseline_prompt_sequence_processed_token_count;
    uint64_t baseline_prompt_sequence_digest_fnv1a64;
    uint64_t baseline_prompt_sequence_weight_bytes_read;
    uint64_t baseline_prompt_sequence_elapsed_ns;
    uint64_t baseline_prompt_linear_attention_state_digest_fnv1a64;
    uint64_t baseline_prompt_full_attention_kv_digest_fnv1a64;
    size_t baseline_prompt_linear_attention_state_update_count;
    size_t baseline_prompt_full_attention_kv_update_count;
    uint64_t baseline_prompt_linear_attention_state_bytes;
    uint64_t baseline_prompt_full_attention_kv_bytes;
    size_t baseline_product_layer1_frontier_token_count;
    size_t baseline_product_layer1_frontier_hidden_value_count;
    uint64_t baseline_product_layer1_frontier_digest_fnv1a64;
    uint64_t baseline_product_layer1_frontier_last_output_fnv1a64;
    unsigned int baseline_product_layer1_frontier_product_path;
    size_t baseline_product_q8192_layer1_frontier_context_token_count;
    size_t baseline_product_q8192_layer1_frontier_source_window_token_count;
    size_t baseline_product_q8192_layer1_frontier_token_count;
    size_t baseline_product_q8192_layer1_frontier_hidden_value_count;
    uint64_t baseline_product_q8192_layer1_frontier_source_window_token_ids_fnv1a64;
    uint64_t baseline_product_q8192_layer1_frontier_token_ids_fnv1a64;
    uint64_t baseline_product_q8192_layer1_frontier_digest_fnv1a64;
    uint64_t baseline_product_q8192_layer1_frontier_last_output_fnv1a64;
    size_t baseline_product_q8192_layer0_source_stage_count;
    size_t baseline_product_q8192_layer0_source_stage_token_counts
        [QRT_QWEN36_PRODUCT_Q8192_LAYER0_SOURCE_STAGE_COUNT];
    uint64_t baseline_product_q8192_layer0_source_stage_digest_fnv1a64
        [QRT_QWEN36_PRODUCT_Q8192_LAYER0_SOURCE_STAGE_COUNT];
    uint64_t baseline_product_q8192_layer0_source_stage_last_fnv1a64
        [QRT_QWEN36_PRODUCT_Q8192_LAYER0_SOURCE_STAGE_COUNT];
    size_t baseline_product_q8192_layer1_frontier_stage_count;
    size_t baseline_product_q8192_layer1_frontier_stage_token_counts
        [QRT_QWEN36_PRODUCT_Q8192_LAYER1_STAGE_COUNT];
    uint64_t baseline_product_q8192_layer1_frontier_stage_digest_fnv1a64
        [QRT_QWEN36_PRODUCT_Q8192_LAYER1_STAGE_COUNT];
    uint64_t baseline_product_q8192_layer1_frontier_stage_last_fnv1a64
        [QRT_QWEN36_PRODUCT_Q8192_LAYER1_STAGE_COUNT];
    unsigned int baseline_product_q8192_layer1_frontier_probe_enabled;
    unsigned int baseline_product_q8192_layer1_frontier_context_match;
    unsigned int baseline_product_q8192_layer1_frontier_synthetic_hidden_seed;
    unsigned int baseline_product_q8192_layer1_frontier_linear_core_state_reset;
    unsigned int baseline_product_q8192_layer1_frontier_output_head_skipped;
    unsigned int baseline_product_q8192_layer1_frontier_complete;
    unsigned int baseline_product_q8192_layer1_frontier_product_path;
    unsigned int baseline_product_q8192_layer1_frontier_continuation_attached;
    unsigned int baseline_product_q8192_layer1_frontier_continuation_attempted;
    unsigned int baseline_product_q8192_layer1_frontier_continuation_completed;
    unsigned int baseline_product_q8192_layer1_frontier_continuation_status;
    uint64_t baseline_product_q8192_layer1_frontier_continuation_elapsed_ns;
    uint64_t baseline_product_q8192_layer1_frontier_continuation_digest_fnv1a64;
    unsigned int baseline_product_q8192_prefill_descriptor_batch_early_entry_attempted;
    unsigned int baseline_product_q8192_prefill_descriptor_batch_early_entry_completed;
    unsigned int baseline_product_q8192_prefill_descriptor_batch_early_entry_status;
    uint64_t baseline_product_q8192_prefill_descriptor_batch_early_entry_elapsed_ns;
    uint64_t baseline_product_q8192_prefill_descriptor_batch_early_entry_digest_fnv1a64;
    unsigned int baseline_product_q8192_prefill_descriptor_batch_early_entry_resident_engine_reused;
    unsigned int baseline_product_q8192_prefill_descriptor_batch_early_entry_scratch_engine_created;
    uint64_t baseline_product_q8192_prefill_descriptor_batch_early_entry_engine_context_elapsed_ns;
    char baseline_product_q8192_prefill_descriptor_batch_early_entry_missing_field[64];
    char baseline_product_q8192_prefill_descriptor_batch_early_entry_failure_stage[64];
    char baseline_product_q8192_prefill_descriptor_batch_early_entry_failure
        [QRT_LOAD_ERROR_CAPACITY];
    uint64_t baseline_product_q8192_layer1_frontier_descriptor_batch_digest_fnv1a64;
    char baseline_product_q8192_layer1_frontier_descriptor_batch_missing_field[64];
    char baseline_product_q8192_layer1_frontier_descriptor_batch_failure_stage[64];
    char baseline_product_q8192_layer1_frontier_descriptor_batch_failure
        [QRT_LOAD_ERROR_CAPACITY];
    qrt_qwen36_prefill_descriptor_batch_timing_t
        baseline_product_q8192_layer1_frontier_descriptor_batch_timing;
    uint64_t baseline_product_q8192_descriptor_batch_resident_tensor_read_count;
    uint64_t baseline_product_q8192_descriptor_batch_resident_tensor_read_bytes;
    uint64_t baseline_product_q8192_descriptor_batch_direct_tensor_read_count;
    uint64_t baseline_product_q8192_descriptor_batch_direct_tensor_read_bytes;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_call_count;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_pass_count;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_f32_hash_match_count;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_bf16_hash_match_count;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_checked_value_count;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_expected_value_count;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_all_within_tolerance;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_all_f32_hash_match;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_all_bf16_hash_match;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_worst_layer_index;
    float baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_max_abs_diff;
    uint64_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_digest_fnv1a64;
    uint64_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_f32_pair_digest_fnv1a64;
    uint64_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_bf16_pair_digest_fnv1a64;
    char baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_worst_stage[64];
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_first_diff_layer_index;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_first_diff_element_index;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_first_diff_token_id;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_first_diff_hidden_index;
    uint32_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_first_diff_cpu_bits;
    uint32_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_first_diff_gpu_bits;
    uint32_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_first_diff_ulp_distance;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_max_ulp_layer_index;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_max_ulp_token_id;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_max_ulp_hidden_index;
    uint32_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_max_ulp_distance;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_layer_checked_value_count
        [QRT_QWEN36_LAYER_COUNT];
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_layer_mismatch_count
        [QRT_QWEN36_LAYER_COUNT];
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_layer_first_diff_element_index
        [QRT_QWEN36_LAYER_COUNT];
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_layer_first_diff_token_id
        [QRT_QWEN36_LAYER_COUNT];
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_layer_first_diff_hidden_index
        [QRT_QWEN36_LAYER_COUNT];
    uint32_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_layer_first_diff_cpu_bits
        [QRT_QWEN36_LAYER_COUNT];
    uint32_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_layer_first_diff_gpu_bits
        [QRT_QWEN36_LAYER_COUNT];
    uint32_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_layer_first_diff_ulp_distance
        [QRT_QWEN36_LAYER_COUNT];
    uint32_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_layer_max_ulp_distance
        [QRT_QWEN36_LAYER_COUNT];
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_contract_layer_ulp_bucket_counts
        [QRT_QWEN36_LAYER_COUNT][QRT_QWEN36_ROUTED_OUTPUT_ULP_BUCKET_COUNT];
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_call_count;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_pass_count;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_combined_f32_hash_match_count;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_combined_bf16_hash_match_count;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_output_residual_f32_hash_match_count;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_output_residual_bf16_hash_match_count;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_checked_value_count;
    size_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_expected_value_count;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_all_within_tolerance;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_all_combined_f32_hash_match;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_all_combined_bf16_hash_match;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_all_output_residual_f32_hash_match;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_all_output_residual_bf16_hash_match;
    unsigned int baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_worst_layer_index;
    float baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_max_abs_diff;
    uint64_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_digest_fnv1a64;
    uint64_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_combined_f32_pair_digest_fnv1a64;
    uint64_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_combined_bf16_pair_digest_fnv1a64;
    uint64_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_output_residual_f32_pair_digest_fnv1a64;
    uint64_t baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_output_residual_bf16_pair_digest_fnv1a64;
    char baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_worst_stage[64];
    char baseline_product_q8192_descriptor_batch_routed_expert_gpu_downstream_contract_first_divergent_stage[64];
    size_t baseline_plain_block_token_position;
    unsigned int baseline_plain_block_token_position_matches_prompt;
    int baseline_output_head_token_emitted;
    unsigned int baseline_output_head_sampled_token_id;
    unsigned int baseline_output_head_topk_token_ids
        [QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK];
    float baseline_output_head_topk_logits
        [QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK];
    float baseline_output_head_topk_lm_head_row_l2
        [QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK];
    float baseline_output_head_topk_cosine_to_final_norm
        [QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK];
    unsigned int baseline_output_head_probe_token_requested;
    unsigned int baseline_output_head_probe_token_id;
    unsigned int baseline_output_head_probe_token_evaluated;
    size_t baseline_output_head_probe_token_rank;
    float baseline_output_head_probe_token_logit;
    float baseline_output_head_probe_token_margin_to_top1;
    unsigned int baseline_output_head_probe_token_in_topk;
    unsigned int baseline_output_head_probe_token_set_requested;
    size_t baseline_output_head_probe_token_set_count;
    unsigned int baseline_output_head_probe_token_set_ids
        [QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK];
    size_t baseline_output_head_probe_token_set_evaluated_count;
    size_t baseline_output_head_probe_token_set_ranks
        [QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK];
    float baseline_output_head_probe_token_set_logits
        [QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK];
    float baseline_output_head_probe_token_set_margin_to_top1
        [QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK];
    float baseline_output_head_probe_token_set_lm_head_row_l2
        [QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK];
    float baseline_output_head_probe_token_set_cosine_to_final_norm
        [QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK];
    unsigned int baseline_output_head_probe_token_set_in_topk
        [QRT_QWEN36_OUTPUT_HEAD_SAMPLER_TOPK];
    size_t baseline_output_head_probe_token_set_min_rank;
    unsigned int baseline_output_head_probe_token_set_min_rank_token_id;
    unsigned int baseline_output_head_probe_token_set_any_in_topk;
    uint64_t baseline_output_head_input_hidden_fnv1a64;
    uint64_t baseline_output_head_final_norm_fnv1a64;
    uint64_t baseline_output_head_logits_fnv1a64;
    float baseline_output_head_input_hidden_min;
    float baseline_output_head_input_hidden_max;
    float baseline_output_head_input_hidden_mean;
    float baseline_output_head_input_hidden_rms;
    float baseline_output_head_input_hidden_max_abs;
    float baseline_output_head_final_norm_min;
    float baseline_output_head_final_norm_max;
    float baseline_output_head_final_norm_mean;
    float baseline_output_head_final_norm_rms;
    float baseline_output_head_final_norm_l2;
    float baseline_output_head_final_norm_max_abs;
    float baseline_output_head_logits_min;
    float baseline_output_head_logits_max;
    float baseline_output_head_logits_mean;
    float baseline_output_head_logits_rms;
    float baseline_output_head_logits_max_abs;
    uint64_t baseline_output_head_sampler_fnv1a64;
    uint64_t baseline_output_head_weight_fnv1a64;
    uint64_t baseline_output_head_final_norm_weight_fnv1a64;
    uint64_t baseline_output_head_lm_head_weight_fnv1a64;
    uint64_t baseline_output_head_weight_bytes_read;
    uint64_t baseline_output_head_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_compute_gate_up_projection_elapsed_ns;
    int baseline_plain_block_moe_expert_gate_up_projection_avx2_enabled;
    int baseline_plain_block_moe_expert_gate_up_projection_hip_requested;
    int baseline_plain_block_moe_expert_gate_up_projection_hip_enabled;
    uint64_t baseline_plain_block_moe_expert_gate_up_projection_hip_dispatch_count;
    uint64_t baseline_plain_block_moe_expert_gate_up_projection_hip_fallback_count;
    uint64_t baseline_plain_block_moe_expert_gate_up_projection_hip_preload_count;
    uint64_t baseline_plain_block_moe_expert_gate_up_projection_hip_preload_failure_count;
    uint64_t baseline_plain_block_moe_expert_gate_up_projection_hip_preload_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_gate_up_projection_hip_preload_bytes;
    uint64_t baseline_plain_block_moe_expert_gate_up_projection_hip_upload_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_gate_up_projection_hip_kernel_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_gate_up_projection_hip_output_copy_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_gate_up_projection_hip_upload_bytes;
    uint64_t baseline_plain_block_moe_expert_compute_activation_elapsed_ns;
    int baseline_plain_block_moe_expert_down_projection_avx2_enabled;
    int baseline_plain_block_moe_expert_down_projection_hip_requested;
    int baseline_plain_block_moe_expert_down_projection_hip_enabled;
    uint64_t baseline_plain_block_moe_expert_down_projection_hip_dispatch_count;
    uint64_t baseline_plain_block_moe_expert_down_projection_hip_fallback_count;
    uint64_t baseline_plain_block_moe_expert_down_projection_hip_preload_count;
    uint64_t baseline_plain_block_moe_expert_down_projection_hip_preload_failure_count;
    uint64_t baseline_plain_block_moe_expert_down_projection_hip_preload_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_down_projection_hip_preload_bytes;
    uint64_t baseline_plain_block_moe_expert_down_projection_hip_upload_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_down_projection_hip_kernel_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_down_projection_hip_output_copy_elapsed_ns;
    uint64_t baseline_plain_block_moe_expert_down_projection_hip_upload_bytes;
    int baseline_plain_block_moe_expert_load_parallel_enabled;
    unsigned int baseline_plain_block_moe_expert_load_parallel_worker_count;
    uint64_t baseline_plain_block_moe_expert_load_parallel_dispatch_count;
    uint64_t baseline_plain_block_moe_expert_load_parallel_task_count;
    uint64_t baseline_plain_block_moe_expert_load_parallel_serial_task_count;
    uint64_t baseline_plain_block_moe_expert_load_coalesce_candidate_group_count;
    uint64_t baseline_plain_block_moe_expert_load_coalesce_candidate_task_count;
    uint64_t baseline_plain_block_moe_expert_load_coalesce_candidate_saved_task_count;
    uint64_t baseline_plain_block_moe_expert_load_coalesce_candidate_bytes;
    int baseline_plain_block_moe_expert_route_parallel_enabled;
    unsigned int baseline_plain_block_moe_expert_route_parallel_worker_count;
    uint64_t baseline_plain_block_moe_expert_route_parallel_dispatch_count;
    uint64_t baseline_plain_block_moe_expert_route_parallel_task_count;
    uint64_t baseline_plain_block_moe_expert_route_parallel_serial_task_count;
    char token_request_failure_stage[64];
    char token_request_failure[QRT_LOAD_ERROR_CAPACITY];
    qrt_load_manifest_result_t manifest;
    qrt_prepack_accounting_result_t prepack;
    unsigned int last_request_descriptor_decode_state_exported;
    char last_request_descriptor_decode_missing_field[64];
    uint64_t last_request_descriptor_decode_state_digest_fnv1a64;
    size_t last_request_descriptor_decode_state_token_count;
    size_t last_request_descriptor_decode_state_position;
    uint32_t last_request_descriptor_decode_state_token_id;
    uint64_t last_request_descriptor_decode_state_token_loop_digest_fnv1a64;
    uint64_t
        last_request_descriptor_decode_state_output_head_input_hidden_fnv1a64;
    uint64_t last_request_descriptor_decode_state_prefix_state_digest_fnv1a64;
    uint64_t
        last_request_descriptor_decode_state_prefix_position_digest_fnv1a64;
    unsigned int
        last_request_descriptor_decode_state_cache_payload_exported;
    uint64_t last_request_descriptor_decode_state_cache_payload_digest_fnv1a64;
    size_t last_request_descriptor_decode_state_cache_layer_count;
    size_t last_request_descriptor_decode_state_cache_token_count;
    unsigned int last_request_descriptor_decode_input_surface_exported;
    uint64_t last_request_descriptor_decode_input_surface_digest_fnv1a64;
    uint64_t
        last_request_descriptor_decode_input_surface_token_embedding_fnv1a64;
    size_t last_request_descriptor_decode_input_surface_token_count;
    size_t last_request_descriptor_decode_input_surface_position;
    uint32_t last_request_descriptor_decode_input_surface_token_id;
    uint64_t
        last_request_descriptor_decode_input_surface_cache_payload_digest_fnv1a64;
} qrt_engine_report_t;

#define QRT_QWEN36_SOURCE_VALUE_TYPE_NONE 0u
#define QRT_QWEN36_SOURCE_VALUE_TYPE_F32 1u
#define QRT_QWEN36_SOURCE_VALUE_TYPE_U32 2u
#define QRT_QWEN36_SOURCE_VALUE_TYPE_BF16 3u

typedef struct qrt_qwen36_layer0_source_stage_export_t {
    uint32_t stage_id;
    const char *stage_name;
    const void *values;
    size_t value_bytes;
    size_t token_count;
    uint64_t digest_fnv1a64;
    uint64_t last_fnv1a64;
    uint32_t value_type;
    uint32_t values_present;
} qrt_qwen36_layer0_source_stage_export_t;

typedef struct qrt_qwen36_layer1_frontier_buffer_export_t {
    uint32_t *token_ids;
    size_t token_capacity;
    float *hidden_values;
    size_t hidden_value_capacity;
    qrt_qwen36_layer0_source_stage_export_t *layer0_source_stages;
    size_t layer0_source_stage_capacity;
    size_t token_count;
    size_t hidden_value_count;
    size_t layer0_source_stage_count;
    size_t source_window_token_count;
    uint64_t token_ids_fnv1a64;
    uint64_t source_window_token_ids_fnv1a64;
    uint64_t hidden_values_fnv1a64;
    uint64_t product_path_digest_fnv1a64;
    uint64_t last_output_fnv1a64;
    uint64_t layer0_source_stage_table_digest_fnv1a64;
    unsigned int product_path;
} qrt_qwen36_layer1_frontier_buffer_export_t;

typedef struct qrt_qwen36_layer1_frontier_continuation_result_t {
    uint64_t digest_fnv1a64;
    unsigned int output_token_emitted;
    uint32_t output_token_id;
    float output_logit;
    uint64_t lm_head_logits_fnv1a64;
    uint64_t sampler_fnv1a64;
    uint64_t token_loop_digest_fnv1a64;
} qrt_qwen36_layer1_frontier_continuation_result_t;

typedef qrt_status_t (*qrt_qwen36_layer1_frontier_continuation_callback_t)(
    const qrt_qwen36_layer1_frontier_buffer_export_t *frontier,
    void *user_data,
    qrt_qwen36_layer1_frontier_continuation_result_t *out_result
);

#define QRT_QWEN36_PREFILL_DESCRIPTOR_BATCH_FLAG_NONE 0u
#define QRT_QWEN36_PREFILL_DESCRIPTOR_BATCH_FLAG_EARLY_ENTRY 1u
/* The request carries caller-owned token IDs rather than a captured fixture. */
#define QRT_QWEN36_PREFILL_DESCRIPTOR_BATCH_FLAG_ARBITRARY_PROMPT 2u

typedef struct qrt_qwen36_prefill_descriptor_batch_request_t {
    const qrt_qwen36_layer1_frontier_buffer_export_t *frontier;
    const char *model_dir;
    qrt_engine_t *resident_engine;
    size_t prefill_tokens;
    uint32_t flags;
    const uint32_t *input_tokens;
    size_t input_token_count;
    unsigned int prefix_cache_state_attached;
    size_t prefix_cache_cached_prefix_tokens;
    size_t prefix_cache_suffix_tokens;
    size_t prefix_cache_executed_input_tokens;
    size_t prefix_cache_output_token_capacity;
    size_t output_token_capacity;
    uint64_t prefix_cache_state_digest_fnv1a64;
    uint64_t prefix_cache_position_digest_fnv1a64;
    qrt_qwen36_layer1_frontier_continuation_callback_t
        early_entry_continuation_callback;
    void *early_entry_continuation_user_data;
} qrt_qwen36_prefill_descriptor_batch_request_t;

typedef struct qrt_qwen36_prefill_descriptor_batch_result_t {
    qrt_qwen36_layer1_frontier_continuation_result_t continuation;
    size_t output_token_capacity;
    size_t output_token_count;
    uint32_t output_tokens[QRT_PREFIX_CACHE_TARGET_OUTPUT_TOKENS];
    uint64_t output_tokens_fnv1a64;
    uint64_t output_head_input_hidden_fnv1a64;
    uint64_t output_sequence_elapsed_ns;
    uint64_t output_sequence_tpot_elapsed_ns;
    size_t output_sequence_tpot_sample_count;
    size_t output_sequence_prefill_sample_count;
    size_t output_sequence_decode_token_count;
    unsigned int output_sequence_autoregressive_decode;
    uint64_t descriptor_batch_digest_fnv1a64;
    uint64_t wall_clock_ns;
    qrt_qwen36_prefill_descriptor_batch_timing_t timing;
    size_t layer1_frontier_token_count;
    size_t layer1_frontier_source_window_token_count;
    size_t layer1_frontier_hidden_value_count;
    uint64_t layer1_frontier_token_ids_fnv1a64;
    uint64_t layer1_frontier_source_window_token_ids_fnv1a64;
    uint64_t layer1_frontier_hidden_values_fnv1a64;
    uint64_t layer1_frontier_digest_fnv1a64;
    uint64_t layer1_frontier_last_output_fnv1a64;
    unsigned int layer1_frontier_product_path;
    size_t layer1_frontier_layer0_source_stage_count;
    size_t layer1_frontier_layer0_source_stage_token_counts
        [QRT_QWEN36_PRODUCT_Q8192_LAYER0_SOURCE_STAGE_COUNT];
    uint64_t layer1_frontier_layer0_source_stage_digest_fnv1a64
        [QRT_QWEN36_PRODUCT_Q8192_LAYER0_SOURCE_STAGE_COUNT];
    uint64_t layer1_frontier_layer0_source_stage_last_fnv1a64
        [QRT_QWEN36_PRODUCT_Q8192_LAYER0_SOURCE_STAGE_COUNT];
    size_t layer1_frontier_stage_count;
    size_t layer1_frontier_stage_token_counts
        [QRT_QWEN36_PRODUCT_Q8192_LAYER1_STAGE_COUNT];
    uint64_t layer1_frontier_stage_digest_fnv1a64
        [QRT_QWEN36_PRODUCT_Q8192_LAYER1_STAGE_COUNT];
    uint64_t layer1_frontier_stage_last_fnv1a64
        [QRT_QWEN36_PRODUCT_Q8192_LAYER1_STAGE_COUNT];
    unsigned int early_entry_resident_engine_reused;
    unsigned int early_entry_scratch_engine_created;
    uint64_t early_entry_engine_context_elapsed_ns;
    uint64_t resident_tensor_read_count;
    uint64_t resident_tensor_read_bytes;
    uint64_t direct_tensor_read_count;
    uint64_t direct_tensor_read_bytes;
    size_t routed_expert_gpu_contract_call_count;
    size_t routed_expert_gpu_contract_pass_count;
    size_t routed_expert_gpu_contract_f32_hash_match_count;
    size_t routed_expert_gpu_contract_bf16_hash_match_count;
    size_t routed_expert_gpu_contract_checked_value_count;
    size_t routed_expert_gpu_contract_expected_value_count;
    unsigned int routed_expert_gpu_contract_all_within_tolerance;
    unsigned int routed_expert_gpu_contract_all_f32_hash_match;
    unsigned int routed_expert_gpu_contract_all_bf16_hash_match;
    unsigned int routed_expert_gpu_contract_worst_layer_index;
    float routed_expert_gpu_contract_max_abs_diff;
    uint64_t routed_expert_gpu_contract_digest_fnv1a64;
    uint64_t routed_expert_gpu_contract_f32_pair_digest_fnv1a64;
    uint64_t routed_expert_gpu_contract_bf16_pair_digest_fnv1a64;
    char routed_expert_gpu_contract_worst_stage[64];
    unsigned int routed_expert_gpu_contract_first_diff_layer_index;
    size_t routed_expert_gpu_contract_first_diff_element_index;
    unsigned int routed_expert_gpu_contract_first_diff_token_id;
    unsigned int routed_expert_gpu_contract_first_diff_hidden_index;
    uint32_t routed_expert_gpu_contract_first_diff_cpu_bits;
    uint32_t routed_expert_gpu_contract_first_diff_gpu_bits;
    uint32_t routed_expert_gpu_contract_first_diff_ulp_distance;
    unsigned int routed_expert_gpu_contract_max_ulp_layer_index;
    unsigned int routed_expert_gpu_contract_max_ulp_token_id;
    unsigned int routed_expert_gpu_contract_max_ulp_hidden_index;
    uint32_t routed_expert_gpu_contract_max_ulp_distance;
    size_t routed_expert_gpu_contract_layer_checked_value_count
        [QRT_QWEN36_LAYER_COUNT];
    size_t routed_expert_gpu_contract_layer_mismatch_count
        [QRT_QWEN36_LAYER_COUNT];
    size_t routed_expert_gpu_contract_layer_first_diff_element_index
        [QRT_QWEN36_LAYER_COUNT];
    unsigned int routed_expert_gpu_contract_layer_first_diff_token_id
        [QRT_QWEN36_LAYER_COUNT];
    unsigned int routed_expert_gpu_contract_layer_first_diff_hidden_index
        [QRT_QWEN36_LAYER_COUNT];
    uint32_t routed_expert_gpu_contract_layer_first_diff_cpu_bits
        [QRT_QWEN36_LAYER_COUNT];
    uint32_t routed_expert_gpu_contract_layer_first_diff_gpu_bits
        [QRT_QWEN36_LAYER_COUNT];
    uint32_t routed_expert_gpu_contract_layer_first_diff_ulp_distance
        [QRT_QWEN36_LAYER_COUNT];
    uint32_t routed_expert_gpu_contract_layer_max_ulp_distance
        [QRT_QWEN36_LAYER_COUNT];
    size_t routed_expert_gpu_contract_layer_ulp_bucket_counts
        [QRT_QWEN36_LAYER_COUNT][QRT_QWEN36_ROUTED_OUTPUT_ULP_BUCKET_COUNT];
    size_t routed_expert_gpu_downstream_contract_call_count;
    size_t routed_expert_gpu_downstream_contract_pass_count;
    size_t routed_expert_gpu_downstream_contract_combined_f32_hash_match_count;
    size_t routed_expert_gpu_downstream_contract_combined_bf16_hash_match_count;
    size_t routed_expert_gpu_downstream_contract_output_residual_f32_hash_match_count;
    size_t routed_expert_gpu_downstream_contract_output_residual_bf16_hash_match_count;
    size_t routed_expert_gpu_downstream_contract_checked_value_count;
    size_t routed_expert_gpu_downstream_contract_expected_value_count;
    unsigned int routed_expert_gpu_downstream_contract_all_within_tolerance;
    unsigned int routed_expert_gpu_downstream_contract_all_combined_f32_hash_match;
    unsigned int routed_expert_gpu_downstream_contract_all_combined_bf16_hash_match;
    unsigned int routed_expert_gpu_downstream_contract_all_output_residual_f32_hash_match;
    unsigned int routed_expert_gpu_downstream_contract_all_output_residual_bf16_hash_match;
    unsigned int routed_expert_gpu_downstream_contract_worst_layer_index;
    float routed_expert_gpu_downstream_contract_max_abs_diff;
    uint64_t routed_expert_gpu_downstream_contract_digest_fnv1a64;
    uint64_t routed_expert_gpu_downstream_contract_combined_f32_pair_digest_fnv1a64;
    uint64_t routed_expert_gpu_downstream_contract_combined_bf16_pair_digest_fnv1a64;
    uint64_t routed_expert_gpu_downstream_contract_output_residual_f32_pair_digest_fnv1a64;
    uint64_t routed_expert_gpu_downstream_contract_output_residual_bf16_pair_digest_fnv1a64;
    char routed_expert_gpu_downstream_contract_worst_stage[64];
    char routed_expert_gpu_downstream_contract_first_divergent_stage[64];
    unsigned int completed;
    char first_missing_runtime_context_field[64];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    unsigned int output_sequence_decode_state_exported;
    char output_sequence_decode_missing_field[64];
    uint64_t output_sequence_decode_state_digest_fnv1a64;
    size_t output_sequence_decode_state_token_count;
    size_t output_sequence_decode_state_position;
    uint32_t output_sequence_decode_state_token_id;
    uint64_t output_sequence_decode_state_token_loop_digest_fnv1a64;
    uint64_t output_sequence_decode_state_output_head_input_hidden_fnv1a64;
    uint64_t output_sequence_decode_state_prefix_state_digest_fnv1a64;
    uint64_t output_sequence_decode_state_prefix_position_digest_fnv1a64;
    unsigned int output_sequence_decode_state_cache_payload_exported;
    uint64_t output_sequence_decode_state_cache_payload_digest_fnv1a64;
    size_t output_sequence_decode_state_cache_layer_count;
    size_t output_sequence_decode_state_cache_token_count;
    unsigned int output_sequence_decode_input_surface_exported;
    uint64_t output_sequence_decode_input_surface_digest_fnv1a64;
    uint64_t output_sequence_decode_input_surface_token_embedding_fnv1a64;
    size_t output_sequence_decode_input_surface_token_count;
    size_t output_sequence_decode_input_surface_position;
    uint32_t output_sequence_decode_input_surface_token_id;
    uint64_t output_sequence_decode_input_surface_cache_payload_digest_fnv1a64;
} qrt_qwen36_prefill_descriptor_batch_result_t;

#define QRT_QWEN36_WHOLE_PROVIDER_FLAG_NONE 0u
#define QRT_QWEN36_WHOLE_PROVIDER_FLAG_COLD_Q8192_PREFILL 1u
#define QRT_QWEN36_WHOLE_PROVIDER_FLAG_ENDPOINT_BF16_GB10_BOUNDARY 2u
/*
 * Negotiates both resident decode v1 and the appended resident_session_*
 * result suffix.  A caller setting this bit guarantees that its result
 * allocation is sizeof(qrt_qwen36_whole_provider_result_t).
 */
#define QRT_QWEN36_WHOLE_PROVIDER_FLAG_RESIDENT_DECODE_V1_RESULT 4u
/*
 * Isolated pure-BF16/F32 cold-q16384 calibration route.  This bit does not
 * imply raw-logit authority; callers must still provide and verify the exact
 * prompt fingerprint and first ordinary greedy token.
 */
#define QRT_QWEN36_WHOLE_PROVIDER_FLAG_COLD_Q16384_PREFILL 8u
/*
 * Isolated pure-BF16/F32 cold-q32768 context-curve route.  Like the q16384
 * bit, this is deliberately incompatible with prefix-cache and endpoint
 * authority flags; it does not enable quantization, MTP, DFlash, or
 * speculative decode.
 */
#define QRT_QWEN36_WHOLE_PROVIDER_FLAG_COLD_Q32768_PREFILL 16u
/*
 * Isolated pure-BF16/F32 cold-q65536 context-curve route.  This extends the
 * same fail-closed long-context contract without changing any decode or
 * result layout.
 */
#define QRT_QWEN36_WHOLE_PROVIDER_FLAG_COLD_Q65536_PREFILL 32u
/*
 * Isolated maximum-context route.  The same bit covers the exact q131072
 * product shape plus the q131071/one-token and q130560/512-token gb10
 * authority shapes used while the reference service is capped at 131072
 * total tokens.
 */
#define QRT_QWEN36_WHOLE_PROVIDER_FLAG_COLD_Q131072_PREFILL 64u
/*
 * General batch-one prompt route. The provider must consume input_tokens
 * verbatim and may not apply captured prompt/token/logit oracle checks.
 */
#define QRT_QWEN36_WHOLE_PROVIDER_FLAG_ARBITRARY_PREFILL 128u

#define QRT_QWEN36_WHOLE_PROVIDER_DECODE_ABI_VERSION 1u
#define QRT_QWEN36_EXACT_FIRST_TOKEN_ABI_VERSION 1u
#define QRT_QWEN36_RESIDENT_PREFIX_CACHE_ABI_VERSION 2u
#define QRT_QWEN36_RESIDENT_PREFIX_CACHE_FALLBACK_ABI_VERSION 1u
#define QRT_QWEN36_RESIDENT_PREFIX_CACHE_RESET_ABI_VERSION 1u
#define QRT_QWEN36_WHOLE_PROVIDER_ENGINE_LIFECYCLE_ABI_VERSION 1u
#define QRT_QWEN36_WHOLE_PROVIDER_THREAD_PREPARE_ABI_VERSION 1u
#define QRT_QWEN36_WHOLE_PROVIDER_DIRECT_ENTRY_ABI_VERSION 1u
#define QRT_QWEN36_RESIDENT_PREFIX_CACHE_CAPACITY_ABI_VERSION 1u
#define QRT_QWEN36_REQUEST_SERIALIZATION_ABI_VERSION 1u
/*
 * The request surface may chain multiple bounded decode-v1 spans without
 * changing the fixed v1 result layout.  This matches the product's 512-token
 * completion shape while keeping each provider ABI call at or below 64.
 */
#define QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS \
    QRT_PREFIX_CACHE_TARGET_OUTPUT_TOKENS
#define QRT_QWEN36_WHOLE_PROVIDER_DECODE_MAX_OUTPUT_TOKENS 64u
#define QRT_QWEN36_WHOLE_PROVIDER_DECODE_FLAG_NONE 0u
#define QRT_QWEN36_EXACT_FIRST_TOKEN_FLAG_NONE 0u
#define QRT_QWEN36_RESIDENT_PREFIX_CACHE_FLAG_NONE 0u
/*
 * Publish stop tokens in the fixed-capacity output stream and continue
 * decoding.  This matches service requests made with ignore_eos=true; it is
 * distinct from the existing minimum-output policy, which substitutes the
 * next ranked non-stop token before the requested minimum length.
 */
#define QRT_QWEN36_RESIDENT_PREFIX_CACHE_FLAG_IGNORE_EOS 1u
#define QRT_QWEN36_RESIDENT_PREFIX_CACHE_RESET_FLAG_NONE 0u
#define QRT_QWEN36_RESIDENT_PREFIX_CACHE_MAX_SUFFIX_TOKENS 1024u
#define QRT_QWEN36_RESIDENT_PREFIX_CACHE_MAX_TAIL_TOKENS 1536u

#define QRT_QWEN36_WHOLE_PROVIDER_ENGINE_LIFECYCLE_ACQUIRE 1u
#define QRT_QWEN36_WHOLE_PROVIDER_ENGINE_LIFECYCLE_RELEASE 2u
#define QRT_QWEN36_WHOLE_PROVIDER_ENGINE_LIFECYCLE_SNAPSHOT 3u
#define QRT_QWEN36_RESIDENT_PREFIX_CACHE_POLICY_SINGLE_SLOT_REPLACE_BEFORE_CAPTURE 1u
#define QRT_QWEN36_REQUEST_SERIALIZATION_OPERATION_ORDINARY 1u
#define QRT_QWEN36_REQUEST_SERIALIZATION_OPERATION_PREFIX 2u
#define QRT_QWEN36_REQUEST_SERIALIZATION_OPERATION_PREFIX_FALLBACK 3u
#define QRT_QWEN36_REQUEST_SERIALIZATION_OPERATION_PREFIX_RESET 4u
#define QRT_QWEN36_REQUEST_SERIALIZATION_OPERATION_STREAM 5u

#define QRT_QWEN36_WHOLE_PROVIDER_REQUEST_ENTRY_NONE 0u
#define QRT_QWEN36_WHOLE_PROVIDER_REQUEST_ENTRY_LEGACY 1u
#define QRT_QWEN36_WHOLE_PROVIDER_REQUEST_ENTRY_DIRECT 2u
#define QRT_QWEN36_WHOLE_PROVIDER_REQUEST_ENTRY_FALLBACK 3u

#define QRT_QWEN36_WHOLE_PROVIDER_SURFACE_SELECTED_MOE 1u
#define QRT_QWEN36_WHOLE_PROVIDER_SURFACE_FULL_ATTENTION 2u
#define QRT_QWEN36_WHOLE_PROVIDER_SURFACE_RESIDENT_STACK 4u
#define QRT_QWEN36_WHOLE_PROVIDER_SURFACE_REQUEST_PATH 8u

typedef int (
    QRT_CDECL *qrt_qwen36_whole_provider_prefill_emit_callback_v1_t
)(
    void *user_data,
    uint32_t token_id,
    uint64_t token_step_elapsed_ns
);

typedef struct qrt_qwen36_whole_provider_request_t {
    const char *model_dir;
    qrt_engine_t *resident_engine;
    const uint32_t *input_tokens;
    size_t input_token_count;
    size_t output_token_capacity;
    uint32_t flags;
    uint32_t required_surfaces;
    uint64_t expected_prompt_token_ids_fnv1a64;
    uint64_t expected_frontier_digest_fnv1a64;
    uint32_t expected_output_token_id;
    float expected_output_logit;
    float output_logit_abs_tolerance;
    unsigned int prefix_cache_state_attached;
    size_t prefix_cache_cached_prefix_tokens;
    size_t prefix_cache_suffix_tokens;
    size_t prefix_cache_executed_input_tokens;
    size_t prefix_cache_output_token_capacity;
    uint64_t prefix_cache_state_digest_fnv1a64;
    uint64_t prefix_cache_position_digest_fnv1a64;
    qrt_qwen36_whole_provider_prefill_emit_callback_v1_t
        prefill_emit_callback;
    void *prefill_emit_user_data;
} qrt_qwen36_whole_provider_request_t;

typedef struct qrt_qwen36_whole_provider_result_t {
    unsigned int completed;
    uint32_t provided_surfaces;
    size_t output_token_capacity;
    size_t output_token_count;
    uint32_t output_tokens[QRT_PREFIX_CACHE_TARGET_OUTPUT_TOKENS];
    uint64_t output_tokens_fnv1a64;
    qrt_qwen36_layer1_frontier_continuation_result_t continuation;
    uint64_t prompt_token_ids_fnv1a64;
    uint64_t frontier_digest_fnv1a64;
    uint64_t provider_digest_fnv1a64;
    uint64_t wall_clock_ns;
    char first_missing_runtime_context_field[64];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    qrt_qwen36_prefill_descriptor_batch_result_t descriptor_result;
    uint64_t preload_wall_clock_ns;
    uint32_t resident_session_valid;
    uint32_t resident_session_prefix_token_count;
    uint64_t resident_session_generation;
    uint64_t resident_session_prompt_token_ids_fnv1a64;
    uint32_t prefill_emit_attempted;
    uint32_t prefill_emit_completed;
    uint32_t prefill_emit_rejected;
    uint32_t reserved0;
    uint64_t prefill_emit_elapsed_ns;
} qrt_qwen36_whole_provider_result_t;

/*
 * Exact single-token requests are a narrow product boundary for deterministic
 * graders and classifiers.  The provider executes the complete prompt at its
 * real token length, samples the final prompt position, and does not create or
 * mutate a resident prefix session.  The full token array plus its digest
 * cross the ABI.
 */
typedef struct qrt_qwen36_exact_first_token_request_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t batch_size;
    qrt_engine_t *resident_engine;
    const uint32_t *input_tokens;
    uint32_t input_token_count;
    uint32_t reserved0;
    uint64_t expected_input_token_ids_fnv1a64;
    uint64_t reserved[2];
} qrt_qwen36_exact_first_token_request_v1_t;

typedef struct qrt_qwen36_exact_first_token_result_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t status;
    uint32_t completed;
    uint32_t batch_size;
    uint32_t output_token_id;
    uint32_t verifier_input_token_count;
    uint32_t reserved0;
    uint64_t input_token_ids_fnv1a64;
    uint64_t total_elapsed_ns;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    uint64_t reserved[2];
} qrt_qwen36_exact_first_token_result_v1_t;

typedef int (
    QRT_CDECL *qrt_qwen36_exact_first_token_v1_t
)(
    const qrt_qwen36_exact_first_token_request_v1_t *request,
    qrt_qwen36_exact_first_token_result_v1_t *result
);

typedef int (QRT_CDECL *qrt_qwen36_whole_provider_decode_emit_callback_v1_t)(
    void *user_data,
    uint64_t session_generation,
    uint32_t output_index,
    uint32_t token_id,
    uint64_t token_step_elapsed_ns,
    uint64_t token_end_elapsed_ns
);

/*
 * The provider owns every resident device allocation.  The core identifies a
 * session only by generation plus prompt digest; no device pointer crosses
 * this ABI.  A non-null callback is invoked synchronously for newly decoded
 * output indices 1..N-1, and returning zero cancels the remaining decode.
 * Any failure or cancellation after a cache mutation must invalidate the
 * provider session; a partially advanced session cannot be reused.
 */
typedef struct qrt_qwen36_whole_provider_decode_request_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t batch_size;
    uint32_t expected_prefix_token_count;
    uint32_t output_token_capacity;
    uint32_t initial_output_token_id;
    uint32_t reserved0;
    uint64_t expected_session_generation;
    uint64_t expected_prompt_token_ids_fnv1a64;
    qrt_qwen36_whole_provider_decode_emit_callback_v1_t emit_callback;
    void *emit_user_data;
    uint64_t reserved[2];
} qrt_qwen36_whole_provider_decode_request_v1_t;

/*
 * output_tokens[0] repeats the already-emitted prefill token.  Timing index
 * zero is therefore zero; indices 1..N-1 are cumulative from decode entry.
 */
typedef struct qrt_qwen36_whole_provider_decode_result_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t status;
    uint32_t completed;
    uint32_t batch_size;
    uint32_t output_token_capacity;
    uint32_t output_token_count;
    uint32_t prefill_token_count;
    uint32_t decode_token_count;
    uint32_t timing_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t session_generation;
    uint64_t prompt_token_ids_fnv1a64;
    uint64_t output_tokens_fnv1a64;
    uint64_t wall_clock_ns;
    uint64_t tpot_elapsed_ns;
    uint32_t output_tokens
        [QRT_QWEN36_WHOLE_PROVIDER_DECODE_MAX_OUTPUT_TOKENS];
    uint64_t token_step_elapsed_ns
        [QRT_QWEN36_WHOLE_PROVIDER_DECODE_MAX_OUTPUT_TOKENS];
    uint64_t token_end_elapsed_ns
        [QRT_QWEN36_WHOLE_PROVIDER_DECODE_MAX_OUTPUT_TOKENS];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    uint64_t reserved[2];
} qrt_qwen36_whole_provider_decode_result_v1_t;

typedef qrt_qwen36_whole_provider_decode_emit_callback_v1_t
    qrt_qwen36_whole_provider_prefix_emit_callback_v1_t;

/*
 * This provider ABI consumes a suffix against a real resident prefix.  The
 * provider owns a copy-on-write transaction and must restore the base session
 * before returning, including on a numerical or callback failure.  A non-null
 * callback publishes output zero immediately after suffix TTFT and publishes
 * each later decode span before the next span starts.  The core validates the
 * complete cached-prefix token identity before invoking it.
 */
typedef struct qrt_qwen36_whole_provider_prefix_request_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t batch_size;
    uint32_t expected_prefix_token_count;
    uint32_t suffix_token_count;
    uint32_t output_token_capacity;
    uint32_t expected_base_committed_token_count;
    uint64_t expected_session_generation;
    uint64_t expected_prompt_token_ids_fnv1a64;
    uint64_t expected_suffix_token_ids_fnv1a64;
    const uint32_t *suffix_tokens;
    uint32_t input_token_count;
    uint32_t reserved0;
    uint64_t expected_input_token_ids_fnv1a64;
    const uint32_t *input_tokens;
    qrt_qwen36_whole_provider_prefix_emit_callback_v1_t emit_callback;
    void *emit_user_data;
} qrt_qwen36_whole_provider_prefix_request_v1_t;

/*
 * teacher_forced_prediction_tokens[i] is the native next-token prediction
 * after suffix_tokens[i].  output_tokens[0] repeats the last such prediction;
 * later output tokens are ordinary autoregressive decode.  All result tokens
 * are copied out before the shadow transaction is rolled back.
 */
typedef struct qrt_qwen36_resident_prefix_cache_result_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t status;
    uint32_t completed;
    uint32_t batch_size;
    uint32_t provider_invoked;
    uint32_t exact_prefix_match;
    uint32_t copy_on_write_transaction;
    uint32_t state_restored;
    uint32_t prefix_token_count;
    uint32_t suffix_token_count;
    uint32_t output_token_capacity;
    uint32_t output_token_count;
    uint32_t teacher_forced_prediction_count;
    uint32_t base_committed_token_count;
    uint32_t mutated_committed_token_count;
    uint32_t restored_committed_token_count;
    uint32_t output_timing_count;
    uint64_t session_generation;
    uint64_t prompt_token_ids_fnv1a64;
    uint64_t input_token_ids_fnv1a64;
    uint64_t suffix_token_ids_fnv1a64;
    uint64_t teacher_forced_prediction_ids_fnv1a64;
    uint64_t output_token_ids_fnv1a64;
    uint64_t shadow_bytes;
    uint64_t clone_elapsed_ns;
    uint64_t suffix_elapsed_ns;
    uint64_t decode_elapsed_ns;
    uint64_t rollback_elapsed_ns;
    uint64_t ttft_elapsed_ns;
    uint64_t tpot_elapsed_ns;
    uint64_t total_elapsed_ns;
    uint64_t tpot_sample_count;
    uint32_t teacher_forced_prediction_tokens
        [QRT_QWEN36_RESIDENT_PREFIX_CACHE_MAX_SUFFIX_TOKENS];
    uint32_t output_tokens[QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS];
    /*
     * Index zero describes the suffix TTFT: step is ttft_elapsed_ns and end is
     * zero because no autoregressive decode token owns it.  Indices 1..N-1
     * retain the provider's exact per-token step and cumulative decode clocks.
     */
    uint64_t output_token_step_elapsed_ns
        [QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS];
    uint64_t output_token_end_elapsed_ns
        [QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    uint64_t reserved[2];
} qrt_qwen36_resident_prefix_cache_result_v1_t;

/*
 * The core sends its complete resident identity to the provider so reset
 * cannot retire another engine's singleton session.  An empty expected state
 * is valid and makes reset idempotent; nonempty identity fields are forbidden
 * in that case.
 */
typedef struct qrt_qwen36_resident_prefix_cache_reset_request_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t expected_session_valid;
    const qrt_engine_t *expected_owner_engine;
    uint32_t expected_prefix_token_count;
    uint32_t reserved0;
    uint64_t expected_session_generation;
    uint64_t expected_prompt_token_ids_fnv1a64;
    uint64_t reserved[2];
} qrt_qwen36_resident_prefix_cache_reset_request_v1_t;

/*
 * Provider-owned state is released under its resident-session mutex.  The
 * public engine entrypoint clears its cached token identity only after this
 * result passes the owner/generation/state contract.  Reset deliberately
 * retains shared model weights and the loaded provider module.
 */
typedef struct qrt_qwen36_resident_prefix_cache_reset_result_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t status;
    uint32_t completed;
    uint32_t provider_invoked;
    uint32_t provider_completed;
    uint32_t provider_owner_match;
    uint32_t idempotent;
    uint32_t core_session_valid_before;
    uint32_t core_session_valid_after;
    uint32_t provider_session_valid_before;
    uint32_t provider_session_valid_after;
    uint32_t core_prefix_token_count_before;
    uint32_t core_prefix_token_count_after;
    uint32_t provider_prefix_token_count_before;
    uint32_t provider_prefix_token_count_after;
    uint32_t provider_committed_token_count_before;
    uint32_t provider_committed_token_count_after;
    uint32_t shared_model_weights_retained;
    uint32_t reserved0;
    uint64_t core_session_generation_before;
    uint64_t core_session_generation_after;
    uint64_t provider_session_generation_before;
    uint64_t provider_session_generation_after;
    uint64_t provider_generation_clock_before;
    uint64_t provider_generation_clock_after;
    uint64_t core_prompt_token_ids_fnv1a64_before;
    uint64_t core_prompt_token_ids_fnv1a64_after;
    uint64_t provider_prompt_token_ids_fnv1a64_before;
    uint64_t provider_prompt_token_ids_fnv1a64_after;
    uint64_t provider_resident_state_bytes_before;
    uint64_t provider_resident_state_bytes_after;
    uint64_t provider_resident_state_bytes_released;
    uint64_t provider_reset_elapsed_ns;
    uint64_t total_elapsed_ns;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    uint64_t reserved[2];
} qrt_qwen36_resident_prefix_cache_reset_result_v1_t;

/*
 * The whole provider owns process-global model weights and one resident
 * session, while each qrt_engine_t owns an independent DLL reference and
 * prefix identity.  This lifecycle result makes engine registration and
 * release explicit so freeing a stale non-owner engine cannot tear down the
 * current owner's session or the shared model weights.
 */
typedef struct qrt_qwen36_whole_provider_engine_lifecycle_result_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t status;
    uint32_t completed;
    uint32_t operation;
    uint32_t engine_registered_before;
    uint32_t engine_registered_after;
    uint32_t engine_session_owner_before;
    uint32_t live_engine_count_before;
    uint32_t live_engine_count_after;
    uint32_t resident_session_valid_before;
    uint32_t resident_session_valid_after;
    uint32_t resident_session_released;
    uint32_t shared_model_weights_released;
    uint32_t shared_model_weights_retained;
    uint32_t reserved0;
    uint64_t resident_session_generation_before;
    uint64_t resident_session_generation_after;
    uint64_t provider_generation_clock;
    uint64_t resident_state_bytes_before;
    uint64_t resident_state_bytes_after;
    uint64_t total_elapsed_ns;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    uint64_t reserved[2];
} qrt_qwen36_whole_provider_engine_lifecycle_result_v1_t;

/*
 * Retained BF16 matrix plans are local to the native caller thread.  Product
 * worker pools use this boundary during engine/thread setup so their first
 * timed request reuses the provider DLL's real per-thread plans.  This call
 * does not mutate resident token/session state and does not establish request
 * concurrency by itself.
 */
typedef struct qrt_qwen36_whole_provider_thread_prepare_result_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t status;
    uint32_t completed;
    uint32_t engine_registered;
    uint32_t plan_count;
    uint32_t weight_bits;
    uint32_t quantized;
    uint32_t dflash_active;
    uint32_t mtp_active;
    uint32_t speculative_decode;
    uint32_t reserved0;
    uint64_t current_thread_id;
    uint64_t total_elapsed_ns;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    uint64_t reserved[2];
} qrt_qwen36_whole_provider_thread_prepare_result_v1_t;

/*
 * The direct-entry candidate bypasses the generic legacy request frame only
 * for an exact whole-provider q8192 request.  This snapshot makes every
 * activation and fallback observable so a performance run cannot silently
 * compare two legacy calls.  It is diagnostic and non-mutating.
 */
typedef struct qrt_qwen36_whole_provider_direct_entry_result_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t status;
    uint32_t completed;
    uint32_t enabled;
    uint32_t last_entry;
    int32_t last_status;
    uint32_t reserved0;
    uint64_t direct_attempt_count;
    uint64_t direct_completion_count;
    uint64_t direct_fallback_count;
    uint64_t legacy_entry_count;
    uint64_t early_prefill_attempt_count;
    uint64_t early_prefill_completion_count;
    uint64_t early_prefill_rejection_count;
    uint64_t last_early_prefill_request_elapsed_ns;
    uint64_t last_prologue_elapsed_ns;
    uint64_t last_total_elapsed_ns;
    uint64_t last_provider_ttft_elapsed_ns;
    uint64_t last_provider_call_start_elapsed_ns;
    uint64_t last_provider_return_elapsed_ns;
    uint64_t last_provider_to_direct_entry_elapsed_ns;
    uint64_t last_direct_pre_layer_stack_elapsed_ns;
    uint64_t last_direct_layer_stack_elapsed_ns;
    uint64_t last_direct_post_layer_stack_elapsed_ns;
    uint64_t last_provider_post_direct_to_reported_wall_elapsed_ns;
    uint64_t last_provider_phase_accounted_elapsed_ns;
    uint64_t last_provider_phase_accounting_error_ns;
    uint64_t last_request_count;
    uint32_t weight_bits;
    uint32_t quantized;
    uint32_t dflash_active;
    uint32_t mtp_active;
    uint32_t speculative_decode;
    uint32_t reserved1;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    uint64_t reserved[2];
} qrt_qwen36_whole_provider_direct_entry_result_v1_t;

/*
 * The current native provider intentionally owns one resident prefix slot.
 * This snapshot exposes the bounded replace-before-capture policy and the
 * most recent completed replacement eviction.  It is diagnostic and
 * non-mutating; token-array identity remains the authority for cache hits.
 */
typedef struct qrt_qwen36_resident_prefix_cache_capacity_result_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t status;
    uint32_t completed;
    uint32_t policy;
    uint32_t capacity_slots;
    uint32_t occupied_slots;
    uint32_t engine_registered;
    uint32_t current_owner_match;
    uint32_t shared_model_weights_retained;
    uint32_t last_eviction_completed;
    uint32_t last_eviction_cleanup_safe;
    uint32_t last_eviction_same_owner;
    uint32_t last_slots_before;
    uint32_t last_slots_after_release;
    uint32_t last_replacement_completed;
    uint64_t provider_generation_clock;
    uint64_t current_session_generation;
    uint64_t current_prompt_token_ids_fnv1a64;
    uint64_t current_resident_state_bytes;
    uint64_t eviction_count;
    uint64_t last_evicted_session_generation;
    uint64_t last_evicted_prompt_token_ids_fnv1a64;
    uint64_t last_resident_state_bytes_before;
    uint64_t last_resident_state_bytes_after_release;
    uint64_t last_resident_state_bytes_released;
    uint64_t last_replacement_session_generation;
    uint64_t last_replacement_prompt_token_ids_fnv1a64;
    uint64_t last_replacement_resident_state_bytes;
    uint64_t last_eviction_elapsed_ns;
    uint64_t total_elapsed_ns;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    uint64_t reserved[2];
} qrt_qwen36_resident_prefix_cache_capacity_result_v1_t;

/*
 * Native Windows prefix-hit, prefix-fallback, explicit-reset, and streaming
 * entry points serialize their mutable engine identity, callback state,
 * request counters, failure strings, and last-request telemetry.  The lock is
 * recursive so the compound fallback operation retains one outer transaction
 * across initial prefix attempt, ordinary full-prefix seed, and prefix retry.
 * A stream transaction likewise owns callback setup, the ordinary request,
 * callback parity validation, and cleanup.  The ordinary operation counter is
 * reserved until that broader concurrency contract is implemented.  This
 * snapshot is diagnostic and non-mutating.
 */
typedef struct qrt_qwen36_request_serialization_result_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t status;
    uint32_t completed;
    uint32_t initialized;
    uint32_t active_top_level_calls;
    uint32_t maximum_active_top_level_calls;
    uint32_t current_recursion_depth;
    uint32_t current_operation;
    uint32_t reserved0;
    uint64_t ticket_clock;
    uint64_t current_ticket;
    uint64_t last_completed_ticket;
    uint64_t acquisition_count;
    uint64_t completion_count;
    uint64_t contended_acquisition_count;
    uint64_t recursive_acquisition_count;
    uint64_t completion_order_violation_count;
    uint64_t total_wait_ns;
    uint64_t maximum_wait_ns;
    uint64_t ordinary_request_count;
    uint64_t prefix_request_count;
    uint64_t prefix_fallback_request_count;
    uint64_t prefix_reset_count;
    uint64_t stream_request_count;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    uint64_t reserved[2];
} qrt_qwen36_request_serialization_result_v1_t;

/*
 * This caller-visible wrapper preserves qrt_engine_request_tokens_prefix_v1:
 * the initial request remains fail-closed.  Only an exact-prefix identity miss
 * may trigger one ordinary full-prefix/output1 request followed by one retry
 * through the unchanged prefix-cache v1 surface.  Every other initial failure
 * is returned without fallback.
 */
typedef struct qrt_qwen36_resident_prefix_cache_fallback_result_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t status;
    uint32_t completed;
    uint32_t fallback_invoked;
    uint32_t retry_invoked;
    int32_t initial_status;
    int32_t seed_status;
    int32_t retry_status;
    uint32_t seed_output_token_count;
    uint32_t seed_output_token;
    uint32_t full_prefill_token_count;
    uint32_t reserved0;
    uint64_t session_generation_before;
    uint64_t session_generation_after_seed;
    uint64_t initial_attempt_elapsed_ns;
    uint64_t seed_elapsed_ns;
    uint64_t retry_elapsed_ns;
    uint64_t total_elapsed_ns;
    qrt_qwen36_resident_prefix_cache_result_v1_t initial_attempt;
    qrt_qwen36_resident_prefix_cache_result_v1_t hit_result;
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    uint64_t reserved[2];
} qrt_qwen36_resident_prefix_cache_fallback_result_v1_t;

typedef struct qrt_qwen36_prefill_descriptor_batch_decode_request_t {
    const char *model_dir;
    qrt_engine_t *resident_engine;
    size_t prefill_tokens;
    size_t output_token_capacity;
    uint32_t initial_output_token_id;
    uint64_t output_head_input_hidden_fnv1a64;
    unsigned int prefix_cache_state_attached;
    size_t prefix_cache_cached_prefix_tokens;
    size_t prefix_cache_suffix_tokens;
    size_t prefix_cache_executed_input_tokens;
    size_t prefix_cache_output_token_capacity;
    uint64_t prefix_cache_state_digest_fnv1a64;
    uint64_t prefix_cache_position_digest_fnv1a64;
    unsigned int decode_state_exported;
    uint64_t decode_state_digest_fnv1a64;
    size_t decode_state_token_count;
    size_t decode_state_position;
    uint32_t decode_state_token_id;
    uint64_t decode_state_token_loop_digest_fnv1a64;
    uint64_t decode_state_output_head_input_hidden_fnv1a64;
    uint64_t decode_state_prefix_state_digest_fnv1a64;
    uint64_t decode_state_prefix_position_digest_fnv1a64;
    unsigned int decode_state_cache_payload_exported;
    uint64_t decode_state_cache_payload_digest_fnv1a64;
    size_t decode_state_cache_layer_count;
    size_t decode_state_cache_token_count;
    unsigned int decode_state_cache_payload_view_attached;
    qrt_qwen36_decode_cache_payload_view_t decode_state_cache_payload_view;
} qrt_qwen36_prefill_descriptor_batch_decode_request_t;

typedef struct qrt_qwen36_prefill_descriptor_batch_decode_result_t {
    size_t output_token_capacity;
    size_t output_token_count;
    uint32_t output_tokens[QRT_PREFIX_CACHE_TARGET_OUTPUT_TOKENS];
    uint64_t output_tokens_fnv1a64;
    uint64_t output_sequence_elapsed_ns;
    uint64_t output_sequence_tpot_elapsed_ns;
    size_t output_sequence_tpot_sample_count;
    size_t output_sequence_prefill_sample_count;
    size_t output_sequence_decode_token_count;
    unsigned int output_sequence_autoregressive_decode;
    unsigned int output_sequence_decode_state_exported;
    char output_sequence_decode_missing_field[64];
    uint64_t output_sequence_decode_state_digest_fnv1a64;
    size_t output_sequence_decode_state_token_count;
    size_t output_sequence_decode_state_position;
    uint32_t output_sequence_decode_state_token_id;
    uint64_t output_sequence_decode_state_token_loop_digest_fnv1a64;
    uint64_t output_sequence_decode_state_output_head_input_hidden_fnv1a64;
    uint64_t output_sequence_decode_state_prefix_state_digest_fnv1a64;
    uint64_t output_sequence_decode_state_prefix_position_digest_fnv1a64;
    unsigned int output_sequence_decode_state_cache_payload_exported;
    uint64_t output_sequence_decode_state_cache_payload_digest_fnv1a64;
    size_t output_sequence_decode_state_cache_layer_count;
    size_t output_sequence_decode_state_cache_token_count;
    unsigned int output_sequence_decode_input_surface_exported;
    uint64_t output_sequence_decode_input_surface_digest_fnv1a64;
    uint64_t output_sequence_decode_input_surface_token_embedding_fnv1a64;
    size_t output_sequence_decode_input_surface_token_count;
    size_t output_sequence_decode_input_surface_position;
    uint32_t output_sequence_decode_input_surface_token_id;
    uint64_t output_sequence_decode_input_surface_cache_payload_digest_fnv1a64;
    unsigned int completed;
    char first_missing_runtime_context_field[64];
    char failure_stage[64];
    char failure[QRT_LOAD_ERROR_CAPACITY];
    uint64_t wall_clock_ns;
    qrt_qwen36_prefill_descriptor_batch_timing_t timing;
} qrt_qwen36_prefill_descriptor_batch_decode_result_t;

QRT_API const qrt_target_contract_t *qrt_target_contract(void);
QRT_API qrt_status_t qrt_engine_create(const qrt_engine_config_t *config, qrt_engine_t **out_engine);
QRT_API void qrt_engine_free(qrt_engine_t *engine);
QRT_API const char *qrt_engine_status(const qrt_engine_t *engine);
QRT_API qrt_status_t qrt_engine_report(const qrt_engine_t *engine, qrt_engine_report_t *out_report);
QRT_API qrt_status_t qrt_qwen36_exact_ok_layer0_second_step_trace(
    qrt_engine_t *engine,
    qrt_engine_report_t *out_report
);
QRT_API qrt_status_t qrt_engine_set_baseline_output_head_probe_token(
    qrt_engine_t *engine,
    uint32_t token_id
);
QRT_API qrt_status_t qrt_engine_set_baseline_output_head_probe_token_set(
    qrt_engine_t *engine,
    const uint32_t *token_ids,
    size_t token_count
);
QRT_API qrt_status_t qrt_engine_clear_baseline_output_head_probe_token(
    qrt_engine_t *engine
);
QRT_API qrt_status_t qrt_engine_attach_decode_prompt_context(
    qrt_engine_t *engine,
    const uint32_t *token_ids,
    size_t token_count
);
QRT_API qrt_status_t qrt_engine_attach_prefix_cache_state(
    qrt_engine_t *engine,
    const qrt_prefix_cache_state_attach_request_t *request,
    qrt_prefix_cache_state_attach_result_t *out_result
);
QRT_API qrt_status_t qrt_engine_set_baseline_product_q8192_layer1_frontier_probe(
    qrt_engine_t *engine,
    unsigned int enabled
);
QRT_API qrt_status_t qrt_engine_attach_baseline_product_q8192_layer1_frontier_continuation(
    qrt_engine_t *engine,
    qrt_qwen36_layer1_frontier_continuation_callback_t callback,
    void *user_data
);
QRT_API qrt_status_t qrt_engine_export_baseline_product_q8192_layer1_frontier(
    const qrt_engine_t *engine,
    qrt_qwen36_layer1_frontier_buffer_export_t *inout_export
);
QRT_API qrt_status_t qrt_engine_prepack_accounting(
    qrt_engine_t *engine,
    qrt_prepack_accounting_result_t *out_result
);
QRT_API qrt_status_t qrt_engine_prepack_selected_moe_weights(
    qrt_engine_t *engine,
    qrt_selected_moe_prepack_result_t *out_result
);
QRT_API qrt_status_t qrt_engine_require_qwen36_descriptor_request_path(
    qrt_engine_t *engine,
    unsigned int enabled
);
QRT_API qrt_status_t qrt_engine_generate(
    qrt_engine_t *engine,
    const char *prompt,
    char *output,
    size_t output_len
);
QRT_API qrt_status_t qrt_engine_request_tokens(
    qrt_engine_t *engine,
    const uint32_t *input_tokens,
    size_t input_token_count,
    uint32_t *output_tokens,
    size_t output_token_capacity,
    size_t *out_output_token_count
);
QRT_API qrt_status_t qrt_engine_request_tokens_stream_v1(
    qrt_engine_t *engine,
    const uint32_t *input_tokens,
    size_t input_token_count,
    uint32_t *output_tokens,
    size_t output_token_capacity,
    size_t *out_output_token_count,
    qrt_token_stream_callback_v1_t callback,
    void *user_data
);
QRT_API qrt_status_t qrt_engine_request_tokens_prefix_v1(
    qrt_engine_t *engine,
    const uint32_t *input_tokens,
    size_t input_token_count,
    size_t prefix_hit_token_count,
    uint32_t *output_tokens,
    size_t output_token_capacity,
    qrt_qwen36_resident_prefix_cache_result_v1_t *out_result
);
QRT_API qrt_status_t qrt_engine_request_tokens_prefix_stream_v1(
    qrt_engine_t *engine,
    const uint32_t *input_tokens,
    size_t input_token_count,
    size_t prefix_hit_token_count,
    uint32_t *output_tokens,
    size_t output_token_capacity,
    qrt_qwen36_resident_prefix_cache_result_v1_t *out_result,
    qrt_token_stream_callback_v1_t callback,
    void *user_data
);
QRT_API qrt_status_t qrt_engine_request_tokens_prefix_fallback_v1(
    qrt_engine_t *engine,
    const uint32_t *input_tokens,
    size_t input_token_count,
    size_t prefix_hit_token_count,
    uint32_t *output_tokens,
    size_t output_token_capacity,
    qrt_qwen36_resident_prefix_cache_fallback_result_v1_t *out_result
);
QRT_API qrt_status_t qrt_engine_reset_resident_prefix_cache_v1(
    qrt_engine_t *engine,
    qrt_qwen36_resident_prefix_cache_reset_result_v1_t *out_result
);
QRT_API qrt_status_t qrt_engine_whole_provider_lifecycle_snapshot_v1(
    qrt_engine_t *engine,
    qrt_qwen36_whole_provider_engine_lifecycle_result_v1_t *out_result
);
QRT_API qrt_status_t qrt_engine_prepare_request_thread_v1(
    qrt_engine_t *engine,
    qrt_qwen36_whole_provider_thread_prepare_result_v1_t *out_result
);
QRT_API qrt_status_t qrt_engine_whole_provider_direct_entry_snapshot_v1(
    qrt_engine_t *engine,
    qrt_qwen36_whole_provider_direct_entry_result_v1_t *out_result
);
QRT_API qrt_status_t qrt_engine_resident_prefix_cache_capacity_snapshot_v1(
    qrt_engine_t *engine,
    qrt_qwen36_resident_prefix_cache_capacity_result_v1_t *out_result
);
QRT_API qrt_status_t qrt_engine_request_serialization_snapshot_v1(
    qrt_engine_t *engine,
    qrt_qwen36_request_serialization_result_v1_t *out_result
);
QRT_API qrt_status_t qrt_engine_prefill(
    qrt_engine_t *engine,
    const uint32_t *input_tokens,
    size_t input_token_count,
    qrt_qwen36_baseline_execution_result_t *out_result
);
QRT_API qrt_status_t qrt_engine_decode_one(
    qrt_engine_t *engine,
    uint32_t input_token,
    uint32_t *out_token,
    qrt_qwen36_baseline_execution_result_t *out_result
);
QRT_API qrt_status_t qrt_qwen36_baseline_engine_surface_check(
    const qrt_engine_t *engine,
    qrt_qwen36_baseline_surface_check_t *out_result
);
QRT_API uint16_t qrt_float_to_bf16(float value);
QRT_API float qrt_bf16_to_float(uint16_t bits);
QRT_API qrt_status_t qrt_qwen36_fill_bf16_qmatvec_fixture(
    uint16_t *weights,
    size_t weight_elements,
    uint16_t *input,
    size_t input_elements
);
QRT_API qrt_status_t qrt_qwen36_bf16_qmatvec_cpu(
    const uint16_t *weights,
    size_t weight_elements,
    const uint16_t *input,
    size_t input_elements,
    float *output,
    size_t output_elements
);
QRT_API uint64_t qrt_fnv1a64_f32(const float *values, size_t count);
QRT_API float qrt_max_abs_diff_f32(const float *left, const float *right, size_t count);
QRT_API qrt_status_t qrt_qwen36_bf16_qmatvec_fixture_cpu(qrt_bf16_qmatvec_result_t *out_result);
QRT_API qrt_status_t qrt_micro_isomorphic_fixture_cpu(qrt_micro_result_t *out_result);
QRT_API qrt_status_t qrt_prefix_cache_owner_selftest(qrt_prefix_cache_selftest_result_t *out_result);
QRT_API qrt_status_t qrt_prefix_cache_full_model_probe(
    const char *model_dir,
    qrt_prefix_cache_full_model_result_t *out_result
);
QRT_API qrt_status_t qrt_prefix_cache_request_execution_probe(
    const char *model_dir,
    qrt_prefix_cache_request_execution_result_t *out_result
);
QRT_API qrt_status_t qrt_prefix_cache_state_injection_probe(
    const char *model_dir,
    qrt_prefix_cache_state_injection_result_t *out_result
);
QRT_API qrt_status_t qrt_prefix_cache_tensor_state_probe(
    const char *model_dir,
    qrt_prefix_cache_tensor_state_result_t *out_result
);
QRT_API qrt_status_t qrt_prefix_cache_full_kv_tensor_state_probe(
    const char *model_dir,
    qrt_prefix_cache_full_kv_tensor_state_result_t *out_result
);
QRT_API qrt_status_t qrt_prefix_cache_full_kv_compressed_state_probe(
    const char *model_dir,
    qrt_prefix_cache_full_kv_compressed_state_result_t *out_result
);
QRT_API const qrt_qwen36_baseline_plan_t *qrt_qwen36_baseline_plan(void);
QRT_API const qrt_qwen36_baseline_layer_descriptor_t *qrt_qwen36_baseline_layer_descriptor(unsigned int layer_index);
QRT_API const char *qrt_qwen36_baseline_attention_kind_name(qrt_qwen36_baseline_attention_kind_t kind);
QRT_API qrt_status_t qrt_qwen36_tensor_name(
    unsigned int layer_index,
    qrt_qwen36_tensor_kind_t kind,
    char *out_name,
    size_t out_name_len
);
QRT_API qrt_status_t qrt_qwen36_baseline_check(qrt_qwen36_baseline_check_t *out_result);
QRT_API qrt_status_t qrt_qwen36_linear_attention_table_check(
    qrt_engine_t *engine,
    qrt_qwen36_linear_attention_table_check_t *out_result
);
QRT_API qrt_status_t qrt_qwen36_full_attention_table_check(
    qrt_engine_t *engine,
    qrt_qwen36_full_attention_table_check_t *out_result
);
QRT_API qrt_status_t qrt_qwen36_full_attention_compute_check(
    qrt_engine_t *engine,
    qrt_qwen36_full_attention_compute_check_t *out_result
);
QRT_API qrt_status_t qrt_qwen36_moe_block_table_check(
    qrt_engine_t *engine,
    qrt_qwen36_moe_block_table_check_t *out_result
);
QRT_API qrt_status_t qrt_qwen36_moe_block_compute_check(
    qrt_engine_t *engine,
    qrt_qwen36_moe_block_compute_check_t *out_result
);
QRT_API qrt_status_t qrt_qwen36_output_head_compute_check(
    qrt_engine_t *engine,
    qrt_qwen36_output_head_compute_check_t *out_result
);
QRT_API qrt_status_t qrt_qwen36_baseline_block_integration_check(
    qrt_engine_t *engine,
    qrt_qwen36_baseline_block_integration_check_t *out_result
);
QRT_API qrt_status_t qrt_qwen36_baseline_hidden_handoff_check(
    const qrt_qwen36_baseline_block_integration_check_t *block_integration,
    qrt_qwen36_baseline_hidden_handoff_check_t *out_result
);
QRT_API qrt_status_t qrt_engine_attach_baseline_request_handoff(
    qrt_engine_t *engine,
    const qrt_qwen36_baseline_hidden_handoff_check_t *hidden_handoff,
    qrt_qwen36_baseline_request_handoff_check_t *out_result
);
QRT_API qrt_status_t qrt_qwen36_load_manifest_probe(
    const char *model_dir,
    qrt_load_manifest_result_t *out_result
);
QRT_API qrt_status_t qrt_qwen36_prepack_accounting_probe(
    const char *model_dir,
    const qrt_load_manifest_result_t *manifest,
    qrt_prepack_accounting_result_t *out_result
);
QRT_API const char *qrt_strerror(qrt_status_t status);

#ifdef __cplusplus
}
#endif

#endif
