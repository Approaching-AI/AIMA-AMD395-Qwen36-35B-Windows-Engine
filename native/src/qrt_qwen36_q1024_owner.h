#ifndef QRT_QWEN36_Q1024_OWNER_H
#define QRT_QWEN36_Q1024_OWNER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QRT_QWEN36_Q1024_OWNER_ABI_VERSION 1u
#define QRT_QWEN36_Q1024_OWNER_LAYER_COUNT 40u
#define QRT_QWEN36_Q1024_OWNER_SUFFIX_TOKENS 1024u
#ifndef QRT_QWEN36_Q1024_OWNER_PREFIX_TOKENS
#define QRT_QWEN36_Q1024_OWNER_PREFIX_TOKENS 16384u
#endif
#define QRT_QWEN36_Q1024_OWNER_FAILURE_CAPACITY 512u

typedef enum qrt_qwen36_q1024_owner_attention_kind_t {
    QRT_QWEN36_Q1024_OWNER_ATTENTION_LINEAR = 1,
    QRT_QWEN36_Q1024_OWNER_ATTENTION_FULL = 2
} qrt_qwen36_q1024_owner_attention_kind_t;

typedef enum qrt_qwen36_q1024_owner_element_kind_t {
    QRT_QWEN36_Q1024_OWNER_ELEMENT_NONE = 0,
    QRT_QWEN36_Q1024_OWNER_ELEMENT_F32 = 1,
    QRT_QWEN36_Q1024_OWNER_ELEMENT_BF16 = 2
} qrt_qwen36_q1024_owner_element_kind_t;

typedef struct qrt_qwen36_q1024_owner_prepare_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t reserved0;
    const char *gdn_kernel_dir;
    const char *gdn_provider_dll;
    const char *ck_provider_dll;
    const char *moe_kernel_dir;
    const char *moe_provider_dll;
    uint64_t reserved[4];
} qrt_qwen36_q1024_owner_prepare_v1_t;

typedef struct qrt_qwen36_q1024_owner_layer_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t layer_index;
    uint32_t attention_kind;
    uint32_t qkv_ring_element_kind;
    uint32_t linear_gate_values_are_decay;
    uint32_t reserved0;
    uint32_t reserved1;

    const uint16_t *input_norm;
    const uint16_t *post_attention_norm;
    const uint16_t *moe_router;
    const uint16_t *moe_expert_gate_up;
    const uint16_t *moe_expert_down;
    const uint16_t *moe_shared_gate;
    const uint16_t *moe_shared_gate_projection;
    const uint16_t *moe_shared_up_projection;
    const uint16_t *moe_shared_down;

    const uint16_t *linear_qkv;
    const uint16_t *linear_z;
    const uint16_t *linear_a;
    const uint16_t *linear_b;
    const uint16_t *linear_conv;
    const uint16_t *linear_a_log;
    const uint16_t *linear_dt_bias;
    const uint16_t *linear_norm;
    const uint16_t *linear_out;
    const float *linear_prefix_state;
    const void *linear_prefix_ring;
    float *linear_final_state;
    void *linear_final_ring;

    const uint16_t *full_q;
    const uint16_t *full_k;
    const uint16_t *full_v;
    const uint16_t *full_q_norm;
    const uint16_t *full_k_norm;
    const uint16_t *full_out;
    const uint16_t *full_prefix_k;
    const uint16_t *full_prefix_v;
    uint16_t *full_suffix_k;
    uint16_t *full_suffix_v;

    uint64_t reserved[4];
} qrt_qwen36_q1024_owner_layer_v1_t;

typedef struct qrt_qwen36_q1024_owner_request_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t batch_size;
    uint32_t prefix_tokens;
    uint32_t suffix_tokens;
    uint32_t absolute_position_base;
    uint32_t layer_count;
    uint64_t session_generation;
    uint64_t prompt_token_ids_fnv1a64;
    uint64_t suffix_token_ids_fnv1a64;
    const uint32_t *suffix_token_ids;
    const uint16_t *token_embedding_weights;
    const uint16_t *final_norm;
    const uint16_t *lm_head;
    const qrt_qwen36_q1024_owner_layer_v1_t *layers;
    uint64_t reserved[4];
} qrt_qwen36_q1024_owner_request_v1_t;

typedef struct qrt_qwen36_q1024_owner_result_v1_t {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t status;
    uint32_t completed;
    uint32_t layer_count;
    uint32_t linear_layer_count;
    uint32_t full_layer_count;
    uint32_t fixed_weight_pointer_count;
    uint32_t routed_weight_pointer_count;
    uint32_t teacher_prediction_count;
    uint32_t first_continuation_token;
    float first_continuation_logit;
    uint32_t nonfinite_top1_count;
    uint64_t session_generation;
    uint64_t prompt_token_ids_fnv1a64;
    uint64_t suffix_token_ids_fnv1a64;
    uint64_t teacher_prediction_ids_fnv1a64;
    uint64_t prefix_kv_copy_bytes;
    uint64_t linear_state_publish_bytes;
    uint64_t linear_ring_publish_bytes;
    uint64_t full_kv_publish_bytes;
    uint64_t total_elapsed_ns;
    uint64_t layer_stack_elapsed_ns;
    uint64_t output_head_elapsed_ns;
    uint32_t teacher_prediction_tokens
        [QRT_QWEN36_Q1024_OWNER_SUFFIX_TOKENS];
    char failure_stage[QRT_QWEN36_Q1024_OWNER_FAILURE_CAPACITY];
    char failure[QRT_QWEN36_Q1024_OWNER_FAILURE_CAPACITY];
    uint64_t reserved[4];
} qrt_qwen36_q1024_owner_result_v1_t;

typedef int (*qrt_qwen36_q1024_owner_prepare_v1_fn)(
    const qrt_qwen36_q1024_owner_prepare_v1_t *request
);
typedef int (*qrt_qwen36_q1024_owner_run_v1_fn)(
    const qrt_qwen36_q1024_owner_request_v1_t *request,
    qrt_qwen36_q1024_owner_result_v1_t *result
);
typedef const char *(*qrt_qwen36_q1024_owner_last_error_v1_fn)(void);
typedef void (*qrt_qwen36_q1024_owner_release_v1_fn)(void);

#ifdef __cplusplus
}
#endif

#endif
