#include "qwen36_baseline.h"

#include <stdio.h>
#include <string.h>

#define QRT_QWEN36_LINEAR_LAYER_MODULES 7u
#define QRT_QWEN36_FULL_LAYER_MODULES 8u

static qrt_qwen36_baseline_layer_descriptor_t g_qwen36_layers[QRT_QWEN36_LAYER_COUNT];
static qrt_qwen36_baseline_plan_t g_qwen36_plan;
static int g_qwen36_plan_initialized = 0;

static uint64_t qwen36_fnv1a64_update(uint64_t hash, const void *data, size_t count) {
    const unsigned char *bytes = (const unsigned char *)data;
    size_t index;

    for (index = 0; index < count; ++index) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int qwen36_is_full_attention_layer(unsigned int layer_index) {
    return layer_index < QRT_QWEN36_LAYER_COUNT && (layer_index % 4u) == 3u;
}

static void qwen36_baseline_init(void) {
    unsigned int layer_index;
    unsigned int linear_count = 0;
    unsigned int full_count = 0;

    if (g_qwen36_plan_initialized) {
        return;
    }

    memset(g_qwen36_layers, 0, sizeof(g_qwen36_layers));
    for (layer_index = 0; layer_index < QRT_QWEN36_LAYER_COUNT; ++layer_index) {
        qrt_qwen36_baseline_layer_descriptor_t *layer = &g_qwen36_layers[layer_index];
        int is_full = qwen36_is_full_attention_layer(layer_index);

        layer->layer_index = layer_index;
        layer->attention_kind = is_full ? QRT_QWEN36_BASELINE_ATTENTION_FULL : QRT_QWEN36_BASELINE_ATTENTION_LINEAR;
        layer->hidden_size = QRT_QWEN36_HIDDEN_SIZE;
        layer->attention_heads = QRT_QWEN36_ATTENTION_HEADS;
        layer->kv_heads = QRT_QWEN36_KV_HEADS;
        layer->head_dim = QRT_QWEN36_HEAD_DIM;
        layer->expert_count = QRT_QWEN36_EXPERT_COUNT;
        layer->experts_per_token = QRT_QWEN36_EXPERTS_PER_TOKEN;
        layer->expert_intermediate_size = QRT_QWEN36_EXPERT_INTERMEDIATE_SIZE;
        layer->has_moe = 1u;
        layer->has_shared_expert = 1u;
        layer->module_count = is_full ? QRT_QWEN36_FULL_LAYER_MODULES : QRT_QWEN36_LINEAR_LAYER_MODULES;

        if (is_full) {
            ++full_count;
        } else {
            ++linear_count;
        }
    }

    memset(&g_qwen36_plan, 0, sizeof(g_qwen36_plan));
    g_qwen36_plan.model_specific = 1u;
    g_qwen36_plan.batch_size = 1u;
    g_qwen36_plan.descriptor_table_driven = 1u;
    g_qwen36_plan.baseline_complete_engine_step = 1u;
    g_qwen36_plan.prefill_entrypoint_planned = 1u;
    g_qwen36_plan.decode_one_entrypoint_planned = 1u;
    g_qwen36_plan.layer_count = QRT_QWEN36_LAYER_COUNT;
    g_qwen36_plan.linear_attention_layer_count = linear_count;
    g_qwen36_plan.full_attention_layer_count = full_count;
    g_qwen36_plan.hidden_size = QRT_QWEN36_HIDDEN_SIZE;
    g_qwen36_plan.vocab_size = QRT_QWEN36_VOCAB_SIZE;
    g_qwen36_plan.expert_count = QRT_QWEN36_EXPERT_COUNT;
    g_qwen36_plan.experts_per_token = QRT_QWEN36_EXPERTS_PER_TOKEN;
    g_qwen36_plan.expert_intermediate_size = QRT_QWEN36_EXPERT_INTERMEDIATE_SIZE;
    g_qwen36_plan.layers = g_qwen36_layers;
    g_qwen36_plan_initialized = 1;
}

static qrt_status_t qwen36_write_name(char *out_name, size_t out_name_len, const char *fmt, unsigned int layer_index) {
    int written;

    if (out_name == NULL || out_name_len == 0u || fmt == NULL) {
        return QRT_STATUS_INVALID_ARGUMENT;
    }
    written = snprintf(out_name, out_name_len, fmt, layer_index);
    if (written < 0 || (size_t)written >= out_name_len) {
        out_name[0] = '\0';
        return QRT_STATUS_OUT_OF_MEMORY;
    }
    return QRT_STATUS_OK;
}

static qrt_status_t qwen36_write_fixed_name(char *out_name, size_t out_name_len, const char *value) {
    size_t value_len;

    if (out_name == NULL || out_name_len == 0u || value == NULL) {
        return QRT_STATUS_INVALID_ARGUMENT;
    }
    value_len = strlen(value);
    if (value_len + 1u > out_name_len) {
        out_name[0] = '\0';
        return QRT_STATUS_OUT_OF_MEMORY;
    }
    memcpy(out_name, value, value_len + 1u);
    return QRT_STATUS_OK;
}

const qrt_qwen36_baseline_plan_t *qrt_qwen36_baseline_plan(void) {
    qwen36_baseline_init();
    return &g_qwen36_plan;
}

const qrt_qwen36_baseline_layer_descriptor_t *qrt_qwen36_baseline_layer_descriptor(unsigned int layer_index) {
    qwen36_baseline_init();
    if (layer_index >= QRT_QWEN36_LAYER_COUNT) {
        return NULL;
    }
    return &g_qwen36_layers[layer_index];
}

const char *qrt_qwen36_baseline_attention_kind_name(qrt_qwen36_baseline_attention_kind_t kind) {
    switch (kind) {
        case QRT_QWEN36_BASELINE_ATTENTION_LINEAR:
            return "linear_attention";
        case QRT_QWEN36_BASELINE_ATTENTION_FULL:
            return "full_attention";
        default:
            return "unknown";
    }
}

qrt_status_t qrt_qwen36_tensor_name(
    unsigned int layer_index,
    qrt_qwen36_tensor_kind_t kind,
    char *out_name,
    size_t out_name_len
) {
    if (kind != QRT_QWEN36_TENSOR_TOKEN_EMBEDDING &&
        kind != QRT_QWEN36_TENSOR_FINAL_NORM &&
        kind != QRT_QWEN36_TENSOR_LM_HEAD &&
        layer_index >= QRT_QWEN36_LAYER_COUNT) {
        return QRT_STATUS_INVALID_ARGUMENT;
    }

    switch (kind) {
        case QRT_QWEN36_TENSOR_TOKEN_EMBEDDING:
            return qwen36_write_fixed_name(out_name, out_name_len, "model.language_model.embed_tokens.weight");
        case QRT_QWEN36_TENSOR_INPUT_NORM:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.input_layernorm.weight", layer_index);
        case QRT_QWEN36_TENSOR_LINEAR_ATTN_QKV:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.linear_attn.in_proj_qkv.weight", layer_index);
        case QRT_QWEN36_TENSOR_LINEAR_ATTN_Z:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.linear_attn.in_proj_z.weight", layer_index);
        case QRT_QWEN36_TENSOR_LINEAR_ATTN_A:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.linear_attn.in_proj_a.weight", layer_index);
        case QRT_QWEN36_TENSOR_LINEAR_ATTN_B:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.linear_attn.in_proj_b.weight", layer_index);
        case QRT_QWEN36_TENSOR_LINEAR_ATTN_CONV:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.linear_attn.conv1d.weight", layer_index);
        case QRT_QWEN36_TENSOR_LINEAR_ATTN_A_LOG:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.linear_attn.A_log", layer_index);
        case QRT_QWEN36_TENSOR_LINEAR_ATTN_DT_BIAS:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.linear_attn.dt_bias", layer_index);
        case QRT_QWEN36_TENSOR_LINEAR_ATTN_NORM:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.linear_attn.norm.weight", layer_index);
        case QRT_QWEN36_TENSOR_LINEAR_ATTN_OUT_PROJ:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.linear_attn.out_proj.weight", layer_index);
        case QRT_QWEN36_TENSOR_FULL_ATTN_Q:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.self_attn.q_proj.weight", layer_index);
        case QRT_QWEN36_TENSOR_FULL_ATTN_K:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.self_attn.k_proj.weight", layer_index);
        case QRT_QWEN36_TENSOR_FULL_ATTN_V:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.self_attn.v_proj.weight", layer_index);
        case QRT_QWEN36_TENSOR_FULL_ATTN_O:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.self_attn.o_proj.weight", layer_index);
        case QRT_QWEN36_TENSOR_FULL_ATTN_Q_NORM:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.self_attn.q_norm.weight", layer_index);
        case QRT_QWEN36_TENSOR_FULL_ATTN_K_NORM:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.self_attn.k_norm.weight", layer_index);
        case QRT_QWEN36_TENSOR_POST_ATTENTION_NORM:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.post_attention_layernorm.weight", layer_index);
        case QRT_QWEN36_TENSOR_MOE_ROUTER:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.mlp.gate.weight", layer_index);
        case QRT_QWEN36_TENSOR_MOE_EXPERT_GATE_UP:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.mlp.experts.gate_up_proj", layer_index);
        case QRT_QWEN36_TENSOR_MOE_EXPERT_DOWN:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.mlp.experts.down_proj", layer_index);
        case QRT_QWEN36_TENSOR_MOE_SHARED_GATE:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.mlp.shared_expert_gate.weight", layer_index);
        case QRT_QWEN36_TENSOR_MOE_SHARED_GATE_PROJ:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.mlp.shared_expert.gate_proj.weight", layer_index);
        case QRT_QWEN36_TENSOR_MOE_SHARED_UP_PROJ:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.mlp.shared_expert.up_proj.weight", layer_index);
        case QRT_QWEN36_TENSOR_MOE_SHARED_DOWN:
            return qwen36_write_name(out_name, out_name_len, "model.language_model.layers.%u.mlp.shared_expert.down_proj.weight", layer_index);
        case QRT_QWEN36_TENSOR_FINAL_NORM:
            return qwen36_write_fixed_name(out_name, out_name_len, "model.language_model.norm.weight");
        case QRT_QWEN36_TENSOR_LM_HEAD:
            return qwen36_write_fixed_name(out_name, out_name_len, "lm_head.weight");
        default:
            return QRT_STATUS_INVALID_ARGUMENT;
    }
}

qrt_status_t qrt_qwen36_baseline_check(qrt_qwen36_baseline_check_t *out_result) {
    const qrt_qwen36_baseline_plan_t *plan;
    uint64_t hash = UINT64_C(1469598103934665603);
    unsigned int index;
    qrt_status_t status;

    if (out_result == NULL) {
        return QRT_STATUS_INVALID_ARGUMENT;
    }
    memset(out_result, 0, sizeof(*out_result));
    plan = qrt_qwen36_baseline_plan();

    out_result->model_specific = plan->model_specific;
    out_result->batch_size = plan->batch_size;
    out_result->descriptor_table_driven = plan->descriptor_table_driven;
    out_result->prefill_entrypoint_planned = plan->prefill_entrypoint_planned;
    out_result->decode_one_entrypoint_planned = plan->decode_one_entrypoint_planned;
    out_result->layer_count = plan->layer_count;
    out_result->linear_attention_layer_count = plan->linear_attention_layer_count;
    out_result->full_attention_layer_count = plan->full_attention_layer_count;
    out_result->first_full_attention_layer = 3u;
    out_result->last_full_attention_layer = 39u;
    out_result->layer_descriptor_count_matches =
        (plan->layer_count == QRT_QWEN36_LAYER_COUNT &&
         plan->linear_attention_layer_count == QRT_QWEN36_LINEAR_ATTENTION_LAYERS &&
         plan->full_attention_layer_count == QRT_QWEN36_FULL_ATTENTION_LAYERS) ? 1u : 0u;

    out_result->layer_schedule_pass = 1u;
    for (index = 0; index < plan->layer_count; ++index) {
        const qrt_qwen36_baseline_layer_descriptor_t *layer = &plan->layers[index];
        int expected_full = qwen36_is_full_attention_layer(index);
        qrt_qwen36_baseline_attention_kind_t expected_kind =
            expected_full ? QRT_QWEN36_BASELINE_ATTENTION_FULL : QRT_QWEN36_BASELINE_ATTENTION_LINEAR;

        if (layer->layer_index != index ||
            layer->attention_kind != expected_kind ||
            layer->hidden_size != QRT_QWEN36_HIDDEN_SIZE ||
            layer->expert_count != QRT_QWEN36_EXPERT_COUNT ||
            layer->experts_per_token != QRT_QWEN36_EXPERTS_PER_TOKEN ||
            layer->has_moe != 1u ||
            layer->has_shared_expert != 1u) {
            out_result->layer_schedule_pass = 0u;
        }
        hash = qwen36_fnv1a64_update(hash, layer, sizeof(*layer));
    }
    out_result->descriptor_fnv1a64 = hash;

    status = qrt_qwen36_tensor_name(0u, QRT_QWEN36_TENSOR_TOKEN_EMBEDDING, out_result->sample_token_embedding, sizeof(out_result->sample_token_embedding));
    if (status == QRT_STATUS_OK) {
        status = qrt_qwen36_tensor_name(0u, QRT_QWEN36_TENSOR_LINEAR_ATTN_QKV, out_result->sample_linear_qkv, sizeof(out_result->sample_linear_qkv));
    }
    if (status == QRT_STATUS_OK) {
        status = qrt_qwen36_tensor_name(3u, QRT_QWEN36_TENSOR_FULL_ATTN_Q, out_result->sample_full_q, sizeof(out_result->sample_full_q));
    }
    if (status == QRT_STATUS_OK) {
        status = qrt_qwen36_tensor_name(10u, QRT_QWEN36_TENSOR_MOE_ROUTER, out_result->sample_moe_router, sizeof(out_result->sample_moe_router));
    }
    if (status == QRT_STATUS_OK) {
        status = qrt_qwen36_tensor_name(0u, QRT_QWEN36_TENSOR_FINAL_NORM, out_result->sample_final_norm, sizeof(out_result->sample_final_norm));
    }
    if (status == QRT_STATUS_OK) {
        status = qrt_qwen36_tensor_name(0u, QRT_QWEN36_TENSOR_LM_HEAD, out_result->sample_lm_head, sizeof(out_result->sample_lm_head));
    }
    out_result->tensor_name_builder_pass = (status == QRT_STATUS_OK) ? 1u : 0u;

    out_result->correctness_pass =
        (out_result->model_specific == 1u &&
         out_result->batch_size == 1u &&
         out_result->descriptor_table_driven == 1u &&
         out_result->prefill_entrypoint_planned == 1u &&
         out_result->decode_one_entrypoint_planned == 1u &&
         out_result->layer_descriptor_count_matches == 1u &&
         out_result->layer_schedule_pass == 1u &&
         out_result->tensor_name_builder_pass == 1u) ? 1 : 0;

    if (!out_result->correctness_pass) {
        qwen36_write_fixed_name(out_result->failure, sizeof(out_result->failure), "baseline descriptor check failed");
        return QRT_STATUS_UNSUPPORTED;
    }
    return QRT_STATUS_OK;
}
