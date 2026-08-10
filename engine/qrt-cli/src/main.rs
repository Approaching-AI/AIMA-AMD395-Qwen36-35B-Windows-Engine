use std::env;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

const QRT_STATUS_OK: c_int = 0;
const QRT_STATUS_NOT_IMPLEMENTED: c_int = 3;
const QRT_STATUS_UNSUPPORTED: c_int = 4;
const QRT_QWEN36_EXECUTION_PREFILL: c_int = 1;
const QRT_QWEN36_EXECUTION_DECODE_ONE: c_int = 2;
const QRT_QWEN36_TENSOR_NAME_CAPACITY: usize = 160;
const QRT_LOAD_ERROR_CAPACITY: usize = 192;

#[repr(C)]
struct QrtTargetContract {
    repo: *const c_char,
    model: *const c_char,
    precision: *const c_char,
    primary_runtime_language: *const c_char,
    tooling_language: *const c_char,
    target_os: *const c_char,
    target_device: *const c_char,
    test_host: *const c_char,
    dependency_policy: *const c_char,
    status: *const c_char,
}

#[repr(C)]
struct QrtEngineConfig {
    model_path: *const c_char,
    context_tokens: usize,
    batch_size: u32,
}

#[repr(C)]
struct QrtBf16QmatvecResult {
    hidden_size: usize,
    expert_intermediate_size: usize,
    weight_elements: usize,
    payload_bytes: usize,
    expected_output_fnv1a64: u64,
    output_fnv1a64: u64,
    max_abs_diff: f32,
    correctness_pass: c_int,
}

#[repr(C)]
struct QrtMicroResult {
    hidden_size: usize,
    expert_count: usize,
    expert_intermediate_size: usize,
    vocab_size: usize,
    prompt_tokens: usize,
    kv_tokens: usize,
    selected_expert: c_int,
    generated_token: c_int,
    streaming_callbacks: c_int,
    exact_prefix_hit_tokens: c_int,
    shared_prefix_hit_tokens: c_int,
    unrelated_prefix_hit_tokens: c_int,
    contamination_guard_pass: c_int,
    expected_trace_fnv1a64: u64,
    trace_fnv1a64: u64,
    rmsnorm_pass: c_int,
    rope_pass: c_int,
    attention_kv_pass: c_int,
    linear_attention_pass: c_int,
    moe_pass: c_int,
    lm_head_pass: c_int,
    prefix_cache_pass: c_int,
    streaming_pass: c_int,
    correctness_pass: c_int,
}

#[repr(C)]
struct QrtQwen36BaselineCheck {
    model_specific: u32,
    batch_size: u32,
    descriptor_table_driven: u32,
    prefill_entrypoint_planned: u32,
    decode_one_entrypoint_planned: u32,
    layer_count: u32,
    linear_attention_layer_count: u32,
    full_attention_layer_count: u32,
    first_full_attention_layer: u32,
    last_full_attention_layer: u32,
    layer_descriptor_count_matches: u32,
    layer_schedule_pass: u32,
    tensor_name_builder_pass: u32,
    descriptor_fnv1a64: u64,
    sample_token_embedding: [c_char; QRT_QWEN36_TENSOR_NAME_CAPACITY],
    sample_linear_qkv: [c_char; QRT_QWEN36_TENSOR_NAME_CAPACITY],
    sample_full_q: [c_char; QRT_QWEN36_TENSOR_NAME_CAPACITY],
    sample_moe_router: [c_char; QRT_QWEN36_TENSOR_NAME_CAPACITY],
    sample_final_norm: [c_char; QRT_QWEN36_TENSOR_NAME_CAPACITY],
    sample_lm_head: [c_char; QRT_QWEN36_TENSOR_NAME_CAPACITY],
    correctness_pass: c_int,
    failure: [c_char; QRT_LOAD_ERROR_CAPACITY],
}

#[repr(C)]
struct QrtQwen36BaselineExecutionResult {
    mode: c_int,
    input_token_count: usize,
    output_token_count: usize,
    baseline_plan_attached: u32,
    descriptor_table_driven: u32,
    module_dispatch_started: u32,
    layer_count: u32,
    linear_attention_layer_count: u32,
    full_attention_layer_count: u32,
    first_missing_layer: u32,
    descriptor_fnv1a64: u64,
    token_embedding_materialized: u32,
    input_embeddings_materialized_count: usize,
    input_embeddings_fnv1a64: u64,
    last_token_embedding_fnv1a64: u64,
    token_embedding_bytes_read: u64,
    token_embedding_elapsed_ns: u64,
    layer0_input_norm_applied: u32,
    layer0_input_norm_materialized_count: usize,
    layer0_input_norm_fnv1a64: u64,
    last_layer0_input_norm_fnv1a64: u64,
    layer0_input_norm_weight_fnv1a64: u64,
    layer0_input_norm_weight_bytes_read: u64,
    layer0_input_norm_elapsed_ns: u64,
    layer0_qkv_projection_applied: u32,
    layer0_qkv_projection_materialized_count: usize,
    layer0_qkv_projection_fnv1a64: u64,
    last_layer0_qkv_projection_fnv1a64: u64,
    layer0_qkv_projection_weight_fnv1a64: u64,
    layer0_qkv_projection_weight_bytes_read: u64,
    layer0_qkv_projection_elapsed_ns: u64,
    layer0_zab_projection_applied: u32,
    layer0_zab_projection_materialized_count: usize,
    layer0_zab_projection_fnv1a64: u64,
    last_layer0_zab_projection_fnv1a64: u64,
    layer0_z_projection_weight_fnv1a64: u64,
    layer0_a_projection_weight_fnv1a64: u64,
    layer0_b_projection_weight_fnv1a64: u64,
    layer0_zab_projection_weight_bytes_read: u64,
    layer0_zab_projection_elapsed_ns: u64,
    layer0_conv_qkv_applied: u32,
    layer0_conv_qkv_materialized_count: usize,
    layer0_conv_qkv_fnv1a64: u64,
    last_layer0_conv_qkv_fnv1a64: u64,
    layer0_conv_qkv_weight_fnv1a64: u64,
    layer0_conv_qkv_weight_bytes_read: u64,
    layer0_conv_qkv_elapsed_ns: u64,
    layer0_postconv_qkv_applied: u32,
    layer0_postconv_qkv_materialized_count: usize,
    layer0_postconv_q_scaled_fnv1a64: u64,
    last_layer0_postconv_q_scaled_fnv1a64: u64,
    layer0_postconv_k_norm_fnv1a64: u64,
    last_layer0_postconv_k_norm_fnv1a64: u64,
    layer0_postconv_value_fnv1a64: u64,
    last_layer0_postconv_value_fnv1a64: u64,
    layer0_postconv_qkv_elapsed_ns: u64,
    layer0_gate_rows_applied: u32,
    layer0_gate_rows_materialized_count: usize,
    layer0_gate_g_fnv1a64: u64,
    last_layer0_gate_g_fnv1a64: u64,
    layer0_gate_beta_fnv1a64: u64,
    last_layer0_gate_beta_fnv1a64: u64,
    layer0_a_log_fnv1a64: u64,
    layer0_dt_bias_fnv1a64: u64,
    layer0_gate_weight_bytes_read: u64,
    layer0_gate_rows_elapsed_ns: u64,
    layer0_core_rows_applied: u32,
    layer0_core_rows_materialized_count: usize,
    layer0_core_rows_fnv1a64: u64,
    last_layer0_core_rows_fnv1a64: u64,
    layer0_core_final_state_fnv1a64: u64,
    layer0_core_rows_elapsed_ns: u64,
    layer0_gated_rmsnorm_applied: u32,
    layer0_gated_rmsnorm_materialized_count: usize,
    layer0_gated_rmsnorm_fnv1a64: u64,
    last_layer0_gated_rmsnorm_fnv1a64: u64,
    layer0_linear_norm_weight_fnv1a64: u64,
    layer0_linear_norm_weight_bytes_read: u64,
    layer0_gated_rmsnorm_elapsed_ns: u64,
    layer0_out_projection_applied: u32,
    layer0_out_projection_materialized_count: usize,
    layer0_out_projection_fnv1a64: u64,
    last_layer0_out_projection_fnv1a64: u64,
    layer0_out_projection_weight_fnv1a64: u64,
    layer0_out_projection_weight_bytes_read: u64,
    layer0_out_projection_elapsed_ns: u64,
    layer0_residual_hidden_applied: u32,
    layer0_residual_hidden_materialized_count: usize,
    layer0_residual_hidden_fnv1a64: u64,
    last_layer0_residual_hidden_fnv1a64: u64,
    layer0_residual_hidden_elapsed_ns: u64,
    layer0_post_attention_rmsnorm_applied: u32,
    layer0_post_attention_rmsnorm_materialized_count: usize,
    layer0_post_attention_rmsnorm_fnv1a64: u64,
    last_layer0_post_attention_rmsnorm_fnv1a64: u64,
    layer0_post_attention_norm_weight_fnv1a64: u64,
    layer0_post_attention_norm_weight_bytes_read: u64,
    layer0_post_attention_rmsnorm_elapsed_ns: u64,
    layer0_moe_router_applied: u32,
    layer0_moe_router_materialized_count: usize,
    layer0_moe_router_logits_fnv1a64: u64,
    last_layer0_moe_router_logits_fnv1a64: u64,
    layer0_moe_router_topk_ids_fnv1a64: u64,
    last_layer0_moe_router_topk_ids_fnv1a64: u64,
    layer0_moe_router_topk_weights_fnv1a64: u64,
    last_layer0_moe_router_topk_weights_fnv1a64: u64,
    layer0_moe_router_weight_fnv1a64: u64,
    layer0_moe_router_weight_bytes_read: u64,
    layer0_moe_router_elapsed_ns: u64,
    layer0_moe_expert_applied: u32,
    layer0_moe_expert_materialized_count: usize,
    layer0_moe_expert_selected_count: usize,
    layer0_moe_expert_routed_fnv1a64: u64,
    last_layer0_moe_expert_routed_fnv1a64: u64,
    layer0_moe_expert_selected_ids_fnv1a64: u64,
    layer0_moe_expert_gate_up_weight_fnv1a64: u64,
    layer0_moe_expert_down_weight_fnv1a64: u64,
    layer0_moe_expert_weight_bytes_read: u64,
    layer0_moe_expert_elapsed_ns: u64,
    layer0_moe_shared_expert_applied: u32,
    layer0_moe_shared_expert_materialized_count: usize,
    layer0_moe_shared_expert_fnv1a64: u64,
    last_layer0_moe_shared_expert_fnv1a64: u64,
    layer0_moe_combined_fnv1a64: u64,
    last_layer0_moe_combined_fnv1a64: u64,
    layer0_moe_shared_gate_fnv1a64: u64,
    last_layer0_moe_shared_gate_fnv1a64: u64,
    layer0_moe_shared_gate_weight_fnv1a64: u64,
    layer0_moe_shared_gate_proj_weight_fnv1a64: u64,
    layer0_moe_shared_up_proj_weight_fnv1a64: u64,
    layer0_moe_shared_down_weight_fnv1a64: u64,
    layer0_moe_shared_expert_weight_bytes_read: u64,
    layer0_moe_shared_expert_elapsed_ns: u64,
    layer0_output_residual_applied: u32,
    layer0_output_residual_materialized_count: usize,
    layer0_output_residual_fnv1a64: u64,
    last_layer0_output_residual_fnv1a64: u64,
    layer0_output_residual_elapsed_ns: u64,
    layer1_input_norm_applied: u32,
    layer1_input_norm_materialized_count: usize,
    layer1_input_norm_fnv1a64: u64,
    last_layer1_input_norm_fnv1a64: u64,
    layer1_input_norm_weight_fnv1a64: u64,
    layer1_input_norm_weight_bytes_read: u64,
    layer1_input_norm_elapsed_ns: u64,
    layer1_qkv_projection_applied: u32,
    layer1_qkv_projection_materialized_count: usize,
    layer1_qkv_projection_fnv1a64: u64,
    last_layer1_qkv_projection_fnv1a64: u64,
    layer1_qkv_projection_weight_fnv1a64: u64,
    layer1_qkv_projection_weight_bytes_read: u64,
    layer1_qkv_projection_elapsed_ns: u64,
    layer1_zab_projection_applied: u32,
    layer1_zab_projection_materialized_count: usize,
    layer1_zab_projection_fnv1a64: u64,
    last_layer1_zab_projection_fnv1a64: u64,
    layer1_z_projection_weight_fnv1a64: u64,
    layer1_a_projection_weight_fnv1a64: u64,
    layer1_b_projection_weight_fnv1a64: u64,
    layer1_zab_projection_weight_bytes_read: u64,
    layer1_zab_projection_elapsed_ns: u64,
    layer1_conv_qkv_applied: u32,
    layer1_conv_qkv_materialized_count: usize,
    layer1_conv_qkv_fnv1a64: u64,
    last_layer1_conv_qkv_fnv1a64: u64,
    layer1_conv_qkv_weight_fnv1a64: u64,
    layer1_conv_qkv_weight_bytes_read: u64,
    layer1_conv_qkv_elapsed_ns: u64,
    layer1_postconv_qkv_applied: u32,
    layer1_postconv_qkv_materialized_count: usize,
    layer1_postconv_q_scaled_fnv1a64: u64,
    last_layer1_postconv_q_scaled_fnv1a64: u64,
    layer1_postconv_k_norm_fnv1a64: u64,
    last_layer1_postconv_k_norm_fnv1a64: u64,
    layer1_postconv_value_fnv1a64: u64,
    last_layer1_postconv_value_fnv1a64: u64,
    layer1_postconv_qkv_elapsed_ns: u64,
    layer1_gate_rows_applied: u32,
    layer1_gate_rows_materialized_count: usize,
    layer1_gate_g_fnv1a64: u64,
    last_layer1_gate_g_fnv1a64: u64,
    layer1_gate_beta_fnv1a64: u64,
    last_layer1_gate_beta_fnv1a64: u64,
    layer1_a_log_fnv1a64: u64,
    layer1_dt_bias_fnv1a64: u64,
    layer1_gate_weight_bytes_read: u64,
    layer1_gate_rows_elapsed_ns: u64,
    layer1_core_rows_applied: u32,
    layer1_core_rows_materialized_count: usize,
    layer1_core_rows_fnv1a64: u64,
    last_layer1_core_rows_fnv1a64: u64,
    layer1_core_final_state_fnv1a64: u64,
    layer1_core_rows_elapsed_ns: u64,
    layer1_gated_rmsnorm_applied: u32,
    layer1_gated_rmsnorm_materialized_count: usize,
    layer1_gated_rmsnorm_fnv1a64: u64,
    last_layer1_gated_rmsnorm_fnv1a64: u64,
    layer1_linear_norm_weight_fnv1a64: u64,
    layer1_linear_norm_weight_bytes_read: u64,
    layer1_gated_rmsnorm_elapsed_ns: u64,
    layer1_out_projection_applied: u32,
    layer1_out_projection_materialized_count: usize,
    layer1_out_projection_fnv1a64: u64,
    last_layer1_out_projection_fnv1a64: u64,
    layer1_out_projection_weight_fnv1a64: u64,
    layer1_out_projection_weight_bytes_read: u64,
    layer1_out_projection_elapsed_ns: u64,
    layer1_residual_hidden_applied: u32,
    layer1_residual_hidden_materialized_count: usize,
    layer1_residual_hidden_fnv1a64: u64,
    last_layer1_residual_hidden_fnv1a64: u64,
    layer1_residual_hidden_elapsed_ns: u64,
    layer1_post_attention_rmsnorm_applied: u32,
    layer1_post_attention_rmsnorm_materialized_count: usize,
    layer1_post_attention_rmsnorm_fnv1a64: u64,
    last_layer1_post_attention_rmsnorm_fnv1a64: u64,
    layer1_post_attention_norm_weight_fnv1a64: u64,
    layer1_post_attention_norm_weight_bytes_read: u64,
    layer1_post_attention_rmsnorm_elapsed_ns: u64,
    layer1_moe_router_applied: u32,
    layer1_moe_router_materialized_count: usize,
    layer1_moe_router_logits_fnv1a64: u64,
    last_layer1_moe_router_logits_fnv1a64: u64,
    layer1_moe_router_topk_ids_fnv1a64: u64,
    last_layer1_moe_router_topk_ids_fnv1a64: u64,
    layer1_moe_router_topk_weights_fnv1a64: u64,
    last_layer1_moe_router_topk_weights_fnv1a64: u64,
    layer1_moe_router_weight_fnv1a64: u64,
    layer1_moe_router_weight_bytes_read: u64,
    layer1_moe_router_elapsed_ns: u64,
    layer1_moe_expert_applied: u32,
    layer1_moe_expert_materialized_count: usize,
    layer1_moe_expert_selected_count: usize,
    layer1_moe_expert_routed_fnv1a64: u64,
    last_layer1_moe_expert_routed_fnv1a64: u64,
    layer1_moe_expert_selected_ids_fnv1a64: u64,
    layer1_moe_expert_gate_up_weight_fnv1a64: u64,
    layer1_moe_expert_down_weight_fnv1a64: u64,
    layer1_moe_expert_weight_bytes_read: u64,
    layer1_moe_expert_elapsed_ns: u64,
    layer1_moe_shared_expert_applied: u32,
    layer1_moe_shared_expert_materialized_count: usize,
    layer1_moe_shared_expert_fnv1a64: u64,
    last_layer1_moe_shared_expert_fnv1a64: u64,
    layer1_moe_combined_fnv1a64: u64,
    last_layer1_moe_combined_fnv1a64: u64,
    layer1_moe_shared_gate_fnv1a64: u64,
    last_layer1_moe_shared_gate_fnv1a64: u64,
    layer1_moe_shared_gate_weight_fnv1a64: u64,
    layer1_moe_shared_gate_proj_weight_fnv1a64: u64,
    layer1_moe_shared_up_proj_weight_fnv1a64: u64,
    layer1_moe_shared_down_weight_fnv1a64: u64,
    layer1_moe_shared_expert_weight_bytes_read: u64,
    layer1_moe_shared_expert_elapsed_ns: u64,
    layer1_output_residual_applied: u32,
    layer1_output_residual_materialized_count: usize,
    layer1_output_residual_fnv1a64: u64,
    last_layer1_output_residual_fnv1a64: u64,
    layer1_output_residual_elapsed_ns: u64,
    layer2_input_norm_applied: u32,
    layer2_input_norm_materialized_count: usize,
    layer2_input_norm_fnv1a64: u64,
    last_layer2_input_norm_fnv1a64: u64,
    layer2_input_norm_weight_fnv1a64: u64,
    layer2_input_norm_weight_bytes_read: u64,
    layer2_input_norm_elapsed_ns: u64,
    layer2_qkv_projection_applied: u32,
    layer2_qkv_projection_materialized_count: usize,
    layer2_qkv_projection_fnv1a64: u64,
    last_layer2_qkv_projection_fnv1a64: u64,
    layer2_qkv_projection_weight_fnv1a64: u64,
    layer2_qkv_projection_weight_bytes_read: u64,
    layer2_qkv_projection_elapsed_ns: u64,
    layer2_zab_projection_applied: u32,
    layer2_zab_projection_materialized_count: usize,
    layer2_zab_projection_fnv1a64: u64,
    last_layer2_zab_projection_fnv1a64: u64,
    layer2_z_projection_weight_fnv1a64: u64,
    layer2_a_projection_weight_fnv1a64: u64,
    layer2_b_projection_weight_fnv1a64: u64,
    layer2_zab_projection_weight_bytes_read: u64,
    layer2_zab_projection_elapsed_ns: u64,
    layer2_conv_qkv_applied: u32,
    layer2_conv_qkv_materialized_count: usize,
    layer2_conv_qkv_fnv1a64: u64,
    last_layer2_conv_qkv_fnv1a64: u64,
    layer2_conv_qkv_weight_fnv1a64: u64,
    layer2_conv_qkv_weight_bytes_read: u64,
    layer2_conv_qkv_elapsed_ns: u64,
    layer2_postconv_qkv_applied: u32,
    layer2_postconv_qkv_materialized_count: usize,
    layer2_postconv_q_scaled_fnv1a64: u64,
    last_layer2_postconv_q_scaled_fnv1a64: u64,
    layer2_postconv_k_norm_fnv1a64: u64,
    last_layer2_postconv_k_norm_fnv1a64: u64,
    layer2_postconv_value_fnv1a64: u64,
    last_layer2_postconv_value_fnv1a64: u64,
    layer2_postconv_qkv_elapsed_ns: u64,
    layer2_gate_rows_applied: u32,
    layer2_gate_rows_materialized_count: usize,
    layer2_gate_g_fnv1a64: u64,
    last_layer2_gate_g_fnv1a64: u64,
    layer2_gate_beta_fnv1a64: u64,
    last_layer2_gate_beta_fnv1a64: u64,
    layer2_a_log_fnv1a64: u64,
    layer2_dt_bias_fnv1a64: u64,
    layer2_gate_weight_bytes_read: u64,
    layer2_gate_rows_elapsed_ns: u64,
    layer2_core_rows_applied: u32,
    layer2_core_rows_materialized_count: usize,
    layer2_core_rows_fnv1a64: u64,
    last_layer2_core_rows_fnv1a64: u64,
    layer2_core_final_state_fnv1a64: u64,
    layer2_core_rows_elapsed_ns: u64,
    layer2_gated_rmsnorm_applied: u32,
    layer2_gated_rmsnorm_materialized_count: usize,
    layer2_gated_rmsnorm_fnv1a64: u64,
    last_layer2_gated_rmsnorm_fnv1a64: u64,
    layer2_linear_norm_weight_fnv1a64: u64,
    layer2_linear_norm_weight_bytes_read: u64,
    layer2_gated_rmsnorm_elapsed_ns: u64,
    layer2_out_projection_applied: u32,
    layer2_out_projection_materialized_count: usize,
    layer2_out_projection_fnv1a64: u64,
    last_layer2_out_projection_fnv1a64: u64,
    layer2_out_projection_weight_fnv1a64: u64,
    layer2_out_projection_weight_bytes_read: u64,
    layer2_out_projection_elapsed_ns: u64,
    layer2_residual_hidden_applied: u32,
    layer2_residual_hidden_materialized_count: usize,
    layer2_residual_hidden_fnv1a64: u64,
    last_layer2_residual_hidden_fnv1a64: u64,
    layer2_residual_hidden_elapsed_ns: u64,
    first_missing_module: [c_char; 64],
    failure_stage: [c_char; 64],
    failure: [c_char; QRT_LOAD_ERROR_CAPACITY],
    inference_success_claimed: c_int,
}

#[repr(C)]
struct QrtQwen36BaselineSurfaceCheck {
    baseline_plan_attached: u32,
    layer_count: u32,
    linear_attention_layer_count: u32,
    full_attention_layer_count: u32,
    prefill_entrypoint_planned: u32,
    decode_one_entrypoint_planned: u32,
    descriptor_fnv1a64: u64,
    prefill_call_count: usize,
    decode_one_call_count: usize,
    failure_stage: [c_char; 64],
    failure: [c_char; QRT_LOAD_ERROR_CAPACITY],
    correctness_pass: c_int,
}

enum QrtEngine {}

unsafe extern "C" {
    fn qrt_target_contract() -> *const QrtTargetContract;
    fn qrt_engine_create(config: *const QrtEngineConfig, out_engine: *mut *mut QrtEngine) -> c_int;
    fn qrt_engine_free(engine: *mut QrtEngine);
    fn qrt_engine_status(engine: *const QrtEngine) -> *const c_char;
    fn qrt_engine_generate(
        engine: *mut QrtEngine,
        prompt: *const c_char,
        output: *mut c_char,
        output_len: usize,
    ) -> c_int;
    fn qrt_engine_prefill(
        engine: *mut QrtEngine,
        input_tokens: *const u32,
        input_token_count: usize,
        out_result: *mut QrtQwen36BaselineExecutionResult,
    ) -> c_int;
    fn qrt_engine_decode_one(
        engine: *mut QrtEngine,
        input_token: u32,
        out_token: *mut u32,
        out_result: *mut QrtQwen36BaselineExecutionResult,
    ) -> c_int;
    fn qrt_qwen36_bf16_qmatvec_fixture_cpu(out_result: *mut QrtBf16QmatvecResult) -> c_int;
    fn qrt_micro_isomorphic_fixture_cpu(out_result: *mut QrtMicroResult) -> c_int;
    fn qrt_qwen36_baseline_check(out_result: *mut QrtQwen36BaselineCheck) -> c_int;
    fn qrt_qwen36_baseline_engine_surface_check(
        engine: *const QrtEngine,
        out_result: *mut QrtQwen36BaselineSurfaceCheck,
    ) -> c_int;
    fn qrt_strerror(status: c_int) -> *const c_char;
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("contract");

    let exit_code = match command {
        "contract" => {
            print_contract();
            0
        }
        "self-test" => match self_test() {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("self-test failed: {err}");
                1
            }
        },
        "slice-self-test" => match slice_self_test() {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("slice-self-test failed: {err}");
                1
            }
        },
        "micro-self-test" => match micro_self_test() {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("micro-self-test failed: {err}");
                1
            }
        },
        "baseline-self-test" => match baseline_self_test() {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("baseline-self-test failed: {err}");
                1
            }
        },
        "help" | "--help" | "-h" => {
            print_help();
            0
        }
        other => {
            eprintln!("unknown command: {other}");
            print_help();
            2
        }
    };

    std::process::exit(exit_code);
}

fn print_help() {
    println!("qrt-cli commands:");
    println!("  contract   print the target contract");
    println!("  self-test  validate Rust-to-C ABI linkage");
    println!("  slice-self-test  run the H-PORT-001 scalar C ABI fixture");
    println!("  micro-self-test  run the H-MICRO-001 CPU/control-flow fixture");
    println!("  baseline-self-test  validate the Qwen3.6 modular baseline descriptor table");
}

fn print_contract() {
    let contract = unsafe {
        qrt_target_contract()
            .as_ref()
            .expect("qrt_target_contract returned null")
    };

    println!("repo={}", cstr(contract.repo));
    println!("model={}", cstr(contract.model));
    println!("precision={}", cstr(contract.precision));
    println!(
        "runtime_language={}",
        cstr(contract.primary_runtime_language)
    );
    println!("tooling_language={}", cstr(contract.tooling_language));
    println!("target_os={}", cstr(contract.target_os));
    println!("target_device={}", cstr(contract.target_device));
    println!("test_host={}", cstr(contract.test_host));
    println!("dependency_policy={}", cstr(contract.dependency_policy));
    println!("status={}", cstr(contract.status));
}

fn self_test() -> Result<(), String> {
    print_contract();

    let model_path = CString::new("model-placeholder-not-loaded").map_err(|err| err.to_string())?;
    let config = QrtEngineConfig {
        model_path: model_path.as_ptr(),
        context_tokens: 8192,
        batch_size: 1,
    };

    let mut engine: *mut QrtEngine = ptr::null_mut();
    let status = unsafe { qrt_engine_create(&config, &mut engine) };
    if status != QRT_STATUS_OK {
        return Err(format!("qrt_engine_create: {}", qrt_error(status)));
    }

    if engine.is_null() {
        return Err("qrt_engine_create returned a null engine".to_string());
    }

    let engine_status = unsafe { cstr(qrt_engine_status(engine)) };
    println!("engine_status={engine_status}");

    let prompt = CString::new("hello").map_err(|err| err.to_string())?;
    let mut output = [0 as c_char; 16];
    let gen_status =
        unsafe { qrt_engine_generate(engine, prompt.as_ptr(), output.as_mut_ptr(), output.len()) };
    unsafe { qrt_engine_free(engine) };

    if gen_status != QRT_STATUS_NOT_IMPLEMENTED {
        return Err(format!(
            "expected not_implemented from qrt_engine_generate, got {}",
            qrt_error(gen_status)
        ));
    }

    println!("abi_self_test=pass");
    println!("legacy_text_generate=unsupported_use_qrt_server_token_api");
    Ok(())
}

fn slice_self_test() -> Result<(), String> {
    let mut result = QrtBf16QmatvecResult {
        hidden_size: 0,
        expert_intermediate_size: 0,
        weight_elements: 0,
        payload_bytes: 0,
        expected_output_fnv1a64: 0,
        output_fnv1a64: 0,
        max_abs_diff: 0.0,
        correctness_pass: 0,
    };

    let status = unsafe { qrt_qwen36_bf16_qmatvec_fixture_cpu(&mut result) };
    if status != QRT_STATUS_OK {
        return Err(format!(
            "qrt_qwen36_bf16_qmatvec_fixture_cpu: {}",
            qrt_error(status)
        ));
    }
    if result.correctness_pass == 0 || result.output_fnv1a64 != result.expected_output_fnv1a64 {
        return Err(format!(
            "fixture hash mismatch: expected {:016x}, got {:016x}",
            result.expected_output_fnv1a64, result.output_fnv1a64
        ));
    }

    println!("slice=hport001_bf16_qmatvec");
    println!("hidden_size={}", result.hidden_size);
    println!(
        "expert_intermediate_size={}",
        result.expert_intermediate_size
    );
    println!("weight_elements={}", result.weight_elements);
    println!("payload_bytes={}", result.payload_bytes);
    println!(
        "expected_output_fnv1a64={:016x}",
        result.expected_output_fnv1a64
    );
    println!("output_fnv1a64={:016x}", result.output_fnv1a64);
    println!("max_abs_diff={:.6}", result.max_abs_diff);
    println!("slice_self_test=pass");
    Ok(())
}

fn micro_self_test() -> Result<(), String> {
    let mut result = QrtMicroResult {
        hidden_size: 0,
        expert_count: 0,
        expert_intermediate_size: 0,
        vocab_size: 0,
        prompt_tokens: 0,
        kv_tokens: 0,
        selected_expert: 0,
        generated_token: 0,
        streaming_callbacks: 0,
        exact_prefix_hit_tokens: 0,
        shared_prefix_hit_tokens: 0,
        unrelated_prefix_hit_tokens: 0,
        contamination_guard_pass: 0,
        expected_trace_fnv1a64: 0,
        trace_fnv1a64: 0,
        rmsnorm_pass: 0,
        rope_pass: 0,
        attention_kv_pass: 0,
        linear_attention_pass: 0,
        moe_pass: 0,
        lm_head_pass: 0,
        prefix_cache_pass: 0,
        streaming_pass: 0,
        correctness_pass: 0,
    };

    let status = unsafe { qrt_micro_isomorphic_fixture_cpu(&mut result) };
    if status != QRT_STATUS_OK {
        return Err(format!(
            "qrt_micro_isomorphic_fixture_cpu: {} trace={:016x}",
            qrt_error(status),
            result.trace_fnv1a64
        ));
    }
    if result.correctness_pass == 0 {
        return Err("micro fixture reported correctness_pass=false".to_string());
    }

    println!("micro=hmicro001_cpu_control_loop");
    println!("hidden_size={}", result.hidden_size);
    println!("expert_count={}", result.expert_count);
    println!(
        "expert_intermediate_size={}",
        result.expert_intermediate_size
    );
    println!("vocab_size={}", result.vocab_size);
    println!("prompt_tokens={}", result.prompt_tokens);
    println!("kv_tokens={}", result.kv_tokens);
    println!("selected_expert={}", result.selected_expert);
    println!("generated_token={}", result.generated_token);
    println!("streaming_callbacks={}", result.streaming_callbacks);
    println!(
        "prefix_hits=exact:{} shared:{} unrelated:{}",
        result.exact_prefix_hit_tokens,
        result.shared_prefix_hit_tokens,
        result.unrelated_prefix_hit_tokens
    );
    println!(
        "contamination_guard_pass={}",
        result.contamination_guard_pass
    );
    println!("trace_fnv1a64={:016x}", result.trace_fnv1a64);
    println!(
        "module_passes=rmsnorm:{} rope:{} attention_kv:{} linear_attention:{} moe:{} lm_head:{} prefix_cache:{} streaming:{}",
        result.rmsnorm_pass,
        result.rope_pass,
        result.attention_kv_pass,
        result.linear_attention_pass,
        result.moe_pass,
        result.lm_head_pass,
        result.prefix_cache_pass,
        result.streaming_pass
    );
    println!("micro_self_test=pass");
    Ok(())
}

fn baseline_self_test() -> Result<(), String> {
    let mut result = QrtQwen36BaselineCheck {
        model_specific: 0,
        batch_size: 0,
        descriptor_table_driven: 0,
        prefill_entrypoint_planned: 0,
        decode_one_entrypoint_planned: 0,
        layer_count: 0,
        linear_attention_layer_count: 0,
        full_attention_layer_count: 0,
        first_full_attention_layer: 0,
        last_full_attention_layer: 0,
        layer_descriptor_count_matches: 0,
        layer_schedule_pass: 0,
        tensor_name_builder_pass: 0,
        descriptor_fnv1a64: 0,
        sample_token_embedding: [0; QRT_QWEN36_TENSOR_NAME_CAPACITY],
        sample_linear_qkv: [0; QRT_QWEN36_TENSOR_NAME_CAPACITY],
        sample_full_q: [0; QRT_QWEN36_TENSOR_NAME_CAPACITY],
        sample_moe_router: [0; QRT_QWEN36_TENSOR_NAME_CAPACITY],
        sample_final_norm: [0; QRT_QWEN36_TENSOR_NAME_CAPACITY],
        sample_lm_head: [0; QRT_QWEN36_TENSOR_NAME_CAPACITY],
        correctness_pass: 0,
        failure: [0; QRT_LOAD_ERROR_CAPACITY],
    };

    let status = unsafe { qrt_qwen36_baseline_check(&mut result) };
    if status != QRT_STATUS_OK {
        return Err(format!(
            "qrt_qwen36_baseline_check: {} failure={}",
            qrt_error(status),
            cstr(result.failure.as_ptr())
        ));
    }
    if result.correctness_pass == 0 {
        return Err("baseline descriptor check reported correctness_pass=false".to_string());
    }

    let model_path = CString::new("model-placeholder-not-loaded").map_err(|err| err.to_string())?;
    let config = QrtEngineConfig {
        model_path: model_path.as_ptr(),
        context_tokens: 8192,
        batch_size: 1,
    };
    let mut engine: *mut QrtEngine = ptr::null_mut();
    let create_status = unsafe { qrt_engine_create(&config, &mut engine) };
    if create_status != QRT_STATUS_OK {
        return Err(format!("qrt_engine_create: {}", qrt_error(create_status)));
    }
    if engine.is_null() {
        return Err("qrt_engine_create returned a null engine".to_string());
    }
    let engine = EngineHandle(engine);

    let prefill_tokens = [248044_u32, 1_u32, 2_u32];
    let mut prefill = baseline_execution_result();
    let prefill_status = unsafe {
        qrt_engine_prefill(
            engine.as_mut_ptr(),
            prefill_tokens.as_ptr(),
            prefill_tokens.len(),
            &mut prefill,
        )
    };
    if prefill_status != QRT_STATUS_UNSUPPORTED {
        return Err(format!(
            "expected unsupported from qrt_engine_prefill without a model manifest, got {}",
            qrt_error(prefill_status)
        ));
    }
    validate_baseline_execution_result(
        "prefill",
        &prefill,
        QRT_QWEN36_EXECUTION_PREFILL,
        prefill_tokens.len(),
        &result,
        "token_embedding",
        "baseline_module_token_embedding_requires_manifest",
        0,
    )?;

    let mut decode = baseline_execution_result();
    let mut decode_token = u32::MAX;
    let decode_status = unsafe {
        qrt_engine_decode_one(engine.as_mut_ptr(), 2_u32, &mut decode_token, &mut decode)
    };
    if decode_status != QRT_STATUS_UNSUPPORTED {
        return Err(format!(
            "expected unsupported from qrt_engine_decode_one without a model manifest, got {}",
            qrt_error(decode_status)
        ));
    }
    if decode_token != 0 {
        return Err(format!(
            "decode_one placeholder token changed: expected 0, got {decode_token}"
        ));
    }
    validate_baseline_execution_result(
        "decode_one",
        &decode,
        QRT_QWEN36_EXECUTION_DECODE_ONE,
        1,
        &result,
        "token_embedding",
        "baseline_module_token_embedding_requires_manifest",
        0,
    )?;

    let mut surface = baseline_surface_check();
    let surface_status =
        unsafe { qrt_qwen36_baseline_engine_surface_check(engine.as_ptr(), &mut surface) };
    if surface_status != QRT_STATUS_OK {
        return Err(format!(
            "qrt_qwen36_baseline_engine_surface_check: {} failure_stage={} failure={}",
            qrt_error(surface_status),
            cstr(surface.failure_stage.as_ptr()),
            cstr(surface.failure.as_ptr())
        ));
    }
    validate_baseline_surface_check(&surface, &result)?;

    println!("baseline=hbaseline001_modular_descriptor_plan");
    println!("model_specific={}", result.model_specific);
    println!("batch_size={}", result.batch_size);
    println!("descriptor_table_driven={}", result.descriptor_table_driven);
    println!(
        "entrypoints=prefill:{} decode_one:{}",
        result.prefill_entrypoint_planned, result.decode_one_entrypoint_planned
    );
    println!("layer_count={}", result.layer_count);
    println!(
        "attention_layers=linear:{} full:{} first_full:{} last_full:{}",
        result.linear_attention_layer_count,
        result.full_attention_layer_count,
        result.first_full_attention_layer,
        result.last_full_attention_layer
    );
    println!(
        "checks=descriptor_count:{} layer_schedule:{} tensor_names:{}",
        result.layer_descriptor_count_matches,
        result.layer_schedule_pass,
        result.tensor_name_builder_pass
    );
    println!("descriptor_fnv1a64={:016x}", result.descriptor_fnv1a64);
    println!(
        "sample_token_embedding={}",
        cstr(result.sample_token_embedding.as_ptr())
    );
    println!(
        "sample_linear_qkv={}",
        cstr(result.sample_linear_qkv.as_ptr())
    );
    println!("sample_full_q={}", cstr(result.sample_full_q.as_ptr()));
    println!(
        "sample_moe_router={}",
        cstr(result.sample_moe_router.as_ptr())
    );
    println!(
        "sample_final_norm={}",
        cstr(result.sample_final_norm.as_ptr())
    );
    println!("sample_lm_head={}", cstr(result.sample_lm_head.as_ptr()));
    println!("prefill_status=unsupported_requires_manifest");
    println!("decode_one_status=unsupported_requires_manifest");
    println!(
        "first_missing_module={}",
        cstr(prefill.first_missing_module.as_ptr())
    );
    println!(
        "token_embedding_materialized={}",
        prefill.token_embedding_materialized + decode.token_embedding_materialized
    );
    println!(
        "layer0_input_norm_applied={}",
        prefill.layer0_input_norm_applied + decode.layer0_input_norm_applied
    );
    println!(
        "layer0_qkv_projection_applied={}",
        prefill.layer0_qkv_projection_applied + decode.layer0_qkv_projection_applied
    );
    println!(
        "layer0_zab_projection_applied={}",
        prefill.layer0_zab_projection_applied + decode.layer0_zab_projection_applied
    );
    println!(
        "layer0_conv_qkv_applied={}",
        prefill.layer0_conv_qkv_applied + decode.layer0_conv_qkv_applied
    );
    println!(
        "layer0_postconv_qkv_applied={}",
        prefill.layer0_postconv_qkv_applied + decode.layer0_postconv_qkv_applied
    );
    println!(
        "layer0_gate_rows_applied={}",
        prefill.layer0_gate_rows_applied + decode.layer0_gate_rows_applied
    );
    println!(
        "layer0_core_rows_applied={}",
        prefill.layer0_core_rows_applied + decode.layer0_core_rows_applied
    );
    println!(
        "layer0_gated_rmsnorm_applied={}",
        prefill.layer0_gated_rmsnorm_applied + decode.layer0_gated_rmsnorm_applied
    );
    println!(
        "layer0_out_projection_applied={}",
        prefill.layer0_out_projection_applied + decode.layer0_out_projection_applied
    );
    println!(
        "layer0_residual_hidden_applied={}",
        prefill.layer0_residual_hidden_applied + decode.layer0_residual_hidden_applied
    );
    println!(
        "layer0_post_attention_rmsnorm_applied={}",
        prefill.layer0_post_attention_rmsnorm_applied
            + decode.layer0_post_attention_rmsnorm_applied
    );
    println!(
        "layer0_moe_router_applied={}",
        prefill.layer0_moe_router_applied + decode.layer0_moe_router_applied
    );
    println!(
        "layer0_moe_expert_applied={}",
        prefill.layer0_moe_expert_applied + decode.layer0_moe_expert_applied
    );
    println!(
        "layer0_moe_shared_expert_applied={}",
        prefill.layer0_moe_shared_expert_applied + decode.layer0_moe_shared_expert_applied
    );
    println!(
        "layer0_output_residual_applied={}",
        prefill.layer0_output_residual_applied + decode.layer0_output_residual_applied
    );
    println!(
        "layer1_input_norm_applied={}",
        prefill.layer1_input_norm_applied + decode.layer1_input_norm_applied
    );
    println!(
        "layer1_qkv_projection_applied={}",
        prefill.layer1_qkv_projection_applied + decode.layer1_qkv_projection_applied
    );
    println!(
        "layer1_zab_projection_applied={}",
        prefill.layer1_zab_projection_applied + decode.layer1_zab_projection_applied
    );
    println!(
        "layer1_conv_qkv_applied={}",
        prefill.layer1_conv_qkv_applied + decode.layer1_conv_qkv_applied
    );
    println!(
        "layer1_postconv_qkv_applied={}",
        prefill.layer1_postconv_qkv_applied + decode.layer1_postconv_qkv_applied
    );
    println!(
        "layer1_gate_rows_applied={}",
        prefill.layer1_gate_rows_applied + decode.layer1_gate_rows_applied
    );
    println!(
        "layer1_core_rows_applied={}",
        prefill.layer1_core_rows_applied + decode.layer1_core_rows_applied
    );
    println!(
        "layer1_gated_rmsnorm_applied={}",
        prefill.layer1_gated_rmsnorm_applied + decode.layer1_gated_rmsnorm_applied
    );
    println!(
        "layer1_out_projection_applied={}",
        prefill.layer1_out_projection_applied + decode.layer1_out_projection_applied
    );
    println!(
        "layer1_residual_hidden_applied={}",
        prefill.layer1_residual_hidden_applied + decode.layer1_residual_hidden_applied
    );
    println!(
        "layer1_post_attention_rmsnorm_applied={}",
        prefill.layer1_post_attention_rmsnorm_applied
            + decode.layer1_post_attention_rmsnorm_applied
    );
    println!(
        "layer1_moe_router_applied={}",
        prefill.layer1_moe_router_applied + decode.layer1_moe_router_applied
    );
    println!(
        "layer1_moe_expert_applied={}",
        prefill.layer1_moe_expert_applied + decode.layer1_moe_expert_applied
    );
    println!(
        "layer1_moe_shared_expert_applied={}",
        prefill.layer1_moe_shared_expert_applied + decode.layer1_moe_shared_expert_applied
    );
    println!(
        "layer1_output_residual_applied={}",
        prefill.layer1_output_residual_applied + decode.layer1_output_residual_applied
    );
    println!(
        "layer2_input_norm_applied={}",
        prefill.layer2_input_norm_applied + decode.layer2_input_norm_applied
    );
    println!(
        "layer2_qkv_projection_applied={}",
        prefill.layer2_qkv_projection_applied + decode.layer2_qkv_projection_applied
    );
    println!(
        "layer2_zab_projection_applied={}",
        prefill.layer2_zab_projection_applied + decode.layer2_zab_projection_applied
    );
    println!(
        "layer2_conv_qkv_applied={}",
        prefill.layer2_conv_qkv_applied + decode.layer2_conv_qkv_applied
    );
    println!(
        "layer2_postconv_qkv_applied={}",
        prefill.layer2_postconv_qkv_applied + decode.layer2_postconv_qkv_applied
    );
    println!(
        "layer2_gate_rows_applied={}",
        prefill.layer2_gate_rows_applied + decode.layer2_gate_rows_applied
    );
    println!(
        "layer2_core_rows_applied={}",
        prefill.layer2_core_rows_applied + decode.layer2_core_rows_applied
    );
    println!(
        "layer2_gated_rmsnorm_applied={}",
        prefill.layer2_gated_rmsnorm_applied + decode.layer2_gated_rmsnorm_applied
    );
    println!(
        "layer2_out_projection_applied={}",
        prefill.layer2_out_projection_applied + decode.layer2_out_projection_applied
    );
    println!(
        "layer2_residual_hidden_applied={}",
        prefill.layer2_residual_hidden_applied + decode.layer2_residual_hidden_applied
    );
    println!(
        "surface_calls=prefill:{} decode_one:{}",
        surface.prefill_call_count, surface.decode_one_call_count
    );
    println!(
        "inference_success_claimed={}",
        prefill.inference_success_claimed + decode.inference_success_claimed
    );
    println!("baseline_self_test=pass");
    Ok(())
}

struct EngineHandle(*mut QrtEngine);

impl EngineHandle {
    fn as_mut_ptr(&self) -> *mut QrtEngine {
        self.0
    }

    fn as_ptr(&self) -> *const QrtEngine {
        self.0
    }
}

impl Drop for EngineHandle {
    fn drop(&mut self) {
        unsafe { qrt_engine_free(self.0) };
    }
}

fn baseline_execution_result() -> QrtQwen36BaselineExecutionResult {
    QrtQwen36BaselineExecutionResult {
        mode: 0,
        input_token_count: 0,
        output_token_count: 0,
        baseline_plan_attached: 0,
        descriptor_table_driven: 0,
        module_dispatch_started: 0,
        layer_count: 0,
        linear_attention_layer_count: 0,
        full_attention_layer_count: 0,
        first_missing_layer: 0,
        descriptor_fnv1a64: 0,
        token_embedding_materialized: 0,
        input_embeddings_materialized_count: 0,
        input_embeddings_fnv1a64: 0,
        last_token_embedding_fnv1a64: 0,
        token_embedding_bytes_read: 0,
        token_embedding_elapsed_ns: 0,
        layer0_input_norm_applied: 0,
        layer0_input_norm_materialized_count: 0,
        layer0_input_norm_fnv1a64: 0,
        last_layer0_input_norm_fnv1a64: 0,
        layer0_input_norm_weight_fnv1a64: 0,
        layer0_input_norm_weight_bytes_read: 0,
        layer0_input_norm_elapsed_ns: 0,
        layer0_qkv_projection_applied: 0,
        layer0_qkv_projection_materialized_count: 0,
        layer0_qkv_projection_fnv1a64: 0,
        last_layer0_qkv_projection_fnv1a64: 0,
        layer0_qkv_projection_weight_fnv1a64: 0,
        layer0_qkv_projection_weight_bytes_read: 0,
        layer0_qkv_projection_elapsed_ns: 0,
        layer0_zab_projection_applied: 0,
        layer0_zab_projection_materialized_count: 0,
        layer0_zab_projection_fnv1a64: 0,
        last_layer0_zab_projection_fnv1a64: 0,
        layer0_z_projection_weight_fnv1a64: 0,
        layer0_a_projection_weight_fnv1a64: 0,
        layer0_b_projection_weight_fnv1a64: 0,
        layer0_zab_projection_weight_bytes_read: 0,
        layer0_zab_projection_elapsed_ns: 0,
        layer0_conv_qkv_applied: 0,
        layer0_conv_qkv_materialized_count: 0,
        layer0_conv_qkv_fnv1a64: 0,
        last_layer0_conv_qkv_fnv1a64: 0,
        layer0_conv_qkv_weight_fnv1a64: 0,
        layer0_conv_qkv_weight_bytes_read: 0,
        layer0_conv_qkv_elapsed_ns: 0,
        layer0_postconv_qkv_applied: 0,
        layer0_postconv_qkv_materialized_count: 0,
        layer0_postconv_q_scaled_fnv1a64: 0,
        last_layer0_postconv_q_scaled_fnv1a64: 0,
        layer0_postconv_k_norm_fnv1a64: 0,
        last_layer0_postconv_k_norm_fnv1a64: 0,
        layer0_postconv_value_fnv1a64: 0,
        last_layer0_postconv_value_fnv1a64: 0,
        layer0_postconv_qkv_elapsed_ns: 0,
        layer0_gate_rows_applied: 0,
        layer0_gate_rows_materialized_count: 0,
        layer0_gate_g_fnv1a64: 0,
        last_layer0_gate_g_fnv1a64: 0,
        layer0_gate_beta_fnv1a64: 0,
        last_layer0_gate_beta_fnv1a64: 0,
        layer0_a_log_fnv1a64: 0,
        layer0_dt_bias_fnv1a64: 0,
        layer0_gate_weight_bytes_read: 0,
        layer0_gate_rows_elapsed_ns: 0,
        layer0_core_rows_applied: 0,
        layer0_core_rows_materialized_count: 0,
        layer0_core_rows_fnv1a64: 0,
        last_layer0_core_rows_fnv1a64: 0,
        layer0_core_final_state_fnv1a64: 0,
        layer0_core_rows_elapsed_ns: 0,
        layer0_gated_rmsnorm_applied: 0,
        layer0_gated_rmsnorm_materialized_count: 0,
        layer0_gated_rmsnorm_fnv1a64: 0,
        last_layer0_gated_rmsnorm_fnv1a64: 0,
        layer0_linear_norm_weight_fnv1a64: 0,
        layer0_linear_norm_weight_bytes_read: 0,
        layer0_gated_rmsnorm_elapsed_ns: 0,
        layer0_out_projection_applied: 0,
        layer0_out_projection_materialized_count: 0,
        layer0_out_projection_fnv1a64: 0,
        last_layer0_out_projection_fnv1a64: 0,
        layer0_out_projection_weight_fnv1a64: 0,
        layer0_out_projection_weight_bytes_read: 0,
        layer0_out_projection_elapsed_ns: 0,
        layer0_residual_hidden_applied: 0,
        layer0_residual_hidden_materialized_count: 0,
        layer0_residual_hidden_fnv1a64: 0,
        last_layer0_residual_hidden_fnv1a64: 0,
        layer0_residual_hidden_elapsed_ns: 0,
        layer0_post_attention_rmsnorm_applied: 0,
        layer0_post_attention_rmsnorm_materialized_count: 0,
        layer0_post_attention_rmsnorm_fnv1a64: 0,
        last_layer0_post_attention_rmsnorm_fnv1a64: 0,
        layer0_post_attention_norm_weight_fnv1a64: 0,
        layer0_post_attention_norm_weight_bytes_read: 0,
        layer0_post_attention_rmsnorm_elapsed_ns: 0,
        layer0_moe_router_applied: 0,
        layer0_moe_router_materialized_count: 0,
        layer0_moe_router_logits_fnv1a64: 0,
        last_layer0_moe_router_logits_fnv1a64: 0,
        layer0_moe_router_topk_ids_fnv1a64: 0,
        last_layer0_moe_router_topk_ids_fnv1a64: 0,
        layer0_moe_router_topk_weights_fnv1a64: 0,
        last_layer0_moe_router_topk_weights_fnv1a64: 0,
        layer0_moe_router_weight_fnv1a64: 0,
        layer0_moe_router_weight_bytes_read: 0,
        layer0_moe_router_elapsed_ns: 0,
        layer0_moe_expert_applied: 0,
        layer0_moe_expert_materialized_count: 0,
        layer0_moe_expert_selected_count: 0,
        layer0_moe_expert_routed_fnv1a64: 0,
        last_layer0_moe_expert_routed_fnv1a64: 0,
        layer0_moe_expert_selected_ids_fnv1a64: 0,
        layer0_moe_expert_gate_up_weight_fnv1a64: 0,
        layer0_moe_expert_down_weight_fnv1a64: 0,
        layer0_moe_expert_weight_bytes_read: 0,
        layer0_moe_expert_elapsed_ns: 0,
        layer0_moe_shared_expert_applied: 0,
        layer0_moe_shared_expert_materialized_count: 0,
        layer0_moe_shared_expert_fnv1a64: 0,
        last_layer0_moe_shared_expert_fnv1a64: 0,
        layer0_moe_combined_fnv1a64: 0,
        last_layer0_moe_combined_fnv1a64: 0,
        layer0_moe_shared_gate_fnv1a64: 0,
        last_layer0_moe_shared_gate_fnv1a64: 0,
        layer0_moe_shared_gate_weight_fnv1a64: 0,
        layer0_moe_shared_gate_proj_weight_fnv1a64: 0,
        layer0_moe_shared_up_proj_weight_fnv1a64: 0,
        layer0_moe_shared_down_weight_fnv1a64: 0,
        layer0_moe_shared_expert_weight_bytes_read: 0,
        layer0_moe_shared_expert_elapsed_ns: 0,
        layer0_output_residual_applied: 0,
        layer0_output_residual_materialized_count: 0,
        layer0_output_residual_fnv1a64: 0,
        last_layer0_output_residual_fnv1a64: 0,
        layer0_output_residual_elapsed_ns: 0,
        layer1_input_norm_applied: 0,
        layer1_input_norm_materialized_count: 0,
        layer1_input_norm_fnv1a64: 0,
        last_layer1_input_norm_fnv1a64: 0,
        layer1_input_norm_weight_fnv1a64: 0,
        layer1_input_norm_weight_bytes_read: 0,
        layer1_input_norm_elapsed_ns: 0,
        layer1_qkv_projection_applied: 0,
        layer1_qkv_projection_materialized_count: 0,
        layer1_qkv_projection_fnv1a64: 0,
        last_layer1_qkv_projection_fnv1a64: 0,
        layer1_qkv_projection_weight_fnv1a64: 0,
        layer1_qkv_projection_weight_bytes_read: 0,
        layer1_qkv_projection_elapsed_ns: 0,
        layer1_zab_projection_applied: 0,
        layer1_zab_projection_materialized_count: 0,
        layer1_zab_projection_fnv1a64: 0,
        last_layer1_zab_projection_fnv1a64: 0,
        layer1_z_projection_weight_fnv1a64: 0,
        layer1_a_projection_weight_fnv1a64: 0,
        layer1_b_projection_weight_fnv1a64: 0,
        layer1_zab_projection_weight_bytes_read: 0,
        layer1_zab_projection_elapsed_ns: 0,
        layer1_conv_qkv_applied: 0,
        layer1_conv_qkv_materialized_count: 0,
        layer1_conv_qkv_fnv1a64: 0,
        last_layer1_conv_qkv_fnv1a64: 0,
        layer1_conv_qkv_weight_fnv1a64: 0,
        layer1_conv_qkv_weight_bytes_read: 0,
        layer1_conv_qkv_elapsed_ns: 0,
        layer1_postconv_qkv_applied: 0,
        layer1_postconv_qkv_materialized_count: 0,
        layer1_postconv_q_scaled_fnv1a64: 0,
        last_layer1_postconv_q_scaled_fnv1a64: 0,
        layer1_postconv_k_norm_fnv1a64: 0,
        last_layer1_postconv_k_norm_fnv1a64: 0,
        layer1_postconv_value_fnv1a64: 0,
        last_layer1_postconv_value_fnv1a64: 0,
        layer1_postconv_qkv_elapsed_ns: 0,
        layer1_gate_rows_applied: 0,
        layer1_gate_rows_materialized_count: 0,
        layer1_gate_g_fnv1a64: 0,
        last_layer1_gate_g_fnv1a64: 0,
        layer1_gate_beta_fnv1a64: 0,
        last_layer1_gate_beta_fnv1a64: 0,
        layer1_a_log_fnv1a64: 0,
        layer1_dt_bias_fnv1a64: 0,
        layer1_gate_weight_bytes_read: 0,
        layer1_gate_rows_elapsed_ns: 0,
        layer1_core_rows_applied: 0,
        layer1_core_rows_materialized_count: 0,
        layer1_core_rows_fnv1a64: 0,
        last_layer1_core_rows_fnv1a64: 0,
        layer1_core_final_state_fnv1a64: 0,
        layer1_core_rows_elapsed_ns: 0,
        layer1_gated_rmsnorm_applied: 0,
        layer1_gated_rmsnorm_materialized_count: 0,
        layer1_gated_rmsnorm_fnv1a64: 0,
        last_layer1_gated_rmsnorm_fnv1a64: 0,
        layer1_linear_norm_weight_fnv1a64: 0,
        layer1_linear_norm_weight_bytes_read: 0,
        layer1_gated_rmsnorm_elapsed_ns: 0,
        layer1_out_projection_applied: 0,
        layer1_out_projection_materialized_count: 0,
        layer1_out_projection_fnv1a64: 0,
        last_layer1_out_projection_fnv1a64: 0,
        layer1_out_projection_weight_fnv1a64: 0,
        layer1_out_projection_weight_bytes_read: 0,
        layer1_out_projection_elapsed_ns: 0,
        layer1_residual_hidden_applied: 0,
        layer1_residual_hidden_materialized_count: 0,
        layer1_residual_hidden_fnv1a64: 0,
        last_layer1_residual_hidden_fnv1a64: 0,
        layer1_residual_hidden_elapsed_ns: 0,
        layer1_post_attention_rmsnorm_applied: 0,
        layer1_post_attention_rmsnorm_materialized_count: 0,
        layer1_post_attention_rmsnorm_fnv1a64: 0,
        last_layer1_post_attention_rmsnorm_fnv1a64: 0,
        layer1_post_attention_norm_weight_fnv1a64: 0,
        layer1_post_attention_norm_weight_bytes_read: 0,
        layer1_post_attention_rmsnorm_elapsed_ns: 0,
        layer1_moe_router_applied: 0,
        layer1_moe_router_materialized_count: 0,
        layer1_moe_router_logits_fnv1a64: 0,
        last_layer1_moe_router_logits_fnv1a64: 0,
        layer1_moe_router_topk_ids_fnv1a64: 0,
        last_layer1_moe_router_topk_ids_fnv1a64: 0,
        layer1_moe_router_topk_weights_fnv1a64: 0,
        last_layer1_moe_router_topk_weights_fnv1a64: 0,
        layer1_moe_router_weight_fnv1a64: 0,
        layer1_moe_router_weight_bytes_read: 0,
        layer1_moe_router_elapsed_ns: 0,
        layer1_moe_expert_applied: 0,
        layer1_moe_expert_materialized_count: 0,
        layer1_moe_expert_selected_count: 0,
        layer1_moe_expert_routed_fnv1a64: 0,
        last_layer1_moe_expert_routed_fnv1a64: 0,
        layer1_moe_expert_selected_ids_fnv1a64: 0,
        layer1_moe_expert_gate_up_weight_fnv1a64: 0,
        layer1_moe_expert_down_weight_fnv1a64: 0,
        layer1_moe_expert_weight_bytes_read: 0,
        layer1_moe_expert_elapsed_ns: 0,
        layer1_moe_shared_expert_applied: 0,
        layer1_moe_shared_expert_materialized_count: 0,
        layer1_moe_shared_expert_fnv1a64: 0,
        last_layer1_moe_shared_expert_fnv1a64: 0,
        layer1_moe_combined_fnv1a64: 0,
        last_layer1_moe_combined_fnv1a64: 0,
        layer1_moe_shared_gate_fnv1a64: 0,
        last_layer1_moe_shared_gate_fnv1a64: 0,
        layer1_moe_shared_gate_weight_fnv1a64: 0,
        layer1_moe_shared_gate_proj_weight_fnv1a64: 0,
        layer1_moe_shared_up_proj_weight_fnv1a64: 0,
        layer1_moe_shared_down_weight_fnv1a64: 0,
        layer1_moe_shared_expert_weight_bytes_read: 0,
        layer1_moe_shared_expert_elapsed_ns: 0,
        layer1_output_residual_applied: 0,
        layer1_output_residual_materialized_count: 0,
        layer1_output_residual_fnv1a64: 0,
        last_layer1_output_residual_fnv1a64: 0,
        layer1_output_residual_elapsed_ns: 0,
        layer2_input_norm_applied: 0,
        layer2_input_norm_materialized_count: 0,
        layer2_input_norm_fnv1a64: 0,
        last_layer2_input_norm_fnv1a64: 0,
        layer2_input_norm_weight_fnv1a64: 0,
        layer2_input_norm_weight_bytes_read: 0,
        layer2_input_norm_elapsed_ns: 0,
        layer2_qkv_projection_applied: 0,
        layer2_qkv_projection_materialized_count: 0,
        layer2_qkv_projection_fnv1a64: 0,
        last_layer2_qkv_projection_fnv1a64: 0,
        layer2_qkv_projection_weight_fnv1a64: 0,
        layer2_qkv_projection_weight_bytes_read: 0,
        layer2_qkv_projection_elapsed_ns: 0,
        layer2_zab_projection_applied: 0,
        layer2_zab_projection_materialized_count: 0,
        layer2_zab_projection_fnv1a64: 0,
        last_layer2_zab_projection_fnv1a64: 0,
        layer2_z_projection_weight_fnv1a64: 0,
        layer2_a_projection_weight_fnv1a64: 0,
        layer2_b_projection_weight_fnv1a64: 0,
        layer2_zab_projection_weight_bytes_read: 0,
        layer2_zab_projection_elapsed_ns: 0,
        layer2_conv_qkv_applied: 0,
        layer2_conv_qkv_materialized_count: 0,
        layer2_conv_qkv_fnv1a64: 0,
        last_layer2_conv_qkv_fnv1a64: 0,
        layer2_conv_qkv_weight_fnv1a64: 0,
        layer2_conv_qkv_weight_bytes_read: 0,
        layer2_conv_qkv_elapsed_ns: 0,
        layer2_postconv_qkv_applied: 0,
        layer2_postconv_qkv_materialized_count: 0,
        layer2_postconv_q_scaled_fnv1a64: 0,
        last_layer2_postconv_q_scaled_fnv1a64: 0,
        layer2_postconv_k_norm_fnv1a64: 0,
        last_layer2_postconv_k_norm_fnv1a64: 0,
        layer2_postconv_value_fnv1a64: 0,
        last_layer2_postconv_value_fnv1a64: 0,
        layer2_postconv_qkv_elapsed_ns: 0,
        layer2_gate_rows_applied: 0,
        layer2_gate_rows_materialized_count: 0,
        layer2_gate_g_fnv1a64: 0,
        last_layer2_gate_g_fnv1a64: 0,
        layer2_gate_beta_fnv1a64: 0,
        last_layer2_gate_beta_fnv1a64: 0,
        layer2_a_log_fnv1a64: 0,
        layer2_dt_bias_fnv1a64: 0,
        layer2_gate_weight_bytes_read: 0,
        layer2_gate_rows_elapsed_ns: 0,
        layer2_core_rows_applied: 0,
        layer2_core_rows_materialized_count: 0,
        layer2_core_rows_fnv1a64: 0,
        last_layer2_core_rows_fnv1a64: 0,
        layer2_core_final_state_fnv1a64: 0,
        layer2_core_rows_elapsed_ns: 0,
        layer2_gated_rmsnorm_applied: 0,
        layer2_gated_rmsnorm_materialized_count: 0,
        layer2_gated_rmsnorm_fnv1a64: 0,
        last_layer2_gated_rmsnorm_fnv1a64: 0,
        layer2_linear_norm_weight_fnv1a64: 0,
        layer2_linear_norm_weight_bytes_read: 0,
        layer2_gated_rmsnorm_elapsed_ns: 0,
        layer2_out_projection_applied: 0,
        layer2_out_projection_materialized_count: 0,
        layer2_out_projection_fnv1a64: 0,
        last_layer2_out_projection_fnv1a64: 0,
        layer2_out_projection_weight_fnv1a64: 0,
        layer2_out_projection_weight_bytes_read: 0,
        layer2_out_projection_elapsed_ns: 0,
        layer2_residual_hidden_applied: 0,
        layer2_residual_hidden_materialized_count: 0,
        layer2_residual_hidden_fnv1a64: 0,
        last_layer2_residual_hidden_fnv1a64: 0,
        layer2_residual_hidden_elapsed_ns: 0,
        first_missing_module: [0; 64],
        failure_stage: [0; 64],
        failure: [0; QRT_LOAD_ERROR_CAPACITY],
        inference_success_claimed: 0,
    }
}

fn baseline_surface_check() -> QrtQwen36BaselineSurfaceCheck {
    QrtQwen36BaselineSurfaceCheck {
        baseline_plan_attached: 0,
        layer_count: 0,
        linear_attention_layer_count: 0,
        full_attention_layer_count: 0,
        prefill_entrypoint_planned: 0,
        decode_one_entrypoint_planned: 0,
        descriptor_fnv1a64: 0,
        prefill_call_count: 0,
        decode_one_call_count: 0,
        failure_stage: [0; 64],
        failure: [0; QRT_LOAD_ERROR_CAPACITY],
        correctness_pass: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_baseline_execution_result(
    label: &str,
    actual: &QrtQwen36BaselineExecutionResult,
    expected_mode: c_int,
    expected_input_tokens: usize,
    descriptor: &QrtQwen36BaselineCheck,
    expected_missing_module: &str,
    expected_failure_stage: &str,
    expected_token_embedding_materialized: u32,
) -> Result<(), String> {
    if actual.mode != expected_mode {
        return Err(format!("{label} mode mismatch: {}", actual.mode));
    }
    if actual.input_token_count != expected_input_tokens {
        return Err(format!(
            "{label} input token count mismatch: {}",
            actual.input_token_count
        ));
    }
    if actual.output_token_count != 0 {
        return Err(format!(
            "{label} output token count should stay zero before module implementation"
        ));
    }
    if actual.baseline_plan_attached != 1
        || actual.descriptor_table_driven != 1
        || actual.module_dispatch_started != 1
    {
        return Err(format!(
            "{label} did not enter descriptor-table module dispatch"
        ));
    }
    if actual.layer_count != descriptor.layer_count
        || actual.linear_attention_layer_count != descriptor.linear_attention_layer_count
        || actual.full_attention_layer_count != descriptor.full_attention_layer_count
        || actual.descriptor_fnv1a64 != descriptor.descriptor_fnv1a64
    {
        return Err(format!(
            "{label} descriptor fields drifted from baseline check"
        ));
    }
    if actual.first_missing_layer != 0 {
        return Err(format!("{label} first missing layer should be 0"));
    }
    if actual.token_embedding_materialized != expected_token_embedding_materialized {
        return Err(format!(
            "{label} token_embedding_materialized mismatch: {}",
            actual.token_embedding_materialized
        ));
    }
    if actual.token_embedding_materialized == 0
        && (actual.input_embeddings_materialized_count != 0
            || actual.input_embeddings_fnv1a64 != 0
            || actual.last_token_embedding_fnv1a64 != 0
            || actual.token_embedding_bytes_read != 0)
    {
        return Err(format!(
            "{label} reported token embedding data without materialization"
        ));
    }
    if actual.layer0_input_norm_applied != 0
        || actual.layer0_input_norm_materialized_count != 0
        || actual.layer0_input_norm_fnv1a64 != 0
        || actual.last_layer0_input_norm_fnv1a64 != 0
        || actual.layer0_input_norm_weight_fnv1a64 != 0
        || actual.layer0_input_norm_weight_bytes_read != 0
    {
        return Err(format!(
            "{label} reported layer0 input norm data before manifest-backed execution"
        ));
    }
    if actual.layer0_qkv_projection_applied != 0
        || actual.layer0_qkv_projection_materialized_count != 0
        || actual.layer0_qkv_projection_fnv1a64 != 0
        || actual.last_layer0_qkv_projection_fnv1a64 != 0
        || actual.layer0_qkv_projection_weight_fnv1a64 != 0
        || actual.layer0_qkv_projection_weight_bytes_read != 0
    {
        return Err(format!(
            "{label} reported layer0 QKV projection data before manifest-backed execution"
        ));
    }
    if actual.layer0_zab_projection_applied != 0
        || actual.layer0_zab_projection_materialized_count != 0
        || actual.layer0_zab_projection_fnv1a64 != 0
        || actual.last_layer0_zab_projection_fnv1a64 != 0
        || actual.layer0_z_projection_weight_fnv1a64 != 0
        || actual.layer0_a_projection_weight_fnv1a64 != 0
        || actual.layer0_b_projection_weight_fnv1a64 != 0
        || actual.layer0_zab_projection_weight_bytes_read != 0
    {
        return Err(format!(
            "{label} reported layer0 Z/A/B projection data before manifest-backed execution"
        ));
    }
    if actual.layer0_conv_qkv_applied != 0
        || actual.layer0_conv_qkv_materialized_count != 0
        || actual.layer0_conv_qkv_fnv1a64 != 0
        || actual.last_layer0_conv_qkv_fnv1a64 != 0
        || actual.layer0_conv_qkv_weight_fnv1a64 != 0
        || actual.layer0_conv_qkv_weight_bytes_read != 0
    {
        return Err(format!(
            "{label} reported layer0 conv_qkv data before manifest-backed execution"
        ));
    }
    if actual.layer0_postconv_qkv_applied != 0
        || actual.layer0_postconv_qkv_materialized_count != 0
        || actual.layer0_postconv_q_scaled_fnv1a64 != 0
        || actual.last_layer0_postconv_q_scaled_fnv1a64 != 0
        || actual.layer0_postconv_k_norm_fnv1a64 != 0
        || actual.last_layer0_postconv_k_norm_fnv1a64 != 0
        || actual.layer0_postconv_value_fnv1a64 != 0
        || actual.last_layer0_postconv_value_fnv1a64 != 0
        || actual.layer0_postconv_qkv_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer0 postconv_qkv data before manifest-backed execution"
        ));
    }
    if actual.layer0_gate_rows_applied != 0
        || actual.layer0_gate_rows_materialized_count != 0
        || actual.layer0_gate_g_fnv1a64 != 0
        || actual.last_layer0_gate_g_fnv1a64 != 0
        || actual.layer0_gate_beta_fnv1a64 != 0
        || actual.last_layer0_gate_beta_fnv1a64 != 0
        || actual.layer0_a_log_fnv1a64 != 0
        || actual.layer0_dt_bias_fnv1a64 != 0
        || actual.layer0_gate_weight_bytes_read != 0
        || actual.layer0_gate_rows_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer0 gate rows data before manifest-backed execution"
        ));
    }
    if actual.layer0_core_rows_applied != 0
        || actual.layer0_core_rows_materialized_count != 0
        || actual.layer0_core_rows_fnv1a64 != 0
        || actual.last_layer0_core_rows_fnv1a64 != 0
        || actual.layer0_core_final_state_fnv1a64 != 0
        || actual.layer0_core_rows_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer0 core rows data before manifest-backed execution"
        ));
    }
    if actual.layer0_gated_rmsnorm_applied != 0
        || actual.layer0_gated_rmsnorm_materialized_count != 0
        || actual.layer0_gated_rmsnorm_fnv1a64 != 0
        || actual.last_layer0_gated_rmsnorm_fnv1a64 != 0
        || actual.layer0_linear_norm_weight_fnv1a64 != 0
        || actual.layer0_linear_norm_weight_bytes_read != 0
        || actual.layer0_gated_rmsnorm_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer0 gated RMSNorm data before manifest-backed execution"
        ));
    }
    if actual.layer0_out_projection_applied != 0
        || actual.layer0_out_projection_materialized_count != 0
        || actual.layer0_out_projection_fnv1a64 != 0
        || actual.last_layer0_out_projection_fnv1a64 != 0
        || actual.layer0_out_projection_weight_fnv1a64 != 0
        || actual.layer0_out_projection_weight_bytes_read != 0
        || actual.layer0_out_projection_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer0 out projection data before manifest-backed execution"
        ));
    }
    if actual.layer0_residual_hidden_applied != 0
        || actual.layer0_residual_hidden_materialized_count != 0
        || actual.layer0_residual_hidden_fnv1a64 != 0
        || actual.last_layer0_residual_hidden_fnv1a64 != 0
        || actual.layer0_residual_hidden_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer0 residual hidden data before manifest-backed execution"
        ));
    }
    if actual.layer0_post_attention_rmsnorm_applied != 0
        || actual.layer0_post_attention_rmsnorm_materialized_count != 0
        || actual.layer0_post_attention_rmsnorm_fnv1a64 != 0
        || actual.last_layer0_post_attention_rmsnorm_fnv1a64 != 0
        || actual.layer0_post_attention_norm_weight_fnv1a64 != 0
        || actual.layer0_post_attention_norm_weight_bytes_read != 0
        || actual.layer0_post_attention_rmsnorm_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer0 post-attention RMSNorm data before manifest-backed execution"
        ));
    }
    if actual.layer0_moe_router_applied != 0
        || actual.layer0_moe_router_materialized_count != 0
        || actual.layer0_moe_router_logits_fnv1a64 != 0
        || actual.last_layer0_moe_router_logits_fnv1a64 != 0
        || actual.layer0_moe_router_topk_ids_fnv1a64 != 0
        || actual.last_layer0_moe_router_topk_ids_fnv1a64 != 0
        || actual.layer0_moe_router_topk_weights_fnv1a64 != 0
        || actual.last_layer0_moe_router_topk_weights_fnv1a64 != 0
        || actual.layer0_moe_router_weight_fnv1a64 != 0
        || actual.layer0_moe_router_weight_bytes_read != 0
        || actual.layer0_moe_router_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer0 MoE router data before manifest-backed execution"
        ));
    }
    if actual.layer0_moe_expert_applied != 0
        || actual.layer0_moe_expert_materialized_count != 0
        || actual.layer0_moe_expert_selected_count != 0
        || actual.layer0_moe_expert_routed_fnv1a64 != 0
        || actual.last_layer0_moe_expert_routed_fnv1a64 != 0
        || actual.layer0_moe_expert_selected_ids_fnv1a64 != 0
        || actual.layer0_moe_expert_gate_up_weight_fnv1a64 != 0
        || actual.layer0_moe_expert_down_weight_fnv1a64 != 0
        || actual.layer0_moe_expert_weight_bytes_read != 0
        || actual.layer0_moe_expert_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer0 MoE expert data before manifest-backed execution"
        ));
    }
    if actual.layer0_moe_shared_expert_applied != 0
        || actual.layer0_moe_shared_expert_materialized_count != 0
        || actual.layer0_moe_shared_expert_fnv1a64 != 0
        || actual.last_layer0_moe_shared_expert_fnv1a64 != 0
        || actual.layer0_moe_combined_fnv1a64 != 0
        || actual.last_layer0_moe_combined_fnv1a64 != 0
        || actual.layer0_moe_shared_gate_fnv1a64 != 0
        || actual.last_layer0_moe_shared_gate_fnv1a64 != 0
        || actual.layer0_moe_shared_gate_weight_fnv1a64 != 0
        || actual.layer0_moe_shared_gate_proj_weight_fnv1a64 != 0
        || actual.layer0_moe_shared_up_proj_weight_fnv1a64 != 0
        || actual.layer0_moe_shared_down_weight_fnv1a64 != 0
        || actual.layer0_moe_shared_expert_weight_bytes_read != 0
        || actual.layer0_moe_shared_expert_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer0 MoE shared expert data before manifest-backed execution"
        ));
    }
    if actual.layer0_output_residual_applied != 0
        || actual.layer0_output_residual_materialized_count != 0
        || actual.layer0_output_residual_fnv1a64 != 0
        || actual.last_layer0_output_residual_fnv1a64 != 0
        || actual.layer0_output_residual_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer0 output residual data before manifest-backed execution"
        ));
    }
    if actual.layer1_input_norm_applied != 0
        || actual.layer1_input_norm_materialized_count != 0
        || actual.layer1_input_norm_fnv1a64 != 0
        || actual.last_layer1_input_norm_fnv1a64 != 0
        || actual.layer1_input_norm_weight_fnv1a64 != 0
        || actual.layer1_input_norm_weight_bytes_read != 0
        || actual.layer1_input_norm_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 input norm data before manifest-backed execution"
        ));
    }
    if actual.layer1_qkv_projection_applied != 0
        || actual.layer1_qkv_projection_materialized_count != 0
        || actual.layer1_qkv_projection_fnv1a64 != 0
        || actual.last_layer1_qkv_projection_fnv1a64 != 0
        || actual.layer1_qkv_projection_weight_fnv1a64 != 0
        || actual.layer1_qkv_projection_weight_bytes_read != 0
        || actual.layer1_qkv_projection_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 QKV projection data before manifest-backed execution"
        ));
    }
    if actual.layer1_zab_projection_applied != 0
        || actual.layer1_zab_projection_materialized_count != 0
        || actual.layer1_zab_projection_fnv1a64 != 0
        || actual.last_layer1_zab_projection_fnv1a64 != 0
        || actual.layer1_z_projection_weight_fnv1a64 != 0
        || actual.layer1_a_projection_weight_fnv1a64 != 0
        || actual.layer1_b_projection_weight_fnv1a64 != 0
        || actual.layer1_zab_projection_weight_bytes_read != 0
        || actual.layer1_zab_projection_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 Z/A/B projection data before manifest-backed execution"
        ));
    }
    if actual.layer1_conv_qkv_applied != 0
        || actual.layer1_conv_qkv_materialized_count != 0
        || actual.layer1_conv_qkv_fnv1a64 != 0
        || actual.last_layer1_conv_qkv_fnv1a64 != 0
        || actual.layer1_conv_qkv_weight_fnv1a64 != 0
        || actual.layer1_conv_qkv_weight_bytes_read != 0
        || actual.layer1_conv_qkv_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 conv_qkv data before manifest-backed execution"
        ));
    }
    if actual.layer1_postconv_qkv_applied != 0
        || actual.layer1_postconv_qkv_materialized_count != 0
        || actual.layer1_postconv_q_scaled_fnv1a64 != 0
        || actual.last_layer1_postconv_q_scaled_fnv1a64 != 0
        || actual.layer1_postconv_k_norm_fnv1a64 != 0
        || actual.last_layer1_postconv_k_norm_fnv1a64 != 0
        || actual.layer1_postconv_value_fnv1a64 != 0
        || actual.last_layer1_postconv_value_fnv1a64 != 0
        || actual.layer1_postconv_qkv_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 postconv_qkv data before manifest-backed execution"
        ));
    }
    if actual.layer1_gate_rows_applied != 0
        || actual.layer1_gate_rows_materialized_count != 0
        || actual.layer1_gate_g_fnv1a64 != 0
        || actual.last_layer1_gate_g_fnv1a64 != 0
        || actual.layer1_gate_beta_fnv1a64 != 0
        || actual.last_layer1_gate_beta_fnv1a64 != 0
        || actual.layer1_a_log_fnv1a64 != 0
        || actual.layer1_dt_bias_fnv1a64 != 0
        || actual.layer1_gate_weight_bytes_read != 0
        || actual.layer1_gate_rows_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 gate rows data before manifest-backed execution"
        ));
    }
    if actual.layer1_core_rows_applied != 0
        || actual.layer1_core_rows_materialized_count != 0
        || actual.layer1_core_rows_fnv1a64 != 0
        || actual.last_layer1_core_rows_fnv1a64 != 0
        || actual.layer1_core_final_state_fnv1a64 != 0
        || actual.layer1_core_rows_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 core rows data before manifest-backed execution"
        ));
    }
    if actual.layer1_gated_rmsnorm_applied != 0
        || actual.layer1_gated_rmsnorm_materialized_count != 0
        || actual.layer1_gated_rmsnorm_fnv1a64 != 0
        || actual.last_layer1_gated_rmsnorm_fnv1a64 != 0
        || actual.layer1_linear_norm_weight_fnv1a64 != 0
        || actual.layer1_linear_norm_weight_bytes_read != 0
        || actual.layer1_gated_rmsnorm_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 gated RMSNorm data before manifest-backed execution"
        ));
    }
    if actual.layer1_out_projection_applied != 0
        || actual.layer1_out_projection_materialized_count != 0
        || actual.layer1_out_projection_fnv1a64 != 0
        || actual.last_layer1_out_projection_fnv1a64 != 0
        || actual.layer1_out_projection_weight_fnv1a64 != 0
        || actual.layer1_out_projection_weight_bytes_read != 0
        || actual.layer1_out_projection_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 out projection data before manifest-backed execution"
        ));
    }
    if actual.layer1_residual_hidden_applied != 0
        || actual.layer1_residual_hidden_materialized_count != 0
        || actual.layer1_residual_hidden_fnv1a64 != 0
        || actual.last_layer1_residual_hidden_fnv1a64 != 0
        || actual.layer1_residual_hidden_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 residual hidden data before manifest-backed execution"
        ));
    }
    if actual.layer1_post_attention_rmsnorm_applied != 0
        || actual.layer1_post_attention_rmsnorm_materialized_count != 0
        || actual.layer1_post_attention_rmsnorm_fnv1a64 != 0
        || actual.last_layer1_post_attention_rmsnorm_fnv1a64 != 0
        || actual.layer1_post_attention_norm_weight_fnv1a64 != 0
        || actual.layer1_post_attention_norm_weight_bytes_read != 0
        || actual.layer1_post_attention_rmsnorm_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 post-attention RMSNorm data before manifest-backed execution"
        ));
    }
    if actual.layer1_moe_router_applied != 0
        || actual.layer1_moe_router_materialized_count != 0
        || actual.layer1_moe_router_logits_fnv1a64 != 0
        || actual.last_layer1_moe_router_logits_fnv1a64 != 0
        || actual.layer1_moe_router_topk_ids_fnv1a64 != 0
        || actual.last_layer1_moe_router_topk_ids_fnv1a64 != 0
        || actual.layer1_moe_router_topk_weights_fnv1a64 != 0
        || actual.last_layer1_moe_router_topk_weights_fnv1a64 != 0
        || actual.layer1_moe_router_weight_fnv1a64 != 0
        || actual.layer1_moe_router_weight_bytes_read != 0
        || actual.layer1_moe_router_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 MoE router data before manifest-backed execution"
        ));
    }
    if actual.layer1_moe_expert_applied != 0
        || actual.layer1_moe_expert_materialized_count != 0
        || actual.layer1_moe_expert_selected_count != 0
        || actual.layer1_moe_expert_routed_fnv1a64 != 0
        || actual.last_layer1_moe_expert_routed_fnv1a64 != 0
        || actual.layer1_moe_expert_selected_ids_fnv1a64 != 0
        || actual.layer1_moe_expert_gate_up_weight_fnv1a64 != 0
        || actual.layer1_moe_expert_down_weight_fnv1a64 != 0
        || actual.layer1_moe_expert_weight_bytes_read != 0
        || actual.layer1_moe_expert_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 MoE expert data before manifest-backed execution"
        ));
    }
    if actual.layer1_moe_shared_expert_applied != 0
        || actual.layer1_moe_shared_expert_materialized_count != 0
        || actual.layer1_moe_shared_expert_fnv1a64 != 0
        || actual.last_layer1_moe_shared_expert_fnv1a64 != 0
        || actual.layer1_moe_combined_fnv1a64 != 0
        || actual.last_layer1_moe_combined_fnv1a64 != 0
        || actual.layer1_moe_shared_gate_fnv1a64 != 0
        || actual.last_layer1_moe_shared_gate_fnv1a64 != 0
        || actual.layer1_moe_shared_gate_weight_fnv1a64 != 0
        || actual.layer1_moe_shared_gate_proj_weight_fnv1a64 != 0
        || actual.layer1_moe_shared_up_proj_weight_fnv1a64 != 0
        || actual.layer1_moe_shared_down_weight_fnv1a64 != 0
        || actual.layer1_moe_shared_expert_weight_bytes_read != 0
        || actual.layer1_moe_shared_expert_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 MoE shared expert data before manifest-backed execution"
        ));
    }
    if actual.layer1_output_residual_applied != 0
        || actual.layer1_output_residual_materialized_count != 0
        || actual.layer1_output_residual_fnv1a64 != 0
        || actual.last_layer1_output_residual_fnv1a64 != 0
        || actual.layer1_output_residual_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer1 output residual data before manifest-backed execution"
        ));
    }
    if actual.layer2_input_norm_applied != 0
        || actual.layer2_input_norm_materialized_count != 0
        || actual.layer2_input_norm_fnv1a64 != 0
        || actual.last_layer2_input_norm_fnv1a64 != 0
        || actual.layer2_input_norm_weight_fnv1a64 != 0
        || actual.layer2_input_norm_weight_bytes_read != 0
        || actual.layer2_input_norm_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer2 input norm data before manifest-backed execution"
        ));
    }
    if actual.layer2_qkv_projection_applied != 0
        || actual.layer2_qkv_projection_materialized_count != 0
        || actual.layer2_qkv_projection_fnv1a64 != 0
        || actual.last_layer2_qkv_projection_fnv1a64 != 0
        || actual.layer2_qkv_projection_weight_fnv1a64 != 0
        || actual.layer2_qkv_projection_weight_bytes_read != 0
        || actual.layer2_qkv_projection_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer2 QKV projection data before manifest-backed execution"
        ));
    }
    if actual.layer2_zab_projection_applied != 0
        || actual.layer2_zab_projection_materialized_count != 0
        || actual.layer2_zab_projection_fnv1a64 != 0
        || actual.last_layer2_zab_projection_fnv1a64 != 0
        || actual.layer2_z_projection_weight_fnv1a64 != 0
        || actual.layer2_a_projection_weight_fnv1a64 != 0
        || actual.layer2_b_projection_weight_fnv1a64 != 0
        || actual.layer2_zab_projection_weight_bytes_read != 0
        || actual.layer2_zab_projection_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer2 Z/A/B projection data before manifest-backed execution"
        ));
    }
    if actual.layer2_conv_qkv_applied != 0
        || actual.layer2_conv_qkv_materialized_count != 0
        || actual.layer2_conv_qkv_fnv1a64 != 0
        || actual.last_layer2_conv_qkv_fnv1a64 != 0
        || actual.layer2_conv_qkv_weight_fnv1a64 != 0
        || actual.layer2_conv_qkv_weight_bytes_read != 0
        || actual.layer2_conv_qkv_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer2 conv_qkv data before manifest-backed execution"
        ));
    }
    if actual.layer2_postconv_qkv_applied != 0
        || actual.layer2_postconv_qkv_materialized_count != 0
        || actual.layer2_postconv_q_scaled_fnv1a64 != 0
        || actual.last_layer2_postconv_q_scaled_fnv1a64 != 0
        || actual.layer2_postconv_k_norm_fnv1a64 != 0
        || actual.last_layer2_postconv_k_norm_fnv1a64 != 0
        || actual.layer2_postconv_value_fnv1a64 != 0
        || actual.last_layer2_postconv_value_fnv1a64 != 0
        || actual.layer2_postconv_qkv_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer2 postconv_qkv data before manifest-backed execution"
        ));
    }
    if actual.layer2_gate_rows_applied != 0
        || actual.layer2_gate_rows_materialized_count != 0
        || actual.layer2_gate_g_fnv1a64 != 0
        || actual.last_layer2_gate_g_fnv1a64 != 0
        || actual.layer2_gate_beta_fnv1a64 != 0
        || actual.last_layer2_gate_beta_fnv1a64 != 0
        || actual.layer2_a_log_fnv1a64 != 0
        || actual.layer2_dt_bias_fnv1a64 != 0
        || actual.layer2_gate_weight_bytes_read != 0
        || actual.layer2_gate_rows_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer2 gate rows data before manifest-backed execution"
        ));
    }
    if actual.layer2_core_rows_applied != 0
        || actual.layer2_core_rows_materialized_count != 0
        || actual.layer2_core_rows_fnv1a64 != 0
        || actual.last_layer2_core_rows_fnv1a64 != 0
        || actual.layer2_core_final_state_fnv1a64 != 0
        || actual.layer2_core_rows_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer2 core rows data before manifest-backed execution"
        ));
    }
    if actual.layer2_gated_rmsnorm_applied != 0
        || actual.layer2_gated_rmsnorm_materialized_count != 0
        || actual.layer2_gated_rmsnorm_fnv1a64 != 0
        || actual.last_layer2_gated_rmsnorm_fnv1a64 != 0
        || actual.layer2_linear_norm_weight_fnv1a64 != 0
        || actual.layer2_linear_norm_weight_bytes_read != 0
        || actual.layer2_gated_rmsnorm_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer2 gated RMSNorm data before manifest-backed execution"
        ));
    }
    if actual.layer2_out_projection_applied != 0
        || actual.layer2_out_projection_materialized_count != 0
        || actual.layer2_out_projection_fnv1a64 != 0
        || actual.last_layer2_out_projection_fnv1a64 != 0
        || actual.layer2_out_projection_weight_fnv1a64 != 0
        || actual.layer2_out_projection_weight_bytes_read != 0
        || actual.layer2_out_projection_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer2 output projection data before manifest-backed execution"
        ));
    }
    if actual.layer2_residual_hidden_applied != 0
        || actual.layer2_residual_hidden_materialized_count != 0
        || actual.layer2_residual_hidden_fnv1a64 != 0
        || actual.last_layer2_residual_hidden_fnv1a64 != 0
        || actual.layer2_residual_hidden_elapsed_ns != 0
    {
        return Err(format!(
            "{label} reported layer2 residual hidden data before manifest-backed execution"
        ));
    }
    if cstr(actual.first_missing_module.as_ptr()) != expected_missing_module {
        return Err(format!(
            "{label} first missing module mismatch: {}",
            cstr(actual.first_missing_module.as_ptr())
        ));
    }
    if cstr(actual.failure_stage.as_ptr()) != expected_failure_stage {
        return Err(format!(
            "{label} failure stage mismatch: {}",
            cstr(actual.failure_stage.as_ptr())
        ));
    }
    if actual.inference_success_claimed != 0 {
        return Err(format!(
            "{label} claimed inference success before implementation"
        ));
    }
    Ok(())
}

fn validate_baseline_surface_check(
    surface: &QrtQwen36BaselineSurfaceCheck,
    descriptor: &QrtQwen36BaselineCheck,
) -> Result<(), String> {
    if surface.correctness_pass == 0 {
        return Err("baseline surface check reported correctness_pass=false".to_string());
    }
    if surface.baseline_plan_attached != 1
        || surface.prefill_entrypoint_planned != 1
        || surface.decode_one_entrypoint_planned != 1
    {
        return Err("baseline surface did not expose both planned entrypoints".to_string());
    }
    if surface.layer_count != descriptor.layer_count
        || surface.linear_attention_layer_count != descriptor.linear_attention_layer_count
        || surface.full_attention_layer_count != descriptor.full_attention_layer_count
        || surface.descriptor_fnv1a64 != descriptor.descriptor_fnv1a64
    {
        return Err("baseline surface descriptor fields drifted from descriptor check".to_string());
    }
    if surface.prefill_call_count != 1 || surface.decode_one_call_count != 1 {
        return Err(format!(
            "baseline surface call counts mismatch: prefill={} decode_one={}",
            surface.prefill_call_count, surface.decode_one_call_count
        ));
    }
    Ok(())
}

fn qrt_error(status: c_int) -> String {
    unsafe { cstr(qrt_strerror(status)).to_string() }
}

fn cstr(ptr: *const c_char) -> &'static str {
    if ptr.is_null() {
        return "<null>";
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .unwrap_or("<invalid-utf8>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_names_baiying() {
        let contract = unsafe {
            qrt_target_contract()
                .as_ref()
                .expect("qrt_target_contract returned null")
        };
        assert_eq!(cstr(contract.test_host), "baiying");
        assert_eq!(cstr(contract.target_os), "Windows");
        assert_eq!(cstr(contract.target_device), "AMD395");
    }

    #[test]
    fn legacy_text_generate_is_explicitly_unsupported() {
        self_test().expect("self-test should pass");
    }

    #[test]
    fn qmatvec_slice_fixture_matches_expected_hash() {
        slice_self_test().expect("slice fixture should pass");
    }

    #[test]
    fn micro_fixture_runs_control_flow_boundaries() {
        micro_self_test().expect("micro fixture should pass");
    }

    #[test]
    fn baseline_descriptor_plan_covers_qwen36_engine_shape() {
        baseline_self_test().expect("baseline descriptor check should pass");
    }
}
