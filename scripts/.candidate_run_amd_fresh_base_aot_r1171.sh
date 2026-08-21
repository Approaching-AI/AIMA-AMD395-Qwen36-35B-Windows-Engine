#!/usr/bin/env bash

set -Eeuo pipefail
umask 022
ulimit -c 0

workspace_root=${QRT_AMD_WORKSPACE_ROOT:?set the bounded AMD workspace root}
base=${workspace_root}/AIMA-dynamic-logical-smoke-r1093-t3J3OB
fresh_root=${QRT_AMD_FRESH_BASE_AOT_ROOT:-${workspace_root}/qrt-dynamic-aot-r1171-REFHyB}
kernel_dir=${fresh_root}/kernels
run_label=${QRT_AMD_REAL_LAYER3_RUN_LABEL:-r1171}
run_dir=${fresh_root}/real-layer3-${run_label}
aux_kernel_dir=${base}/zero-correction-finalize-r1150/build/kernels
provider=${QRT_AMD_NATIVE_SHARED_PROVIDER:-${base}/real-q8192-layer3-native-shared-r1160/build/provider-group-m1-native-shared.so}
replay=${base}/real-q8192-layer3-correction-boundary-r1148/run/q8192_real_moe_replay
capture_dir=${base}/real-q8192-layer3-correction-boundary-r1148/capture
weights_dir=${base}/real-q8192-layer3-correction-boundary-r1148/weights

expected_generator_sha256=d5497e32c0bf5514a2cc136be95d2a02b66538906b21e42224b4362826add6a4
expected_metadata_sha256=${QRT_AMD_BASE_AOT_METADATA_SHA256:-37cf35ffd130f6df51bd95661efd924dc398ee818ae2b3086ddb0af1bbc57199}
expected_provider_sha256=${QRT_AMD_NATIVE_SHARED_PROVIDER_SHA256:-74e38c27ada6963158cffe4ad936c8acca55e3d4946c7937e8d784878c8563b7}
expected_capture_input_sha256=356bacc6f40b177676d5146854a716bf37e15e1ff7db488da6e4689c2c4b2f28
expected_capture_output_sha256=557d9176e1ddbe969e3651b546dda03ca568b96e99c85059b69b20cf640ba11d
expected_row_major_gate_sha256=${QRT_AMD_AUX_ROW_MAJOR_GATE_SHA256:-e6504e2da5345509304155153323b60c0d084527c55ed1db6918699294e21a6c}
expected_zero_finalize_sha256=${QRT_AMD_AUX_ZERO_FINALIZE_SHA256:-032b60f88cce479b19e94854813ea55d1c1790082a533c826ef326c4a0eb51ba}
expected_conditional_down_sha256=${QRT_AMD_AUX_CONDITIONAL_DOWN_SHA256:-60252375edfffc86f16a17da85e8fcdd321129b2f49c81d898eaf8404d9a7289}
aux_debug_sections_stripped=${QRT_AMD_AUX_DEBUG_SECTIONS_STRIPPED:-0}

case "${fresh_root}" in
    "${workspace_root}"/qrt-dynamic-aot-r1171-*) ;;
    *) echo "fresh AOT root escaped the r1171 staging family" >&2; exit 2 ;;
esac
[[ ${run_label} =~ ^r[0-9]+(-[a-z0-9-]+)?$ ]] || {
    echo "invalid real-layer3 run label" >&2
    exit 2
}
[[ ${aux_debug_sections_stripped} == 0 || \
    ${aux_debug_sections_stripped} == 1 ]] || {
    echo "invalid auxiliary debug-section status" >&2
    exit 2
}
[[ ! -e ${run_dir} ]] || {
    echo "refusing to overwrite ${run_dir}" >&2
    exit 2
}
for required in "${fresh_root}/compile_q8192_triton_selected_moe.py" \
    "${kernel_dir}/metadata.json" "${provider}" "${replay}" \
    "${capture_dir}/full-moe-input-bf16.bin" \
    "${capture_dir}/full-moe-output-bf16.bin"; do
    [[ -f ${required} ]] || { echo "missing ${required}" >&2; exit 3; }
done

verify_sha256() {
    local path=$1
    local expected=$2
    local actual
    actual=$(sha256sum "${path}" | awk '{print $1}')
    [[ ${actual} == "${expected}" ]] || {
        echo "sha256 mismatch path=${path} actual=${actual}" >&2
        exit 3
    }
}
verify_sha256 "${fresh_root}/compile_q8192_triton_selected_moe.py" \
    "${expected_generator_sha256}"
verify_sha256 "${kernel_dir}/metadata.json" "${expected_metadata_sha256}"
verify_sha256 "${provider}" "${expected_provider_sha256}"
verify_sha256 "${capture_dir}/full-moe-input-bf16.bin" \
    "${expected_capture_input_sha256}"
verify_sha256 "${capture_dir}/full-moe-output-bf16.bin" \
    "${expected_capture_output_sha256}"

declare -A auxiliary_hashes=(
    [q8192_triton_0626_row_major_sorted_conditional_exact_gate_rows256.hsaco]=${expected_row_major_gate_sha256}
    [q8192_triton_0626_zero_correction_gate_finalize.hsaco]=${expected_zero_finalize_sha256}
    [q8192_triton_0626_conditional_exact_down_rows4.hsaco]=${expected_conditional_down_sha256}
)
for name in "${!auxiliary_hashes[@]}"; do
    if [[ -e ${kernel_dir}/${name} ]]; then
        verify_sha256 "${kernel_dir}/${name}" "${auxiliary_hashes[${name}]}"
    else
        verify_sha256 "${aux_kernel_dir}/${name}" \
            "${auxiliary_hashes[${name}]}"
        cp -- "${aux_kernel_dir}/${name}" "${kernel_dir}/${name}"
    fi
done

mkdir -p -- "${run_dir}"
{
    printf 'record_type=qrt_amd395_fresh_base_aot_real_layer3\n'
    printf 'schema_version=1\n'
    printf 'host=%s\n' "$(hostname)"
    printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'repo_commit=6cdce444177f54e8985a2d6148da863369bece32\n'
    printf 'run_label=%s\n' "${run_label}"
    printf 'generator_sha256=%s\n' "${expected_generator_sha256}"
    printf 'metadata_sha256=%s\n' "${expected_metadata_sha256}"
    printf 'provider_sha256=%s\n' "${expected_provider_sha256}"
    printf 'base_aot_block_m=64\nbase_aot_group_m=1\n'
    printf 'base_aot_dynamic_logical_abi=1\n'
    printf 'auxiliary_aot_debug_sections_stripped=%s\n' \
        "${aux_debug_sections_stripped}"
    printf 'auxiliary_row_major_gate_sha256=%s\n' \
        "${expected_row_major_gate_sha256}"
    printf 'auxiliary_zero_finalize_sha256=%s\n' \
        "${expected_zero_finalize_sha256}"
    printf 'auxiliary_conditional_down_sha256=%s\n' \
        "${expected_conditional_down_sha256}"
    printf 'gb10_capture_host=gb10-4t\n'
    printf 'gb10_capture_layer=3\n'
    printf 'gb10_capture_prompt_tokens=8192\n'
    printf 'component_only=1\ninference_success_claimed=0\n'
} > "${run_dir}/provenance.env"
sha256sum "${kernel_dir}"/*.hsaco > "${run_dir}/kernel-artifacts.sha256"

idle_streak=0
idle_checks=0
while [[ ${idle_streak} -lt 3 && ${idle_checks} -lt 15 ]]; do
    owners=$(fuser /dev/kfd 2>/dev/null || true)
    owners=${owners//[[:space:]]/}
    idle_checks=$((idle_checks + 1))
    if [[ -z ${owners} ]]; then
        idle_streak=$((idle_streak + 1))
        echo "gpu_idle_check streak=${idle_streak}/3 check=${idle_checks}/15"
    else
        idle_streak=0
        echo "gpu_busy_check owners=${owners} check=${idle_checks}/15"
    fi
    [[ ${idle_streak} -ge 3 ]] || sleep 20
done
[[ ${idle_streak} -eq 3 ]] || { echo "GPU did not become idle" >&2; exit 75; }

export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export HIP_VISIBLE_DEVICES=0
export QRT_QWEN36_Q8192_ROUTER_HIPBLASLT_BF16=1
export QRT_QWEN36_CUDA_VLLM_ROUTER_HAWKEYE_MIDPOINT_RADIUS=173
export QRT_QWEN36_CUDA_VLLM_SHARED_HAWKEYE_MIDPOINT_RADIUS=0
export QRT_QWEN36_Q8192_VLLM_BF16_RESIDUAL_CARRIER=1
export QRT_QWEN36_EXACT_ARBITRARY_VLLM_SPLIT_VARIANCE=1
export QRT_QWEN36_Q8192_VLLM_SORTED_BF16_ROUTE_SUM=1
export QRT_QWEN36_CUDA_VLLM_MOE_HAWKEYE_MIDPOINT_RADIUS=0
export QRT_QWEN36_CUDA_VLLM_MOE_UP_HAWKEYE_MIDPOINT_RADIUS=0
export QRT_QWEN36_CUDA_VLLM_ROUTED_DOWN_CONTRIBUTION_HAWKEYE_MIDPOINT_RADIUS=0
export QRT_QWEN36_CUDA_VLLM_ROUTED_GATE_HAWKEYE_LOW_EXPONENT_THRESHOLD=0
export QRT_QWEN36_CUDA_VLLM_ROUTED_UP_HAWKEYE_LOW_EXPONENT_THRESHOLD=0
export QRT_QWEN36_CUDA_VLLM_ROUTED_DOWN_HAWKEYE_LOW_EXPONENT_THRESHOLD=0

set +e
timeout --signal=TERM --kill-after=20s 180s \
    "${replay}" "${kernel_dir}" "${provider}" \
    "${capture_dir}/full-moe-input-bf16.bin" \
    "${capture_dir}/full-moe-output-bf16.bin" \
    "${weights_dir}/router.bf16" \
    "${weights_dir}/routed-gate-up.bf16" \
    "${weights_dir}/routed-down.bf16" \
    "${weights_dir}/shared-gate.bf16" \
    "${weights_dir}/shared-gate-projection.bf16" \
    "${weights_dir}/shared-up-projection.bf16" \
    "${weights_dir}/shared-down.bf16" \
    "${run_dir}/output.bf16" > "${run_dir}/replay.json" \
    2> "${run_dir}/replay.stderr.log"
replay_rc=$?
set -e
printf 'replay_exit_code=%s\n' "${replay_rc}" >> "${run_dir}/provenance.env"
[[ ${replay_rc} -eq 0 || ${replay_rc} -eq 3 ]] || exit "${replay_rc}"

python3 - "${run_dir}/replay.json" <<'PY'
import json
import sys

path = sys.argv[1]
record = json.load(open(path, encoding="utf-8"))
assert record["gb10_real_input"] is True
assert record["component_only"] is True
assert record["inference_success_claimed"] is False
assert record["nonfinite_count"] == 0
assert record["abs_difference_over_0_125_count"] == 0
assert record["max_abs_difference_to_reference_bf16"] <= 0.125
print(json.dumps({
    "status": "pass",
    "elapsed_ms": record["elapsed_ms"],
    "output_bf16_fnv1a64": record["output_bf16_fnv1a64"],
    "max_abs_difference_to_gb10_bf16": (
        record["max_abs_difference_to_reference_bf16"]
    ),
    "abs_difference_over_0_125_count": (
        record["abs_difference_over_0_125_count"]
    ),
    "component_only": True,
    "inference_success_claimed": False,
}, sort_keys=True))
PY
