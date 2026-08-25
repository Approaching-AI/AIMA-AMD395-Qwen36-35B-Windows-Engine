#!/usr/bin/env bash

set -Eeuo pipefail
umask 022
ulimit -c 0

workspace_root=${QRT_AMD_WORKSPACE_ROOT:?set the bounded AMD workspace root}
stage_root=${QRT_AMD_PUBLIC_SILU_STAGE_ROOT:?set the bounded public SiLU AOT stage root}
candidate_commit=${QRT_PUBLIC_CANDIDATE_COMMIT:?set the exact candidate commit}
run_label=${QRT_AMD_PUBLIC_SILU_RUN_LABEL:-r1193}
kernel_dir=${stage_root}/kernels
run_dir=${stage_root}/qualification-${run_label}
base=${workspace_root}/AIMA-dynamic-logical-smoke-r1093-t3J3OB
accepted_root=${workspace_root}/qrt-dynamic-aot-r1171-REFHyB
provider=${accepted_root}/provider-r1175b/provider-native-shared-v2-sorted-parity.so
smoke=${accepted_root}/smoke-r1196b/moe-v2-sorted-oracle-smoke
replay=${base}/real-q8192-layer3-correction-boundary-r1148/run/q8192_real_moe_replay
capture_dir=${base}/real-q8192-layer3-correction-boundary-r1148/capture
weights_dir=${base}/real-q8192-layer3-correction-boundary-r1148/weights

expected_metadata_sha256=41e77d3afdecba63ee15c63e5f585f44ae29a6220e296e2752dcc18cfd24dc14
expected_provider_source_sha256=894760e70f42d6f92a8a3c322ae3533960f311d6704fb251cd2b8c6bb3894c19
expected_provider_sha256=5170bbf386a40837cfbb142ece292fa88217849b5bb771fbe68406e688ca2e84
expected_smoke_source_sha256=ccdc0fe31be05ff2a808c483a59adeae29bd0f4232b8c2ad169291359808a3ab
expected_smoke_sha256=ec64172ab8cdb9b037bce169e51307cb8c66abeb8e74e40003fcf0a34c3bfaa9
expected_replay_sha256=06af52b32427d8c86e9de0e0cfb127bccf1b43156871082dab8f9eebaa795b49
expected_capture_input_sha256=356bacc6f40b177676d5146854a716bf37e15e1ff7db488da6e4689c2c4b2f28
expected_capture_output_sha256=557d9176e1ddbe969e3651b546dda03ca568b96e99c85059b69b20cf640ba11d

case "${stage_root}" in
    "${workspace_root}"/qrt-dynamic-aot-r1171-silu-r1193-*) ;;
    *) echo "public SiLU AOT stage escaped the r1193 family" >&2; exit 2 ;;
esac
[[ ${candidate_commit} =~ ^[0-9a-f]{40}$ ]] || {
    echo "candidate commit must be a full lowercase SHA" >&2
    exit 2
}
[[ ${run_label} =~ ^r[0-9]+(-[a-z0-9-]+)?$ ]] || {
    echo "invalid public SiLU qualification label" >&2
    exit 2
}
[[ ! -e ${run_dir} ]] || {
    echo "refusing to overwrite ${run_dir}" >&2
    exit 2
}

verify_sha256() {
    local path=$1
    local expected=$2
    local actual
    [[ -f ${path} ]] || { echo "missing ${path}" >&2; exit 3; }
    actual=$(sha256sum "${path}" | awk '{print $1}')
    [[ ${actual} == "${expected}" ]] || {
        echo "sha256 mismatch path=${path} actual=${actual}" >&2
        exit 3
    }
}

verify_sha256 "${kernel_dir}/metadata.json" "${expected_metadata_sha256}"
verify_sha256 "${provider}" "${expected_provider_sha256}"
verify_sha256 "${smoke}" "${expected_smoke_sha256}"
verify_sha256 "${replay}" "${expected_replay_sha256}"
verify_sha256 "${capture_dir}/full-moe-input-bf16.bin" \
    "${expected_capture_input_sha256}"
verify_sha256 "${capture_dir}/full-moe-output-bf16.bin" \
    "${expected_capture_output_sha256}"

declare -A kernel_hashes=(
    [q8192_selected_moe_route_count.hsaco]=48cc7d3660fa6051f085996e827fe9d0485f52be306273d4d1a9bea8d7b39a39
    [q8192_selected_moe_route_prefix_by_program.hsaco]=3aaec2826f234daad9a308ebc8957e0cc309d524ddc92984c39e5027eb80db4d
    [q8192_selected_moe_route_padded_prefix.hsaco]=8c6fd362ea0b5e9b7d43ec99ae404ddef021ffed9844ad1203a82c307483b914
    [q8192_selected_moe_route_scatter.hsaco]=dd87e1236cda5eb292b0c918ee98d04218e1ee174a2e9b86e98bf21644d15c7f
    [q8192_selected_moe_gate_up_silu.hsaco]=945acb545a0cdbdb333ad7ed863a081b4e4a6a0b0872142482e307713529c0fe
    [q8192_selected_moe_down.hsaco]=bd8b6970d1bcc86fe8eb8f9ee4a8d70fb24222badeddf9e477e9596c29d6c30a
    [q8192_triton_0626_row_major_sorted_conditional_exact_gate_rows256.hsaco]=cba71fdc8d510bc8f41d1e2501d3a64a8e7f2d3eb1078d4a940cc1c23a25be54
    [q8192_triton_0626_zero_correction_gate_finalize.hsaco]=8105adcbd809bb01982c065bcd8f03b168313ff8f31fd78a4ffd04728846a4c6
    [q8192_triton_0626_conditional_exact_down_rows4.hsaco]=2b430b0226d09af12b36018eed1a9f761141bee37793d5879c142cdd9cb8c1c4
)
for name in "${!kernel_hashes[@]}"; do
    verify_sha256 "${kernel_dir}/${name}" "${kernel_hashes[${name}]}"
done

declare -A weight_hashes=(
    [router.bf16]=86a1ba236777d0207ee3661e316b294b3e684c642ed0b27e9063fb5117383bbe
    [routed-gate-up.bf16]=89f62413f132671b333660ed20f23bac9cb454bc55afe818ee1917aa66c3265a
    [routed-down.bf16]=c18b5507075f0ec984c5eb823b96b18c2f7f4bbadc23c980c6f28e35a665aa60
    [shared-gate.bf16]=ac59b9445a977ad36d1bd18270b0bce42374cb45a7d845a552eaedade01330ca
    [shared-gate-projection.bf16]=eed6781e6f5b9023796c373b8c8a077371c3ce22de594c4cbc7be28e738a0a1a
    [shared-up-projection.bf16]=861c1c8e63be1e757769e84ed4411e4a496a55c540ff90d55c9957c444302ac8
    [shared-down.bf16]=c31e2776c9d317678cdb30eefd4024752de9d47de7e3b797053b0dfe92c8ee89
)
for name in "${!weight_hashes[@]}"; do
    verify_sha256 "${weights_dir}/${name}" "${weight_hashes[${name}]}"
done

idle_streak=0
for check in 1 2 3; do
    owners=$(fuser /dev/kfd 2>/dev/null || true)
    owners=${owners//[[:space:]]/}
    if [[ -z ${owners} ]]; then
        idle_streak=$((idle_streak + 1))
        echo "gpu_idle_check streak=${idle_streak}/3 check=${check}/3"
    else
        echo "gpu_busy_check owners=${owners} check=${check}/3" >&2
        exit 75
    fi
    [[ ${check} -eq 3 ]] || sleep 2
done
[[ ${idle_streak} -eq 3 ]]

mkdir -p -- "${run_dir}"
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
unset QRT_PRODUCT_RADIUS_LIGHT_TOKEN_SEQUENCE
unset QRT_PREFILL_DESCRIPTOR_BATCH_PROFILE_MOE_SUBPHASES

timeout --signal=TERM --kill-after=20s 300s \
    "${smoke}" "${kernel_dir}" 1 "${provider}" \
    > "${run_dir}/smoke.stdout.log" \
    2> "${run_dir}/smoke.stderr.log"

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
    "${run_dir}/output.bf16" \
    > "${run_dir}/replay.json" \
    2> "${run_dir}/replay.stderr.log"
replay_rc=$?
set -e
[[ ${replay_rc} -eq 0 || ${replay_rc} -eq 3 ]] || exit "${replay_rc}"

python3 - "${run_dir}/smoke.stdout.log" "${run_dir}/replay.json" \
    "${run_dir}/qualification.json" "${candidate_commit}" \
    "${replay_rc}" "${run_dir}/output.bf16" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    smoke_path,
    replay_path,
    output_path,
    candidate_commit,
    replay_rc,
    replay_output_path,
) = sys.argv[1:]
text = Path(smoke_path).read_text(encoding="utf-8")
expected_tokens = [
    2073, 2156, 2560, 3073, 4609, 6145, 2049, 2175,
    2176, 2177, 2559, 2561, 3071, 3072, 3583, 3584,
    3585, 4095, 4096, 4097, 4607, 4608, 6143, 6144,
    7167, 7168, 7169, 7679, 7680, 7681, 8191, 8192,
]
cases = []
summary_fields = None
for line in text.splitlines():
    fields = dict(item.split("=", 1) for item in line.split() if "=" in item)
    if line.startswith("dynamic_logical_case "):
        cases.append({
            "index": int(fields["index"]),
            "tokens": int(fields["tokens"]),
            "total_ms": float(fields["total_ms"]),
            "timing_samples": int(fields["timing_samples"]),
            "timing_stat": fields["timing_stat"],
            "component_only": int(fields["component_only"]),
            "inference_success_claimed": int(fields["inference_success_claimed"]),
        })
    elif line.startswith("q8192_triton_selected_moe_smoke "):
        summary_fields = fields
assert [case["index"] for case in cases] == list(range(32))
assert [case["tokens"] for case in cases] == expected_tokens
assert all(case["total_ms"] > 0.0 for case in cases)
assert all(case["timing_samples"] == 3 for case in cases)
assert all(case["timing_stat"] == "median" for case in cases)
assert all(case["component_only"] == 1 for case in cases)
assert all(case["inference_success_claimed"] == 0 for case in cases)
assert summary_fields is not None
for field in (
    "dynamic_logical_mismatches",
    "dynamic_logical_nonfinite",
    "dynamic_logical_q8192_mismatches",
    "full_provider_async_mismatches",
    "full_provider_v3_mismatches",
    "full_provider_v3_async_mismatches",
    "router_debug_id_mismatches",
    "router_debug_weight_mismatches",
):
    assert int(summary_fields[field]) == 0, (field, summary_fields[field])
assert float(summary_fields["dynamic_logical_max_abs_diff"]) == 0.0
assert float(summary_fields["dynamic_logical_q8192_max_abs_diff"]) == 0.0
assert int(summary_fields["dynamic_logical_tail_guard_pass"]) == 1
assert int(summary_fields["provider_backend_mask"]) == 15

replay = json.loads(Path(replay_path).read_text(encoding="utf-8"))
assert replay["gb10_real_input"] is True
assert replay["component_only"] is True
assert replay["inference_success_claimed"] is False
assert replay["nonfinite_count"] == 0
assert replay["abs_difference_over_0_125_count"] == 0
assert replay["max_abs_difference_to_reference_bf16"] <= 0.125

record = {
    "schema_version": 1,
    "record_type": "qrt_amd395_public_silu_aot_component_qualification",
    "status": "pass",
    "host": "quings",
    "candidate_commit": candidate_commit,
    "dynamic_logical": {
        "case_count": len(cases),
        "tokens": expected_tokens,
        "q7169_total_ms": next(
            case["total_ms"] for case in cases if case["tokens"] == 7169
        ),
        "q8192_total_ms": next(
            case["total_ms"] for case in cases if case["tokens"] == 8192
        ),
        "minimum_total_ms": min(case["total_ms"] for case in cases),
        "maximum_total_ms": max(case["total_ms"] for case in cases),
        "mismatches": 0,
        "nonfinite": 0,
        "tail_guard_pass": True,
        "provider_backend_mask": 15,
    },
    "gb10_layer3": {
        "capture_host": "gb10-4t",
        "prompt_tokens": 8192,
        "replay_exit_code": int(replay_rc),
        "elapsed_ms": replay["elapsed_ms"],
        "output_bf16_fnv1a64": replay["output_bf16_fnv1a64"],
        "output_sha256": hashlib.sha256(
            Path(replay_output_path).read_bytes()
        ).hexdigest(),
        "nonfinite_count": replay["nonfinite_count"],
        "max_abs_difference_to_reference_bf16": (
            replay["max_abs_difference_to_reference_bf16"]
        ),
        "abs_difference_over_0_125_count": (
            replay["abs_difference_over_0_125_count"]
        ),
    },
    "component_only": True,
    "inference_success_claimed": False,
    "windows_product_requalification_required": True,
}
Path(output_path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, sort_keys=True))
PY

{
    printf 'record_type=qrt_amd395_public_silu_aot_component_qualification\n'
    printf 'schema_version=1\nhost=%s\n' "$(hostname)"
    printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'candidate_commit=%s\n' "${candidate_commit}"
    printf 'run_label=%s\n' "${run_label}"
    printf 'base_aot_metadata_sha256=%s\n' "${expected_metadata_sha256}"
    printf 'provider_source_sha256=%s\n' "${expected_provider_source_sha256}"
    printf 'provider_sha256=%s\n' "${expected_provider_sha256}"
    printf 'smoke_source_sha256=%s\n' "${expected_smoke_source_sha256}"
    printf 'smoke_sha256=%s\n' "${expected_smoke_sha256}"
    printf 'replay_sha256=%s\n' "${expected_replay_sha256}"
    printf 'row_major_gate_sha256=%s\n' \
        "${kernel_hashes[q8192_triton_0626_row_major_sorted_conditional_exact_gate_rows256.hsaco]}"
    printf 'zero_finalize_sha256=%s\n' \
        "${kernel_hashes[q8192_triton_0626_zero_correction_gate_finalize.hsaco]}"
    printf 'gb10_capture_host=gb10-4t\ngb10_capture_layer=3\n'
    printf 'gb10_capture_prompt_tokens=8192\n'
    printf 'component_only=1\ninference_success_claimed=0\n'
    printf 'windows_product_requalification_required=1\n'
} > "${run_dir}/provenance.env"
sha256sum "${kernel_dir}"/*.hsaco > "${run_dir}/kernel-artifacts.sha256"
sha256sum "${run_dir}/smoke.stdout.log" "${run_dir}/replay.json" \
    "${run_dir}/output.bf16" "${run_dir}/qualification.json" \
    "${run_dir}/provenance.env" "${run_dir}/kernel-artifacts.sha256" \
    > "${run_dir}/SHA256SUMS"
