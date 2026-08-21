#!/usr/bin/env bash

set -Eeuo pipefail
umask 022
ulimit -c 0

workspace_root=${QRT_AMD_WORKSPACE_ROOT:?set the bounded AMD workspace root}
base=${workspace_root}/AIMA-dynamic-logical-smoke-r1093-t3J3OB
fresh_root=${QRT_AMD_FRESH_BASE_AOT_ROOT:-${workspace_root}/qrt-dynamic-aot-r1171-REFHyB}
kernel_dir=${QRT_AMD_BASE_AOT_KERNEL_DIR:-${fresh_root}/kernels}
run_label=${QRT_AMD_DYNAMIC_SMOKE_LABEL:-r1172-matched}
run_dir=${fresh_root}/dynamic-smoke-${run_label}
expected_metadata_sha256=${QRT_AMD_BASE_AOT_METADATA_SHA256:-37cf35ffd130f6df51bd95661efd924dc398ee818ae2b3086ddb0af1bbc57199}
expected_group_m=${QRT_AMD_BASE_AOT_GROUP_M:-1}
provider=${QRT_AMD_NATIVE_SHARED_PROVIDER:-${base}/real-q8192-layer3-native-shared-r1160/build/provider-group-m1-native-shared.so}
expected_provider_sha256=${QRT_AMD_NATIVE_SHARED_PROVIDER_SHA256:-74e38c27ada6963158cffe4ad936c8acca55e3d4946c7937e8d784878c8563b7}
smoke=${QRT_AMD_DYNAMIC_SMOKE_EXE:-${base}/native-shared-interleaved-random-r1165/build/moe-product-radius-light-interleaved-smoke}
expected_smoke_sha256=${QRT_AMD_DYNAMIC_SMOKE_SHA256:-dffdd64086eb43e74b79f995fa74dc798565dcb148900673bd183e9ebc97eb9b}
expected_smoke_source_sha256=${QRT_AMD_DYNAMIC_SMOKE_SOURCE_SHA256:-f1b0c9abe11d41c041a9c8d03eb13bc36a81b15fe188a72a3e76f2e52bc2f862}

case "${fresh_root}" in
    "${workspace_root}"/qrt-dynamic-aot-r1171-*) ;;
    *) echo "fresh AOT root escaped the r1171 staging family" >&2; exit 2 ;;
esac
case "${kernel_dir}" in
    "${fresh_root}"/*) ;;
    *) echo "base AOT kernel directory escaped the staging root" >&2; exit 2 ;;
esac
[[ ${run_label} =~ ^r[0-9]+(-[a-z0-9-]+)?$ ]] || {
    echo "invalid dynamic smoke run label" >&2
    exit 2
}
[[ ${expected_group_m} == 1 || ${expected_group_m} == 8 ]] || {
    echo "unsupported base AOT group-M" >&2
    exit 2
}
[[ ! -e ${run_dir} ]] || {
    echo "refusing to overwrite ${run_dir}" >&2
    exit 2
}
[[ $(sha256sum "${provider}" | awk '{print $1}') == \
    "${expected_provider_sha256}" ]]
[[ $(sha256sum "${smoke}" | awk '{print $1}') == \
    "${expected_smoke_sha256}" ]]
[[ $(sha256sum "${kernel_dir}/metadata.json" | awk '{print $1}') == \
    "${expected_metadata_sha256}" ]]
mkdir -p -- "${run_dir}"

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
unset QRT_PRODUCT_RADIUS_LIGHT_TOKEN_SEQUENCE
unset QRT_PREFILL_DESCRIPTOR_BATCH_PROFILE_MOE_SUBPHASES

timeout --signal=TERM --kill-after=20s 300s \
    "${smoke}" "${kernel_dir}" 1 "${provider}" \
    > "${run_dir}/smoke.stdout.log" \
    2> "${run_dir}/smoke.stderr.log"

python3 - "${run_dir}/smoke.stdout.log" \
    "${run_dir}/summary.json" <<'PY'
import json
import re
import sys

stdout_path, summary_path = sys.argv[1:]
text = open(stdout_path, encoding="utf-8").read()
expected = [
    2073, 2156, 2560, 3073, 4609, 6145, 2049, 2175,
    2176, 2177, 2559, 2561, 3071, 3072, 3583, 3584,
    3585, 4095, 4096, 4097, 4607, 4608, 6143, 6144,
    7167, 7168, 7169, 7679, 7680, 7681, 8191, 8192,
]
cases = []
for line in text.splitlines():
    if not line.startswith("dynamic_logical_case "):
        continue
    fields = dict(item.split("=", 1) for item in line.split() if "=" in item)
    cases.append({
        "index": int(fields["index"]),
        "tokens": int(fields["tokens"]),
        "total_ms": float(fields["total_ms"]),
        "timing_samples": int(fields["timing_samples"]),
        "timing_stat": fields["timing_stat"],
        "reset_included": int(fields["reset_included"]),
        "component_only": int(fields["component_only"]),
        "inference_success_claimed": int(
            fields["inference_success_claimed"]
        ),
    })
assert [record["index"] for record in cases] == list(range(32))
assert [record["tokens"] for record in cases] == expected
assert all(record["total_ms"] > 0 for record in cases)
assert all(record["timing_samples"] == 3 for record in cases)
assert all(record["timing_stat"] == "median" for record in cases)
assert all(record["reset_included"] == 0 for record in cases)
assert all(record["component_only"] == 1 for record in cases)
assert all(record["inference_success_claimed"] == 0 for record in cases)
required = (
    r"dynamic_logical_q8192_mismatches=0",
    r"dynamic_logical_q8192_max_abs_diff=0",
    r"dynamic_logical_case_count=32",
    r"dynamic_logical_nonfinite=0",
    r"dynamic_logical_tail_guard_pass=1",
    r"provider_backend_mask=15",
)
assert all(re.search(pattern, text) for pattern in required)
summary = {
    "record_type": "qrt_amd395_fresh_base_dynamic_logical_smoke",
    "schema_version": 1,
    "status": "pass",
    "case_count": len(cases),
    "tokens": expected,
    "minimum_total_ms": min(record["total_ms"] for record in cases),
    "maximum_total_ms": max(record["total_ms"] for record in cases),
    "q8192_exact_pass": True,
    "nonfinite": 0,
    "tail_guard_pass": True,
    "provider_backend_mask": 15,
    "component_only": True,
    "inference_success_claimed": False,
}
open(summary_path, "w", encoding="utf-8").write(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, sort_keys=True))
PY

{
    printf 'record_type=qrt_amd395_fresh_base_dynamic_logical_smoke\n'
    printf 'schema_version=1\nhost=%s\n' "$(hostname)"
    printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'repo_commit=6cdce444177f54e8985a2d6148da863369bece32\n'
    printf 'smoke_source_sha256=%s\n' "${expected_smoke_source_sha256}"
    printf 'smoke_sha256=%s\n' "${expected_smoke_sha256}"
    printf 'provider_sha256=%s\n' "${expected_provider_sha256}"
    printf 'base_aot_block_m=64\nbase_aot_group_m=%s\n' \
        "${expected_group_m}"
    printf 'dynamic_logical_case_count=32\nq8192_exact_pass=1\n'
    printf 'component_only=1\ninference_success_claimed=0\n'
} > "${run_dir}/provenance.env"
sha256sum "${run_dir}"/smoke.stdout.log "${run_dir}"/summary.json \
    "${run_dir}"/provenance.env > "${run_dir}/SHA256SUMS"
