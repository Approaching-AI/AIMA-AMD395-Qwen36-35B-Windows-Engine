#!/usr/bin/env bash

set -Eeuo pipefail
umask 022
ulimit -c 0

workspace_root=${QRT_AMD_WORKSPACE_ROOT:?set the bounded AMD workspace root}
stage_root=${QRT_AMD_FRESH_BASE_AOT_ROOT:-${workspace_root}/qrt-dynamic-aot-r1171-REFHyB}
run_dir=${stage_root}/q2523-clock-normalized-r1183
kernel_dir=${stage_root}/kernels
provider=${stage_root}/provider-r1175b/provider-native-shared-v2-sorted-parity.so
smoke=${stage_root}/smoke-r1178b/moe-v2-sorted-oracle-smoke
expected_provider_sha256=5170bbf386a40837cfbb142ece292fa88217849b5bb771fbe68406e688ca2e84
expected_smoke_sha256=a98ea57776a1e3b6041774d67c6840d73ccd79dab4e7dad67f4ff000fbe3e369

case "${stage_root}" in
    "${workspace_root}"/qrt-dynamic-aot-r1171-*) ;;
    *) echo "q2523 stage escaped the r1171 family" >&2; exit 2 ;;
esac
[[ ! -e ${run_dir} ]] || {
    echo "refusing to overwrite ${run_dir}" >&2
    exit 2
}
[[ $(sha256sum "${provider}" | awk '{print $1}') == \
    "${expected_provider_sha256}" ]]
[[ $(sha256sum "${smoke}" | awk '{print $1}') == \
    "${expected_smoke_sha256}" ]]
mkdir -p -- "${run_dir}"

python3 - "${run_dir}" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
sequence = [8192, 8192]
comparisons = []
for repeat in range(15):
    comparison = {
        "repeat": repeat,
        "reference_before_index": len(sequence),
    }
    sequence.append(2560)
    comparison["candidate_index"] = len(sequence)
    sequence.append(2523)
    comparison["reference_after_index"] = len(sequence)
    sequence.append(2560)
    comparisons.append(comparison)
plan = {
    "candidate": 2523,
    "reference": 2560,
    "repeat_count": 15,
    "launches_per_sample": 16,
    "measurement_sequence": sequence,
    "comparisons": comparisons,
}
(run / "point-plan.json").write_text(
    json.dumps(plan, separators=(",", ":")) + "\n", encoding="utf-8"
)
(run / "token-sequence.txt").write_text(
    ",".join(map(str, sequence)) + "\n", encoding="utf-8"
)
PY

{
    printf 'record_type=qrt_amd395_q2523_clock_normalized\n'
    printf 'schema_version=1\nhost=%s\n' "$(hostname)"
    printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'repo_commit=6cdce444177f54e8985a2d6148da863369bece32\n'
    printf 'provider_sha256=%s\nsmoke_sha256=%s\n' \
        "${expected_provider_sha256}" "${expected_smoke_sha256}"
    printf 'candidate=2523\nreference=2560\nrepeat_count=15\n'
    printf 'launches_per_sample=16\ntiming_samples=3\n'
    printf 'ratio_minimum=0.97\nratio_maximum=1.03\n'
    printf 'component_only=1\ngb10_random_length_boundary_attached=0\n'
    printf 'inference_success_claimed=0\n'
} > "${run_dir}/provenance.env"

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
export QRT_PRODUCT_RADIUS_LIGHT_TOKEN_SEQUENCE
QRT_PRODUCT_RADIUS_LIGHT_TOKEN_SEQUENCE=$(tr -d '\n' \
    < "${run_dir}/token-sequence.txt")
unset QRT_PREFILL_DESCRIPTOR_BATCH_PROFILE_MOE_SUBPHASES

set +e
timeout --signal=TERM --kill-after=30s 600s \
    "${smoke}" "${kernel_dir}" 16 "${provider}" 8192 \
    > "${run_dir}/measurement.stdout.log" \
    2> "${run_dir}/measurement.stderr.log"
measurement_exit=$?
set -e

python3 - "${run_dir}" "${measurement_exit}" <<'PY'
import json
import math
from pathlib import Path
import statistics
import sys

run = Path(sys.argv[1])
measurement_exit = int(sys.argv[2])
plan = json.loads((run / "point-plan.json").read_text(encoding="utf-8"))
records = []
semantic_failures = 0
for line in (run / "measurement.stdout.log").read_text(
    encoding="utf-8"
).splitlines():
    if not line.startswith("moe_product_radius_light "):
        continue
    fields = dict(item.split("=", 1) for item in line.split() if "=" in item)
    try:
        record = {
            "sequence_index": int(fields["sequence_index"]),
            "tokens": int(fields["tokens"]),
            "total_ms": float(fields["total_ms"]),
            "hash": fields["output_f32_fnv1a64"],
        }
    except (KeyError, ValueError):
        semantic_failures += 1
        continue
    semantic_failures += int(not (
        fields.get("status") == "pass"
        and fields.get("sequence_count") == "47"
        and fields.get("timing_samples") == "3"
        and fields.get("launches_per_sample") == "16"
        and fields.get("nonfinite") == "0"
        and fields.get("tail_guard_pass") == "1"
        and fields.get("provider_backend_mask") == "15"
        and fields.get("component_only") == "1"
        and fields.get("inference_success_claimed") == "0"
    ))
    records.append(record)
records.sort(key=lambda record: record["sequence_index"])
sequence_pass = (
    len(records) == 47
    and all(
        record["sequence_index"] == index
        and record["tokens"] == plan["measurement_sequence"][index]
        and math.isfinite(record["total_ms"])
        and record["total_ms"] > 0.0
        for index, record in enumerate(records)
    )
)
hashes = {}
for record in records:
    hashes.setdefault(record["tokens"], set()).add(record["hash"])
hash_stability_pass = bool(records) and all(
    len(values) == 1 for values in hashes.values()
)
ratios = []
drifts = []
comparisons = []
if sequence_pass:
    for planned in plan["comparisons"]:
        before = records[planned["reference_before_index"]]
        candidate = records[planned["candidate_index"]]
        after = records[planned["reference_after_index"]]
        before_speed = before["tokens"] * 1000.0 / before["total_ms"]
        after_speed = after["tokens"] * 1000.0 / after["total_ms"]
        candidate_speed = candidate["tokens"] * 1000.0 / candidate["total_ms"]
        ratio = candidate_speed / math.sqrt(before_speed * after_speed)
        drift = max(before_speed, after_speed) / min(
            before_speed, after_speed
        )
        ratios.append(ratio)
        drifts.append(drift)
        comparisons.append({
            **planned,
            "before_ms": before["total_ms"],
            "candidate_ms": candidate["total_ms"],
            "after_ms": after["total_ms"],
            "ratio": ratio,
            "anchor_drift": drift,
        })
median_ratio = statistics.median(ratios) if ratios else None
median_drift = statistics.median(drifts) if drifts else None
strict_pass = bool(ratios) and (
    0.97 <= median_ratio <= 1.03 and median_drift <= 1.03
)
summary = {
    "record_type": "qrt_amd395_q2523_clock_normalized",
    "schema_version": 1,
    "measurement_exit_code": measurement_exit,
    "measurement_count": len(records),
    "semantic_failures": semantic_failures,
    "sequence_pass": sequence_pass,
    "output_hash_stability_by_tokens_pass": hash_stability_pass,
    "repeat_count": len(ratios),
    "launches_per_sample": 16,
    "ratio_min": min(ratios) if ratios else None,
    "ratio_median": median_ratio,
    "ratio_max": max(ratios) if ratios else None,
    "anchor_drift_min": min(drifts) if drifts else None,
    "anchor_drift_median": median_drift,
    "anchor_drift_max": max(drifts) if drifts else None,
    "strict_median_pass": strict_pass,
    "comparisons": comparisons,
    "component_only": True,
    "gb10_random_length_boundary_attached": False,
    "inference_success_claimed": False,
}
(run / "summary.json").write_text(
    json.dumps(summary, separators=(",", ":")) + "\n", encoding="utf-8"
)
print(json.dumps({
    key: value for key, value in summary.items() if key != "comparisons"
}, separators=(",", ":")))
if (
    measurement_exit != 0 or semantic_failures != 0 or not sequence_pass
    or not hash_stability_pass or len(ratios) != 15 or not strict_pass
):
    raise SystemExit(1)
PY

find "${run_dir}" -maxdepth 1 -type f ! -name SHA256SUMS -print0 |
    sort -z | xargs -0 sha256sum > "${run_dir}/SHA256SUMS"
echo "q2523_clock_normalized_finished exit_code=0"
