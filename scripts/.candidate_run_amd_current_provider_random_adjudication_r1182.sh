#!/usr/bin/env bash

set -Eeuo pipefail
umask 022
ulimit -c 0

workspace_root=${QRT_AMD_WORKSPACE_ROOT:?set the bounded AMD workspace root}
stage_root=${QRT_AMD_FRESH_BASE_AOT_ROOT:-${workspace_root}/qrt-dynamic-aot-r1171-REFHyB}
source_run=${stage_root}/interleaved-random-r1181
run_dir=${stage_root}/interleaved-random-adjudication-r1182
kernel_dir=${stage_root}/kernels
provider=${stage_root}/provider-r1175b/provider-native-shared-v2-sorted-parity.so
smoke=${stage_root}/smoke-r1178b/moe-v2-sorted-oracle-smoke
expected_provider_sha256=5170bbf386a40837cfbb142ece292fa88217849b5bb771fbe68406e688ca2e84
expected_smoke_sha256=a98ea57776a1e3b6041774d67c6840d73ccd79dab4e7dad67f4ff000fbe3e369
expected_source_summary_sha256=5a2ef09657be66da816894389d7c64f265abb88260ecd6cf3d74b84de4382283

case "${stage_root}" in
    "${workspace_root}"/qrt-dynamic-aot-r1171-*) ;;
    *) echo "adjudication stage escaped the r1171 family" >&2; exit 2 ;;
esac
[[ ! -e ${run_dir} ]] || {
    echo "refusing to overwrite ${run_dir}" >&2
    exit 2
}
[[ $(sha256sum "${provider}" | awk '{print $1}') == \
    "${expected_provider_sha256}" ]]
[[ $(sha256sum "${smoke}" | awk '{print $1}') == \
    "${expected_smoke_sha256}" ]]
[[ $(sha256sum "${source_run}/summary.json" | awk '{print $1}') == \
    "${expected_source_summary_sha256}" ]]

mkdir -p -- "${run_dir}"
exec > >(tee -a "${run_dir}/driver.stdout.log") \
    2> >(tee -a "${run_dir}/driver.stderr.log" >&2)
script_exit=99
finalize() {
    local shell_status=$?
    local final_status=${script_exit}
    if [[ ${final_status} -eq 99 ]]; then
        final_status=${shell_status}
        [[ ${final_status} -ne 0 ]] || final_status=98
    fi
    printf 'finished_utc=%s\nexit_code=%s\n' \
        "$(date -u +%FT%TZ)" "${final_status}" > "${run_dir}/result.env"
    find "${run_dir}" -maxdepth 1 -type f \
        ! -name driver.stdout.log ! -name driver.stderr.log \
        ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
        > "${run_dir}/SHA256SUMS"
}
trap finalize EXIT
trap 'script_exit=130; exit 130' HUP INT TERM

{
    printf 'record_type=qrt_amd395_current_provider_random_adjudication\n'
    printf 'schema_version=1\nhost=%s\n' "$(hostname)"
    printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'repo_commit=6cdce444177f54e8985a2d6148da863369bece32\n'
    printf 'provider_sha256=%s\nsmoke_sha256=%s\n' \
        "${expected_provider_sha256}" "${expected_smoke_sha256}"
    printf 'source_summary_sha256=%s\n' "${expected_source_summary_sha256}"
    printf 'candidate_count=11\nrepeat_count_per_candidate=15\n'
    printf 'comparison_shuffle_seed=39536120\nprocess_count=1\n'
    printf 'launches_per_sample=4\ntiming_samples=3\n'
    printf 'ratio_minimum=0.97\nratio_maximum=1.03\n'
    printf 'component_only=1\ngb10_random_length_boundary_attached=0\n'
    printf 'inference_success_claimed=0\n'
} > "${run_dir}/provenance.env"

python3 - "${source_run}/summary.json" "${run_dir}" <<'PY'
import json
from pathlib import Path
import random
import sys

source_path, run_path = map(Path, sys.argv[1:])
source = json.loads(source_path.read_text(encoding="utf-8"))
failed = [
    {
        "pair_id": int(record["pair_id"]),
        "candidate": int(record["random_tokens"]),
        "reference": int(record["upper"]),
        "source_ratio": float(
            record["random_to_bracketed_anchor_throughput_ratio"]
        ),
        "source_anchor_drift": float(
            record["anchor_max_to_min_throughput_ratio"]
        ),
    }
    for record in source["failed_pairs"]
]
assert len(failed) == 11
comparisons = [
    {**record, "repeat": repeat}
    for record in failed
    for repeat in range(15)
]
random.Random(39_536_120).shuffle(comparisons)
sequence = [8192, 8192]
for index, comparison in enumerate(comparisons):
    comparison["comparison_index"] = index
    comparison["reference_before_index"] = len(sequence)
    sequence.append(comparison["reference"])
    comparison["candidate_index"] = len(sequence)
    sequence.append(comparison["candidate"])
    comparison["reference_after_index"] = len(sequence)
    sequence.append(comparison["reference"])
assert len(comparisons) == 165
assert len(sequence) == 497
plan = {
    "source_summary_sha256": (
        "5a2ef09657be66da816894389d7c64f265abb88260ecd6cf3d74b84de4382283"
    ),
    "candidate_count": 11,
    "repeat_count_per_candidate": 15,
    "comparison_shuffle_seed": 39_536_120,
    "measurement_sequence": sequence,
    "comparisons": comparisons,
}
(run_path / "point-plan.json").write_text(
    json.dumps(plan, separators=(",", ":")) + "\n", encoding="utf-8"
)
(run_path / "token-sequence.txt").write_text(
    ",".join(map(str, sequence)) + "\n", encoding="utf-8"
)
PY

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
[[ ${idle_streak} -eq 3 ]] || {
    echo "GPU did not become idle" >&2
    script_exit=75
    exit 75
}

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
timeout --signal=TERM --kill-after=30s 1200s \
    "${smoke}" "${kernel_dir}" 4 "${provider}" 8192 \
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
            "output_f32_fnv1a64": fields["output_f32_fnv1a64"],
        }
    except (KeyError, ValueError):
        semantic_failures += 1
        continue
    semantic_failures += int(not (
        fields.get("status") == "pass"
        and fields.get("sequence_count") == str(
            len(plan["measurement_sequence"])
        )
        and fields.get("timing_samples") == "3"
        and fields.get("launches_per_sample") == "4"
        and fields.get("nonfinite") == "0"
        and fields.get("tail_guard_pass") == "1"
        and fields.get("provider_backend_mask") == "15"
        and fields.get("component_only") == "1"
        and fields.get("inference_success_claimed") == "0"
    ))
    records.append(record)
records.sort(key=lambda record: record["sequence_index"])
sequence_pass = (
    len(records) == len(plan["measurement_sequence"])
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
    hashes.setdefault(record["tokens"], set()).add(
        record["output_f32_fnv1a64"]
    )
hash_stability_pass = bool(records) and all(
    len(values) == 1 for values in hashes.values()
)

comparisons = []
if sequence_pass:
    for planned in plan["comparisons"]:
        before = records[planned["reference_before_index"]]
        candidate = records[planned["candidate_index"]]
        after = records[planned["reference_after_index"]]
        before_speed = before["tokens"] * 1000.0 / before["total_ms"]
        after_speed = after["tokens"] * 1000.0 / after["total_ms"]
        reference_speed = math.sqrt(before_speed * after_speed)
        candidate_speed = candidate["tokens"] * 1000.0 / candidate["total_ms"]
        comparisons.append({
            **planned,
            "candidate_ms": candidate["total_ms"],
            "reference_before_ms": before["total_ms"],
            "reference_after_ms": after["total_ms"],
            "ratio": candidate_speed / reference_speed,
            "anchor_drift": max(before_speed, after_speed) /
                min(before_speed, after_speed),
        })
by_candidate = {}
for comparison in comparisons:
    by_candidate.setdefault(comparison["candidate"], []).append(comparison)
adjudication = {}
for candidate, candidate_records in sorted(by_candidate.items()):
    ratios = sorted(record["ratio"] for record in candidate_records)
    drifts = sorted(record["anchor_drift"] for record in candidate_records)
    median_ratio = statistics.median(ratios)
    median_drift = statistics.median(drifts)
    adjudication[str(candidate)] = {
        "reference": candidate_records[0]["reference"],
        "sample_count": len(candidate_records),
        "source_ratio": candidate_records[0]["source_ratio"],
        "ratio_min": ratios[0],
        "ratio_p10": ratios[1],
        "ratio_median": median_ratio,
        "ratio_p90": ratios[-2],
        "ratio_max": ratios[-1],
        "anchor_drift_median": median_drift,
        "within_0_97_to_1_03_count": sum(
            0.97 <= value <= 1.03 for value in ratios
        ),
        "median_ratio_pass": 0.97 <= median_ratio <= 1.03,
        "median_anchor_stability_pass": median_drift <= 1.03,
    }
reproducible_failures = {
    candidate: record for candidate, record in adjudication.items()
    if not record["median_ratio_pass"]
    or not record["median_anchor_stability_pass"]
}
integer_spike_detected = any(
    int(candidate) % 64 == 0 and not record["median_ratio_pass"]
    for candidate, record in adjudication.items()
)
strict_pass = len(adjudication) == 11 and not reproducible_failures
summary = {
    "record_type": "qrt_amd395_current_provider_random_adjudication",
    "schema_version": 1,
    "measurement_exit_code": measurement_exit,
    "measurement_count": len(records),
    "expected_measurement_count": 497,
    "comparison_count": len(comparisons),
    "expected_comparison_count": 165,
    "candidate_count": len(adjudication),
    "repeat_count_per_candidate": 15,
    "semantic_failures": semantic_failures,
    "sequence_pass": sequence_pass,
    "output_hash_stability_by_tokens_pass": hash_stability_pass,
    "candidate_adjudication": adjudication,
    "reproducible_failure_count": len(reproducible_failures),
    "reproducible_failures": reproducible_failures,
    "integer_length_performance_spike_detected": integer_spike_detected,
    "all_candidate_medians_strict_0_97_to_1_03_pass": strict_pass,
    "component_only": True,
    "gb10_random_length_boundary_attached": False,
    "inference_success_claimed": False,
}
(run / "summary.json").write_text(
    json.dumps(summary, separators=(",", ":")) + "\n", encoding="utf-8"
)
print(json.dumps({
    key: value for key, value in summary.items()
    if key not in ("candidate_adjudication", "reproducible_failures")
}, separators=(",", ":")))
if (
    measurement_exit != 0 or semantic_failures != 0 or not sequence_pass
    or not hash_stability_pass or len(comparisons) != 165 or not strict_pass
):
    raise SystemExit(1)
PY

script_exit=0
echo "current_provider_random_adjudication_finished exit_code=0"
exit 0
