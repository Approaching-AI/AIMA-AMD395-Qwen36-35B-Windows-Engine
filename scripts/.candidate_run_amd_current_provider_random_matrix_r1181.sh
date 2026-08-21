#!/usr/bin/env bash

set -Eeuo pipefail
umask 022
ulimit -c 0

workspace_root=${QRT_AMD_WORKSPACE_ROOT:?set the bounded AMD workspace root}
stage_root=${QRT_AMD_FRESH_BASE_AOT_ROOT:-${workspace_root}/qrt-dynamic-aot-r1171-REFHyB}
run_dir=${stage_root}/interleaved-random-r1181
kernel_dir=${stage_root}/kernels
provider=${stage_root}/provider-r1175b/provider-native-shared-v2-sorted-parity.so
smoke=${stage_root}/smoke-r1178b/moe-v2-sorted-oracle-smoke
expected_provider_sha256=5170bbf386a40837cfbb142ece292fa88217849b5bb771fbe68406e688ca2e84
expected_smoke_sha256=a98ea57776a1e3b6041774d67c6840d73ccd79dab4e7dad67f4ff000fbe3e369
expected_metadata_sha256=37cf35ffd130f6df51bd95661efd924dc398ee818ae2b3086ddb0af1bbc57199

case "${stage_root}" in
    "${workspace_root}"/qrt-dynamic-aot-r1171-*) ;;
    *) echo "random-matrix stage escaped the r1171 family" >&2; exit 2 ;;
esac
[[ ! -e ${run_dir} ]] || {
    echo "refusing to overwrite ${run_dir}" >&2
    exit 2
}
for required in "${provider}" "${smoke}" "${kernel_dir}/metadata.json"; do
    [[ -f ${required} ]] || { echo "missing ${required}" >&2; exit 3; }
done
[[ $(sha256sum "${provider}" | awk '{print $1}') == \
    "${expected_provider_sha256}" ]]
[[ $(sha256sum "${smoke}" | awk '{print $1}') == \
    "${expected_smoke_sha256}" ]]
[[ $(sha256sum "${kernel_dir}/metadata.json" | awk '{print $1}') == \
    "${expected_metadata_sha256}" ]]

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
    printf 'record_type=qrt_amd395_current_provider_random_matrix\n'
    printf 'schema_version=1\nhost=%s\n' "$(hostname)"
    printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'repo_commit=6cdce444177f54e8985a2d6148da863369bece32\n'
    printf 'provider_sha256=%s\n' "${expected_provider_sha256}"
    printf 'smoke_sha256=%s\n' "${expected_smoke_sha256}"
    printf 'metadata_sha256=%s\n' "${expected_metadata_sha256}"
    printf 'base_seed=39536119\nrandom_per_interval=6\n'
    printf 'pair_count=78\nformal_measurement_count=234\n'
    printf 'warm_measurement_count=2\nprocess_count=1\n'
    printf 'launches_per_sample=8\ntiming_samples=3\n'
    printf 'ratio_minimum=0.97\nratio_maximum=1.03\n'
    printf 'component_only=1\ngb10_random_length_boundary_attached=0\n'
    printf 'inference_success_claimed=0\n'
} > "${run_dir}/provenance.env"

python3 - "${run_dir}" <<'PY'
import json
from pathlib import Path
import random
import sys

run = Path(sys.argv[1])
boundaries = (
    2048, 2560, 3072, 3328, 3584, 4096, 4608,
    5120, 5632, 6144, 6656, 7168, 7680, 8192,
)
base_seed = 39_536_119
generator = random.Random(base_seed)
pairs = []
for interval_index, (lower, upper) in enumerate(
    zip(boundaries, boundaries[1:])
):
    candidates = list(range(lower + 1, upper))
    aligned_candidates = [
        value for value in candidates
        if value % 64 == 0 and lower + 2 <= value <= upper - 2
    ]
    aligned = generator.choice(aligned_candidates)
    selected = {lower + 1, aligned - 1, aligned, aligned + 1, upper - 1}
    ordinary = [
        value for value in candidates
        if value not in selected and value % 64 not in (0, 1, 63)
    ]
    selected.add(generator.choice(ordinary))
    assert len(selected) == 6
    for pair_in_interval, prompt_tokens in enumerate(sorted(selected)):
        pairs.append({
            "pair_id": interval_index * 6 + pair_in_interval,
            "interval_index": interval_index,
            "lower": lower,
            "upper": upper,
            "random_tokens": prompt_tokens,
        })
random.Random(base_seed ^ 0xA11CE).shuffle(pairs)

sequence = [8192, 8192]
for pair in pairs:
    pair["anchor_before_index"] = len(sequence)
    sequence.append(pair["upper"])
    pair["random_index"] = len(sequence)
    sequence.append(pair["random_tokens"])
    pair["anchor_after_index"] = len(sequence)
    sequence.append(pair["upper"])
assert len(pairs) == 78
assert len(sequence) == 236
plan = {
    "boundaries": boundaries,
    "base_seed": base_seed,
    "random_per_interval": 6,
    "warm_measurements": 2,
    "formal_measurements": 234,
    "measurement_sequence": sequence,
    "pairs": pairs,
}
(run / "point-plan.json").write_text(
    json.dumps(plan, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
(run / "token-sequence.txt").write_text(
    ",".join(map(str, sequence)) + "\n",
    encoding="utf-8",
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

/opt/rocm/bin/rocm-smi --showclocks --showtemp --showpower --showuse \
    > "${run_dir}/device-before.log" 2>&1 || true
set +e
timeout --signal=TERM --kill-after=30s 900s \
    "${smoke}" "${kernel_dir}" 8 "${provider}" 8192 \
    > "${run_dir}/measurement.stdout.log" \
    2> "${run_dir}/measurement.stderr.log"
measurement_exit=$?
set -e
/opt/rocm/bin/rocm-smi --showclocks --showtemp --showpower --showuse \
    > "${run_dir}/device-after.log" 2>&1 || true

python3 - "${run_dir}" "${measurement_exit}" <<'PY'
import json
import math
from pathlib import Path
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
        and fields.get("launches_per_sample") == "8"
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
hashes_by_tokens = {}
for record in records:
    hashes_by_tokens.setdefault(record["tokens"], set()).add(
        record["output_f32_fnv1a64"]
    )
hash_stability_pass = bool(records) and all(
    len(values) == 1 for values in hashes_by_tokens.values()
)

comparisons = []
if sequence_pass:
    for pair in plan["pairs"]:
        before = records[pair["anchor_before_index"]]
        random_record = records[pair["random_index"]]
        after = records[pair["anchor_after_index"]]
        before_speed = before["tokens"] * 1000.0 / before["total_ms"]
        after_speed = after["tokens"] * 1000.0 / after["total_ms"]
        reference_speed = math.sqrt(before_speed * after_speed)
        random_speed = (
            random_record["tokens"] * 1000.0 / random_record["total_ms"]
        )
        ratio = random_speed / reference_speed
        anchor_drift = max(before_speed, after_speed) / min(
            before_speed, after_speed
        )
        comparisons.append({
            **pair,
            "anchor_before_ms": before["total_ms"],
            "random_ms": random_record["total_ms"],
            "anchor_after_ms": after["total_ms"],
            "random_to_bracketed_anchor_throughput_ratio": ratio,
            "anchor_max_to_min_throughput_ratio": anchor_drift,
            "ratio_pass": 0.97 <= ratio <= 1.03,
            "anchor_stability_pass": anchor_drift <= 1.03,
        })
ratios = [
    record["random_to_bracketed_anchor_throughput_ratio"]
    for record in comparisons
]
anchor_drifts = [
    record["anchor_max_to_min_throughput_ratio"]
    for record in comparisons
]
failed_pairs = [
    record for record in comparisons
    if not record["ratio_pass"] or not record["anchor_stability_pass"]
]
aligned = [
    record for record in comparisons if record["random_tokens"] % 64 == 0
]
integer_spike_detected = any(not record["ratio_pass"] for record in aligned)
strict_pass = bool(comparisons) and not failed_pairs
summary = {
    "record_type": "qrt_amd395_current_provider_random_matrix",
    "schema_version": 1,
    "measurement_exit_code": measurement_exit,
    "warm_measurement_count": 2,
    "formal_measurement_count": max(0, len(records) - 2),
    "expected_formal_measurement_count": 234,
    "pair_count": len(comparisons),
    "expected_pair_count": 78,
    "unique_token_count": len(hashes_by_tokens),
    "semantic_failures": semantic_failures,
    "sequence_pass": sequence_pass,
    "output_hash_stability_by_tokens_pass": hash_stability_pass,
    "ratio_min": min(ratios) if ratios else None,
    "ratio_max": max(ratios) if ratios else None,
    "anchor_drift_max": max(anchor_drifts) if anchor_drifts else None,
    "failed_pair_count": len(failed_pairs),
    "failed_pairs": failed_pairs,
    "aligned_probe_count": len(aligned),
    "integer_length_performance_spike_detected": integer_spike_detected,
    "strict_0_97_to_1_03_pass": strict_pass,
    "comparisons": comparisons,
    "component_only": True,
    "gb10_random_length_boundary_attached": False,
    "inference_success_claimed": False,
}
(run / "summary.json").write_text(
    json.dumps(summary, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    key: value for key, value in summary.items()
    if key not in ("comparisons", "failed_pairs")
}, separators=(",", ":")))
if (
    measurement_exit != 0 or semantic_failures != 0 or not sequence_pass
    or not hash_stability_pass or len(comparisons) != 78 or not strict_pass
):
    raise SystemExit(1)
PY

script_exit=0
echo "current_provider_random_matrix_finished exit_code=0"
exit 0
