#!/usr/bin/env bash

set -Eeuo pipefail
umask 022
ulimit -c 0

stage_root=${QRT_AMD_PUBLIC_AOT_ROOT:?set the bounded /var/tmp AOT root}
triton_python=${QRT_TRITON_PYTHON:?set the Triton 3.6 Python executable}
llvm_strip=${QRT_ROCM_LLVM_STRIP:-/opt/rocm/llvm/bin/llvm-strip}
source_dir=${stage_root}/sources
output_dir=${stage_root}/kernels
metadata_dir=${stage_root}/metadata
log_dir=${stage_root}/logs
expected_base_source_sha256=d5497e32c0bf5514a2cc136be95d2a02b66538906b21e42224b4362826add6a4
expected_row_source_sha256=6d5ae64a61b05f86689f5237566d531b135b763796016192b2c908aa3cd16a37
expected_sorted_source_sha256=4882f55ba0b4a8ebb8e446891446cdd7a4552e05ed0cbde41c477714a2179ca2
expected_zero_source_sha256=f1c100a3e3c916000d5e993414e60f245c513fd8066039bd7a5ff71c82511b0f
expected_down_source_sha256=5edafd02560c5937347233fe2f40ee3eb111a530bd10effeb28c18f6e0a431af

case "${stage_root}" in
    /var/tmp/qrt-public-aot-r1184-*) ;;
    *) echo "public AOT root must use the bounded r1184 /var/tmp family" >&2; exit 2 ;;
esac
[[ -x ${triton_python} ]] || {
    echo "Triton Python is not executable" >&2
    exit 2
}
[[ -x ${llvm_strip} ]] || {
    echo "ROCm llvm-strip is not executable" >&2
    exit 2
}
[[ ! -e ${output_dir} && ! -e ${metadata_dir} && ! -e ${log_dir} ]] || {
    echo "refusing to overwrite generated public AOT outputs" >&2
    exit 2
}
verify_sha256() {
    local path=$1
    local expected=$2
    local actual
    actual=$(sha256sum "${path}" | awk '{print $1}')
    [[ ${actual} == "${expected}" ]] || {
        echo "source hash mismatch path=${path} actual=${actual}" >&2
        exit 3
    }
}
base_source=${source_dir}/compile_q8192_triton_selected_moe.py
row_source=${source_dir}/compile_q8192_row_major_sorted_conditional_gate.py
sorted_source=${source_dir}/compile_q8192_sorted_conditional_gate.py
zero_source=${source_dir}/compile_q8192_zero_correction_gate_finalize.py
down_source=${source_dir}/compile_q8192_conditional_exact_down.py
verify_sha256 "${base_source}" "${expected_base_source_sha256}"
verify_sha256 "${row_source}" "${expected_row_source_sha256}"
verify_sha256 "${sorted_source}" "${expected_sorted_source_sha256}"
verify_sha256 "${zero_source}" "${expected_zero_source_sha256}"
verify_sha256 "${down_source}" "${expected_down_source_sha256}"
[[ $("${triton_python}" -c 'import triton; print(triton.__version__)') == 3.6.0 ]]

mkdir -p -- "${output_dir}" "${metadata_dir}" "${log_dir}" \
    "${stage_root}/cache" "${stage_root}/tmp"
export TRITON_CACHE_DIR=${stage_root}/cache
export TMPDIR=${stage_root}/tmp

timeout --signal=TERM --kill-after=30s 600s \
    "${triton_python}" "${base_source}" \
    --output-dir "${output_dir}" --metadata "${output_dir}/metadata.json" \
    --tokens 8192 --block-m 64 --block-n 64 --block-k 64 --group-m 1 \
    --num-warps 4 --num-stages 1 --waves-per-eu 0 \
    --gate-num-warps 4 --gate-num-stages 1 --gate-waves-per-eu 0 \
    --down-num-warps 4 --down-num-stages 1 --down-waves-per-eu 0 \
    > "${log_dir}/base-selected-moe.stdout.log" \
    2> "${log_dir}/base-selected-moe.stderr.log"
timeout --signal=TERM --kill-after=30s 300s \
    "${triton_python}" "${row_source}" \
    --output-dir "${output_dir}" \
    --metadata "${metadata_dir}/row-major-gate.json" --rows 256 \
    > "${log_dir}/row-major-gate.stdout.log" \
    2> "${log_dir}/row-major-gate.stderr.log"
timeout --signal=TERM --kill-after=30s 300s \
    "${triton_python}" "${zero_source}" \
    --output-dir "${output_dir}" \
    --metadata "${metadata_dir}/zero-finalize.json" \
    > "${log_dir}/zero-finalize.stdout.log" \
    2> "${log_dir}/zero-finalize.stderr.log"
timeout --signal=TERM --kill-after=30s 300s \
    "${triton_python}" "${down_source}" \
    --output-dir "${output_dir}" \
    --metadata "${metadata_dir}/conditional-down.json" --rows 4 \
    > "${log_dir}/conditional-down.stdout.log" \
    2> "${log_dir}/conditional-down.stderr.log"

expected_files=(
    q8192_selected_moe_route_count.hsaco
    q8192_selected_moe_route_prefix_by_program.hsaco
    q8192_selected_moe_route_padded_prefix.hsaco
    q8192_selected_moe_route_scatter.hsaco
    q8192_selected_moe_gate_up_silu.hsaco
    q8192_selected_moe_down.hsaco
    q8192_triton_0626_row_major_sorted_conditional_exact_gate_rows256.hsaco
    q8192_triton_0626_zero_correction_gate_finalize.hsaco
    q8192_triton_0626_conditional_exact_down_rows4.hsaco
)
for name in "${expected_files[@]}"; do
    [[ -s ${output_dir}/${name} ]] || {
        echo "missing generated ${name}" >&2
        exit 3
    }
    "${llvm_strip}" --strip-debug "${output_dir}/${name}"
    if strings -a "${output_dir}/${name}" |
            grep -E '/home/[^/]+|/Users/[^/]+' >/dev/null; then
        echo "generated ${name} contains a private home path" >&2
        exit 4
    fi
done

strip_version=$("${llvm_strip}" --version | sed -n '1p')
python3 - "${metadata_dir}" "${output_dir}" "${strip_version}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

metadata_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
strip_version = sys.argv[3]
base_metadata_path = output_dir / "metadata.json"
base_record = json.loads(base_metadata_path.read_text())
assert base_record["target"] == "gfx1151"
assert base_record["compiler"] == "Triton 3.6.0 HIP backend"
assert base_record["shape"]["tokens"] == 8192
assert base_record["shape"]["block_m"] == 64
assert base_record["shape"]["group_m"] == 1
assert len(base_record["kernels"]) == 6
for candidate in base_record["kernels"]:
    payload = (output_dir / candidate["file"]).read_bytes()
    candidate["bytes"] = len(payload)
    candidate["sha256"] = hashlib.sha256(payload).hexdigest()
base_record["postprocess"] = {
    "debug_sections_stripped": True,
    "tool": strip_version,
    "arguments": ["--strip-debug"],
}
base_metadata_path.write_text(json.dumps(base_record, indent=2) + "\n")

checks = (
    ("row-major-gate.json", "variants", 256),
    ("zero-finalize.json", None, None),
    ("conditional-down.json", "variants", 4),
)
for metadata_name, variants_key, rows in checks:
    metadata_path = metadata_dir / metadata_name
    record = json.loads(metadata_path.read_text())
    assert record["target"] == "gfx1151"
    assert record["compiler"] == "Triton 3.6.0 HIP backend"
    if variants_key is None:
        candidates = [record]
    else:
        candidates = [
            item for item in record[variants_key]
            if item["rows_per_program"] == rows
        ]
        assert len(candidates) == 1
    for candidate in candidates:
        payload = (output_dir / candidate["file"]).read_bytes()
        candidate["bytes"] = len(payload)
        candidate["sha256"] = hashlib.sha256(payload).hexdigest()
    record["postprocess"] = {
        "debug_sections_stripped": True,
        "tool": strip_version,
        "arguments": ["--strip-debug"],
    }
    metadata_path.write_text(json.dumps(record, indent=2) + "\n")

for metadata_name, variants_key, rows in checks:
    record = json.loads((metadata_dir / metadata_name).read_text())
    if variants_key is None:
        candidates = [record]
    else:
        candidates = [
            item for item in record[variants_key]
            if item["rows_per_program"] == rows
        ]
        assert len(candidates) == 1
    assert record["postprocess"]["debug_sections_stripped"] is True
    for candidate in candidates:
        payload = (output_dir / candidate["file"]).read_bytes()
        assert len(payload) == candidate["bytes"]
        assert hashlib.sha256(payload).hexdigest() == candidate["sha256"]

base_record = json.loads(base_metadata_path.read_text())
assert base_record["postprocess"]["debug_sections_stripped"] is True
for candidate in base_record["kernels"]:
    payload = (output_dir / candidate["file"]).read_bytes()
    assert len(payload) == candidate["bytes"]
    assert hashlib.sha256(payload).hexdigest() == candidate["sha256"]
PY

{
    printf 'record_type=qrt_amd395_public_complete_aot_build\n'
    printf 'schema_version=1\nhost=%s\n' "$(hostname)"
    printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'repo_commit=6cdce444177f54e8985a2d6148da863369bece32\n'
    printf 'triton_version=3.6.0\ntarget=gfx1151\n'
    printf 'llvm_strip_version=%s\n' "${strip_version}"
    printf 'source_root_class=neutral_var_tmp\n'
    printf 'kernel_count=9\nprivate_home_path_count=0\n'
    printf 'component_only=1\ninference_success_claimed=0\n'
} > "${stage_root}/provenance.env"
sha256sum "${output_dir}"/*.hsaco "${output_dir}/metadata.json" \
    "${metadata_dir}"/*.json \
    "${stage_root}/provenance.env" > "${stage_root}/SHA256SUMS"
cat "${stage_root}/SHA256SUMS"
