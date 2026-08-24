#!/usr/bin/env bash

set -Eeuo pipefail
umask 022
ulimit -c 0

stage_root=${QRT_AMD_PUBLIC_SILU_AOT_ROOT:?set the bounded /var/tmp AOT root}
triton_python=${QRT_TRITON_PYTHON:?set the Triton 3.6 Python executable}
llvm_strip=${QRT_ROCM_LLVM_STRIP:-/opt/rocm/llvm/bin/llvm-strip}
source_dir=${stage_root}/sources
output_dir=${stage_root}/kernels
metadata_dir=${stage_root}/metadata
log_dir=${stage_root}/logs

expected_row_source_sha256=d60013df4167a7679eca94a10fe1e3b4df05eef8c4990e217c4ac3c9a94dc258
expected_sorted_source_sha256=f33b78e290fe7ce34fb926b4c2704558dcc79c64558a656f1d6e2b05625f885b
expected_zero_source_sha256=4882de7577a97bfbc0d91ff224df4e5c45b055f496a080ecf3e852dfd886e6eb

case "${stage_root}" in
    /var/tmp/qrt-public-silu-aot-r1192-*) ;;
    *) echo "public SiLU AOT root escaped the bounded r1192 family" >&2; exit 2 ;;
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
    echo "refusing to overwrite generated public SiLU AOT outputs" >&2
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

row_source=${source_dir}/compile_q8192_row_major_sorted_conditional_gate.py
sorted_source=${source_dir}/compile_q8192_sorted_conditional_gate.py
zero_source=${source_dir}/compile_q8192_zero_correction_gate_finalize.py
verify_sha256 "${row_source}" "${expected_row_source_sha256}"
verify_sha256 "${sorted_source}" "${expected_sorted_source_sha256}"
verify_sha256 "${zero_source}" "${expected_zero_source_sha256}"
[[ $("${triton_python}" -c 'import triton; print(triton.__version__)') == 3.6.0 ]]

mkdir -p -- "${output_dir}" "${metadata_dir}" "${log_dir}" \
    "${stage_root}/cache" "${stage_root}/tmp"
export TRITON_CACHE_DIR=${stage_root}/cache
export TMPDIR=${stage_root}/tmp

timeout --signal=TERM --kill-after=30s 300s \
    "${triton_python}" "${row_source}" \
    --output-dir "${output_dir}" \
    --metadata "${metadata_dir}/row-major-gate.json" --rows 256 \
    > "${log_dir}/row-major-gate.stdout.log" \
    2> "${log_dir}/row-major-gate.stderr.log"
timeout --signal=TERM --kill-after=30s 300s \
    "${triton_python}" "${sorted_source}" \
    --output-dir "${output_dir}" \
    --metadata "${metadata_dir}/sorted-gate.json" --rows 256 \
    > "${log_dir}/sorted-gate.stdout.log" \
    2> "${log_dir}/sorted-gate.stderr.log"
timeout --signal=TERM --kill-after=30s 300s \
    "${triton_python}" "${zero_source}" \
    --output-dir "${output_dir}" \
    --metadata "${metadata_dir}/zero-finalize.json" \
    > "${log_dir}/zero-finalize.stdout.log" \
    2> "${log_dir}/zero-finalize.stderr.log"

expected_files=(
    q8192_triton_0626_row_major_sorted_conditional_exact_gate_rows256.hsaco
    q8192_triton_0626_sorted_conditional_exact_gate_rows256.hsaco
    q8192_triton_0626_zero_correction_gate_finalize.hsaco
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
checks = (
    ("row-major-gate.json", "variants", 256),
    ("sorted-gate.json", "variants", 256),
    ("zero-finalize.json", None, None),
)
for metadata_name, variants_key, rows in checks:
    metadata_path = metadata_dir / metadata_name
    record = json.loads(metadata_path.read_text())
    assert record["target"] == "gfx1151"
    assert record["compiler"] == "Triton 3.6.0 HIP backend"
    candidates = (
        [record]
        if variants_key is None
        else [
            item
            for item in record[variants_key]
            if item["rows_per_program"] == rows
        ]
    )
    assert len(candidates) == 1
    for candidate in candidates:
        payload = (output_dir / candidate["file"]).read_bytes()
        candidate["bytes"] = len(payload)
        candidate["sha256"] = hashlib.sha256(payload).hexdigest()
    record["postprocess"] = {
        "debug_sections_stripped": True,
        "tool": strip_version,
        "arguments": ["--strip-debug"],
        "private_home_path_count": 0,
    }
    metadata_path.write_text(json.dumps(record, indent=2) + "\n")
PY

{
    printf 'record_type=qrt_amd395_public_silu_aot_build\n'
    printf 'schema_version=1\nhost=%s\n' "$(hostname)"
    printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'triton_version=3.6.0\ntarget=gfx1151\n'
    printf 'llvm_strip_version=%s\n' "${strip_version}"
    printf 'source_root_class=neutral_var_tmp\n'
    printf 'source_count=3\nkernel_count=3\nprivate_home_path_count=0\n'
    printf 'component_only=1\ninference_success_claimed=0\n'
} > "${stage_root}/provenance.env"
sha256sum "${source_dir}"/*.py "${output_dir}"/*.hsaco \
    "${metadata_dir}"/*.json "${stage_root}/provenance.env" \
    > "${stage_root}/SHA256SUMS"
cat "${stage_root}/SHA256SUMS"
