#!/usr/bin/env bash

set -Eeuo pipefail
umask 022
ulimit -c 0

workspace_root=${QRT_AMD_WORKSPACE_ROOT:?set the bounded AMD workspace root}
stage_root=${QRT_AMD_FRESH_BASE_AOT_ROOT:-${workspace_root}/qrt-dynamic-aot-r1171-REFHyB}
source_label=${QRT_AMD_DYNAMIC_SMOKE_SOURCE_LABEL:-r1176}
build_label=${QRT_AMD_DYNAMIC_SMOKE_BUILD_LABEL:-r1176b}
source=${stage_root}/stage-${source_label}/native/providers/triton_moe/q8192_triton_selected_moe_smoke.cpp
build=${stage_root}/smoke-${build_label}
smoke=${build}/moe-v2-sorted-oracle-smoke
expected_source_sha256=${QRT_AMD_DYNAMIC_SMOKE_SOURCE_SHA256:-b003e065a7f38389775a9f41b7f119941fa0652ce0ea9c69a7e765b6a565aa03}

case "${stage_root}" in
    "${workspace_root}"/qrt-dynamic-aot-r1171-*) ;;
    *) echo "smoke stage escaped the r1171 family" >&2; exit 2 ;;
esac
[[ ${source_label} =~ ^r[0-9]+$ ]] || {
    echo "invalid smoke source label" >&2
    exit 2
}
[[ ${build_label} =~ ^r[0-9]+b$ ]] || {
    echo "invalid smoke build label" >&2
    exit 2
}
[[ ! -e ${build} ]] || {
    echo "refusing to overwrite ${build}" >&2
    exit 2
}
[[ $(sha256sum "${source}" | awk '{print $1}') == \
    "${expected_source_sha256}" ]] || {
    echo "smoke source hash differs" >&2
    exit 3
}
mkdir -p -- "${build}"

defines=(
    -DQRT_TRITON_MOE_BLOCK_M=64
    -DQRT_TRITON_MOE_GATE_BLOCK_N=64
    -DQRT_TRITON_MOE_DOWN_BLOCK_N=64
    -DQRT_TRITON_MOE_GROUP_M=1
    -DQRT_TRITON_MOE_ROUTE_THREADS=128
    -DQRT_TRITON_MOE_GATE_THREADS=128
    -DQRT_TRITON_MOE_DOWN_THREADS=128
    -DQRT_TRITON_MOE_GATE_SHARED_BYTES=8192
    -DQRT_TRITON_MOE_DOWN_SHARED_BYTES=8192
    -DQRT_TRITON_MOE_ROUTER_THREADS=256
    -DQRT_TRITON_MOE_ROUTER_TOKEN_TILE=8
    -DQRT_TRITON_MOE_FUSED_COMBINE_WIDTH=4
    -DQRT_TRITON_MOE_FULL_V3_EVENT_SLOTS=16
    -DQRT_TRITON_MOE_NATIVE_WMMA_K_STAGE=32
    -DQRT_TRITON_MOE_NATIVE_WMMA_GATE=1
    -DQRT_TRITON_MOE_NATIVE_WMMA_DOWN=1
    -DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B=1
    -DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_LOAD_THREADS=192
    -DQRT_TRITON_MOE_NATIVE_FUSED_ROUTE_LAYOUT=1
    -DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SPLIT_GATE_PASSES=1
    -DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SERIAL_GATE_N32=1
    -DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SERIAL_DOWN_N32=1
    -DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SKIP_INACTIVE_A_STORES=1
    -DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_FUSED_OVERFLOW32=1
    -DQRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_PALETTE=1
    -DQRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE=1
    -DQRT_TRITON_MOE_TRANSPOSED_ROUTER=1
    -DQRT_TRITON_MOE_FULL_V3_FUSED_COMBINE=1
    -DQRT_TRITON_MOE_BATCHED_HAWKEYE=1
    -DQRT_TRITON_MOE_CONDITIONAL_EXACT_GATE=1
    -DQRT_TRITON_MOE_SORTED_CONDITIONAL_EXACT_GATE=1
    -DQRT_TRITON_MOE_ROW_MAJOR_SORTED_CONDITIONAL_EXACT_GATE=1
    -DQRT_TRITON_MOE_CONDITIONAL_EXACT_GATE_ROWS=256
    -DQRT_TRITON_MOE_CONDITIONAL_EXACT_DOWN=1
    -DQRT_TRITON_MOE_CONDITIONAL_EXACT_DOWN_ROWS=4
    -DQRT_TRITON_MOE_Q1024_EXACT_SHARED=0
)
timeout --signal=TERM --kill-after=30s 900s \
    /opt/rocm/bin/hipcc -std=c++17 -O3 --offload-arch=gfx1151 \
    "${defines[@]}" "${source}" -o "${smoke}" -ldl \
    > "${build}/compile.stdout.log" \
    2> "${build}/compile.stderr.log"

{
    printf 'record_type=qrt_amd395_dynamic_smoke_sorted_oracle_build\n'
    printf 'schema_version=1\nhost=%s\n' "$(hostname)"
    printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'repo_commit=6cdce444177f54e8985a2d6148da863369bece32\n'
    printf 'smoke_source_sha256=%s\n' "${expected_source_sha256}"
    printf 'smoke_sha256=%s\n' "$(sha256sum "${smoke}" | awk '{print $1}')"
    printf 'provider_group_m=1\nexact_shared=0\n'
    printf 'sorted_bf16_provider_oracle=1\n'
    printf 'component_only=1\ninference_success_claimed=0\n'
} > "${build}/provenance.env"
sha256sum "${smoke}" "${build}/provenance.env"
