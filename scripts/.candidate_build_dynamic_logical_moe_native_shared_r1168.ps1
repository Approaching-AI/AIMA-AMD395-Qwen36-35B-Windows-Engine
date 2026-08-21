param(
    [string]$Repo = 'D:\projects\AIMA-dynamic-logical-r1',
    [ValidateRange(120, 1800)]
    [int]$TimeoutSeconds = 1200
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$qualificationScript = Join-Path $Repo `
    'scripts\.candidate_build_dynamic_logical_moe_r1093.ps1'
if (-not (Test-Path -LiteralPath $qualificationScript -PathType Leaf)) {
    throw "dynamic-logical qualification script not found: $qualificationScript"
}

& $qualificationScript `
    -Repo $Repo `
    -RunLabel r1168 `
    -TimeoutSeconds $TimeoutSeconds `
    -NativeSharedSelected
if ($LASTEXITCODE -ne 0) {
    throw "r1168 native-shared qualification exited $LASTEXITCODE"
}

$recordPath = Join-Path $Repo `
    'build\dynamic-logical-moe-r1168\qualification-provenance.json'
if (-not (Test-Path -LiteralPath $recordPath -PathType Leaf)) {
    throw "r1168 qualification record not found: $recordPath"
}
$record = Get-Content -Raw -LiteralPath $recordPath | ConvertFrom-Json
$expectedAuxiliary = @(
    'q8192_triton_0626_row_major_sorted_conditional_exact_gate_rows256.hsaco',
    'q8192_triton_0626_zero_correction_gate_finalize.hsaco',
    'q8192_triton_0626_conditional_exact_down_rows4.hsaco'
)
if (-not [bool]$record.native_shared_selected -or
    [bool]$record.aot_reused -or
    [string]$record.base_aot_mode -cne `
        'generated_from_current_source' -or
    [int]$record.base_aot_block_m -ne 64 -or
    [int]$record.base_aot_group_m -ne 1 -or
    -not [bool]$record.base_aot_dynamic_logical_abi -or
    (@($record.auxiliary_kernel_files) -join ',') -cne
        ($expectedAuxiliary -join ',') -or
    @($record.required_runtime_environment).Count -ne 12 -or
    @($record.artifacts).Count -ne 11 -or
    -not [bool]$record.dynamic_logical_pass -or
    -not [bool]$record.dynamic_logical_q8192_exact_pass -or
    [bool]$record.inference_success_claimed) {
    throw 'r1168 native-shared qualification record did not pass'
}
Write-Output $recordPath
