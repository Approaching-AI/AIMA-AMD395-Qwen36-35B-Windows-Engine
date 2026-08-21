param(
    [string]$Repo = 'D:\projects\AIMA-dynamic-logical-r1',
    [ValidatePattern('^r[0-9]+$')]
    [string]$RunLabel = 'r1091',
    [ValidatePattern('^r[0-9]+$')]
    [string]$ProductRunLabel = 'r1089',
    [ValidatePattern('^r[0-9]+$')]
    [string]$WholeProviderBuildLabel = 'r1088',
    [ValidatePattern('^r[0-9]+$')]
    [string]$AttentionBuildLabel = 'r1087',
    [ValidatePattern('^r[0-9]+$')]
    [string]$DynamicMoeBuildLabel = 'r1093',
    [switch]$DynamicMoeNativeSharedSelected,
    [ValidatePattern('^r[0-9]+$')]
    [string]$GdnQualificationLabel = 'r1094',
    [ValidateSet('fla', 'aiter')]
    [string]$GdnRoute = 'aiter',
    [ValidateRange(1, 65535)]
    [uint16]$Port = 8000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$releaseRoot = 'D:\projects\AIMA-AMD395-Qwen36-35B-Windows-Engine-v1.0.1-release-2bf0457\build\runtime-v1.0.1-2bf0457'
$serviceExe = Join-Path $releaseRoot 'engine\qrt.exe'
$modelPath = 'D:\models\Qwen3.6-35B-A3B'
$wholeProvider = Join-Path $Repo `
    "build\whole-provider-fla-boundary-contract-$WholeProviderBuildLabel\qrt_qwen36_whole_provider.dll"
$attentionRoot = Join-Path $Repo `
    "build\ck-fmha-q1-terminal-exact-$AttentionBuildLabel"
$attentionProvider = Join-Path $attentionRoot `
    'qrt_ck_fmha_q1_terminal_exact.dll'
$attentionBuildProvenance = Join-Path $attentionRoot `
    'build-provenance.json'
$productRun = Join-Path $Repo `
    "build\product-q8192-$GdnRoute-boundary-triton-ck-$ProductRunLabel"
$qualifiedEnv = Join-Path $productRun 'runtime.env'
$productRecord = Join-Path $productRun 'run-record.json'
$gb10OraclePath = Join-Path $Repo `
    'contracts\hprefill_q8192_gb10_oracle.json'
$gb10ContinuationCaptureScriptPath = Join-Path $Repo `
    'scripts\capture_gb10_q8192_continuation.py'
$serviceDir = Join-Path $Repo "build\service-random-length-terminal-$RunLabel"
$statePath = Join-Path $serviceDir 'service.json'
$logPath = Join-Path $serviceDir 'service.log'
$routeLogAudit = Join-Path $Repo `
    'scripts\verify_prefill_random_length_route_log.py'
$dynamicMoeRoot = Join-Path $Repo `
    "build\dynamic-logical-moe-$DynamicMoeBuildLabel"
$dynamicMoeProvider = Join-Path $dynamicMoeRoot `
    'qrt_triton_moe_q8192_provider.dll'
$dynamicMoeMetadata = Join-Path $dynamicMoeRoot 'metadata.json'
$dynamicMoeQualification = Join-Path $dynamicMoeRoot `
    'qualification-provenance.json'
$dynamicMoeGenerator = Join-Path $Repo `
    'native\generators\compile_q8192_triton_selected_moe.py'
$dynamicMoeProviderSource = Join-Path $Repo `
    'native\providers\triton_moe\qrt_triton_moe_q8192_provider.cpp'
$dynamicMoeSmokeSource = Join-Path $Repo `
    'native\providers\triton_moe\q8192_triton_selected_moe_smoke.cpp'
$dynamicMoeBuildScript = Join-Path $Repo `
    'scripts\.candidate_build_dynamic_logical_moe_r1093.ps1'
$dynamicMoeBuilderScript = Join-Path $Repo `
    'scripts\baiying_build_triton_moe_q8192.ps1'
$gdnQualificationRoot = Join-Path $Repo `
    "build\dynamic-logical-gdn-$GdnQualificationLabel"
$gdnQualification = Join-Path $gdnQualificationRoot `
    'qualification-provenance.json'
$smoothTailRoot = Join-Path $releaseRoot 'smooth-tail'
$arbitraryMoeProvider = Join-Path $releaseRoot `
    'q1024-moe\qrt_triton_moe_q1024_exact_provider_slots64.dll'
$arbitraryMoeKernelDir = Join-Path $releaseRoot 'q1024-moe\moe-kernels'
$modelEngineLoadMaxMs = 30000.0
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false
$expectedDynamicMoeAuxiliaryKernelFiles = if (
    $DynamicMoeNativeSharedSelected
) {
    @(
        'q8192_triton_0626_row_major_sorted_conditional_exact_gate_rows256.hsaco',
        'q8192_triton_0626_zero_correction_gate_finalize.hsaco',
        'q8192_triton_0626_conditional_exact_down_rows4.hsaco'
    )
} else {
    @()
}
$expectedDynamicMoeRuntimeEnvironment = [ordered]@{}
if ($DynamicMoeNativeSharedSelected) {
    $expectedDynamicMoeRuntimeEnvironment = [ordered]@{
        QRT_QWEN36_Q8192_ROUTER_HIPBLASLT_BF16 = '1'
        QRT_QWEN36_CUDA_VLLM_ROUTER_HAWKEYE_MIDPOINT_RADIUS = '173'
        QRT_QWEN36_CUDA_VLLM_SHARED_HAWKEYE_MIDPOINT_RADIUS = '0'
        QRT_QWEN36_Q8192_VLLM_BF16_RESIDUAL_CARRIER = '1'
        QRT_QWEN36_EXACT_ARBITRARY_VLLM_SPLIT_VARIANCE = '1'
        QRT_QWEN36_Q8192_VLLM_SORTED_BF16_ROUTE_SUM = '1'
        QRT_QWEN36_CUDA_VLLM_MOE_HAWKEYE_MIDPOINT_RADIUS = '0'
        QRT_QWEN36_CUDA_VLLM_MOE_UP_HAWKEYE_MIDPOINT_RADIUS = '0'
        QRT_QWEN36_CUDA_VLLM_ROUTED_DOWN_CONTRIBUTION_HAWKEYE_MIDPOINT_RADIUS = '0'
        QRT_QWEN36_CUDA_VLLM_ROUTED_GATE_HAWKEYE_LOW_EXPONENT_THRESHOLD = '0'
        QRT_QWEN36_CUDA_VLLM_ROUTED_UP_HAWKEYE_LOW_EXPONENT_THRESHOLD = '0'
        QRT_QWEN36_CUDA_VLLM_ROUTED_DOWN_HAWKEYE_LOW_EXPONENT_THRESHOLD = '0'
    }
}
$expectedDynamicMoeRuntimeEnvironmentRecords = @(
    foreach ($entry in $expectedDynamicMoeRuntimeEnvironment.GetEnumerator()) {
        [ordered]@{
            name = [string]$entry.Key
            value = [string]$entry.Value
        }
    }
)

function Assert-DynamicMoeRuntimeEnvironmentRecords {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [object[]]$Records,
        [Parameter(Mandatory = $true)][string]$Source
    )
    if ($Records.Count -ne $expectedDynamicMoeRuntimeEnvironment.Count) {
        throw "$Source runtime environment count differs"
    }
    $seen = @{}
    foreach ($entry in $Records) {
        $name = [string]$entry.name
        $value = [string]$entry.value
        if ([string]::IsNullOrWhiteSpace($name) -or
            $seen.ContainsKey($name) -or
            -not $expectedDynamicMoeRuntimeEnvironment.Contains($name) -or
            $value -cne [string]$expectedDynamicMoeRuntimeEnvironment[$name]) {
            throw "$Source runtime environment differs: $name"
        }
        $seen[$name] = $true
    }
}

if (Test-Path -LiteralPath $serviceDir) {
    throw "refusing to overwrite random-length service: $serviceDir"
}
foreach ($leaf in @(
        $serviceExe,
        $wholeProvider,
        $attentionProvider,
        $attentionBuildProvenance,
        $qualifiedEnv,
        $productRecord,
        $gb10OraclePath,
        $gb10ContinuationCaptureScriptPath,
        $routeLogAudit,
        $dynamicMoeProvider,
        $dynamicMoeMetadata,
        $dynamicMoeQualification,
        $dynamicMoeGenerator,
        $dynamicMoeProviderSource,
        $dynamicMoeSmokeSource,
        $dynamicMoeBuildScript,
        $dynamicMoeBuilderScript,
        $gdnQualification
    )) {
    if (-not (Test-Path -LiteralPath $leaf -PathType Leaf)) {
        throw "random-length service input is missing: $leaf"
    }
}
foreach ($container in @(
        $modelPath,
        $smoothTailRoot,
        $dynamicMoeRoot,
        $arbitraryMoeKernelDir
    )) {
    if (-not (Test-Path -LiteralPath $container -PathType Container)) {
        throw "random-length service directory is missing: $container"
    }
}
if (-not (Test-Path -LiteralPath $arbitraryMoeProvider -PathType Leaf)) {
    throw "arbitrary MoE provider is missing: $arbitraryMoeProvider"
}

$repoCommit = (& git -C $Repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $repoCommit -notmatch '^[0-9a-f]{40}$') {
    throw "could not resolve a lowercase 40-hex repository commit: $repoCommit"
}

$productEvidence = Get-Content -Raw -LiteralPath $productRecord |
    ConvertFrom-Json
$gb10Oracle = Get-Content -Raw -LiteralPath $gb10OraclePath |
    ConvertFrom-Json
$gb10OracleSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $gb10OraclePath
).Hash.ToLowerInvariant()
$gb10ContinuationCaptureScriptSha256 = (
    Get-FileHash -Algorithm SHA256 `
        -LiteralPath $gb10ContinuationCaptureScriptPath
).Hash.ToLowerInvariant()
$requiredProductFields = @(
    'record_type',
    'host',
    'repo_commit',
    'model_path',
    'gdn_route',
    'moe_arithmetic',
    'q1_attention',
    'prompt_u32le_sha256',
    'prompt_u32le_fnv1a64',
    'gb10_oracle',
    'gb10_oracle_sha256',
    'gb10_oracle_capture_request_sha256',
    'gb10_oracle_capture_response_sha256',
    'gb10_oracle_capture_command_file_sha256',
    'gb10_continuation_capture_request_sha256',
    'gb10_continuation_capture_response_sha256',
    'gb10_continuation_capture_command_file_sha256',
    'gb10_first_token',
    'gb10_first_token_raw_logit',
    'first_token_raw_logit_absolute_difference',
    'first_token_raw_logit_tolerance',
    'expected_output_tokens',
    'native_output_tokens',
    'expected_output_token_ids_u32le_sha256',
    'expected_output_token_ids_u32le_fnv1a64',
    'continuation_token_count',
    'decode_step_count',
    'engine_self_hashes_diagnostic_only',
    'continuation_token_for_token_pass',
    'correctness_pass',
    'q8192_ttft_ms',
    'q8192_ttft_max_ms',
    'q8192_prefill_tokens_per_second',
    'q8192_prefill_metric_relative_error',
    'q8192_prefill_metric_relative_error_max',
    'q8192_prefill_min_tokens_per_second',
    'q8192_tpot_ms',
    'q8192_tpot_max_ms',
    'q8192_decode_tokens_per_second',
    'q8192_decode_min_tokens_per_second',
    'performance_pass',
    'model_engine_load_ms',
    'model_engine_load_max_ms',
    'model_engine_load_pass',
    'route_pass',
    'decode_route_pass',
    'q8192_moe_fixed_route_pass',
    'prefix_tokens',
    'prefix_suffix_tokens',
    'prefix_hit_count',
    'prefix_output_tokens',
    'prefix_tpot_ms',
    'prefix_decode_tokens_per_second',
    'prefix_model_engine_load_ms',
    'prefix_continuation_token_for_token_pass',
    'prefix_continuation_correctness_pass',
    'prefix_continuation_performance_pass',
    'prefix_route_pass',
    'prefix_continuation_pass',
    'acceptance_pass',
    'environment_sha256',
    'whole_provider_sha256',
    'attention_provider_sha256',
    'attention_build_label',
    'attention_build_provenance',
    'attention_build_provenance_sha256',
    'attention_dynamic_logical_case_count',
    'attention_dynamic_logical_reset_included',
    'attention_dynamic_logical_component_only',
    'gdn_qualification_label',
    'gdn_build_provenance',
    'gdn_build_provenance_sha256',
    'gdn_provider',
    'gdn_provider_sha256',
    'gdn_kernel_dir',
    'gdn_dynamic_logical_case_count',
    'gdn_dynamic_logical_mode_count',
    'gdn_dynamic_logical_reset_included',
    'gdn_dynamic_logical_component_only',
    'gdn_artifacts',
    'dynamic_moe_build_label',
    'q8192_moe_build_provenance_sha256',
    'q8192_moe_provider',
    'q8192_moe_provider_sha256',
    'q8192_moe_dynamic_logical_component_pass',
    'q8192_moe_dynamic_logical_q8192_exact_pass',
    'q8192_moe_dynamic_logical_case_count',
    'q8192_moe_expected_full_provider_hash_diagnostic_only',
    'q8192_moe_dynamic_logical_timing_samples',
    'q8192_moe_dynamic_logical_timing_stat',
    'q8192_moe_dynamic_logical_nonfinite',
    'q8192_moe_dynamic_logical_component_only',
    'q8192_moe_dynamic_logical_reset_included',
    'q8192_moe_artifacts'
)
if ($DynamicMoeNativeSharedSelected) {
    $requiredProductFields += @(
        'dynamic_moe_native_shared_selected',
        'dynamic_moe_base_aot_mode',
        'dynamic_moe_base_aot_block_m',
        'dynamic_moe_base_aot_group_m',
        'dynamic_moe_base_aot_dynamic_logical_abi',
        'dynamic_moe_required_runtime_environment'
    )
}
foreach ($name in $requiredProductFields) {
    if ($productEvidence.PSObject.Properties.Name -notcontains $name) {
        throw "q8192 product evidence lacks required field: $name"
    }
}
$productOracleSnapshot = [string]$productEvidence.gb10_oracle
if (-not (Test-Path -LiteralPath $productOracleSnapshot -PathType Leaf)) {
    throw "q8192 product GB10 oracle snapshot is missing: $productOracleSnapshot"
}
$productOracleSnapshotSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $productOracleSnapshot
).Hash.ToLowerInvariant()
$productEvidencePropertyNames = @($productEvidence.PSObject.Properties.Name)
$recordedProductNativeSharedSelected =
    if ($productEvidencePropertyNames -contains
            'dynamic_moe_native_shared_selected') {
        [bool]$productEvidence.dynamic_moe_native_shared_selected
    } else {
        $false
    }
$recordedProductRuntimeEnvironment =
    if ($productEvidencePropertyNames -contains
            'dynamic_moe_required_runtime_environment') {
        @($productEvidence.dynamic_moe_required_runtime_environment)
    } else {
        @()
    }
if ($recordedProductNativeSharedSelected -ne
        [bool]$DynamicMoeNativeSharedSelected) {
    throw 'q8192 product selected-MoE route differs from this service'
}
Assert-DynamicMoeRuntimeEnvironmentRecords `
    -Records $recordedProductRuntimeEnvironment `
    -Source 'q8192 product'
if ([string]$productEvidence.record_type -ne `
        'qrt_q8192_gb10_product_acceptance' -or
    [string]$productEvidence.host -ine [Environment]::MachineName -or
    [string]$productEvidence.repo_commit -ne $repoCommit -or
    [string]$productEvidence.model_path -ine $modelPath -or
    [string]$productEvidence.gdn_route -ne $GdnRoute -or
    [string]$productEvidence.moe_arithmetic -ne 'triton' -or
    [string]$productEvidence.q1_attention -ne 'ck' -or
    [string]$productEvidence.dynamic_moe_build_label -ne `
        $DynamicMoeBuildLabel -or
    [string]$productEvidence.attention_build_label -ne `
        $AttentionBuildLabel -or
    [string]$productEvidence.gdn_qualification_label -ne `
        $GdnQualificationLabel -or
    [string]$productEvidence.prompt_u32le_sha256 -ne `
        'dda20edc609f935f34d3d41ca4a84ffefa66726676756fe26ff3bef4fbff0b96' -or
    [string]$productEvidence.prompt_u32le_fnv1a64 -ne `
        '1584e34d56e5d78b' -or
    [string]$productEvidence.gb10_oracle_sha256 -ne `
        $gb10OracleSha256 -or
    $productOracleSnapshotSha256 -ne $gb10OracleSha256 -or
    [string]$productEvidence.gb10_oracle_capture_request_sha256 -ne `
        [string]$gb10Oracle.capture.request_sha256 -or
    [string]$productEvidence.gb10_oracle_capture_response_sha256 -ne `
        [string]$gb10Oracle.capture.response_sha256 -or
    [string]$productEvidence.gb10_oracle_capture_command_file_sha256 -ne `
        [string]$gb10Oracle.capture.command_file_sha256 -or
    [string]$productEvidence.gb10_continuation_capture_request_sha256 -ne `
        [string]$gb10Oracle.continuation_capture.request_sha256 -or
    [string]$productEvidence.gb10_continuation_capture_response_sha256 -ne `
        [string]$gb10Oracle.continuation_capture.response_sha256 -or
    [string]$productEvidence.gb10_continuation_capture_command_file_sha256 `
        -ne $gb10ContinuationCaptureScriptSha256 -or
    [string]$gb10Oracle.continuation_capture.command_file_sha256 -ne `
        $gb10ContinuationCaptureScriptSha256 -or
    [int]$productEvidence.gb10_first_token -ne `
        [int]$gb10Oracle.expected.first_token_id -or
    [double]$productEvidence.gb10_first_token_raw_logit -ne `
        [double]$gb10Oracle.expected.first_token_raw_logit -or
    [string]$productEvidence.expected_output_token_ids_u32le_sha256 -ne `
        [string]$gb10Oracle.expected.output_token_ids_u32le_sha256 -or
    [string]$productEvidence.expected_output_token_ids_u32le_fnv1a64 -ne `
        [string]$gb10Oracle.expected.output_token_ids_u32le_fnv1a64 -or
    [int]$productEvidence.continuation_token_count -ne 32 -or
    [int]$productEvidence.decode_step_count -ne 31) {
    throw 'q8192 product evidence identity does not match this qualified service'
}
$gb10OutputTokens = @(
    $gb10Oracle.expected.output_token_ids | ForEach-Object { [int]$_ }
)
$productExpectedOutputTokens = @(
    $productEvidence.expected_output_tokens | ForEach-Object { [int]$_ }
)
$productNativeOutputTokens = @(
    $productEvidence.native_output_tokens | ForEach-Object { [int]$_ }
)
$productPrefixOutputTokens = @(
    $productEvidence.prefix_output_tokens | ForEach-Object { [int]$_ }
)
if ($gb10OutputTokens.Count -ne 32 -or
    $productExpectedOutputTokens.Count -ne $gb10OutputTokens.Count -or
    $productNativeOutputTokens.Count -ne $gb10OutputTokens.Count -or
    $productPrefixOutputTokens.Count -ne $gb10OutputTokens.Count) {
    throw 'q8192 product continuation token counts differ from the GB10 oracle'
}
for ($tokenIndex = 0; $tokenIndex -lt $gb10OutputTokens.Count; `
        $tokenIndex++) {
    if ($productExpectedOutputTokens[$tokenIndex] -ne `
            $gb10OutputTokens[$tokenIndex] -or
        $productNativeOutputTokens[$tokenIndex] -ne `
            $gb10OutputTokens[$tokenIndex] -or
        $productPrefixOutputTokens[$tokenIndex] -ne `
            $gb10OutputTokens[$tokenIndex]) {
        throw "q8192 product continuation differs at token $tokenIndex"
    }
}
if (-not [bool]$productEvidence.correctness_pass -or
    -not [bool]$productEvidence.engine_self_hashes_diagnostic_only -or
    -not [bool]$productEvidence.continuation_token_for_token_pass -or
    -not [bool]$productEvidence.prefix_continuation_token_for_token_pass -or
    -not [bool]$productEvidence.prefix_continuation_correctness_pass -or
    [double]$productEvidence.first_token_raw_logit_absolute_difference -gt
        [double]$productEvidence.first_token_raw_logit_tolerance -or
    [double]$productEvidence.first_token_raw_logit_tolerance -gt 0.125) {
    throw 'q8192 product evidence did not pass the gb10 raw-logit boundary'
}
if (-not [bool]$productEvidence.performance_pass -or
    [double]$productEvidence.q8192_prefill_tokens_per_second -lt 1506.407 -or
    [double]$productEvidence.q8192_prefill_metric_relative_error -gt `
        0.001 -or
    [double]$productEvidence.q8192_prefill_metric_relative_error_max -gt `
        0.001 -or
    [double]$productEvidence.q8192_prefill_min_tokens_per_second -lt `
        1506.407 -or
    [double]$productEvidence.q8192_ttft_ms -gt 4187.416 -or
    [double]$productEvidence.q8192_ttft_max_ms -gt 4187.416 -or
    [double]$productEvidence.q8192_decode_tokens_per_second -lt 28.168 -or
    [double]$productEvidence.q8192_decode_min_tokens_per_second -lt `
        28.168 -or
    [double]$productEvidence.q8192_tpot_ms -gt 35.502 -or
    [double]$productEvidence.q8192_tpot_max_ms -gt 35.502 -or
    -not [bool]$productEvidence.prefix_continuation_performance_pass -or
    [double]$productEvidence.prefix_decode_tokens_per_second -lt 28.168 -or
    [double]$productEvidence.prefix_tpot_ms -gt 35.502 -or
    [double]$productEvidence.prefix_model_engine_load_ms -gt `
        $modelEngineLoadMaxMs) {
    throw 'q8192 product evidence did not retain the 1506.407 tok/s and 4187.416 ms performance boundary'
}
if (-not [bool]$productEvidence.model_engine_load_pass -or
    [double]$productEvidence.model_engine_load_ms -gt `
        $modelEngineLoadMaxMs -or
    [double]$productEvidence.model_engine_load_max_ms -gt `
        $modelEngineLoadMaxMs) {
    throw 'q8192 product evidence did not pass the 30000 ms model-plus-engine load boundary'
}
if (-not [bool]$productEvidence.route_pass -or
    -not [bool]$productEvidence.decode_route_pass -or
    -not [bool]$productEvidence.prefix_route_pass -or
    -not [bool]$productEvidence.prefix_continuation_pass -or
    -not [bool]$productEvidence.q8192_moe_fixed_route_pass -or
    -not [bool]$productEvidence.acceptance_pass -or
    [int]$productEvidence.prefix_tokens -ne 8191 -or
    [int]$productEvidence.prefix_suffix_tokens -ne 1 -or
    [int]$productEvidence.prefix_hit_count -ne 2) {
    throw 'q8192 product evidence did not use every qualified terminal route without fallback'
}
$qualifiedEnvSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $qualifiedEnv
).Hash.ToLowerInvariant()
if ([string]$productEvidence.environment_sha256 -ne $qualifiedEnvSha256) {
    throw 'qualified environment differs from the q8192 product record'
}
if ([string]$productEvidence.whole_provider_sha256 -ne (
        Get-FileHash -Algorithm SHA256 -LiteralPath $wholeProvider
    ).Hash.ToLowerInvariant()) {
    throw 'whole provider differs from the q8192-qualified product run'
}
if ([string]$productEvidence.attention_provider_sha256 -ne (
        Get-FileHash -Algorithm SHA256 -LiteralPath $attentionProvider
    ).Hash.ToLowerInvariant()) {
    throw 'attention provider differs from the q8192-qualified product run'
}
$attentionBuildProvenanceSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $attentionBuildProvenance
).Hash.ToLowerInvariant()
$attentionBuildEvidence = Get-Content -Raw `
    -LiteralPath $attentionBuildProvenance | ConvertFrom-Json
if ([string]$productEvidence.attention_build_provenance -ine `
        $attentionBuildProvenance -or
    [string]$productEvidence.attention_build_provenance_sha256 -ne `
        $attentionBuildProvenanceSha256 -or
    [int]$productEvidence.attention_dynamic_logical_case_count -ne 32 -or
    [bool]$productEvidence.attention_dynamic_logical_reset_included -or
    -not [bool]$productEvidence.attention_dynamic_logical_component_only -or
    [string]$attentionBuildEvidence.repo_commit -ne $repoCommit -or
    [string]$attentionBuildEvidence.run_label -ne $AttentionBuildLabel -or
    [string]$attentionBuildEvidence.path -ine $attentionProvider -or
    [string]$attentionBuildEvidence.sha256 -ne `
        [string]$productEvidence.attention_provider_sha256 -or
    [int]$attentionBuildEvidence.dynamic_logical_case_count -ne 32 -or
    @($attentionBuildEvidence.dynamic_logical_cases).Count -ne 32 -or
    [bool]$attentionBuildEvidence.dynamic_logical_reset_included -or
    -not [bool]$attentionBuildEvidence.dynamic_logical_component_only -or
    [bool]$attentionBuildEvidence.inference_success_claimed -or
    -not [bool]$attentionBuildEvidence.smoke.completed -or
    [bool]$attentionBuildEvidence.smoke.timed_out -or
    [int]$attentionBuildEvidence.smoke.exit_code -ne 0) {
    throw 'attention dynamic-logical provenance differs from the qualified product run'
}
$expectedAttentionDynamicTokens = @(
    2073, 2156, 2560, 3073, 4609, 6145, 2049, 2175,
    2176, 2177, 2559, 2561, 3071, 3072, 3583, 3584,
    3585, 4095, 4096, 4097, 4607, 4608, 6143, 6144,
    7167, 7168, 7169, 7679, 7680, 7681, 8191, 8192
)
for ($caseIndex = 0; $caseIndex -lt $expectedAttentionDynamicTokens.Count; `
        $caseIndex++) {
    $case = @($attentionBuildEvidence.dynamic_logical_cases)[$caseIndex]
    $coldMs = [double]$case.cold_total_ms
    $submitMs = [double]$case.cold_submit_ms
    $deviceMs = [double]$case.cold_device_completion_ms
    $warmMs = [double]$case.warm_mean_ms
    if ([int]$case.tokens -ne $expectedAttentionDynamicTokens[$caseIndex] -or
        [double]::IsNaN($coldMs) -or [double]::IsInfinity($coldMs) -or
        $coldMs -le 0.0 -or
        [double]::IsNaN($submitMs) -or [double]::IsInfinity($submitMs) -or
        $submitMs -lt 0.0 -or $submitMs -gt $coldMs -or
        [double]::IsNaN($deviceMs) -or [double]::IsInfinity($deviceMs) -or
        $deviceMs -lt 0.0 -or
        [double]::IsNaN($warmMs) -or [double]::IsInfinity($warmMs) -or
        $warmMs -le 0.0 -or [bool]$case.reset_included) {
        throw "attention dynamic-logical timing differs at case $caseIndex"
    }
}
$gdnQualificationSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $gdnQualification
).Hash.ToLowerInvariant()
$gdnEvidence = Get-Content -Raw -LiteralPath $gdnQualification |
    ConvertFrom-Json
if ([string]$productEvidence.gdn_build_provenance -ine $gdnQualification -or
    [string]$productEvidence.gdn_build_provenance_sha256 -ne `
        $gdnQualificationSha256 -or
    [string]$productEvidence.gdn_provider -ine `
        [string]$gdnEvidence.provider -or
    [string]$productEvidence.gdn_provider_sha256 -ne `
        [string]$gdnEvidence.provider_sha256 -or
    [string]$productEvidence.gdn_kernel_dir -ine `
        [string]$gdnEvidence.kernel_dir -or
    [int]$productEvidence.gdn_dynamic_logical_case_count -ne 32 -or
    [int]$productEvidence.gdn_dynamic_logical_mode_count -ne 64 -or
    [bool]$productEvidence.gdn_dynamic_logical_reset_included -or
    -not [bool]$productEvidence.gdn_dynamic_logical_component_only -or
    [string]$gdnEvidence.record_type -ne `
        'qrt_dynamic_logical_gdn_qualification' -or
    [string]$gdnEvidence.host -ine [Environment]::MachineName -or
    [string]$gdnEvidence.repo_commit -ne $repoCommit -or
    [string]$gdnEvidence.run_label -ne $GdnQualificationLabel -or
    [int]$gdnEvidence.dynamic_logical_case_count -ne 32 -or
    [int]$gdnEvidence.dynamic_logical_mode_count -ne 64 -or
    [bool]$gdnEvidence.dynamic_logical_reset_included -or
    -not [bool]$gdnEvidence.component_only -or
    [bool]$gdnEvidence.inference_success_claimed -or
    -not (Test-Path -LiteralPath ([string]$gdnEvidence.provider) `
        -PathType Leaf) -or
    [string]$gdnEvidence.provider_sha256 -ne (
        Get-FileHash -Algorithm SHA256 `
            -LiteralPath ([string]$gdnEvidence.provider)
    ).Hash.ToLowerInvariant() -or
    -not (Test-Path -LiteralPath ([string]$gdnEvidence.kernel_dir) `
        -PathType Container)) {
    throw 'dynamic-logical AITER GDN differs from the qualified product run'
}
$gdnArtifacts = @($gdnEvidence.artifacts)
$productGdnArtifacts = @($productEvidence.gdn_artifacts)
if ($gdnArtifacts.Count -ne 5 -or $productGdnArtifacts.Count -ne 5) {
    throw 'dynamic-logical AITER GDN artifact evidence is incomplete'
}
foreach ($artifact in $gdnArtifacts) {
    $artifactPath = [string]$artifact.path
    if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
        throw "dynamic-logical GDN artifact is missing: $artifactPath"
    }
    $item = Get-Item -LiteralPath $artifactPath
    $actualSha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $item.FullName).Hash.ToLowerInvariant()
    $productArtifact = @(
        $productGdnArtifacts | Where-Object {
            [string]$_.path -ieq $item.FullName
        }
    )
    if ([uint64]$artifact.bytes -ne [uint64]$item.Length -or
        [string]$artifact.sha256 -ne $actualSha256 -or
        $productArtifact.Count -ne 1 -or
        [uint64]$productArtifact[0].bytes -ne [uint64]$item.Length -or
        [string]$productArtifact[0].sha256 -ne $actualSha256) {
        throw "dynamic-logical GDN artifact differs: $artifactPath"
    }
}

$dynamicMoeEvidence = Get-Content -Raw -LiteralPath $dynamicMoeQualification |
    ConvertFrom-Json
if ([string]$dynamicMoeEvidence.record_type -ne `
        'qrt_dynamic_logical_q8192_moe_build' -or
    [string]$dynamicMoeEvidence.host -ine [Environment]::MachineName -or
    [string]$dynamicMoeEvidence.repo_commit -ne $repoCommit -or
    [string]$dynamicMoeEvidence.run_label -ne $DynamicMoeBuildLabel -or
    [string]$dynamicMoeEvidence.command_file -ine $dynamicMoeBuildScript -or
    [string]$dynamicMoeEvidence.command_file_sha256 -ne (
        Get-FileHash -Algorithm SHA256 -LiteralPath $dynamicMoeBuildScript
    ).Hash.ToLowerInvariant() -or
    [string]$dynamicMoeEvidence.builder -ine $dynamicMoeBuilderScript -or
    [string]$dynamicMoeEvidence.builder_sha256 -ne (
        Get-FileHash -Algorithm SHA256 -LiteralPath $dynamicMoeBuilderScript
    ).Hash.ToLowerInvariant() -or
    [string]$dynamicMoeEvidence.provider -ine $dynamicMoeProvider -or
    @($dynamicMoeEvidence.PSObject.Properties.Name) -notcontains `
        'expected_full_provider_hash_diagnostic_only' -or
    -not [bool]$dynamicMoeEvidence.expected_full_provider_hash_diagnostic_only -or
    -not [bool]$dynamicMoeEvidence.dynamic_logical_pass -or
    -not [bool]$dynamicMoeEvidence.dynamic_logical_q8192_exact_pass -or
    [int]$dynamicMoeEvidence.dynamic_logical_case_count -ne 32 -or
    [int]$dynamicMoeEvidence.dynamic_logical_timing_samples -ne 3 -or
    [string]$dynamicMoeEvidence.dynamic_logical_timing_stat -ne 'median' -or
    @($dynamicMoeEvidence.PSObject.Properties.Name) -notcontains `
        'dynamic_logical_nonfinite' -or
    [int]$dynamicMoeEvidence.dynamic_logical_nonfinite -ne 0 -or
    @($dynamicMoeEvidence.PSObject.Properties.Name) -notcontains `
        'dynamic_logical_component_only' -or
    -not [bool]$dynamicMoeEvidence.dynamic_logical_component_only -or
    @($dynamicMoeEvidence.PSObject.Properties.Name) -notcontains `
        'inference_success_claimed' -or
    [bool]$dynamicMoeEvidence.inference_success_claimed -or
    @($dynamicMoeEvidence.PSObject.Properties.Name) -notcontains `
        'dynamic_logical_reset_included' -or
    [bool]$dynamicMoeEvidence.dynamic_logical_reset_included -or
    [int]$dynamicMoeEvidence.dynamic_logical_min_tokens -ne 2049 -or
    [int]$dynamicMoeEvidence.dynamic_logical_max_tokens -ne 8192 -or
    [int]$dynamicMoeEvidence.provider_backend_mask -ne 15) {
    throw 'dynamic-logical q8192 MoE provenance does not qualify this service'
}
$dynamicMoeEvidencePropertyNames = @(
    $dynamicMoeEvidence.PSObject.Properties.Name
)
$recordedDynamicMoeNativeSharedSelected =
    if ($dynamicMoeEvidencePropertyNames -contains 'native_shared_selected') {
        [bool]$dynamicMoeEvidence.native_shared_selected
    } else {
        $false
    }
$recordedDynamicMoeAotReused =
    if ($dynamicMoeEvidencePropertyNames -contains 'aot_reused') {
        [bool]$dynamicMoeEvidence.aot_reused
    } else {
        $false
    }
$recordedDynamicMoeBaseAotMode =
    if ($dynamicMoeEvidencePropertyNames -contains 'base_aot_mode') {
        [string]$dynamicMoeEvidence.base_aot_mode
    } else {
        ''
    }
$recordedDynamicMoeAuxiliaryKernelFiles =
    if ($dynamicMoeEvidencePropertyNames -contains 'auxiliary_kernel_files') {
        @($dynamicMoeEvidence.auxiliary_kernel_files)
    } else {
        @()
    }
$recordedDynamicMoeRuntimeEnvironment =
    if ($dynamicMoeEvidencePropertyNames -contains
            'required_runtime_environment') {
        @($dynamicMoeEvidence.required_runtime_environment)
    } else {
        @()
    }
if ($recordedDynamicMoeNativeSharedSelected -ne
        [bool]$DynamicMoeNativeSharedSelected -or
    ($DynamicMoeNativeSharedSelected -and (
        $recordedDynamicMoeAotReused -or
        $recordedDynamicMoeBaseAotMode -cne
            'generated_from_current_source' -or
        [int]$dynamicMoeEvidence.base_aot_block_m -ne 64 -or
        [int]$dynamicMoeEvidence.base_aot_group_m -ne 1 -or
        -not [bool]$dynamicMoeEvidence.base_aot_dynamic_logical_abi)) -or
    (-not $DynamicMoeNativeSharedSelected -and
        $recordedDynamicMoeAotReused) -or
    ($recordedDynamicMoeAuxiliaryKernelFiles -join ',') -cne
        ($expectedDynamicMoeAuxiliaryKernelFiles -join ',')) {
    throw 'dynamic-logical q8192 MoE selected-route provenance differs'
}
if ($DynamicMoeNativeSharedSelected -and (
        [string]$productEvidence.dynamic_moe_base_aot_mode -cne
            $recordedDynamicMoeBaseAotMode -or
        [int]$productEvidence.dynamic_moe_base_aot_block_m -ne
            [int]$dynamicMoeEvidence.base_aot_block_m -or
        [int]$productEvidence.dynamic_moe_base_aot_group_m -ne
            [int]$dynamicMoeEvidence.base_aot_group_m -or
        [bool]$productEvidence.dynamic_moe_base_aot_dynamic_logical_abi -ne
            [bool]$dynamicMoeEvidence.base_aot_dynamic_logical_abi)) {
    throw 'q8192 product base-AOT provenance differs from qualification'
}
Assert-DynamicMoeRuntimeEnvironmentRecords `
    -Records $recordedDynamicMoeRuntimeEnvironment `
    -Source 'dynamic-logical q8192 MoE qualification'
$dynamicMoeInputHashes = [ordered]@{
    generator_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $dynamicMoeGenerator
    ).Hash.ToLowerInvariant()
    provider_source_sha256 = (
        Get-FileHash -Algorithm SHA256 `
            -LiteralPath $dynamicMoeProviderSource
    ).Hash.ToLowerInvariant()
    smoke_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $dynamicMoeSmokeSource
    ).Hash.ToLowerInvariant()
}
foreach ($name in $dynamicMoeInputHashes.Keys) {
    if ([string]$dynamicMoeEvidence.$name -ne `
        [string]$dynamicMoeInputHashes[$name]) {
        throw "dynamic-logical q8192 MoE build input hash differs: $name"
    }
}
$dynamicMoeQualificationSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $dynamicMoeQualification
).Hash.ToLowerInvariant()
if ([string]$productEvidence.q8192_moe_build_provenance_sha256 -ne `
        $dynamicMoeQualificationSha256 -or
    [string]$productEvidence.q8192_moe_provider -ine $dynamicMoeProvider -or
    [string]$productEvidence.q8192_moe_provider_sha256 -ne (
        Get-FileHash -Algorithm SHA256 -LiteralPath $dynamicMoeProvider
    ).Hash.ToLowerInvariant() -or
    -not [bool]$productEvidence.q8192_moe_dynamic_logical_component_pass -or
    -not [bool]$productEvidence.q8192_moe_dynamic_logical_q8192_exact_pass -or
    [int]$productEvidence.q8192_moe_dynamic_logical_case_count -ne 32 -or
    -not [bool]$productEvidence.q8192_moe_expected_full_provider_hash_diagnostic_only -or
    [int]$productEvidence.q8192_moe_dynamic_logical_timing_samples -ne 3 -or
    [string]$productEvidence.q8192_moe_dynamic_logical_timing_stat -ne `
        'median' -or
    [int]$productEvidence.q8192_moe_dynamic_logical_nonfinite -ne 0 -or
    -not [bool]$productEvidence.q8192_moe_dynamic_logical_component_only -or
    [bool]$productEvidence.q8192_moe_dynamic_logical_reset_included) {
    throw 'dynamic-logical q8192 MoE differs from the qualified product run'
}
$dynamicMoeRuntimeRecords = @(
    foreach ($artifact in @($dynamicMoeEvidence.artifacts)) {
        $artifactPath = Join-Path $dynamicMoeRoot ([string]$artifact.file)
        if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
            throw "dynamic-logical q8192 MoE artifact is missing: $artifactPath"
        }
        $item = Get-Item -LiteralPath $artifactPath
        $actual = [ordered]@{
            file = $item.Name
            bytes = [uint64]$item.Length
            sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $item.FullName
            ).Hash.ToLowerInvariant()
        }
        if ([uint64]$artifact.bytes -ne [uint64]$actual.bytes -or
            [string]$artifact.sha256 -ne [string]$actual.sha256) {
            throw "dynamic-logical q8192 MoE artifact hash differs: $($item.Name)"
        }
        $actual
    }
)
$qualifiedDynamicArtifacts = @($productEvidence.q8192_moe_artifacts)
$expectedDynamicMoeArtifactCount = 8 +
    $expectedDynamicMoeAuxiliaryKernelFiles.Count
if ($dynamicMoeRuntimeRecords.Count -ne $expectedDynamicMoeArtifactCount -or
    $qualifiedDynamicArtifacts.Count -ne $expectedDynamicMoeArtifactCount) {
    throw 'dynamic-logical q8192 MoE artifact evidence is incomplete'
}
foreach ($actual in $dynamicMoeRuntimeRecords) {
    $qualified = @(
        $qualifiedDynamicArtifacts | Where-Object {
            [string]$_.file -ceq [string]$actual.file
        }
    )
    if ($qualified.Count -ne 1 -or
        [uint64]$qualified[0].bytes -ne [uint64]$actual.bytes -or
        [string]$qualified[0].sha256 -ne [string]$actual.sha256) {
        throw "dynamic-logical q8192 MoE product artifact differs: $($actual.file)"
    }
}

$environment = @{}
foreach ($line in Get-Content -LiteralPath $qualifiedEnv) {
    $trimmed = $line.Trim()
    if ($trimmed.Length -eq 0 -or $trimmed.StartsWith('#')) {
        continue
    }
    $equals = $trimmed.IndexOf('=')
    if ($equals -le 0) {
        throw "invalid qualified environment entry: $line"
    }
    $name = $trimmed.Substring(0, $equals)
    if ($environment.ContainsKey($name)) {
        throw "duplicate qualified environment entry: $name"
    }
    $environment[$name] = $trimmed.Substring($equals + 1)
}
$requiredEnvironment = [ordered]@{
    QRT_PREFILL_DESCRIPTOR_BATCH_LAYER39_Q1_KV8192 = '1'
    QRT_PREFILL_DESCRIPTOR_BATCH_LAYER39_Q1_CK_FMHA = '1'
    QRT_PREFILL_DESCRIPTOR_BATCH_LAYER39_Q1_TRITON_0626_ROUTED = '1'
    QRT_PREFILL_DESCRIPTOR_BATCH_Q1_TERMINAL_DEVICE_CORRIDOR = '1'
    QRT_PREFILL_DESCRIPTOR_BATCH_RESIDENT_BF16_MATRIX_PROVIDER = '1'
    QRT_PREFILL_DESCRIPTOR_BATCH_RESIDENT_LAYER_STACK_CARRIER = '1'
    QRT_PREFILL_DESCRIPTOR_BATCH_RESIDENT_LAYER_STACK_SOURCE_LAYOUT = '1'
    QRT_PREFILL_DESCRIPTOR_BATCH_TRUST_EXACT_ROUTED_GPU_HANDOFF = '1'
    QRT_PREFILL_DESCRIPTOR_BATCH_REQUIRED_MARKERS_ONLY = '1'
    QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_AITER_FUSED_GDN_PROVIDER = '1'
    QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_AITER_FUSED_GDN_MIN_LAYER = '0'
    QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_AITER_FUSED_GDN_MAX_LAYER = '38'
    QRT_PREFILL_DESCRIPTOR_BATCH_SELECTED_MOE_FAST_ARITHMETIC_BACKEND = '0'
    QRT_PREFILL_DESCRIPTOR_BATCH_ROUTED_CPU_REFERENCE_ARITHMETIC = '0'
    QRT_QWEN36_DYNAMIC_LOGICAL_MOE_PROVIDER = '0'
    QRT_QWEN36_WHOLE_PROVIDER_ARBITRARY_CONTEXT = '1'
    QRT_QWEN36_EXACT_ARBITRARY_RETAINED_Q8192 = '1'
    QRT_QWEN36_EXACT_ARBITRARY_PRODUCT_PATH = '1'
    QRT_QWEN36_LAYER39_DYNAMIC_TERMINAL_COMPACT_Q = '1'
    QRT_QWEN36_LAYER39_DYNAMIC_TERMINAL_PACKED_MOE = '1'
    QRT_QWEN36_LAYER39_DYNAMIC_TERMINAL_DEVICE_CORRIDOR = '1'
}
foreach ($name in $requiredEnvironment.Keys) {
    if (-not $environment.ContainsKey($name) -or
        [string]$environment[$name] -ne [string]$requiredEnvironment[$name]) {
        throw "qualified environment does not activate $name"
    }
}
if ([string]$environment[
        'QRT_PREFILL_DESCRIPTOR_BATCH_FULL_ATTENTION_CK_FMHA_DLL'
    ] -ne $attentionProvider) {
    throw 'qualified environment does not bind the q8192-qualified CK DLL'
}
if ($GdnRoute -eq 'aiter' -and
    ([string]$environment[
        'QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_AITER_FUSED_GDN_DLL'
    ] -ine [string]$gdnEvidence.provider -or
     [string]$environment[
        'QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_AITER_FUSED_GDN_KERNEL_DIR'
     ] -ine [string]$gdnEvidence.kernel_dir)) {
    throw 'qualified environment does not bind the dynamic-logical AITER GDN artifacts'
}

$qualifiedDynamicMoeProvider = [string]$environment[
    'QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_TRITON_SELECTED_MOE_DLL'
]
$qualifiedDynamicMoeKernelDir = [string]$environment[
    'QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_TRITON_SELECTED_MOE_KERNEL_DIR'
]
if ($qualifiedDynamicMoeProvider -ine $dynamicMoeProvider -or
    $qualifiedDynamicMoeKernelDir -ine $dynamicMoeRoot) {
    throw 'qualified environment does not bind the dynamic-logical q8192 MoE build'
}
foreach ($entry in $expectedDynamicMoeRuntimeEnvironment.GetEnumerator()) {
    $name = [string]$entry.Key
    if (-not $environment.ContainsKey($name) -or
        [string]$environment[$name] -cne [string]$entry.Value) {
        throw "qualified environment does not bind selected-MoE numeric route: $name"
    }
}

$environment['QRT_QWEN36_DYNAMIC_LOGICAL_MOE_PROVIDER'] = '1'
$environment['QRT_QWEN36_SMOOTH_TAIL_MOE_PROVIDER'] = '0'
$environment['QRT_QWEN36_SMOOTH_TAIL_DENSE_CEIL_PROVIDER'] = '0'
$environment['QRT_QWEN36_SMOOTH_TAIL_SINGLE_CEIL_PROVIDER'] = '0'
$environment['QRT_QWEN36_SMOOTH_TAIL_BOUNDED_TRANSACTIONS'] = '0'
$environment['QRT_QWEN36_SMOOTH_TAIL_PARALLEL_BASE'] = '0'
$environment['QRT_QWEN36_SMOOTH_TAIL_PARALLEL_TRANSACTIONS'] = '0'
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_SHORT_WEIGHT_INT8'] = '0'
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_SHORT_LOSSLESS_PALETTE'] = '0'
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_SHORT_LOSSLESS_ROW_PALETTE'] = '0'
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_SHORT_LOSSLESS_ROW_PALETTE_GATE_ONLY'] = '0'
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_LOSSLESS_ROW_PALETTE_REPLACE_RAW'] = '0'
$environment['QRT_SERVER_EXACT_FIRST_TOKEN_PREFILL'] = '1'
$environment['QRT_SERVER_EXACT_LETTER_CLASSIFIER'] = '0'
$environment['QRT_SERVER_PREFIX_CACHE'] = '0'
$dynamicEnvironmentKeys = @(
    'QRT_QWEN36_DYNAMIC_LOGICAL_MOE_PROVIDER',
    'QRT_QWEN36_SMOOTH_TAIL_MOE_PROVIDER',
    'QRT_QWEN36_SMOOTH_TAIL_DENSE_CEIL_PROVIDER',
    'QRT_QWEN36_SMOOTH_TAIL_SINGLE_CEIL_PROVIDER',
    'QRT_QWEN36_SMOOTH_TAIL_BOUNDED_TRANSACTIONS',
    'QRT_QWEN36_SMOOTH_TAIL_PARALLEL_BASE',
    'QRT_QWEN36_SMOOTH_TAIL_PARALLEL_TRANSACTIONS',
    'QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_SHORT_WEIGHT_INT8',
    'QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_SHORT_LOSSLESS_PALETTE',
    'QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_SHORT_LOSSLESS_ROW_PALETTE',
    'QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_SHORT_LOSSLESS_ROW_PALETTE_GATE_ONLY',
    'QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_LOSSLESS_ROW_PALETTE_REPLACE_RAW'
) + @(
    $expectedDynamicMoeRuntimeEnvironment.Keys | ForEach-Object {
        [string]$_
    }
)
$coldPrefillEnvironmentKeys = @(
    'QRT_SERVER_EXACT_FIRST_TOKEN_PREFILL',
    'QRT_SERVER_EXACT_LETTER_CLASSIFIER',
    'QRT_SERVER_PREFIX_CACHE'
)

$listeners = @(
    Get-NetTCPConnection -State Listen -LocalPort $Port `
        -ErrorAction SilentlyContinue
)
if ($listeners.Count -ne 0) {
    throw "port $Port is already occupied"
}

[void](New-Item -ItemType Directory -Path $serviceDir)
$serviceEnv = Join-Path $serviceDir 'runtime.env'
$serviceEnvironmentLines = @(
    foreach ($name in @($environment.Keys | Sort-Object)) {
        "$name=$($environment[$name])"
    }
)
[IO.File]::WriteAllText(
    $serviceEnv,
    ($serviceEnvironmentLines -join [Environment]::NewLine) +
        [Environment]::NewLine,
    $utf8
)
$arguments = @(
    'start',
    '--model', $modelPath,
    '--provider', $wholeProvider,
    '--arbitrary-moe-provider', $arbitraryMoeProvider,
    '--arbitrary-moe-kernel-dir', $arbitraryMoeKernelDir,
    '--smooth-tail-moe-root', $smoothTailRoot,
    '--env-file', $serviceEnv,
    '--model-id', 'qwen3.6-35b-a3b',
    '--host', '127.0.0.1',
    '--port', [string]$Port,
    '--max-model-len', '262144',
    '--max-queue-depth', '64',
    '--queue-timeout-seconds', '600',
    '--state-file', $statePath,
    '--log-file', $logPath,
    '--wait-seconds', '120'
)
$serviceStartWatch = [Diagnostics.Stopwatch]::StartNew()
& $serviceExe @arguments
$serviceStartExitCode = $LASTEXITCODE
$serviceStartWatch.Stop()
$serviceStartWallMs = [Math]::Round(
    $serviceStartWatch.Elapsed.TotalMilliseconds,
    6
)
if ($serviceStartExitCode -ne 0) {
    throw "random-length qrt start exited $serviceStartExitCode"
}
$state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
if ($state.status -ne 'ready') {
    throw "random-length service is not ready: $($state.status)"
}
$readyListeners = @(
    Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop |
        Where-Object { [uint32]$_.OwningProcess -eq [uint32]$state.pid }
)
if ($readyListeners.Count -ne 1) {
    throw "random-length service listener count differs: $($readyListeners.Count)"
}
$serviceNativeLoadMs = $null
$serviceTokenizerMs = $null
if ([string]$state.message -match `
        'tokenizer_ms=(?<tokenizer>[0-9]+(?:\.[0-9]+)?); native_load_ms=(?<native>[0-9]+(?:\.[0-9]+)?)') {
    $serviceTokenizerMs = [double]::Parse(
        $Matches['tokenizer'],
        [Globalization.CultureInfo]::InvariantCulture
    )
    $serviceNativeLoadMs = [double]::Parse(
        $Matches['native'],
        [Globalization.CultureInfo]::InvariantCulture
    )
}
$serviceModelEngineLoadPass = `
    $serviceStartWallMs -le $modelEngineLoadMaxMs
if (-not $serviceModelEngineLoadPass) {
    & $serviceExe stop --state-file $statePath --wait-seconds 30
    $serviceStopExitCode = $LASTEXITCODE
    if ($serviceStopExitCode -ne 0) {
        throw "random-length service load took $serviceStartWallMs ms and stop failed with $serviceStopExitCode"
    }
    throw "random-length service model-plus-engine load took $serviceStartWallMs ms, above $modelEngineLoadMaxMs ms"
}

$startRecordPath = Join-Path $serviceDir 'start-record.json'
$startRecord = [ordered]@{
    schema_version = 1
    record_type = 'qrt_random_length_terminal_service_start'
    host = [Environment]::MachineName
    repo_commit = $repoCommit
    dirty_tree = @(& git -C $Repo status --porcelain).Count -ne 0
    command_file = $PSCommandPath
    command = @($serviceExe) + $arguments
    model_path = $modelPath
    model_id = 'qwen3.6-35b-a3b'
    gdn_route = $GdnRoute
    dynamic_moe_native_shared_selected = `
        [bool]$DynamicMoeNativeSharedSelected
    dynamic_moe_base_aot_mode = $recordedDynamicMoeBaseAotMode
    dynamic_moe_base_aot_block_m = `
        [int]$dynamicMoeEvidence.base_aot_block_m
    dynamic_moe_base_aot_group_m = `
        [int]$dynamicMoeEvidence.base_aot_group_m
    dynamic_moe_base_aot_dynamic_logical_abi = `
        [bool]$dynamicMoeEvidence.base_aot_dynamic_logical_abi
    pid = [uint32]$state.pid
    address = [string]$state.address
    state = $statePath
    log = $logPath
    q8192_product_record = $productRecord
    q8192_product_record_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $productRecord
    ).Hash.ToLowerInvariant()
    q8192_gb10_oracle = $gb10OraclePath
    q8192_gb10_oracle_sha256 = $gb10OracleSha256
    q8192_gb10_oracle_snapshot = $productOracleSnapshot
    q8192_gb10_oracle_capture_request_sha256 = `
        [string]$gb10Oracle.capture.request_sha256
    q8192_gb10_oracle_capture_response_sha256 = `
        [string]$gb10Oracle.capture.response_sha256
    q8192_gb10_oracle_capture_command_file_sha256 = `
        [string]$gb10Oracle.capture.command_file_sha256
    q8192_gb10_continuation_capture_request_sha256 = `
        [string]$gb10Oracle.continuation_capture.request_sha256
    q8192_gb10_continuation_capture_response_sha256 = `
        [string]$gb10Oracle.continuation_capture.response_sha256
    q8192_gb10_continuation_capture_command_file_sha256 = `
        $gb10ContinuationCaptureScriptSha256
    q8192_expected_output_token_ids_u32le_sha256 = `
        [string]$gb10Oracle.expected.output_token_ids_u32le_sha256
    q8192_expected_output_token_ids_u32le_fnv1a64 = `
        [string]$gb10Oracle.expected.output_token_ids_u32le_fnv1a64
    q8192_continuation_token_count = `
        [int]$productEvidence.continuation_token_count
    q8192_decode_step_count = [int]$productEvidence.decode_step_count
    engine_self_hashes_diagnostic_only = `
        [bool]$productEvidence.engine_self_hashes_diagnostic_only
    q8192_correctness_pass = [bool]$productEvidence.correctness_pass
    q8192_continuation_token_for_token_pass = `
        [bool]$productEvidence.continuation_token_for_token_pass
    q8192_decode_route_pass = [bool]$productEvidence.decode_route_pass
    q8192_prefix_continuation_pass = `
        [bool]$productEvidence.prefix_continuation_pass
    q8192_prefix_continuation_correctness_pass = `
        [bool]$productEvidence.prefix_continuation_correctness_pass
    q8192_prefix_continuation_performance_pass = `
        [bool]$productEvidence.prefix_continuation_performance_pass
    q8192_prefix_route_pass = [bool]$productEvidence.prefix_route_pass
    q8192_prefix_tokens = [int]$productEvidence.prefix_tokens
    q8192_prefix_suffix_tokens = `
        [int]$productEvidence.prefix_suffix_tokens
    q8192_prefix_hit_count = [int]$productEvidence.prefix_hit_count
    q8192_performance_pass = [bool]$productEvidence.performance_pass
    q8192_model_engine_load_pass = `
        [bool]$productEvidence.model_engine_load_pass
    q8192_route_pass = [bool]$productEvidence.route_pass
    q8192_moe_fixed_route_pass = `
        [bool]$productEvidence.q8192_moe_fixed_route_pass
    q8192_first_token = [int]$productEvidence.gb10_first_token
    q8192_first_token_raw_logit = `
        [double]$productEvidence.gb10_first_token_raw_logit
    q8192_first_token_raw_logit_absolute_difference = `
        [double]$productEvidence.first_token_raw_logit_absolute_difference
    q8192_first_token_raw_logit_tolerance = `
        [double]$productEvidence.first_token_raw_logit_tolerance
    q8192_prefill_tokens_per_second = `
        [double]$productEvidence.q8192_prefill_tokens_per_second
    q8192_prefill_metric_relative_error = `
        [double]$productEvidence.q8192_prefill_metric_relative_error
    q8192_prefill_metric_relative_error_max = `
        [double]$productEvidence.q8192_prefill_metric_relative_error_max
    q8192_prefill_min_tokens_per_second = `
        [double]$productEvidence.q8192_prefill_min_tokens_per_second
    q8192_ttft_ms = [double]$productEvidence.q8192_ttft_ms
    q8192_ttft_max_ms = [double]$productEvidence.q8192_ttft_max_ms
    q8192_decode_tokens_per_second = `
        [double]$productEvidence.q8192_decode_tokens_per_second
    q8192_decode_min_tokens_per_second = `
        [double]$productEvidence.q8192_decode_min_tokens_per_second
    q8192_tpot_ms = [double]$productEvidence.q8192_tpot_ms
    q8192_tpot_max_ms = [double]$productEvidence.q8192_tpot_max_ms
    q8192_prefix_decode_tokens_per_second = `
        [double]$productEvidence.prefix_decode_tokens_per_second
    q8192_prefix_tpot_ms = [double]$productEvidence.prefix_tpot_ms
    q8192_prefix_model_engine_load_ms = `
        [double]$productEvidence.prefix_model_engine_load_ms
    q8192_model_engine_load_ms = `
        [double]$productEvidence.model_engine_load_ms
    model_engine_load_ms = $serviceStartWallMs
    model_engine_load_max_ms = $modelEngineLoadMaxMs
    model_engine_load_pass = $serviceModelEngineLoadPass
    native_load_ms = $serviceNativeLoadMs
    tokenizer_load_ms = $serviceTokenizerMs
    qualified_environment = $qualifiedEnv
    qualified_environment_sha256 = $qualifiedEnvSha256
    service_environment = $serviceEnv
    service_environment_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $serviceEnv
    ).Hash.ToLowerInvariant()
    dynamic_logical_moe_provider = $true
    q8192_moe_root = $dynamicMoeRoot
    q8192_moe_build_provenance = $dynamicMoeQualification
    q8192_moe_build_provenance_sha256 = `
        $dynamicMoeQualificationSha256
    q8192_moe_provider = $dynamicMoeProvider
    q8192_moe_provider_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $dynamicMoeProvider
    ).Hash.ToLowerInvariant()
    q8192_moe_dynamic_logical_component_pass = `
        [bool]$dynamicMoeEvidence.dynamic_logical_pass
    q8192_moe_dynamic_logical_q8192_exact_pass = `
        [bool]$dynamicMoeEvidence.dynamic_logical_q8192_exact_pass
    q8192_moe_dynamic_logical_case_count = `
        [int]$dynamicMoeEvidence.dynamic_logical_case_count
    q8192_moe_expected_full_provider_hash_diagnostic_only = `
        [bool]$dynamicMoeEvidence.expected_full_provider_hash_diagnostic_only
    q8192_moe_dynamic_logical_timing_samples = `
        [int]$dynamicMoeEvidence.dynamic_logical_timing_samples
    q8192_moe_dynamic_logical_timing_stat = `
        [string]$dynamicMoeEvidence.dynamic_logical_timing_stat
    q8192_moe_dynamic_logical_nonfinite = `
        [int]$dynamicMoeEvidence.dynamic_logical_nonfinite
    q8192_moe_dynamic_logical_component_only = `
        [bool]$dynamicMoeEvidence.dynamic_logical_component_only
    q8192_moe_dynamic_logical_reset_included = `
        [bool]$dynamicMoeEvidence.dynamic_logical_reset_included
    q8192_moe_required_runtime_environment = `
        $expectedDynamicMoeRuntimeEnvironmentRecords
    q8192_moe_environment = $dynamicEnvironmentKeys
    q8192_moe_artifacts = $dynamicMoeRuntimeRecords
    whole_provider_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $wholeProvider
    ).Hash.ToLowerInvariant()
    attention_provider_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $attentionProvider
    ).Hash.ToLowerInvariant()
    attention_build_label = $AttentionBuildLabel
    attention_build_provenance = $attentionBuildProvenance
    attention_build_provenance_sha256 = `
        $attentionBuildProvenanceSha256
    attention_dynamic_logical_case_count = `
        [int]$attentionBuildEvidence.dynamic_logical_case_count
    attention_dynamic_logical_reset_included = `
        [bool]$attentionBuildEvidence.dynamic_logical_reset_included
    attention_dynamic_logical_component_only = `
        [bool]$attentionBuildEvidence.dynamic_logical_component_only
    gdn_qualification_label = $GdnQualificationLabel
    gdn_build_provenance = $gdnQualification
    gdn_build_provenance_sha256 = $gdnQualificationSha256
    gdn_provider = [string]$gdnEvidence.provider
    gdn_provider_sha256 = [string]$gdnEvidence.provider_sha256
    gdn_kernel_dir = [string]$gdnEvidence.kernel_dir
    gdn_dynamic_logical_case_count = `
        [int]$gdnEvidence.dynamic_logical_case_count
    gdn_dynamic_logical_mode_count = `
        [int]$gdnEvidence.dynamic_logical_mode_count
    gdn_dynamic_logical_reset_included = `
        [bool]$gdnEvidence.dynamic_logical_reset_included
    gdn_dynamic_logical_component_only = `
        [bool]$gdnEvidence.component_only
    gdn_artifacts = $productGdnArtifacts
    server_prefix_cache_enabled = $false
    exact_first_token_prefill = $true
    exact_letter_classifier = $false
    cold_prefix_contract = `
        'server_prefix_cache_disabled_and_globally_unique_first_token_id'
    cold_prefill_environment = $coldPrefillEnvironmentKeys
    dynamic_terminal_contract = @($requiredEnvironment.Keys)
    route_log_audit = $routeLogAudit
    route_log_required_markers = @(
        'full_attention_ck_q1_dynamic',
        'full_attention_ck_q1_kv8192',
        'q8192_aiter_fused_gdn_provider',
        'q8192_triton_selected_moe_full_provider_v2',
        'layer39_q1_triton_0626_routed_backend',
        'q1_terminal_device_corridor_activate',
        'q1_terminal_device_corridor_triton_metadata_upload',
        'q1_terminal_device_corridor_output_activate',
        'q1_terminal_device_corridor_final_norm_activate',
        'exact_first_token_prefill'
    )
    route_log_forbidden_markers = @(
        'prefix_cache_seed',
        'prefix_cache_hit'
    )
}
[IO.File]::WriteAllText(
    $startRecordPath,
    ($startRecord | ConvertTo-Json -Depth 8) + "`n",
    $utf8
)
$startRecord | ConvertTo-Json -Depth 8 -Compress
