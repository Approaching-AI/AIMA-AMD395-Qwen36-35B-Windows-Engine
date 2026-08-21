param(
    [string]$Repo = 'D:\projects\AIMA-dynamic-logical-r1',
    [ValidatePattern('^r[0-9]+$')]
    [string]$RunLabel = 'r1089',
    [ValidatePattern('^r[0-9]+$')]
    [string]$ProductBuildLabel = 'r1075',
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
    [ValidateSet('fast', 'exact', 'triton')]
    [string]$MoeArithmetic = 'triton',
    [ValidateSet('native', 'ck')]
    [string]$Q1Attention = 'ck',
    [switch]$LayerOutputTrace
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$releaseRoot = 'D:\projects\AIMA-AMD395-Qwen36-35B-Windows-Engine-v1.0.1-release-2bf0457\build\runtime-v1.0.1-2bf0457'
$serviceExe = Join-Path $releaseRoot 'engine\qrt.exe'
$productExe = Join-Path $Repo `
    "build\product-cli-$ProductBuildLabel\qrt-product.exe"
$modelPath = 'D:\models\Qwen3.6-35B-A3B'
$gb10OraclePath = Join-Path $Repo `
    'contracts\hprefill_q8192_gb10_oracle.json'
$gb10CaptureScriptPath = Join-Path $Repo `
    'scripts\capture_gb10_q8192_oracle.py'
$gb10ContinuationCaptureScriptPath = Join-Path $Repo `
    'scripts\capture_gb10_q8192_continuation.py'
$promptGeneratorPath = Join-Path $Repo `
    'scripts\.candidate_generate_random_prompt.py'
$q1MoeModuleDir = Join-Path $releaseRoot 'aot\gfx1151'
$q1MoeGateUpPath = Join-Path $q1MoeModuleDir `
    'q1_triton_0626_gate_up_silu_w8.hsaco'
$q1MoeDownPath = Join-Path $q1MoeModuleDir `
    'q1_triton_0626_down_sum.hsaco'
$q8192PrefillMinTokensPerSecond = 1506.407
$q8192TtftMaxMs = 4187.416
$q8192DecodeMinTokensPerSecond = 28.168
$q8192TpotMaxMs = 35.502
$modelEngineLoadMaxMs = 30000.0
$prefixTokens = 8191
$prefixSuffixTokens = 1
$prefixHitCount = 2
$wholeProvider = Join-Path $Repo `
    "build\whole-provider-fla-boundary-contract-$WholeProviderBuildLabel\qrt_qwen36_whole_provider.dll"
$attentionRoot = Join-Path $Repo `
    "build\ck-fmha-q1-terminal-exact-$AttentionBuildLabel"
$attentionProvider = Join-Path $attentionRoot `
    'qrt_ck_fmha_q1_terminal_exact.dll'
$attentionBuildProvenance = Join-Path $attentionRoot `
    'build-provenance.json'
$attentionBuildScript = Join-Path $Repo `
    'scripts\.candidate_build_ck_fmha_q1_terminal_exact_r1087.ps1'
$attentionProviderSource = Join-Path $Repo `
    'native\providers\ck_fmha\qrt_ck_fmha_q8192_provider.cpp'
$attentionApiSource = Join-Path $Repo `
    'native\providers\ck_fmha\fmha_fwd_api.cpp'
$attentionInstanceSource = Join-Path $Repo `
    'native\providers\ck_fmha\fmha_fwd_gfx1151_d256_bf16_f32out.cpp'
$attentionSmokeSource = Join-Path $Repo `
    'native\providers\ck_fmha\q8192_ck_fmha_direct_smoke.cpp'
$attentionAccumulatorHeader = Join-Path $Repo `
    'native\providers\moe_accumulator\q1_moe_hawkeye_bf16_accumulator.h'
$dynamicMoeRoot = Join-Path $Repo `
    "build\dynamic-logical-moe-$DynamicMoeBuildLabel"
$dynamicMoeProvider = Join-Path $dynamicMoeRoot `
    'qrt_triton_moe_q8192_provider.dll'
$dynamicMoeMetadata = Join-Path $dynamicMoeRoot 'metadata.json'
$dynamicMoeQualification = Join-Path $dynamicMoeRoot `
    'qualification-provenance.json'
$dynamicMoeProviderSource = Join-Path $Repo `
    'native\providers\triton_moe\qrt_triton_moe_q8192_provider.cpp'
$dynamicMoeGeneratorSource = Join-Path $Repo `
    'native\generators\compile_q8192_triton_selected_moe.py'
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
$gdnQualificationScript = Join-Path $Repo `
    'scripts\.candidate_qualify_dynamic_logical_gdn_r1094.ps1'
$gdnSmokeSource = Join-Path $Repo `
    'native\providers\gdn\q8192_aiter_fused_gdn_smoke.cpp'
$statePath = Join-Path $Repo `
    'build\service-q8192-fla-boundary-stage-trace-r1074\service.json'
$sourceCommandPath = Join-Path $Repo `
    'build\service-clean-ck-terminal-exact-r1055\source-commandline.txt'
$baseEnvPath = Join-Path $Repo 'engine\runtime.env'
$outputDir = Join-Path $Repo `
    "build\product-q8192-$GdnRoute-boundary-$MoeArithmetic-$Q1Attention-$RunLabel"
if (Test-Path -LiteralPath $outputDir) {
    throw "refusing to overwrite $RunLabel product run: $outputDir"
}
foreach ($required in @(
        $serviceExe,
        $productExe,
        $wholeProvider,
        $attentionProvider,
        $attentionBuildProvenance,
        $attentionBuildScript,
        $attentionProviderSource,
        $attentionApiSource,
        $attentionInstanceSource,
        $attentionSmokeSource,
        $attentionAccumulatorHeader,
        $dynamicMoeProvider,
        $dynamicMoeMetadata,
        $dynamicMoeQualification,
        $dynamicMoeProviderSource,
        $dynamicMoeGeneratorSource,
        $dynamicMoeSmokeSource,
        $dynamicMoeBuildScript,
        $dynamicMoeBuilderScript,
        $gdnQualificationScript,
        $gdnSmokeSource,
        $statePath,
        $sourceCommandPath,
        $baseEnvPath,
        $q1MoeGateUpPath,
        $q1MoeDownPath,
        $gb10OraclePath,
        $gb10CaptureScriptPath,
        $gb10ContinuationCaptureScriptPath,
        $promptGeneratorPath
    )) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "r1076 input not found: $required"
    }
}
if ($GdnRoute -eq 'aiter' -and
    -not (Test-Path -LiteralPath $gdnQualification -PathType Leaf)) {
    throw "AITER GDN qualification is missing: $gdnQualification"
}
if (-not (Test-Path -LiteralPath $modelPath -PathType Container)) {
    throw "r1076 model not found: $modelPath"
}
$repoCommit = (& git -C $Repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $repoCommit -notmatch '^[0-9a-f]{40}$') {
    throw "could not resolve repository commit: $repoCommit"
}
$gb10Oracle = Get-Content -Raw -LiteralPath $gb10OraclePath |
    ConvertFrom-Json
$gb10OracleSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $gb10OraclePath
).Hash.ToLowerInvariant()
$gb10CaptureScriptSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $gb10CaptureScriptPath
).Hash.ToLowerInvariant()
$gb10ContinuationCaptureScriptSha256 = (
    Get-FileHash -Algorithm SHA256 `
        -LiteralPath $gb10ContinuationCaptureScriptPath
).Hash.ToLowerInvariant()
$promptGeneratorSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $promptGeneratorPath
).Hash.ToLowerInvariant()
$gb10FirstToken = [int]$gb10Oracle.expected.first_token_id
$gb10Logit = [double]$gb10Oracle.expected.first_token_raw_logit
$gb10LogitTolerance = `
    [double]$gb10Oracle.expected.first_token_raw_logit_tolerance
$gb10OutputTokens = @(
    $gb10Oracle.expected.output_token_ids | ForEach-Object { [int]$_ }
)
$gb10ContinuationTokenCount = `
    [int]$gb10Oracle.expected.continuation_token_count
$gb10DecodeStepCount = [int]$gb10Oracle.expected.decode_step_count
$gb10OutputSha256 = `
    [string]$gb10Oracle.expected.output_token_ids_u32le_sha256
$gb10OutputFnv1a64 = `
    [string]$gb10Oracle.expected.output_token_ids_u32le_fnv1a64
if ([int]$gb10Oracle.schema_version -ne 1 -or
    [string]$gb10Oracle.record_type -ne `
        'qrt_hprefill_q8192_gb10_oracle' -or
    [string]$gb10Oracle.status -ne 'pass' -or
    [string]$gb10Oracle.correctness_authority.host -ne 'gb10-4t' -or
    [string]$gb10Oracle.correctness_authority.model -ne `
        'qwen3.6-35b-a3b' -or
    [string]$gb10Oracle.correctness_authority.model_reference -ne `
        '/models' -or
    [string]$gb10Oracle.correctness_authority.dtype -ne 'bfloat16' -or
    [int]$gb10Oracle.prompt.token_count -ne 8192 -or
    [int]$gb10Oracle.prompt.seed -ne 395518 -or
    [int]$gb10Oracle.prompt.first_token_id -ne 84411 -or
    [string]$gb10Oracle.prompt.u32le_sha256 -ne `
        'dda20edc609f935f34d3d41ca4a84ffefa66726676756fe26ff3bef4fbff0b96' -or
    [string]$gb10Oracle.prompt.u32le_fnv1a64 -ne `
        '1584e34d56e5d78b' -or
    [string]$gb10Oracle.prompt.generator.path -ne `
        'scripts/.candidate_generate_random_prompt.py' -or
    [string]$gb10Oracle.prompt.generator.sha256 -ne `
        $promptGeneratorSha256 -or
    $gb10FirstToken -ne 144 -or
    $gb10Logit -ne 10.375 -or
    $gb10LogitTolerance -ne 0.125 -or
    [string]$gb10Oracle.expected.continuation_comparison -ne `
        'token_for_token' -or
    $gb10ContinuationTokenCount -ne 32 -or
    $gb10DecodeStepCount -ne 31 -or
    $gb10OutputTokens.Count -ne $gb10ContinuationTokenCount -or
    $gb10OutputTokens[0] -ne $gb10FirstToken -or
    $gb10OutputSha256 -ne `
        '97d1c7e3ae51a6aa3d55e41d38e410c3995c9dc017a133d8d1533ad3cc3ea9b4' -or
    $gb10OutputFnv1a64 -ne 'e60389077f247d2c' -or
    [string]$gb10Oracle.capture.record_type -ne `
        'gb10_q8192_first_token_raw_logit_capture' -or
    [string]$gb10Oracle.capture.command_file -ne `
        'scripts/capture_gb10_q8192_oracle.py' -or
    [string]$gb10Oracle.capture.command_file_sha256 -ne `
        $gb10CaptureScriptSha256 -or
    [string]$gb10Oracle.capture.request_sha256 -notmatch `
        '^[0-9a-f]{64}$' -or
    [string]$gb10Oracle.capture.response_sha256 -notmatch `
        '^[0-9a-f]{64}$' -or
    [int]$gb10Oracle.capture.http_status -ne 200 -or
    [int]$gb10Oracle.capture.usage.prompt_tokens -ne 8192 -or
    [int]$gb10Oracle.capture.usage.completion_tokens -ne 1 -or
    @($gb10Oracle.capture.output_token_ids).Count -ne 1 -or
    [int]$gb10Oracle.capture.output_token_ids[0] -ne $gb10FirstToken -or
    [int]$gb10Oracle.capture.base_lm_head_capture.argmax_token_id -ne `
        $gb10FirstToken -or
    [int]$gb10Oracle.capture.base_lm_head_capture.selected_token_id -ne `
        $gb10FirstToken -or
    [double]$gb10Oracle.capture.base_lm_head_capture.argmax_raw_logit -ne `
        $gb10Logit -or
    [double]$gb10Oracle.capture.base_lm_head_capture.selected_token_raw_logit_bf16 -ne `
        $gb10Logit -or
    [string]$gb10Oracle.capture.base_lm_head_capture.hidden_bf16_sha256 `
        -notmatch '^[0-9a-f]{64}$' -or
    [int]$gb10Oracle.continuation_capture.schema_version -ne 1 -or
    [string]$gb10Oracle.continuation_capture.record_type -ne `
        'gb10_q8192_continuation_capture' -or
    [string]$gb10Oracle.continuation_capture.authority.host -ne `
        'gb10-4t' -or
    [string]$gb10Oracle.continuation_capture.authority.model -ne `
        'qwen3.6-35b-a3b' -or
    [string]$gb10Oracle.continuation_capture.authority.model_reference -ne `
        '/models' -or
    [string]$gb10Oracle.continuation_capture.authority.dtype -ne `
        'bfloat16' -or
    [string]$gb10Oracle.continuation_capture.command_file -ne `
        'scripts/capture_gb10_q8192_continuation.py' -or
    [string]$gb10Oracle.continuation_capture.command_file_sha256 -ne `
        $gb10ContinuationCaptureScriptSha256 -or
    [bool]$gb10Oracle.continuation_capture.capture_hook_armed -or
    [int]$gb10Oracle.continuation_capture.request_policy.max_tokens -ne 32 -or
    [double]$gb10Oracle.continuation_capture.request_policy.temperature -ne `
        0.0 -or
    [double]$gb10Oracle.continuation_capture.request_policy.top_p -ne 1.0 -or
    [string]$gb10Oracle.continuation_capture.request_sha256 -notmatch `
        '^[0-9a-f]{64}$' -or
    [string]$gb10Oracle.continuation_capture.response_sha256 -notmatch `
        '^[0-9a-f]{64}$' -or
    [int]$gb10Oracle.continuation_capture.http_status -ne 200 -or
    [int]$gb10Oracle.continuation_capture.usage.prompt_tokens -ne 8192 -or
    [int]$gb10Oracle.continuation_capture.usage.completion_tokens -ne 32 -or
    @($gb10Oracle.continuation_capture.output_token_ids).Count -ne 32 -or
    [string]$gb10Oracle.continuation_capture.output_token_ids_u32le_sha256 `
        -ne $gb10OutputSha256 -or
    [string]$gb10Oracle.continuation_capture.output_token_ids_u32le_fnv1a64 `
        -ne $gb10OutputFnv1a64 -or
    [int]$gb10Oracle.continuation_capture.reproducibility.request_count `
        -ne 2 -or
    -not [bool]$gb10Oracle.continuation_capture.reproducibility.token_for_token_repeat_pass -or
    [string]$gb10Oracle.continuation_capture.reproducibility.repeat_request_sha256 `
        -ne [string]$gb10Oracle.continuation_capture.request_sha256 -or
    [string]$gb10Oracle.continuation_capture.reproducibility.repeat_response_sha256 `
        -notmatch '^[0-9a-f]{64}$' -or
    [string]$gb10Oracle.continuation_capture.reproducibility.repeat_output_token_ids_u32le_sha256 `
        -ne $gb10OutputSha256 -or
    [string]$gb10Oracle.continuation_capture.reproducibility.repeat_output_token_ids_u32le_fnv1a64 `
        -ne $gb10OutputFnv1a64 -or
    [string]$gb10Oracle.model_evidence.weight_tensor_key -ne `
        'lm_head.weight' -or
    [string]$gb10Oracle.model_evidence.weight_dtype -ne `
        'torch.bfloat16' -or
    [bool]$gb10Oracle.diagnostic_policy.openai_logprob_is_raw_logit -or
    [bool]$gb10Oracle.diagnostic_policy.engine_self_hashes_are_correctness_authority -or
    -not [bool]$gb10Oracle.diagnostic_policy.continuation_captured_in_this_contract -or
    -not [bool]$gb10Oracle.diagnostic_policy.decode_and_prefix_continuation_correctness_active) {
    throw 'q8192 GB10 raw-logit/continuation provenance is invalid'
}
for ($tokenIndex = 0; $tokenIndex -lt $gb10OutputTokens.Count; `
        $tokenIndex++) {
    if ([int]$gb10Oracle.continuation_capture.output_token_ids[$tokenIndex] `
            -ne $gb10OutputTokens[$tokenIndex]) {
        throw "q8192 GB10 continuation token differs at index $tokenIndex"
    }
}
$attentionEvidence = Get-Content -Raw `
    -LiteralPath $attentionBuildProvenance | ConvertFrom-Json
$expectedAttentionDynamicTokens = @(
    2073, 2156, 2560, 3073, 4609, 6145, 2049, 2175,
    2176, 2177, 2559, 2561, 3071, 3072, 3583, 3584,
    3585, 4095, 4096, 4097, 4607, 4608, 6143, 6144,
    7167, 7168, 7169, 7679, 7680, 7681, 8191, 8192
)
$attentionExpectedSourceHashes = [ordered]@{
    command_file_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $attentionBuildScript
    ).Hash.ToLowerInvariant()
    provider_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $attentionProviderSource
    ).Hash.ToLowerInvariant()
    api_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $attentionApiSource
    ).Hash.ToLowerInvariant()
    instance_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $attentionInstanceSource
    ).Hash.ToLowerInvariant()
    direct_smoke_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $attentionSmokeSource
    ).Hash.ToLowerInvariant()
    accumulator_header_sha256 = (
        Get-FileHash -Algorithm SHA256 `
            -LiteralPath $attentionAccumulatorHeader
    ).Hash.ToLowerInvariant()
}
$attentionSmokeStdout = Join-Path $attentionRoot `
    'dynamic-logical-smoke.stdout.log'
$attentionSmokeStderr = Join-Path $attentionRoot `
    'dynamic-logical-smoke.stderr.log'
if ([int]$attentionEvidence.schema_version -ne 1 -or
    [string]$attentionEvidence.host -ine [Environment]::MachineName -or
    [string]$attentionEvidence.repo_commit -ne $repoCommit -or
    [string]$attentionEvidence.run_label -ne $AttentionBuildLabel -or
    [string]$attentionEvidence.command_file -ine $attentionBuildScript -or
    [string]$attentionEvidence.path -ine $attentionProvider -or
    [string]$attentionEvidence.sha256 -ne (
        Get-FileHash -Algorithm SHA256 -LiteralPath $attentionProvider
    ).Hash.ToLowerInvariant() -or
    [int]$attentionEvidence.ck_tile_n -ne 32 -or
    [int]$attentionEvidence.blackwell_exact_terminal_rows -ne 16 -or
    [int]$attentionEvidence.dynamic_logical_case_count -ne 32 -or
    @($attentionEvidence.dynamic_logical_cases).Count -ne 32 -or
    [bool]$attentionEvidence.dynamic_logical_reset_included -or
    -not [bool]$attentionEvidence.dynamic_logical_component_only -or
    [bool]$attentionEvidence.inference_success_claimed -or
    -not [bool]$attentionEvidence.smoke.completed -or
    [bool]$attentionEvidence.smoke.timed_out -or
    [int]$attentionEvidence.smoke.exit_code -ne 0 -or
    (@($attentionEvidence.dynamic_logical_tokens) -join ',') -ne
        ($expectedAttentionDynamicTokens -join ',') -or
    [string]$attentionEvidence.smoke_stdout -ine $attentionSmokeStdout -or
    [string]$attentionEvidence.smoke_stderr -ine $attentionSmokeStderr -or
    -not (Test-Path -LiteralPath $attentionSmokeStdout -PathType Leaf) -or
    -not (Test-Path -LiteralPath $attentionSmokeStderr -PathType Leaf) -or
    [string]$attentionEvidence.smoke_stdout_sha256 -ne (
        Get-FileHash -Algorithm SHA256 -LiteralPath $attentionSmokeStdout
    ).Hash.ToLowerInvariant() -or
    [string]$attentionEvidence.smoke_stderr_sha256 -ne (
        Get-FileHash -Algorithm SHA256 -LiteralPath $attentionSmokeStderr
    ).Hash.ToLowerInvariant()) {
    throw 'product CK attention dynamic-logical qualification is invalid'
}
foreach ($name in $attentionExpectedSourceHashes.Keys) {
    if ([string]$attentionEvidence.$name -ne
        [string]$attentionExpectedSourceHashes[$name]) {
        throw "product CK attention source hash differs: $name"
    }
}
for ($caseIndex = 0; $caseIndex -lt $expectedAttentionDynamicTokens.Count; `
        $caseIndex++) {
    $case = @($attentionEvidence.dynamic_logical_cases)[$caseIndex]
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
        throw "product CK attention timing is invalid at case $caseIndex"
    }
}
$attentionBuildProvenanceSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $attentionBuildProvenance
).Hash.ToLowerInvariant()
$gdnEvidence = $null
$gdnArtifacts = @()
$gdnQualificationSha256 = ''
if ($GdnRoute -eq 'aiter') {
    $gdnEvidence = Get-Content -Raw -LiteralPath $gdnQualification |
        ConvertFrom-Json
    $expectedGdnTokens = @(
        2073, 2156, 2560, 3073, 4609, 6145, 2049, 2175,
        2176, 2177, 2559, 2561, 3071, 3072, 3583, 3584,
        3585, 4095, 4096, 4097, 4607, 4608, 6143, 6144,
        7167, 7168, 7169, 7679, 7680, 7681, 8191, 8192
    )
    if ([int]$gdnEvidence.schema_version -ne 1 -or
        [string]$gdnEvidence.record_type -ne `
            'qrt_dynamic_logical_gdn_qualification' -or
        [string]$gdnEvidence.host -ine [Environment]::MachineName -or
        [string]$gdnEvidence.repo_commit -ne $repoCommit -or
        [string]$gdnEvidence.run_label -ne $GdnQualificationLabel -or
        [string]$gdnEvidence.command_file -ine $gdnQualificationScript -or
        [string]$gdnEvidence.command_file_sha256 -ne (
            Get-FileHash -Algorithm SHA256 `
                -LiteralPath $gdnQualificationScript
        ).Hash.ToLowerInvariant() -or
        [string]$gdnEvidence.smoke_source -ine $gdnSmokeSource -or
        [string]$gdnEvidence.smoke_source_sha256 -ne (
            Get-FileHash -Algorithm SHA256 -LiteralPath $gdnSmokeSource
        ).Hash.ToLowerInvariant() -or
        [int]$gdnEvidence.dynamic_logical_case_count -ne 32 -or
        [int]$gdnEvidence.dynamic_logical_mode_count -ne 64 -or
        [bool]$gdnEvidence.dynamic_logical_reset_included -or
        [int]$gdnEvidence.unmeasured_warmup_per_mode -ne 1 -or
        -not [bool]$gdnEvidence.component_only -or
        [bool]$gdnEvidence.inference_success_claimed -or
        (@($gdnEvidence.dynamic_logical_tokens) -join ',') -ne
            ($expectedGdnTokens -join ',') -or
        @($gdnEvidence.cases).Count -ne 32 -or
        @($gdnEvidence.artifacts).Count -ne 5 -or
        -not [bool]$gdnEvidence.compile.completed -or
        [bool]$gdnEvidence.compile.timed_out -or
        [int]$gdnEvidence.compile.exit_code -ne 0) {
        throw 'dynamic-logical AITER GDN qualification identity is invalid'
    }
    $gdnArtifacts = @(
        foreach ($artifact in @($gdnEvidence.artifacts)) {
            $artifactPath = [string]$artifact.path
            if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
                throw "dynamic-logical GDN artifact is missing: $artifactPath"
            }
            $item = Get-Item -LiteralPath $artifactPath
            $actual = [ordered]@{
                path = $item.FullName
                file = $item.Name
                bytes = [uint64]$item.Length
                sha256 = (Get-FileHash -Algorithm SHA256 `
                    -LiteralPath $item.FullName).Hash.ToLowerInvariant()
            }
            if ([string]$artifact.file -cne [string]$actual.file -or
                [uint64]$artifact.bytes -ne [uint64]$actual.bytes -or
                [string]$artifact.sha256 -ne [string]$actual.sha256) {
                throw "dynamic-logical GDN artifact hash differs: $artifactPath"
            }
            $actual
        }
    )
    $expectedGdnKernelNames = @(
        'q1024_seeded_aiter_fused_gdn.hsaco',
        'q262144_aiter_fused_gdn_bf16.hsaco',
        'q32768_seeded_aiter_fused_gdn_bf16.hsaco',
        'q8192_aiter_fused_gdn.hsaco'
    )
    $observedGdnKernelNames = @(
        $gdnArtifacts | Where-Object {
            [string]$_.path -ine [string]$gdnEvidence.provider
        } | ForEach-Object { [string]$_.file } | Sort-Object
    )
    if (($observedGdnKernelNames -join ',') -ne
            ($expectedGdnKernelNames -join ',') -or
        [string]$gdnEvidence.provider_sha256 -ne (
            Get-FileHash -Algorithm SHA256 `
                -LiteralPath ([string]$gdnEvidence.provider)
        ).Hash.ToLowerInvariant()) {
        throw 'dynamic-logical AITER GDN runtime artifact set is incomplete'
    }
    for ($caseIndex = 0; $caseIndex -lt $expectedGdnTokens.Count; `
            $caseIndex++) {
        $case = @($gdnEvidence.cases)[$caseIndex]
        if ([int]$case.tokens -ne $expectedGdnTokens[$caseIndex] -or
            @($case.modes).Count -ne 2 -or
            -not [bool]$case.run.completed -or
            [bool]$case.run.timed_out -or
            [int]$case.run.exit_code -ne 0 -or
            -not (Test-Path -LiteralPath ([string]$case.stdout) `
                -PathType Leaf) -or
            [string]$case.stdout_sha256 -ne (
                Get-FileHash -Algorithm SHA256 `
                    -LiteralPath ([string]$case.stdout)
            ).Hash.ToLowerInvariant() -or
            -not (Test-Path -LiteralPath ([string]$case.stderr) `
                -PathType Leaf) -or
            [string]$case.stderr_sha256 -ne (
                Get-FileHash -Algorithm SHA256 `
                    -LiteralPath ([string]$case.stderr)
            ).Hash.ToLowerInvariant()) {
            throw "dynamic-logical GDN q$($expectedGdnTokens[$caseIndex]) case evidence is invalid"
        }
    }
    $gdnQualificationSha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $gdnQualification
    ).Hash.ToLowerInvariant()
}
$dynamicMoeEvidence = Get-Content -Raw `
    -LiteralPath $dynamicMoeQualification | ConvertFrom-Json
$dynamicMoeExpectedInputHashes = [ordered]@{
    provider_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $dynamicMoeProviderSource
    ).Hash.ToLowerInvariant()
    generator_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $dynamicMoeGeneratorSource
    ).Hash.ToLowerInvariant()
    smoke_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $dynamicMoeSmokeSource
    ).Hash.ToLowerInvariant()
}
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
    throw 'dynamic-logical q8192 MoE qualification identity is invalid'
}
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
$seenDynamicMoeRuntimeEnvironment = @{}
if ($recordedDynamicMoeRuntimeEnvironment.Count -ne
        $expectedDynamicMoeRuntimeEnvironment.Count) {
    throw 'dynamic-logical q8192 MoE runtime environment count differs'
}
foreach ($entry in $recordedDynamicMoeRuntimeEnvironment) {
    $name = [string]$entry.name
    $value = [string]$entry.value
    if ([string]::IsNullOrWhiteSpace($name) -or
        $seenDynamicMoeRuntimeEnvironment.ContainsKey($name) -or
        -not $expectedDynamicMoeRuntimeEnvironment.Contains($name) -or
        $value -cne [string]$expectedDynamicMoeRuntimeEnvironment[$name]) {
        throw "dynamic-logical q8192 MoE runtime environment differs: $name"
    }
    $seenDynamicMoeRuntimeEnvironment[$name] = $true
}
foreach ($name in $dynamicMoeExpectedInputHashes.Keys) {
    if ([string]$dynamicMoeEvidence.$name -ne `
        [string]$dynamicMoeExpectedInputHashes[$name]) {
        throw "dynamic-logical q8192 MoE source hash differs: $name"
    }
}
$dynamicMoeArtifacts = @(
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
$expectedDynamicMoeArtifactCount =
    8 + $expectedDynamicMoeAuxiliaryKernelFiles.Count
$observedDynamicMoeArtifactNames = @(
    $dynamicMoeArtifacts | ForEach-Object { [string]$_.file }
)
foreach ($name in $expectedDynamicMoeAuxiliaryKernelFiles) {
    if ($observedDynamicMoeArtifactNames -cnotcontains $name) {
        throw "dynamic-logical q8192 MoE auxiliary artifact is missing: $name"
    }
}
if ($dynamicMoeArtifacts.Count -ne $expectedDynamicMoeArtifactCount -or
    [string]$dynamicMoeEvidence.provider_sha256 -ne (
        Get-FileHash -Algorithm SHA256 -LiteralPath $dynamicMoeProvider
    ).Hash.ToLowerInvariant()) {
    throw 'dynamic-logical q8192 MoE runtime artifact set is incomplete'
}
$dynamicMoeQualificationSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $dynamicMoeQualification
).Hash.ToLowerInvariant()
$dynamicMoeProviderSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $dynamicMoeProvider
).Hash.ToLowerInvariant()
$q1MoeGateUpSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $q1MoeGateUpPath
).Hash.ToLowerInvariant()
$q1MoeDownSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $q1MoeDownPath
).Hash.ToLowerInvariant()
if ($MoeArithmetic -eq 'triton' -and
    ($q1MoeGateUpSha256 -ne `
        'd8121b0d85c0a2c5e80e9a2b49177d72890a78d128a7fa309978b89829b87cad' -or
     $q1MoeDownSha256 -ne `
        '4760439fdce74da05aec4747eccd317227829d50fdcb97a63c72ba5df342d125')) {
    throw 'terminal q1 Triton-0626 AOT hashes do not match the packaged gfx1151 contract'
}

$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$listening = @(
    Get-NetTCPConnection -State Listen -LocalPort 8000 `
        -ErrorAction SilentlyContinue
)
if ($state.status -eq 'ready') {
    if ($listening.Count -eq 0) {
        Write-Host 'recorded r1074 service is stale after host restart; port 8000 is free'
    } elseif ($listening.Count -ne 1 -or
              [int]$listening[0].OwningProcess -ne [int]$state.pid) {
        throw 'port 8000 does not belong to the recorded r1074 service'
    } else {
        & $serviceExe stop --state-file $statePath --wait-seconds 30
        if ($LASTEXITCODE -ne 0) {
            throw "could not stop r1074 service: $LASTEXITCODE"
        }
    }
} elseif ($state.status -eq 'stopped') {
    if ($listening.Count -ne 0) {
        throw 'r1074 is stopped but port 8000 is occupied'
    }
} else {
    throw "r1074 has an unexpected state before product run: $($state.status)"
}
if (@(Get-NetTCPConnection -State Listen -LocalPort 8000 `
        -ErrorAction SilentlyContinue).Count -ne 0) {
    throw 'port 8000 remained occupied after stopping r1074'
}

[void](New-Item -ItemType Directory -Path $outputDir)
$promptPath = Join-Path $outputDir 'prompt-q8192-seed395518-first84411.json'
$expectedPath = Join-Path $outputDir `
    'expected-output-gb10-q8192-32tokens.json'
$promptRecordPath = Join-Path $outputDir 'prompt-record.json'
$gb10OracleSnapshotPath = Join-Path $outputDir 'gb10-q8192-oracle.json'
$envPath = Join-Path $outputDir 'runtime.env'
$stdoutPath = Join-Path $outputDir 'product.stdout.json'
$stderrPath = Join-Path $outputDir 'product.stderr.log'
$prefixStdoutPath = Join-Path $outputDir 'prefix-continuation.stdout.json'
$prefixStderrPath = Join-Path $outputDir 'prefix-continuation.stderr.log'
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false
Copy-Item -LiteralPath $gb10OraclePath -Destination $gb10OracleSnapshotPath
if ((Get-FileHash -Algorithm SHA256 `
        -LiteralPath $gb10OracleSnapshotPath).Hash.ToLowerInvariant() -ne `
        $gb10OracleSha256) {
    throw 'q8192 GB10 oracle snapshot hash differs from the qualified source'
}

$python = (Get-Command python.exe -ErrorAction Stop).Source
$promptRecord = & $python `
    $promptGeneratorPath `
    --output $promptPath --tokens 8192 --seed 395518 --first-token 84411
if ($LASTEXITCODE -ne 0) {
    throw "q8192 prompt generation failed: $LASTEXITCODE"
}
$promptRecord = ($promptRecord -join [Environment]::NewLine).Trim()
$promptMetadata = $promptRecord | ConvertFrom-Json
if ($promptMetadata.u32le_sha256 -ne `
        [string]$gb10Oracle.prompt.u32le_sha256 -or
    $promptMetadata.u32le_fnv1a64 -ne `
        [string]$gb10Oracle.prompt.u32le_fnv1a64) {
    throw 'generated q8192 prompt does not match the hashed GB10 oracle prompt'
}
[IO.File]::WriteAllText($promptRecordPath, $promptRecord + "`n", $utf8)
$expectedJson = ConvertTo-Json -InputObject @($gb10OutputTokens) -Compress
[IO.File]::WriteAllText(
    $expectedPath,
    $expectedJson + "`n",
    $utf8
)

$environment = [ordered]@{}
foreach ($line in Get-Content -LiteralPath $baseEnvPath) {
    $trimmed = $line.Trim()
    if ($trimmed.Length -eq 0 -or $trimmed.StartsWith('#')) {
        continue
    }
    $equals = $trimmed.IndexOf('=')
    if ($equals -le 0) {
        throw "invalid base environment entry: $line"
    }
    $environment[$trimmed.Substring(0, $equals)] = `
        $trimmed.Substring($equals + 1)
}
$sourceCommand = (Get-Content -LiteralPath $sourceCommandPath -Raw).Trim()
foreach ($match in [regex]::Matches(
        $sourceCommand,
        '(?:^|\s)--set-env\s+(?<entry>[^\s]+)'
    )) {
    $entry = $match.Groups['entry'].Value
    $equals = $entry.IndexOf('=')
    if ($equals -le 0) {
        throw "invalid saved environment entry: $entry"
    }
    $name = $entry.Substring(0, $equals)
    $value = $entry.Substring($equals + 1).Replace(
        (Join-Path $Repo 'build\triton-style-attention-blackwell-fused-pv-r1049\qrt_triton_style_attention_blackwell_fused_pv.dll'),
        $attentionProvider
    )
    $environment[$name] = $value
}
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_TRITON_SELECTED_MOE_DLL'] = `
    $dynamicMoeProvider
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_TRITON_SELECTED_MOE_KERNEL_DIR'] = `
    $dynamicMoeRoot
foreach ($entry in $expectedDynamicMoeRuntimeEnvironment.GetEnumerator()) {
    $environment[[string]$entry.Key] = [string]$entry.Value
}
$environment['QRT_QWEN36_DYNAMIC_LOGICAL_MOE_PROVIDER'] = '0'
$environment['QRT_QWEN36_WHOLE_PROVIDER_ARBITRARY_CONTEXT'] = '1'
$environment['QRT_QWEN36_EXACT_ARBITRARY_RETAINED_Q8192'] = '1'
$environment['QRT_QWEN36_EXACT_ARBITRARY_PRODUCT_PATH'] = '1'
$environment['QRT_QWEN36_EXACT_ARBITRARY_Q1024_MOE_PROVIDER'] = '1'
$environment['QRT_QWEN36_EXACT_ARBITRARY_Q1024_MOE_DLL'] = `
    (Join-Path $releaseRoot 'q1024-moe\qrt_triton_moe_q1024_exact_provider_slots64.dll')
$environment['QRT_QWEN36_EXACT_ARBITRARY_Q1024_MOE_KERNEL_DIR'] = `
    (Join-Path $releaseRoot 'q1024-moe\moe-kernels')
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_AITER_FUSED_GDN_PROVIDER'] = '1'
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_AITER_FUSED_GDN_MIN_LAYER'] = '0'
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_AITER_FUSED_GDN_MAX_LAYER'] = '38'
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_FLA_CHUNK_GDN_ARITHMETIC'] = `
    if ($GdnRoute -eq 'fla') { '1' } else { '0' }
if ($GdnRoute -eq 'aiter') {
    $qualifiedGdnProvider = [string]$environment[
        'QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_AITER_FUSED_GDN_DLL'
    ]
    $qualifiedGdnKernelDir = [string]$environment[
        'QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_AITER_FUSED_GDN_KERNEL_DIR'
    ]
    if ([string]::IsNullOrWhiteSpace($qualifiedGdnProvider) -or
        [string]::IsNullOrWhiteSpace($qualifiedGdnKernelDir) -or
        $qualifiedGdnProvider -ine [string]$gdnEvidence.provider -or
        $qualifiedGdnKernelDir -ine [string]$gdnEvidence.kernel_dir) {
        throw 'qualified environment does not bind the dynamic-logical AITER GDN artifacts'
    }
}
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_SELECTED_MOE_FAST_ARITHMETIC_BACKEND'] = `
    if ($MoeArithmetic -eq 'fast') { '1' } else { '0' }
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_ROUTED_CPU_REFERENCE_ARITHMETIC'] = `
    if ($MoeArithmetic -eq 'exact') { '1' } else { '0' }
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_ROUTED_EXACT_ROUTE_TILE'] = `
    if ($MoeArithmetic -eq 'exact') { '1' } else { '0' }
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_GROUPED_EXACT_ROUTED_BACKEND'] = `
    if ($MoeArithmetic -eq 'exact') { '1' } else { '0' }
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_WHOLE_SELECTED_MOE_EXACT_BACKEND'] = `
    if ($MoeArithmetic -eq 'exact') { '1' } else { '0' }
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_LAYER39_Q1_TRITON_0626_ROUTED'] = `
    if ($MoeArithmetic -eq 'triton') { '1' } else { '0' }
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_Q1_MOE_TRITON_0626_MODULE_DIR'] = `
    $q1MoeModuleDir
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_Q1_MOE_TRITON_0626_SELECTED_GATE_WARPS8'] = '1'
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_LAYER39_Q1_CK_FMHA'] = `
    if ($Q1Attention -eq 'ck') { '1' } else { '0' }
$environment['QRT_PREFILL_DESCRIPTOR_BATCH_FULL_ATTENTION_CK_FMHA_DLL'] = `
    $attentionProvider
$environment['QRT_QWEN36_LAYER39_DYNAMIC_TERMINAL_COMPACT_Q'] = '1'
$environment['QRT_QWEN36_LAYER39_DYNAMIC_TERMINAL_PACKED_MOE'] = '1'
$environment['QRT_QWEN36_LAYER39_DYNAMIC_TERMINAL_DEVICE_CORRIDOR'] = '1'
$environment['AMD_SERIALIZE_KERNEL'] = '0'
$environment['QRT_QWEN36_EXACT_ARBITRARY_LAYER_OUTPUT_TRACE'] = `
    if ($LayerOutputTrace) { '1' } else { '0' }
$environment['QRT_QWEN36_EXACT_ARBITRARY_LAYER_OUTPUT_TRACE_POSITION'] = '8191'
$environment['QRT_QWEN36_EXACT_ARBITRARY_LAYER_BOUNDARY_TRACE'] = '0'
$environment['QRT_QWEN36_EXACT_ARBITRARY_LINEAR_STAGE_TRACE_LAYER'] = '-1'
$environment['QRT_QWEN36_GB10_TEACHER_FORCED_CONTINUATION'] = ''
$environment['QRT_QWEN36_Q16384_Q1024_GB10_TEACHER_FORCED_CONTINUATION'] = ''
$environment['QRT_QWEN36_RESIDENT_DECODE_TOKEN_MARKERS_ELIDED'] = '0'
$environment['QRT_QWEN36_SERVICE_COMPACT_LOG'] = '0'
$environmentLines = @(
    foreach ($name in $environment.Keys) {
        "$name=$($environment[$name])"
    }
)
[IO.File]::WriteAllText(
    $envPath,
    ($environmentLines -join [Environment]::NewLine) + [Environment]::NewLine,
    $utf8
)

function Quote-ProcessArg {
    param([Parameter(Mandatory = $true)][string]$Value)
    return [string]::Concat('"', $Value.Replace('"', '\"'), '"')
}

function Invoke-CapturedProduct {
    param(
        [Parameter(Mandatory = $true)][string[]]$NativeArguments,
        [Parameter(Mandatory = $true)][string]$StdOutPath,
        [Parameter(Mandatory = $true)][string]$StdErrPath
    )
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $productExe
    $startInfo.Arguments = @($NativeArguments | ForEach-Object {
            Quote-ProcessArg $_
        }) -join ' '
    $startInfo.WorkingDirectory = $Repo
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    $watch = [Diagnostics.Stopwatch]::StartNew()
    [void]$process.Start()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $completed = $process.WaitForExit(360000)
    if (-not $completed) {
        try { $process.Kill() } catch {}
        $process.WaitForExit()
    } else {
        $process.WaitForExit()
    }
    $watch.Stop()
    $capturedStdout = $stdoutTask.GetAwaiter().GetResult()
    $capturedStderr = $stderrTask.GetAwaiter().GetResult()
    [IO.File]::WriteAllText($StdOutPath, $capturedStdout, $utf8)
    [IO.File]::WriteAllText($StdErrPath, $capturedStderr, $utf8)
    return [ordered]@{
        arguments = $NativeArguments
        completed = $completed
        exit_code = if ($completed) { [int]$process.ExitCode } else { 124 }
        wall_ms = [Math]::Round($watch.Elapsed.TotalMilliseconds, 6)
        stdout = $capturedStdout
        stderr = $capturedStderr
    }
}

$arguments = @(
    'run',
    '--model', $modelPath,
    '--tokens', $promptPath,
    '--output-tokens', ([string]$gb10ContinuationTokenCount),
    '--expected-output', $expectedPath,
    '--expected-prompt-fnv', ([string]$gb10Oracle.prompt.u32le_fnv1a64),
    '--expected-output-fnv', $gb10OutputFnv1a64,
    '--env-file', $envPath,
    '--provider-dll', $wholeProvider
)
$baseRun = Invoke-CapturedProduct -NativeArguments $arguments `
    -StdOutPath $stdoutPath -StdErrPath $stderrPath
$stdout = [string]$baseRun.stdout
$stderr = [string]$baseRun.stderr
$exitCode = [int]$baseRun.exit_code
$prefixArguments = @($arguments) + @(
    '--prefix-tokens', ([string]$prefixTokens),
    '--prefix-hits', ([string]$prefixHitCount),
    '--prefix-negative-guard'
)
$prefixRun = Invoke-CapturedProduct -NativeArguments $prefixArguments `
    -StdOutPath $prefixStdoutPath -StdErrPath $prefixStderrPath
$prefixStdout = [string]$prefixRun.stdout
$prefixStderr = [string]$prefixRun.stderr
$prefixExitCode = [int]$prefixRun.exit_code
$requiredRouteMarkers = @(
    'BATCH_MARK full_attention_ck_q1_kv8192',
    'BATCH_MARK q8192_triton_selected_moe_full_provider_v2',
    'BATCH_MARK layer39_q1_triton_0626_routed_backend',
    'BATCH_MARK q1_terminal_device_corridor_activate',
    'BATCH_MARK q1_terminal_device_corridor_triton_metadata_upload',
    'BATCH_MARK q1_terminal_device_corridor_output_activate',
    'BATCH_MARK q1_terminal_device_corridor_final_norm_activate',
    'BATCH_MARK qwen36_whole_provider_resident_decode_token',
    'BATCH_MARK qwen36_whole_provider_resident_decode_complete'
)
$forbiddenRouteMarkers = @(
    'BATCH_MARK layer39_q1_kv8192_fallback',
    'BATCH_MARK q1_terminal_device_corridor_fallback',
    'BATCH_MARK q1_terminal_device_corridor_output_fallback',
    'BATCH_MARK q1_terminal_device_corridor_final_norm_fallback',
    'BATCH_MARK qwen36_whole_provider_gb10_teacher_forced_begin',
    'BATCH_MARK qwen36_whole_provider_gb10_teacher_forced_complete'
)
$missingRouteMarkers = @(
    $requiredRouteMarkers | Where-Object { -not $stderr.Contains($_) }
)
$observedForbiddenRouteMarkers = @(
    $forbiddenRouteMarkers | Where-Object { $stderr.Contains($_) }
)
$q8192MoeRouteLines = @(
    $stderr -split "`r?`n" | Where-Object {
        $_.Contains(
            'BATCH_MARK q8192_triton_selected_moe_full_provider_v2'
        ) -and $_.Contains(' selected_tokens=8192 ')
    }
)
$q8192MoeFixedRoutePass = $q8192MoeRouteLines.Count -gt 0 -and @(
    $q8192MoeRouteLines | Where-Object {
        -not $_.Contains(' provider_tile_tokens=8192 ') -or
        -not $_.Contains(' provider_tile_count=1 ') -or
        -not $_.Contains(' provider_tail_tokens=0 ') -or
        -not $_.Contains(' provider_tail_padded=0 ') -or
        -not $_.Contains(' dynamic_logical_moe_provider=0 ')
    }
).Count -eq 0
$decodeTokenRouteLines = @(
    $stderr -split "`r?`n" | Where-Object {
        $_.Contains(
            'BATCH_MARK qwen36_whole_provider_resident_decode_token'
        )
    }
)
$decodeGreedyRoutePass = @(
    $decodeTokenRouteLines | Where-Object {
        -not $_.Contains(' gb10_teacher_forced_input=0 ')
    }
).Count -eq 0
$decodeCompleteRouteLines = @(
    $stderr -split "`r?`n" | Where-Object {
        $_.Contains(
            'BATCH_MARK qwen36_whole_provider_resident_decode_complete'
        ) -and $_.Contains(
            " output_tokens=$gb10ContinuationTokenCount "
        ) -and $_.Contains(" decode_tokens=$gb10DecodeStepCount ")
    }
)
$decodeRoutePass = $decodeTokenRouteLines.Count -eq $gb10DecodeStepCount `
    -and $decodeCompleteRouteLines.Count -eq 1 -and $decodeGreedyRoutePass
$routePass = $missingRouteMarkers.Count -eq 0 -and `
    $observedForbiddenRouteMarkers.Count -eq 0 -and `
    $q8192MoeFixedRoutePass -and $decodeRoutePass
$summary = $null
if (-not [string]::IsNullOrWhiteSpace($stdout)) {
    $summaryLine = @(
        $stdout -split "`r?`n" | Where-Object {
            -not [string]::IsNullOrWhiteSpace($_)
        }
    )[-1]
    $summary = $summaryLine | ConvertFrom-Json
}
$prefixSummary = $null
if (-not [string]::IsNullOrWhiteSpace($prefixStdout)) {
    $prefixSummaryLine = @(
        $prefixStdout -split "`r?`n" | Where-Object {
            -not [string]::IsNullOrWhiteSpace($_)
        }
    )[-1]
    $prefixSummary = $prefixSummaryLine | ConvertFrom-Json
}
$prefixOutputTokens = if ($null -ne $prefixSummary) {
    @($prefixSummary.output_token_ids | ForEach-Object { [int]$_ })
} else {
    @()
}
$prefixContinuationExact = `
    $prefixOutputTokens.Count -eq $gb10OutputTokens.Count
if ($prefixContinuationExact) {
    for ($tokenIndex = 0; $tokenIndex -lt $gb10OutputTokens.Count; `
            $tokenIndex++) {
        if ($prefixOutputTokens[$tokenIndex] -ne `
                $gb10OutputTokens[$tokenIndex]) {
            $prefixContinuationExact = $false
            break
        }
    }
}
$prefixTpotMs = if ($null -ne $prefixSummary -and
    $null -ne $prefixSummary.tpot_ms) {
    [double]$prefixSummary.tpot_ms
} else {
    [double]::NaN
}
$prefixDecodeTokensPerSecond = if ($null -ne $prefixSummary -and
    $null -ne $prefixSummary.decode_tokens_per_second) {
    [double]$prefixSummary.decode_tokens_per_second
} else {
    [double]::NaN
}
$prefixModelEngineLoadMs = if ($null -ne $prefixSummary -and
    $null -ne $prefixSummary.engine_load_ms) {
    [double]$prefixSummary.engine_load_ms
} else {
    [double]::NaN
}
$prefixProviderTtftValues = if ($null -ne $prefixSummary) {
    @(
        $prefixSummary.prefix_hit_provider_ttft_ms |
            ForEach-Object { [double]$_ }
    )
} else {
    @()
}
$prefixNumericalMetricsPass = (
    -not [double]::IsNaN($prefixTpotMs) -and
    -not [double]::IsInfinity($prefixTpotMs) -and
    -not [double]::IsNaN($prefixDecodeTokensPerSecond) -and
    -not [double]::IsInfinity($prefixDecodeTokensPerSecond) -and
    -not [double]::IsNaN($prefixModelEngineLoadMs) -and
    -not [double]::IsInfinity($prefixModelEngineLoadMs) -and
    $prefixTpotMs -le $q8192TpotMaxMs -and
    $prefixDecodeTokensPerSecond -ge $q8192DecodeMinTokensPerSecond -and
    $prefixModelEngineLoadMs -le $modelEngineLoadMaxMs
)
$prefixProviderTtftPass = `
    $prefixProviderTtftValues.Count -eq $prefixHitCount
if ($prefixProviderTtftPass) {
    foreach ($prefixTtft in $prefixProviderTtftValues) {
        if ([double]::IsNaN($prefixTtft) -or
            [double]::IsInfinity($prefixTtft) -or
            $prefixTtft -le 0.0 -or $prefixTtft -gt $q8192TtftMaxMs) {
            $prefixProviderTtftPass = $false
            break
        }
    }
}
$prefixForbiddenMarkers = @(
    'BATCH_MARK qwen36_whole_provider_gb10_teacher_forced_begin',
    'BATCH_MARK qwen36_whole_provider_gb10_teacher_forced_complete',
    'BATCH_MARK q1_terminal_device_corridor_fallback',
    'BATCH_MARK q1_terminal_device_corridor_output_fallback',
    'BATCH_MARK q1_terminal_device_corridor_final_norm_fallback'
)
$prefixObservedForbiddenMarkers = @(
    $prefixForbiddenMarkers | Where-Object {
        $prefixStderr.Contains($_)
    }
)
$prefixRequestRouteLines = @(
    $prefixStderr -split "`r?`n" | Where-Object {
        $_.Contains(
            'BATCH_MARK qwen36_resident_prefix_cache_request'
        ) -and $_.Contains(" prefix_tokens=$prefixTokens ") -and
        $_.Contains(" suffix_tokens=$prefixSuffixTokens ") -and
        $_.Contains(" output_tokens=$gb10ContinuationTokenCount ") -and
        $_.Contains(' state_restored=1 ') -and
        $_.Contains(' copy_on_write=1 ') -and
        $_.Contains(' gb10_continuation_teacher_forced=0 ')
    }
)
$prefixDecodeCompleteRouteLines = @(
    $prefixStderr -split "`r?`n" | Where-Object {
        $_.Contains(
            'BATCH_MARK qwen36_whole_provider_resident_decode_complete'
        ) -and $_.Contains(
            " output_tokens=$gb10ContinuationTokenCount "
        ) -and $_.Contains(" decode_tokens=$gb10DecodeStepCount ")
    }
)
$prefixRoutePass = (
    $prefixObservedForbiddenMarkers.Count -eq 0 -and
    $prefixRequestRouteLines.Count -eq $prefixHitCount -and
    $prefixDecodeCompleteRouteLines.Count -eq $prefixHitCount
)
$prefixCorrectnessPass = (
    $prefixExitCode -eq 0 -and
    $null -ne $prefixSummary -and
    [string]$prefixSummary.status -eq 'pass' -and
    [int]$prefixSummary.input_tokens -eq 8192 -and
    [int]$prefixSummary.prefix_tokens -eq $prefixTokens -and
    [int]$prefixSummary.suffix_tokens -eq $prefixSuffixTokens -and
    [int]$prefixSummary.output_tokens -eq $gb10ContinuationTokenCount -and
    [int]$prefixSummary.prefix_hit_count -eq $prefixHitCount -and
    [int]$prefixSummary.prefix_contract_pass_count -eq $prefixHitCount -and
    [int]$prefixSummary.prefix_output_match_count -eq $prefixHitCount -and
    [int]$prefixSummary.prefix_stream_match_count -eq $prefixHitCount -and
    [int]$prefixSummary.prefix_state_restored_count -eq $prefixHitCount -and
    [bool]$prefixSummary.prefix_negative_guard_requested -and
    [bool]$prefixSummary.prefix_negative_guard_pass -and
    [int]$prefixSummary.prefix_negative_guard_provider_invoked -eq 0 -and
    [bool]$prefixSummary.expected_prompt_match -and
    [bool]$prefixSummary.expected_output_tokens_match -and
    [bool]$prefixSummary.stream_matches_output -and
    [bool]$prefixSummary.callbacks_before_return -and
    [bool]$prefixSummary.state_restored -and
    $prefixContinuationExact -and
    $prefixRoutePass
)
$prefixPerformancePass = $prefixProviderTtftPass -and `
    $prefixNumericalMetricsPass
$prefixContinuationPass = $prefixCorrectnessPass -and `
    $prefixPerformancePass
$nativeLogit = if ($null -ne $summary) {
    [double]$summary.first_token_raw_logit
} else {
    [double]::NaN
}
$nativeTtftMs = if ($null -ne $summary -and
    $null -ne $summary.ttft_ms) {
    [double]$summary.ttft_ms
} else {
    [double]::NaN
}
$nativePrefillTokensPerSecond = if ($null -ne $summary -and
    $null -ne $summary.prefill_tokens_per_second) {
    [double]$summary.prefill_tokens_per_second
} else {
    [double]::NaN
}
$nativeTpotMs = if ($null -ne $summary -and $null -ne $summary.tpot_ms) {
    [double]$summary.tpot_ms
} else {
    [double]::NaN
}
$nativeDecodeTokensPerSecond = if ($null -ne $summary -and
    $null -ne $summary.decode_tokens_per_second) {
    [double]$summary.decode_tokens_per_second
} else {
    [double]::NaN
}
$nativeOutputTokens = if ($null -ne $summary) {
    @($summary.output_token_ids | ForEach-Object { [int]$_ })
} else {
    @()
}
$nativeContinuationExact = `
    $nativeOutputTokens.Count -eq $gb10OutputTokens.Count
if ($nativeContinuationExact) {
    for ($tokenIndex = 0; $tokenIndex -lt $gb10OutputTokens.Count; `
            $tokenIndex++) {
        if ($nativeOutputTokens[$tokenIndex] -ne `
                $gb10OutputTokens[$tokenIndex]) {
            $nativeContinuationExact = $false
            break
        }
    }
}
$modelEngineLoadMs = if ($null -ne $summary -and
    $null -ne $summary.engine_load_ms) {
    [double]$summary.engine_load_ms
} else {
    [double]::NaN
}
$derivedPrefillTokensPerSecond = if (
    -not [double]::IsNaN($nativeTtftMs) -and
    -not [double]::IsInfinity($nativeTtftMs) -and
    $nativeTtftMs -gt 0.0
) {
    8192.0 * 1000.0 / $nativeTtftMs
} else {
    [double]::NaN
}
$prefillMetricRelativeError = if (
    -not [double]::IsNaN($derivedPrefillTokensPerSecond) -and
    -not [double]::IsNaN($nativePrefillTokensPerSecond) -and
    $derivedPrefillTokensPerSecond -gt 0.0
) {
    [Math]::Abs(
        $nativePrefillTokensPerSecond - $derivedPrefillTokensPerSecond
    ) / $derivedPrefillTokensPerSecond
} else {
    [double]::NaN
}
$logitDifference = [Math]::Abs($nativeLogit - $gb10Logit)
$correctnessPass = (
    $exitCode -eq 0 -and
    $null -ne $summary -and
    [string]$summary.status -eq 'pass' -and
    [bool]$summary.first_token_report_available -and
    [bool]$summary.first_token_report_matches_output -and
    [bool]$summary.first_token_raw_logit_available -and
    [bool]$summary.expected_prompt_match -and
    [bool]$summary.expected_output_tokens_match -and
    [bool]$summary.stream_matches_output -and
    [bool]$summary.callbacks_before_return -and
    [bool]$summary.state_restored -and
    [int]$summary.output_tokens -eq $gb10ContinuationTokenCount -and
    $nativeContinuationExact -and
    [int]$summary.first_token_report_token_id -eq $gb10FirstToken -and
    $logitDifference -le $gb10LogitTolerance
)
$performancePass = (
    $exitCode -eq 0 -and
    -not [double]::IsNaN($nativeTtftMs) -and
    -not [double]::IsInfinity($nativeTtftMs) -and
    -not [double]::IsNaN($nativePrefillTokensPerSecond) -and
    -not [double]::IsInfinity($nativePrefillTokensPerSecond) -and
    -not [double]::IsNaN($nativeTpotMs) -and
    -not [double]::IsInfinity($nativeTpotMs) -and
    -not [double]::IsNaN($nativeDecodeTokensPerSecond) -and
    -not [double]::IsInfinity($nativeDecodeTokensPerSecond) -and
    -not [double]::IsNaN($prefillMetricRelativeError) -and
    $prefillMetricRelativeError -le 0.001 -and
    $nativeTtftMs -le $q8192TtftMaxMs -and
    $nativePrefillTokensPerSecond -ge $q8192PrefillMinTokensPerSecond -and
    $nativeTpotMs -le $q8192TpotMaxMs -and
    $nativeDecodeTokensPerSecond -ge $q8192DecodeMinTokensPerSecond
)
$modelEngineLoadPass = (
    $exitCode -eq 0 -and
    -not [double]::IsNaN($modelEngineLoadMs) -and
    -not [double]::IsInfinity($modelEngineLoadMs) -and
    $modelEngineLoadMs -le $modelEngineLoadMaxMs
)
$acceptancePass = $correctnessPass -and $performancePass -and `
    $modelEngineLoadPass -and $routePass -and $prefixContinuationPass
$record = [ordered]@{
    schema_version = 1
    record_type = 'qrt_q8192_gb10_product_acceptance'
    host = [Environment]::MachineName
    repo_commit = $repoCommit
    dirty_tree = @(& git -C $Repo status --porcelain).Count -ne 0
    command_file = $PSCommandPath
    command = @($productExe) + $arguments
    prefix_continuation_command = @($productExe) + $prefixArguments
    gdn_route = $GdnRoute
    gdn_qualification_label = $GdnQualificationLabel
    moe_arithmetic = $MoeArithmetic
    q1_attention = $Q1Attention
    attention_build_label = $AttentionBuildLabel
    dynamic_moe_build_label = $DynamicMoeBuildLabel
    dynamic_moe_native_shared_selected = `
        [bool]$DynamicMoeNativeSharedSelected
    dynamic_moe_base_aot_mode = $recordedDynamicMoeBaseAotMode
    dynamic_moe_base_aot_block_m = `
        [int]$dynamicMoeEvidence.base_aot_block_m
    dynamic_moe_base_aot_group_m = `
        [int]$dynamicMoeEvidence.base_aot_group_m
    dynamic_moe_base_aot_dynamic_logical_abi = `
        [bool]$dynamicMoeEvidence.base_aot_dynamic_logical_abi
    dynamic_moe_required_runtime_environment = `
        $expectedDynamicMoeRuntimeEnvironmentRecords
    layer_output_trace = [bool]$LayerOutputTrace
    model_path = $modelPath
    prompt_u32le_sha256 = $promptMetadata.u32le_sha256
    prompt_u32le_fnv1a64 = $promptMetadata.u32le_fnv1a64
    gb10_oracle = $gb10OracleSnapshotPath
    gb10_oracle_sha256 = $gb10OracleSha256
    gb10_oracle_capture_request_sha256 = `
        [string]$gb10Oracle.capture.request_sha256
    gb10_oracle_capture_response_sha256 = `
        [string]$gb10Oracle.capture.response_sha256
    gb10_oracle_capture_command_file_sha256 = $gb10CaptureScriptSha256
    gb10_continuation_capture_request_sha256 = `
        [string]$gb10Oracle.continuation_capture.request_sha256
    gb10_continuation_capture_response_sha256 = `
        [string]$gb10Oracle.continuation_capture.response_sha256
    gb10_continuation_capture_command_file_sha256 = `
        $gb10ContinuationCaptureScriptSha256
    expected_first_token = $gb10FirstToken
    gb10_first_token = $gb10FirstToken
    gb10_first_token_raw_logit = $gb10Logit
    native_first_token_raw_logit = $nativeLogit
    first_token_raw_logit_absolute_difference = $logitDifference
    first_token_raw_logit_tolerance = $gb10LogitTolerance
    expected_output_tokens = $gb10OutputTokens
    native_output_tokens = $nativeOutputTokens
    expected_output_token_ids_u32le_sha256 = $gb10OutputSha256
    expected_output_token_ids_u32le_fnv1a64 = $gb10OutputFnv1a64
    continuation_token_count = $gb10ContinuationTokenCount
    decode_step_count = $gb10DecodeStepCount
    engine_self_hashes_diagnostic_only = $true
    continuation_token_for_token_pass = $nativeContinuationExact
    prefix_continuation_token_for_token_pass = $prefixContinuationExact
    prefix_continuation_correctness_pass = $prefixCorrectnessPass
    prefix_continuation_performance_pass = $prefixPerformancePass
    prefix_continuation_pass = $prefixContinuationPass
    prefix_tokens = $prefixTokens
    prefix_suffix_tokens = $prefixSuffixTokens
    prefix_hit_count = $prefixHitCount
    prefix_output_tokens = $prefixOutputTokens
    prefix_hit_provider_ttft_ms = $prefixProviderTtftValues
    prefix_hit_provider_ttft_pass = $prefixProviderTtftPass
    prefix_tpot_ms = $prefixTpotMs
    prefix_decode_tokens_per_second = $prefixDecodeTokensPerSecond
    prefix_numerical_metrics_pass = $prefixNumericalMetricsPass
    prefix_model_engine_load_ms = $prefixModelEngineLoadMs
    prefix_forbidden_route_markers = $prefixForbiddenMarkers
    prefix_observed_forbidden_route_markers = `
        $prefixObservedForbiddenMarkers
    prefix_request_route_line_count = $prefixRequestRouteLines.Count
    prefix_decode_complete_route_line_count = `
        $prefixDecodeCompleteRouteLines.Count
    prefix_route_pass = $prefixRoutePass
    correctness_pass = $correctnessPass
    q8192_ttft_ms = $nativeTtftMs
    q8192_ttft_max_ms = $q8192TtftMaxMs
    q8192_prefill_tokens_per_second = $nativePrefillTokensPerSecond
    q8192_derived_prefill_tokens_per_second = `
        $derivedPrefillTokensPerSecond
    q8192_prefill_metric_relative_error = $prefillMetricRelativeError
    q8192_prefill_metric_relative_error_max = 0.001
    q8192_prefill_min_tokens_per_second = `
        $q8192PrefillMinTokensPerSecond
    q8192_tpot_ms = $nativeTpotMs
    q8192_tpot_max_ms = $q8192TpotMaxMs
    q8192_decode_tokens_per_second = $nativeDecodeTokensPerSecond
    q8192_decode_min_tokens_per_second = `
        $q8192DecodeMinTokensPerSecond
    performance_pass = $performancePass
    model_engine_load_ms = $modelEngineLoadMs
    model_engine_load_max_ms = $modelEngineLoadMaxMs
    model_engine_load_pass = $modelEngineLoadPass
    required_route_markers = $requiredRouteMarkers
    missing_route_markers = $missingRouteMarkers
    forbidden_route_markers = $forbiddenRouteMarkers
    observed_forbidden_route_markers = $observedForbiddenRouteMarkers
    q8192_moe_fixed_route_line_count = $q8192MoeRouteLines.Count
    q8192_moe_fixed_route_pass = $q8192MoeFixedRoutePass
    decode_token_route_line_count = $decodeTokenRouteLines.Count
    decode_complete_route_line_count = $decodeCompleteRouteLines.Count
    decode_greedy_route_pass = $decodeGreedyRoutePass
    decode_route_pass = $decodeRoutePass
    route_pass = $routePass
    acceptance_pass = $acceptancePass
    process_exit_code = $exitCode
    process_wall_ms = [double]$baseRun.wall_ms
    prefix_process_exit_code = $prefixExitCode
    prefix_process_wall_ms = [double]$prefixRun.wall_ms
    environment_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $envPath
    ).Hash.ToLowerInvariant()
    product_executable_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $productExe
    ).Hash.ToLowerInvariant()
    whole_provider_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $wholeProvider
    ).Hash.ToLowerInvariant()
    attention_provider_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $attentionProvider
    ).Hash.ToLowerInvariant()
    attention_build_provenance = $attentionBuildProvenance
    attention_build_provenance_sha256 = `
        $attentionBuildProvenanceSha256
    attention_dynamic_logical_case_count = `
        [int]$attentionEvidence.dynamic_logical_case_count
    attention_dynamic_logical_reset_included = `
        [bool]$attentionEvidence.dynamic_logical_reset_included
    attention_dynamic_logical_component_only = `
        [bool]$attentionEvidence.dynamic_logical_component_only
    gdn_build_provenance = if ($GdnRoute -eq 'aiter') {
        $gdnQualification
    } else { '' }
    gdn_build_provenance_sha256 = $gdnQualificationSha256
    gdn_provider = if ($GdnRoute -eq 'aiter') {
        [string]$gdnEvidence.provider
    } else { '' }
    gdn_provider_sha256 = if ($GdnRoute -eq 'aiter') {
        [string]$gdnEvidence.provider_sha256
    } else { '' }
    gdn_kernel_dir = if ($GdnRoute -eq 'aiter') {
        [string]$gdnEvidence.kernel_dir
    } else { '' }
    gdn_dynamic_logical_case_count = if ($GdnRoute -eq 'aiter') {
        [int]$gdnEvidence.dynamic_logical_case_count
    } else { 0 }
    gdn_dynamic_logical_mode_count = if ($GdnRoute -eq 'aiter') {
        [int]$gdnEvidence.dynamic_logical_mode_count
    } else { 0 }
    gdn_dynamic_logical_reset_included = if ($GdnRoute -eq 'aiter') {
        [bool]$gdnEvidence.dynamic_logical_reset_included
    } else { $null }
    gdn_dynamic_logical_component_only = if ($GdnRoute -eq 'aiter') {
        [bool]$gdnEvidence.component_only
    } else { $null }
    gdn_artifacts = $gdnArtifacts
    q8192_moe_build_provenance = $dynamicMoeQualification
    q8192_moe_build_provenance_sha256 = `
        $dynamicMoeQualificationSha256
    q8192_moe_provider = $dynamicMoeProvider
    q8192_moe_provider_sha256 = $dynamicMoeProviderSha256
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
    q8192_moe_artifacts = $dynamicMoeArtifacts
    q1_moe_triton_0626_gate_up_sha256 = $q1MoeGateUpSha256
    q1_moe_triton_0626_down_sha256 = $q1MoeDownSha256
    product_summary = $summary
    prefix_product_summary = $prefixSummary
    stdout = $stdoutPath
    stderr = $stderrPath
    prefix_stdout = $prefixStdoutPath
    prefix_stderr = $prefixStderrPath
}
$recordPath = Join-Path $outputDir 'run-record.json'
[IO.File]::WriteAllText(
    $recordPath,
    ($record | ConvertTo-Json -Depth 10) + "`n",
    $utf8
)
$record | ConvertTo-Json -Depth 10 -Compress
if ($exitCode -ne 0) {
    exit $exitCode
}
if (-not [bool]$record.correctness_pass) {
    exit 3
}
if (-not [bool]$record.performance_pass) {
    exit 3
}
if (-not [bool]$record.model_engine_load_pass) {
    exit 3
}
if (-not [bool]$record.route_pass) {
    exit 3
}
if (-not [bool]$record.prefix_continuation_pass) {
    exit 3
}
