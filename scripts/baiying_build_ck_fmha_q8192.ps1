param(
    [Parameter(Mandatory = $true)][string]$CkRoot,
    [Parameter(Mandatory = $false)][string]$OutPath = "",
    [Parameter(Mandatory = $false)][string]$OffloadArch = "gfx1151",
    [Parameter(Mandatory = $false)][string]$HipccPath = "",
    [Parameter(Mandatory = $false)][ValidateSet(0, 1)][int]$DppReduction = 0,
    [Parameter(Mandatory = $false)][switch]$Sm121InterpolatedExp2,
    [Parameter(Mandatory = $false)][int]$RunDirectSmoke = 0,
    [Parameter(Mandatory = $false)][string]$DirectSmokePath = "",
    [Parameter(Mandatory = $false)][ValidateRange(1, 100)][int]$DirectSmokeRepetitions = 5,
    [Parameter(Mandatory = $false)][int]$RunQ16384MetamorphicSmoke = 0,
    [Parameter(Mandatory = $false)][string]$Q16384MetamorphicSmokePath = "",
    [Parameter(Mandatory = $false)]
        [ValidateRange(1, 100)]
        [int]$Q16384MetamorphicSmokeRepetitions = 3,
    [Parameter(Mandatory = $false)][int]$RunQ16384Suffix1024Smoke = 0,
    [Parameter(Mandatory = $false)][string]$Q16384Suffix1024SmokePath = "",
    [Parameter(Mandatory = $false)]
        [ValidateRange(5, 100)]
        [int]$Q16384Suffix1024SmokeRepetitions = 5
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $repo "native\providers\ck_fmha"
$directSmokeSource = Join-Path $sourceDir "q8192_ck_fmha_direct_smoke.cpp"
$q16384MetamorphicSmokeSource = Join-Path `
    $sourceDir `
    "q16384_ck_fmha_metamorphic_smoke.cpp"
$q16384Suffix1024SmokeSource = Join-Path `
    $sourceDir `
    "q16384_suffix1024_ck_fmha_smoke.cpp"
$ckInclude = Join-Path $CkRoot "include"
$ckExample = Join-Path $CkRoot "example\ck_tile\01_fmha"
$ckArchHeader = Join-Path $ckInclude "ck_tile\core\arch\arch.hpp"
if (-not (Test-Path -LiteralPath $ckExample -PathType Container)) {
    $ckExample = Join-Path $CkRoot "01_fmha"
}

if ([string]::IsNullOrWhiteSpace($OutPath)) {
    $OutPath = Join-Path $repo `
        "native\aot\gfx1151\q8192_ck_tile_fmha_bf16_f32out.dll"
}
if ([string]::IsNullOrWhiteSpace($DirectSmokePath)) {
    $DirectSmokePath = Join-Path $repo `
        "build\ck-fmha\q8192_ck_fmha_direct_smoke.exe"
}
if ([string]::IsNullOrWhiteSpace($Q16384MetamorphicSmokePath)) {
    $Q16384MetamorphicSmokePath = Join-Path $repo `
        "build\ck-fmha\q16384_ck_fmha_metamorphic_smoke.exe"
}
if ([string]::IsNullOrWhiteSpace($Q16384Suffix1024SmokePath)) {
    $Q16384Suffix1024SmokePath = Join-Path $repo `
        "build\ck-fmha\q16384_suffix1024_ck_fmha_smoke.exe"
}
if ([string]::IsNullOrWhiteSpace($HipccPath)) {
    $hipcc = Get-Command hipcc.exe -ErrorAction Stop
    $HipccPath = $hipcc.Source
}

foreach ($required in @(
    (Join-Path $ckInclude "ck_tile\core.hpp"),
    (Join-Path $ckExample "fmha_fwd.hpp"),
    (Join-Path $sourceDir "qrt_ck_fmha_q8192_provider.cpp"),
    (Join-Path $sourceDir "blackwell_attention.h"),
    (Join-Path $sourceDir "..\sm121_attention_capacity.h"),
    (Join-Path $sourceDir "..\moe_accumulator\sm121_native_product.h"),
    (Join-Path $sourceDir "..\moe_accumulator\sm121_mantissa_parts.h"),
    (Join-Path $sourceDir "..\moe_accumulator\sm121_integer_parts.h"),
    (Join-Path $sourceDir "..\moe_accumulator\sm121_pv_error_bound.h"),
    (Join-Path $sourceDir "..\moe_accumulator\sm121_prepared_bf16.h"),
    (Join-Path $sourceDir "..\gdn\sm121_exp2_table.h"),
    (Join-Path $sourceDir "..\gdn\sm121_exp2_interpolated.h"),
    (Join-Path $sourceDir "..\gdn\sm121_attention_rcp.h"),
    (Join-Path $sourceDir "fmha_fwd_api.cpp"),
    (Join-Path $sourceDir "fmha_fwd_gfx1151_d256_bf16_f32out.cpp"),
    $directSmokeSource,
    $q16384MetamorphicSmokeSource,
    $q16384Suffix1024SmokeSource
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "required CK-Tile provider input not found: $required"
    }
}

$ckArchSource = Get-Content -Raw -LiteralPath $ckArchHeader
if ($ckArchSource -match "struct\s+gfx115_t") {
    $ckArchType = "ck_tile::gfx115_t"
} elseif ($ckArchSource -match "struct\s+gfx11_t") {
    $ckArchType = "ck_tile::gfx11_t"
} else {
    throw "CK-Tile architecture type for gfx1151 was not found in $ckArchHeader"
}

$arguments = @(
    "-std=c++17",
    "-DQRT_SM121_DPP_REDUCTION=$DppReduction",
    "-O3",
    "--offload-arch=$OffloadArch",
    "-DCK_TILE_FMHA_FWD_FAST_EXP2=0",
    "-DQRT_CK_FMHA_VLLM_N32=1",
    "-DQRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL=1",
    "-DQRT_CK_ARCH_TYPE=$ckArchType",
    "-I", $ckInclude,
    "-I", $ckExample,
    "-shared",
    "-lbcrypt",
    (Join-Path $sourceDir "qrt_ck_fmha_q8192_provider.cpp"),
    (Join-Path $sourceDir "fmha_fwd_api.cpp"),
    (Join-Path $sourceDir "fmha_fwd_gfx1151_d256_bf16_f32out.cpp"),
    "-o", $OutPath
)

if ($Sm121InterpolatedExp2) { $arguments += "-DQRT_CK_SM121_INTERPOLATED_EXP2=1" }
& $HipccPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "hipcc exited $LASTEXITCODE while building the q8192 CK-Tile provider"
}

$directSmokeOutput = ""
$directSmokeDynamicLogicalCaseCount = 0
$directSmokeDynamicLogicalResetIncluded = $null
$directSmokeDynamicLogicalTokens = @()
if ($RunDirectSmoke -ne 0) {
    $directSmokeArguments = @(
        "-std=c++17",
        "-O3",
        "--offload-arch=$OffloadArch",
        $directSmokeSource,
        "-o", $DirectSmokePath
    )
    & $HipccPath @directSmokeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "hipcc exited $LASTEXITCODE while building the q8192 CK-Tile direct smoke"
    }
    $resolvedProvider = (Resolve-Path -LiteralPath $OutPath).Path
    $directSmokeOutput = (& $DirectSmokePath `
        $resolvedProvider `
        $DirectSmokeRepetitions 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "q8192 CK-Tile direct smoke exited $LASTEXITCODE`: $directSmokeOutput"
    }
    $dynamicLogicalMatches = [regex]::Matches(
        $directSmokeOutput,
        '(?m)^ck_fmha_dynamic_logical_case\s+' +
        'tokens=(?<tokens>\d+)\b[^\r\n]*' +
        '\breset_included=0\b[^\r\n]*' +
        '\bcomponent_only=1\b[^\r\n]*' +
        '\binference_success_claimed=0\b[^\r\n]*' +
        '\btiming_finite_positive=1\b[^\r\n]*' +
        '\btotal_nonfinite=0\b' +
        '[^\r\n]*\bclose=1\s*$'
    )
    $directSmokeDynamicLogicalCaseCount = $dynamicLogicalMatches.Count
    if ($directSmokeDynamicLogicalCaseCount -ne 32) {
        throw "q8192 CK-Tile direct smoke reported $directSmokeDynamicLogicalCaseCount/32 passing dynamic-logical cases"
    }
    $directSmokeDynamicLogicalTokens = @(
        $dynamicLogicalMatches | ForEach-Object {
            [int]$_.Groups['tokens'].Value
        }
    )
    $expectedDynamicLogicalTokens = @(
        2073, 2156, 2560, 3073, 4609, 6145, 2049, 2175,
        2176, 2177, 2559, 2561, 3071, 3072, 3583, 3584,
        3585, 4095, 4096, 4097, 4607, 4608, 6143, 6144,
        7167, 7168, 7169, 7679, 7680, 7681, 8191, 8192
    )
    if (($directSmokeDynamicLogicalTokens -join ',') -ne
            ($expectedDynamicLogicalTokens -join ',')) {
        throw "q8192 CK-Tile direct smoke dynamic-logical token sequence is incomplete or reordered"
    }
    if ($directSmokeOutput -notmatch (
        '(?m)^ck_fmha_dynamic_logical_smoke\s+' +
        'cases=32\s+all_close=1\s+' +
        'dynamic_logical_reset_included=0\s+' +
        'terminal_authority=product_gb10\s+' +
        'component_only=1\s+inference_success_claimed=0\s*$'
    )) {
        throw "q8192 CK-Tile direct smoke is missing the passing dynamic-logical summary"
    }
    $directSmokeDynamicLogicalResetIncluded = $false
}

$q16384MetamorphicSmokeOutput = ""
if ($RunQ16384MetamorphicSmoke -ne 0) {
    $q16384MetamorphicSmokeArguments = @(
        "-std=c++17",
        "-O3",
        "--offload-arch=$OffloadArch",
        $q16384MetamorphicSmokeSource,
        "-o", $Q16384MetamorphicSmokePath
    )
    & $HipccPath @q16384MetamorphicSmokeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "hipcc exited $LASTEXITCODE while building the q16384 CK-Tile metamorphic smoke"
    }
    $resolvedProvider = (Resolve-Path -LiteralPath $OutPath).Path
    $q16384MetamorphicSmokeOutput = (& $Q16384MetamorphicSmokePath `
        $resolvedProvider `
        $Q16384MetamorphicSmokeRepetitions 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "q16384 CK-Tile metamorphic smoke exited $LASTEXITCODE`: $q16384MetamorphicSmokeOutput"
    }
}

$q16384Suffix1024SmokeOutput = ""
if ($RunQ16384Suffix1024Smoke -ne 0) {
    $q16384Suffix1024SmokeArguments = @(
        "-std=c++17",
        "-O3",
        "--offload-arch=$OffloadArch",
        $q16384Suffix1024SmokeSource,
        "-o", $Q16384Suffix1024SmokePath
    )
    & $HipccPath @q16384Suffix1024SmokeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "hipcc exited $LASTEXITCODE while building the q16384 suffix1024 CK-Tile smoke"
    }
    $resolvedProvider = (Resolve-Path -LiteralPath $OutPath).Path
    $q16384Suffix1024SmokeOutput = (& $Q16384Suffix1024SmokePath `
        $resolvedProvider `
        $Q16384Suffix1024SmokeRepetitions 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "q16384 suffix1024 CK-Tile smoke exited $LASTEXITCODE`: $q16384Suffix1024SmokeOutput"
    }
}

$item = Get-Item -LiteralPath $OutPath
$sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $OutPath).Hash.ToLowerInvariant()
[pscustomobject]@{
    path = $item.FullName
    bytes = $item.Length
    sha256 = $sha256
    offload_arch = $OffloadArch
    ck_root = (Resolve-Path -LiteralPath $CkRoot).Path
    ck_arch_type = $ckArchType
    ck_tile_n = 32
    blackwell_exact_terminal = $true
    sm121_dpp_reduction = ($DppReduction -ne 0)
    sm121_lane_reduce_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir '..\moe_accumulator\sm121_lane_reduce.h')).Hash.ToLowerInvariant()
    attention_capacity_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir "..\sm121_attention_capacity.h")).Hash.ToLowerInvariant()
    blackwell_attention_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir "blackwell_attention.h")).Hash.ToLowerInvariant()
    sm121_wave16_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir '..\moe_accumulator\sm121_wave16.h')).Hash.ToLowerInvariant()
    sm121_subgroup_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir '..\moe_accumulator\sm121_subgroup.h')).Hash.ToLowerInvariant()
    sm121_paired_products_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir '..\moe_accumulator\sm121_paired_products.h')).Hash.ToLowerInvariant()
    sm121_group16_modulo_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir '..\moe_accumulator\sm121_group16_modulo.h')).Hash.ToLowerInvariant()
    sm121_native_product_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir '..\moe_accumulator\sm121_native_product.h')).Hash.ToLowerInvariant()
    sm121_mantissa_parts_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir '..\moe_accumulator\sm121_mantissa_parts.h')).Hash.ToLowerInvariant()
    sm121_integer_parts_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir '..\moe_accumulator\sm121_integer_parts.h')).Hash.ToLowerInvariant()
    sm121_pv_error_bound_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir '..\moe_accumulator\sm121_pv_error_bound.h')).Hash.ToLowerInvariant()
    sm121_prepared_bf16_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir '..\moe_accumulator\sm121_prepared_bf16.h')).Hash.ToLowerInvariant()
    sm121_exp2_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir "..\gdn\sm121_exp2_table.h")).Hash.ToLowerInvariant()
    sm121_interpolated_exp2 = [bool]$Sm121InterpolatedExp2
    sm121_interpolated_exp2_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir "..\gdn\sm121_exp2_interpolated.h")).Hash.ToLowerInvariant()
    sm121_rcp_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $sourceDir "..\gdn\sm121_attention_rcp.h")).Hash.ToLowerInvariant()
    direct_smoke_ran = ($RunDirectSmoke -ne 0)
    direct_smoke_path = if ($RunDirectSmoke -ne 0) {
        (Resolve-Path -LiteralPath $DirectSmokePath).Path
    } else {
        ""
    }
    direct_smoke_repetitions = if ($RunDirectSmoke -ne 0) {
        $DirectSmokeRepetitions
    } else {
        0
    }
    direct_smoke_output = $directSmokeOutput
    direct_smoke_dynamic_logical_case_count = `
        $directSmokeDynamicLogicalCaseCount
    direct_smoke_dynamic_logical_reset_included = `
        $directSmokeDynamicLogicalResetIncluded
    direct_smoke_dynamic_logical_tokens = `
        $directSmokeDynamicLogicalTokens
    dynamic_logical_component_only = $true
    inference_success_claimed = $false
    q16384_metamorphic_smoke_ran = ($RunQ16384MetamorphicSmoke -ne 0)
    q16384_metamorphic_smoke_path = if ($RunQ16384MetamorphicSmoke -ne 0) {
        (Resolve-Path -LiteralPath $Q16384MetamorphicSmokePath).Path
    } else {
        ""
    }
    q16384_metamorphic_smoke_repetitions = if (
        $RunQ16384MetamorphicSmoke -ne 0
    ) {
        $Q16384MetamorphicSmokeRepetitions
    } else {
        0
    }
    q16384_metamorphic_smoke_output = $q16384MetamorphicSmokeOutput
    q16384_suffix1024_smoke_ran = ($RunQ16384Suffix1024Smoke -ne 0)
    q16384_suffix1024_smoke_path = if ($RunQ16384Suffix1024Smoke -ne 0) {
        (Resolve-Path -LiteralPath $Q16384Suffix1024SmokePath).Path
    } else {
        ""
    }
    q16384_suffix1024_smoke_repetitions = if (
        $RunQ16384Suffix1024Smoke -ne 0
    ) {
        $Q16384Suffix1024SmokeRepetitions
    } else {
        0
    }
    q16384_suffix1024_smoke_output = $q16384Suffix1024SmokeOutput
} | ConvertTo-Json
