param(
    [Parameter(Mandatory = $true)][string]$CkRoot,
    [Parameter(Mandatory = $false)][string]$OutPath = "",
    [Parameter(Mandatory = $false)][string]$OffloadArch = "gfx1151",
    [Parameter(Mandatory = $false)][string]$HipccPath = "",
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

$arguments = @(
    "-std=c++17",
    "-O3",
    "--offload-arch=$OffloadArch",
    "-DCK_TILE_FMHA_FWD_FAST_EXP2=0",
    "-I", $ckInclude,
    "-I", $ckExample,
    "-shared",
    (Join-Path $sourceDir "qrt_ck_fmha_q8192_provider.cpp"),
    (Join-Path $sourceDir "fmha_fwd_api.cpp"),
    (Join-Path $sourceDir "fmha_fwd_gfx1151_d256_bf16_f32out.cpp"),
    "-o", $OutPath
)

& $HipccPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "hipcc exited $LASTEXITCODE while building the q8192 CK-Tile provider"
}

$directSmokeOutput = ""
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
