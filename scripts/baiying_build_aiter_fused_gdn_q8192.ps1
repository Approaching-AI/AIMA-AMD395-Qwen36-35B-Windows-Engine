param(
    [Parameter(Mandatory = $false)][string]$WslDistribution = "Ubuntu-24.04",
    [Parameter(Mandatory = $false)][string]$TritonPython = "/opt/qwen36-vllm/bin/python",
    [Parameter(Mandatory = $false)][string]$HipccPath = "",
    [Parameter(Mandatory = $false)][string]$BuildDir = "",
    [Parameter(Mandatory = $false)][string]$OffloadArch = "gfx1151",
    [Parameter(Mandatory = $false)][int]$Repetitions = 3,
    [Parameter(Mandatory = $false)][int]$SeededSuffixRepetitions = 11
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $repo "build\aiter-fused-gdn-aot"
}
if ([string]::IsNullOrWhiteSpace($HipccPath)) {
    $HipccPath = (Get-Command hipcc.exe -ErrorAction Stop).Source
}
if ($Repetitions -lt 3) {
    throw "Repetitions must be at least 3"
}
if ($SeededSuffixRepetitions -lt 11) {
    throw "SeededSuffixRepetitions must be at least 11"
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "cannot convert non-drive Windows path to WSL: $resolved"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $tail = $Matches[2].Replace('\', '/')
    return "/mnt/$drive/$tail"
}

$generator = Join-Path $repo "native\generators\compile_q8192_aiter_fused_gdn.py"
$seededGenerator = Join-Path `
    $repo `
    "native\generators\compile_q1024_seeded_aiter_fused_gdn.py"
$providerSource = Join-Path $repo "native\providers\gdn\qrt_aiter_fused_gdn_q8192_provider.cpp"
$smokeSource = Join-Path $repo "native\providers\gdn\q8192_aiter_fused_gdn_smoke.cpp"
$seededSuffixSmokeSource = Join-Path `
    $repo `
    "native\providers\gdn\q16384_suffix1024_seeded_aiter_fused_gdn_smoke.cpp"
$chunkedBf16SmokeSource = Join-Path `
    $repo `
    "native\providers\gdn\q65536_chunked_seeded_bf16_aiter_fused_gdn_smoke.cpp"
$license = Join-Path $repo "native\providers\gdn\LICENSE.amd-aiter"
foreach ($required in @(
    $generator,
    $seededGenerator,
    $providerSource,
    $smokeSource,
    $seededSuffixSmokeSource,
    $chunkedBf16SmokeSource,
    $license
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "required AITER fused-GDN input not found: $required"
    }
}

$wslGenerator = Convert-ToWslPath $generator
$wslSeededGenerator = Convert-ToWslPath $seededGenerator
$wslBuildDir = Convert-ToWslPath $BuildDir
$wslMetadata = "$wslBuildDir/metadata.json"
$wslSeededMetadata = "$wslBuildDir/seeded-metadata.json"
& wsl.exe -d $WslDistribution -- $TritonPython $wslGenerator `
    --output-dir $wslBuildDir --metadata $wslMetadata
if ($LASTEXITCODE -ne 0) {
    throw "Triton AOT generation exited $LASTEXITCODE"
}
& wsl.exe -d $WslDistribution -- $TritonPython $wslSeededGenerator `
    --output-dir $wslBuildDir --metadata $wslSeededMetadata
if ($LASTEXITCODE -ne 0) {
    throw "seeded suffix Triton AOT generation exited $LASTEXITCODE"
}

$providerDll = Join-Path $BuildDir "qrt_aiter_fused_gdn_q8192_provider.dll"
$smokeExe = Join-Path $BuildDir "q8192_aiter_fused_gdn_smoke.exe"
$seededSuffixSmokeExe = Join-Path `
    $BuildDir `
    "q16384_suffix1024_seeded_aiter_fused_gdn_smoke.exe"
$chunkedBf16SmokeExe = Join-Path `
    $BuildDir `
    "q65536_chunked_seeded_bf16_aiter_fused_gdn_smoke.exe"
& $HipccPath -std=c++17 -O3 "--offload-arch=$OffloadArch" `
    -shared $providerSource -o $providerDll
if ($LASTEXITCODE -ne 0) {
    throw "hipcc exited $LASTEXITCODE while building the fused-GDN provider"
}
& $HipccPath -std=c++17 -O3 "--offload-arch=$OffloadArch" `
    $smokeSource -o $smokeExe
if ($LASTEXITCODE -ne 0) {
    throw "hipcc exited $LASTEXITCODE while building the fused-GDN smoke"
}
& $HipccPath -std=c++17 -O3 "--offload-arch=$OffloadArch" `
    $seededSuffixSmokeSource -o $seededSuffixSmokeExe
if ($LASTEXITCODE -ne 0) {
    throw "hipcc exited $LASTEXITCODE while building the seeded suffix fused-GDN smoke"
}
& $HipccPath -std=c++17 -O3 "--offload-arch=$OffloadArch" `
    $chunkedBf16SmokeSource -o $chunkedBf16SmokeExe
if ($LASTEXITCODE -ne 0) {
    throw "hipcc exited $LASTEXITCODE while building the chunked BF16 fused-GDN smoke"
}

$smokeOutputQ8191 = & $smokeExe $BuildDir $providerDll $Repetitions 8191
$smokeExitCodeQ8191 = $LASTEXITCODE
$smokeOutputQ8192 = & $smokeExe $BuildDir $providerDll $Repetitions 8192
$smokeExitCodeQ8192 = $LASTEXITCODE
$smokeOutputQ8193 = & $smokeExe $BuildDir $providerDll $Repetitions 8193
$smokeExitCodeQ8193 = $LASTEXITCODE
$smokeOutputQ16384 = & $smokeExe $BuildDir $providerDll $Repetitions 16384
$smokeExitCodeQ16384 = $LASTEXITCODE
$seededSuffixSmokeOutput = & $seededSuffixSmokeExe `
    $BuildDir `
    $providerDll `
    $SeededSuffixRepetitions
$seededSuffixSmokeExitCode = $LASTEXITCODE
$chunkedBf16SmokeOutput = & $chunkedBf16SmokeExe $BuildDir $providerDll
$chunkedBf16SmokeExitCode = $LASTEXITCODE
$smokeExitCode = if (
    $smokeExitCodeQ8191 -eq 0 -and
    $smokeExitCodeQ8192 -eq 0 -and
    $smokeExitCodeQ8193 -eq 0 -and
    $smokeExitCodeQ16384 -eq 0 -and
    $seededSuffixSmokeExitCode -eq 0 -and
    $chunkedBf16SmokeExitCode -eq 0
) { 0 } else { 1 }
$smokeText = @(
    $smokeOutputQ8191
    $smokeOutputQ8192
    $smokeOutputQ8193
    $smokeOutputQ16384
) -join "`n"
$asyncParityModeCount = [regex]::Matches(
    $smokeText,
    '(?m)\basync_exact=1\b'
).Count
if ($smokeExitCode -eq 0 -and $asyncParityModeCount -ne 8) {
    throw "AITER fused-GDN smoke did not report exact repeat parity for all four shapes and both gate modes"
}
$seededSuffixPassCount = [regex]::Matches(
    ($seededSuffixSmokeOutput -join "`n"),
    '(?m)\bpass=1\b'
).Count
if ($smokeExitCode -eq 0 -and $seededSuffixPassCount -ne 2) {
    throw "seeded suffix fused-GDN smoke did not pass both gate modes"
}
$chunkedBf16PassCount = [regex]::Matches(
    ($chunkedBf16SmokeOutput -join "`n"),
    '(?m)\bpass=1\b'
).Count
if ($smokeExitCode -eq 0 -and $chunkedBf16PassCount -ne 2) {
    throw "chunked BF16 fused-GDN smoke did not pass both gate modes"
}
$artifactNames = @(
    "q8192_aiter_fused_gdn.hsaco",
    "q262144_aiter_fused_gdn_bf16.hsaco",
    "q1024_seeded_aiter_fused_gdn.hsaco",
    "q32768_seeded_aiter_fused_gdn_bf16.hsaco",
    "qrt_aiter_fused_gdn_q8192_provider.dll",
    "q8192_aiter_fused_gdn_smoke.exe",
    "q16384_suffix1024_seeded_aiter_fused_gdn_smoke.exe",
    "q65536_chunked_seeded_bf16_aiter_fused_gdn_smoke.exe",
    "metadata.json",
    "seeded-metadata.json"
)
$artifacts = foreach ($name in $artifactNames) {
    $path = Join-Path $BuildDir $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "AITER fused-GDN build output is missing: $path"
    }
    $item = Get-Item -LiteralPath $path
    [ordered]@{
        file = $name
        bytes = $item.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    }
}

[ordered]@{
    host = [Environment]::MachineName
    offload_arch = $OffloadArch
    wsl_distribution = $WslDistribution
    triton_python = $TritonPython
    hipcc = $HipccPath
    build_dir = (Resolve-Path -LiteralPath $BuildDir).Path
    repetitions = $Repetitions
    seeded_suffix_repetitions = $SeededSuffixRepetitions
    smoke_exit_code = $smokeExitCode
    q8191_smoke_exit_code = $smokeExitCodeQ8191
    q8192_smoke_exit_code = $smokeExitCodeQ8192
    q8193_smoke_exit_code = $smokeExitCodeQ8193
    q16384_smoke_exit_code = $smokeExitCodeQ16384
    seeded_suffix_smoke_exit_code = $seededSuffixSmokeExitCode
    chunked_bf16_smoke_exit_code = $chunkedBf16SmokeExitCode
    async_parity_mode_count = $asyncParityModeCount
    seeded_suffix_pass_count = $seededSuffixPassCount
    chunked_bf16_pass_count = $chunkedBf16PassCount
    smoke = $smokeText
    seeded_suffix_smoke = ($seededSuffixSmokeOutput -join "`n")
    chunked_bf16_smoke = ($chunkedBf16SmokeOutput -join "`n")
    artifacts = $artifacts
} | ConvertTo-Json -Depth 6

if ($smokeExitCode -ne 0) {
    throw "AITER fused-GDN smoke suite failed"
}
