param(
    [Parameter(Mandatory = $false)][string]$WslDistribution = "Ubuntu-24.04",
    [Parameter(Mandatory = $false)][string]$TritonPython = "/opt/qwen36-vllm/bin/python",
    [Parameter(Mandatory = $false)][string]$HipccPath = "",
    [Parameter(Mandatory = $false)][string]$BuildDir = "",
    [Parameter(Mandatory = $false)][string]$OutDir = "",
    [Parameter(Mandatory = $false)][string]$OffloadArch = "gfx1151",
    [Parameter(Mandatory = $false)][int]$Repetitions = 5,
    [Parameter(Mandatory = $false)][int]$BlockM = 64,
    [Parameter(Mandatory = $false)][int]$BlockN = 64,
    [Parameter(Mandatory = $false)][int]$GateBlockN = 0,
    [Parameter(Mandatory = $false)][int]$DownBlockN = 0,
    [Parameter(Mandatory = $false)][int]$GateBlockK = 64,
    [Parameter(Mandatory = $false)][int]$DownBlockK = 64,
    [Parameter(Mandatory = $false)][int]$GroupM = 8,
    [Parameter(Mandatory = $false)][int]$NumWarps = 4,
    [Parameter(Mandatory = $false)][int]$NumStages = 1,
    [Parameter(Mandatory = $false)][int]$WavesPerEu = 0,
    [Parameter(Mandatory = $false)][int]$GateNumWarps = 0,
    [Parameter(Mandatory = $false)][int]$GateNumStages = 0,
    [Parameter(Mandatory = $false)][int]$GateWavesPerEu = -1,
    [Parameter(Mandatory = $false)][int]$DownNumWarps = 0,
    [Parameter(Mandatory = $false)][int]$DownNumStages = 0,
    [Parameter(Mandatory = $false)][int]$DownWavesPerEu = -1,
    [Parameter(Mandatory = $false)]
        [ValidateRange(0, 1)]
        [int]$VllmSiluBf16Intermediate = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaRouted = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaGate = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaDown = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$TransposedRouter = 0,
    [Parameter(Mandatory = $false)][ValidateSet(16, 32, 64, 128, 256)][int]$RouterThreads = 256,
    [Parameter(Mandatory = $false)][ValidateSet(1, 2, 4, 8)][int]$RouterTokenTile = 1,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$FullV3FusedCombine = 0,
    [Parameter(Mandatory = $false)][ValidateSet(1, 4)][int]$FusedCombineWidth = 1,
    [Parameter(Mandatory = $false)][ValidateRange(1, 256)][int]$FullV3EventSlots = 16,
    [Parameter(Mandatory = $false)][string]$ReuseAotDir = "",
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$RequireExpectedFullProviderHash = 1
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $repo "build\triton-moe-aot"
}
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = $BuildDir
}
if ([string]::IsNullOrWhiteSpace($HipccPath)) {
    $HipccPath = (Get-Command hipcc.exe -ErrorAction Stop).Source
}
if ($Repetitions -le 0) {
    throw "Repetitions must be positive"
}
if ($GateBlockN -le 0) { $GateBlockN = $BlockN }
if ($DownBlockN -le 0) { $DownBlockN = $BlockN }
if ($GateNumWarps -le 0) { $GateNumWarps = $NumWarps }
if ($DownNumWarps -le 0) { $DownNumWarps = $NumWarps }
if ($GateNumStages -le 0) { $GateNumStages = $NumStages }
if ($DownNumStages -le 0) { $DownNumStages = $NumStages }
if ($GateWavesPerEu -lt 0) { $GateWavesPerEu = $WavesPerEu }
if ($DownWavesPerEu -lt 0) { $DownWavesPerEu = $WavesPerEu }
if ($NativeWmmaRouted -ne 0) {
    $NativeWmmaGate = 1
    $NativeWmmaDown = 1
}
if (($NativeWmmaGate -ne 0 -or $NativeWmmaDown -ne 0) -and $BlockM -ne 64) {
    throw "Native WMMA selected-MoE kernels require -BlockM 64."
}
if ($TransposedRouter -eq 0 -and
        ($RouterThreads -ne 256 -or $RouterTokenTile -ne 1)) {
    throw "RouterThreads below 256 or token tiles above 1 require -TransposedRouter 1."
}
if ($FullV3FusedCombine -eq 0 -and $FusedCombineWidth -ne 1) {
    throw "FusedCombineWidth above 1 requires -FullV3FusedCombine 1."
}

New-Item -ItemType Directory -Force -Path $BuildDir, $OutDir | Out-Null

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

$generator = Join-Path $repo (
    "native\generators\compile_q8192_triton_selected_moe.py"
)
$providerSource = Join-Path $repo (
    "native\providers\triton_moe\qrt_triton_moe_q8192_provider.cpp"
)
$smokeSource = Join-Path $repo (
    "native\providers\triton_moe\q8192_triton_selected_moe_smoke.cpp"
)
foreach ($required in @($generator, $providerSource, $smokeSource)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "required selected-MoE provider input not found: $required"
    }
}

$aotFiles = @(
    "q8192_selected_moe_route_count.hsaco",
    "q8192_selected_moe_route_prefix_by_program.hsaco",
    "q8192_selected_moe_route_padded_prefix.hsaco",
    "q8192_selected_moe_route_scatter.hsaco",
    "q8192_selected_moe_gate_up_silu.hsaco",
    "q8192_selected_moe_down.hsaco"
)
$aotReused = -not [string]::IsNullOrWhiteSpace($ReuseAotDir)
$resolvedReuseAotDir = $null
if ($aotReused) {
    $resolvedReuseAotDir = (Resolve-Path -LiteralPath $ReuseAotDir).Path
    $reuseMetadataCandidates = @(
        (Join-Path $resolvedReuseAotDir "metadata.json"),
        (Join-Path $resolvedReuseAotDir "q8192_triton_selected_moe_f32out.json")
    )
    $reuseMetadata = $reuseMetadataCandidates |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
    if ($null -eq $reuseMetadata) {
        throw "reused AOT directory has no selected-MoE metadata: $resolvedReuseAotDir"
    }
    foreach ($name in $aotFiles) {
        $sourceAot = Join-Path $resolvedReuseAotDir $name
        if (-not (Test-Path -LiteralPath $sourceAot -PathType Leaf)) {
            throw "reused AOT directory is missing $name"
        }
        $destinationAot = Join-Path $BuildDir $name
        if (-not [StringComparer]::OrdinalIgnoreCase.Equals(
                [IO.Path]::GetFullPath($sourceAot),
                [IO.Path]::GetFullPath($destinationAot)
            )) {
            Copy-Item -LiteralPath $sourceAot -Destination $destinationAot -Force
        }
    }
    Copy-Item -LiteralPath $reuseMetadata `
        -Destination (Join-Path $BuildDir "metadata.json") -Force
} else {
    $wslGenerator = Convert-ToWslPath $generator
    $wslBuildDir = Convert-ToWslPath $BuildDir
    $wslMetadata = "$wslBuildDir/metadata.json"
    $generatorArgs = @(
        $wslGenerator,
        "--output-dir", $wslBuildDir,
        "--metadata", $wslMetadata,
        "--block-m", $BlockM,
        "--block-n", $BlockN,
        "--gate-block-n", $GateBlockN,
        "--down-block-n", $DownBlockN,
        "--gate-block-k", $GateBlockK,
        "--down-block-k", $DownBlockK,
        "--group-m", $GroupM,
        "--num-warps", $NumWarps,
        "--num-stages", $NumStages,
        "--waves-per-eu", $WavesPerEu,
        "--gate-num-warps", $GateNumWarps,
        "--gate-num-stages", $GateNumStages,
        "--gate-waves-per-eu", $GateWavesPerEu,
        "--down-num-warps", $DownNumWarps,
        "--down-num-stages", $DownNumStages,
        "--down-waves-per-eu", $DownWavesPerEu
    )
    if ($VllmSiluBf16Intermediate -ne 0) {
        $generatorArgs += "--vllm-silu-bf16-intermediate"
    }
    & wsl.exe -d $WslDistribution -- $TritonPython @generatorArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Triton AOT generation exited $LASTEXITCODE"
    }
}

$compiledMetadata = Get-Content -Raw -LiteralPath (Join-Path $BuildDir "metadata.json") | ConvertFrom-Json
$routeMetadata = $compiledMetadata.kernels | Where-Object { $_.name -eq "route_count" }
$gateMetadata = $compiledMetadata.kernels | Where-Object { $_.name -eq "gate_up_silu" }
$downMetadata = $compiledMetadata.kernels | Where-Object { $_.name -eq "down" }
if ($null -eq $routeMetadata -or $null -eq $gateMetadata -or $null -eq $downMetadata) {
    throw "Triton metadata is missing route, gate/up, or down kernel records"
}
$routeThreads = [int]$routeMetadata.threads
$routeKernelNames = @("route_count", "route_prefix_by_program", "route_padded_prefix", "route_scatter")
$routeKernels = @($compiledMetadata.kernels | Where-Object { $_.name -in $routeKernelNames })
if ($routeKernels.Count -ne $routeKernelNames.Count) {
    throw "Triton metadata is missing one or more route kernels"
}
foreach ($routeKernel in $routeKernels) {
    if ([int]$routeKernel.threads -ne $routeThreads) {
        throw "route kernels disagree on workgroup size"
    }
}
$gateThreads = [int]$gateMetadata.threads
$downThreads = [int]$downMetadata.threads
if ($routeThreads -le 0 -or $gateThreads -le 0 -or $downThreads -le 0) {
    throw "Triton metadata reported a non-positive workgroup size"
}
$variantDefines = @(
    "-DQRT_TRITON_MOE_BLOCK_M=$BlockM",
    "-DQRT_TRITON_MOE_GATE_BLOCK_N=$GateBlockN",
    "-DQRT_TRITON_MOE_DOWN_BLOCK_N=$DownBlockN",
    "-DQRT_TRITON_MOE_GROUP_M=$GroupM",
    "-DQRT_TRITON_MOE_ROUTE_THREADS=$routeThreads",
    "-DQRT_TRITON_MOE_GATE_THREADS=$gateThreads",
    "-DQRT_TRITON_MOE_DOWN_THREADS=$downThreads",
    "-DQRT_TRITON_MOE_GATE_SHARED_BYTES=$([int]$gateMetadata.dynamic_shared_bytes)",
    "-DQRT_TRITON_MOE_DOWN_SHARED_BYTES=$([int]$downMetadata.dynamic_shared_bytes)",
    "-DQRT_TRITON_MOE_ROUTER_THREADS=$RouterThreads",
    "-DQRT_TRITON_MOE_ROUTER_TOKEN_TILE=$RouterTokenTile",
    "-DQRT_TRITON_MOE_FUSED_COMBINE_WIDTH=$FusedCombineWidth",
    "-DQRT_TRITON_MOE_FULL_V3_EVENT_SLOTS=$FullV3EventSlots"
)
if ($NativeWmmaGate -ne 0) { $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_GATE=1" }
if ($NativeWmmaDown -ne 0) { $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_DOWN=1" }
if ($TransposedRouter -ne 0) { $variantDefines += "-DQRT_TRITON_MOE_TRANSPOSED_ROUTER=1" }
if ($FullV3FusedCombine -ne 0) { $variantDefines += "-DQRT_TRITON_MOE_FULL_V3_FUSED_COMBINE=1" }

$providerDll = Join-Path $BuildDir "qrt_triton_moe_q8192_provider.dll"
$smokeExe = Join-Path $BuildDir "q8192_triton_selected_moe_smoke.exe"
$rocmRoot = Split-Path -Parent (Split-Path -Parent $HipccPath)
$rocmInclude = Join-Path $rocmRoot "include"
$hipblasLtImport = Join-Path $rocmRoot "lib\libhipblaslt.dll.a"
if (-not (Test-Path -LiteralPath $hipblasLtImport -PathType Leaf)) {
    throw "libhipblaslt.dll.a not found at $hipblasLtImport"
}
$hipblasLtLink = Join-Path $BuildDir "hipblaslt.lib"
Copy-Item -LiteralPath $hipblasLtImport -Destination $hipblasLtLink -Force
& $HipccPath -std=c++17 -O3 "--offload-arch=$OffloadArch" `
    $variantDefines -I $rocmInclude -L $BuildDir -shared $providerSource -o $providerDll `
    -lhipblaslt
if ($LASTEXITCODE -ne 0) {
    throw "hipcc exited $LASTEXITCODE while building the selected-MoE provider"
}
& $HipccPath -std=c++17 -O3 "--offload-arch=$OffloadArch" `
    $variantDefines $smokeSource -o $smokeExe
if ($LASTEXITCODE -ne 0) {
    throw "hipcc exited $LASTEXITCODE while building the selected-MoE smoke"
}

$smokeOutput = & $smokeExe $BuildDir $Repetitions $providerDll
$smokeExitCode = $LASTEXITCODE
$smokeText = $smokeOutput -join "`n"
if ($smokeExitCode -ne 0) {
    throw "selected-MoE full-shape smoke exited $smokeExitCode"
}
$routerDebugPass =
    $smokeText -match '(?m)\brouter_debug_id_mismatches=0\b' -and
    $smokeText -match '(?m)\brouter_debug_weight_mismatches=0\b'
if (-not $routerDebugPass) {
    throw "selected-MoE smoke did not report exact router-debug parity"
}
$syncHashMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_sync_hash=([0-9a-f]+)\b'
)
$asyncHashMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_async_hash=([0-9a-f]+)\b'
)
$v3HashMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_v3_hash=([0-9a-f]+)\b'
)
$v3AsyncHashMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_v3_async_hash=([0-9a-f]+)\b'
)
$asyncParityPass =
    $smokeText -match '(?m)\bfull_provider_async_mismatches=0\b' -and
    $syncHashMatch.Success -and $asyncHashMatch.Success -and
    $syncHashMatch.Groups[1].Value -eq $asyncHashMatch.Groups[1].Value
if (-not $asyncParityPass) {
    throw "selected-MoE smoke did not report exact sync/async full-v2 parity"
}
$v3ParityPass =
    $smokeText -match '(?m)\bfull_provider_v3_mismatches=0\b' -and
    $syncHashMatch.Success -and $v3HashMatch.Success -and
    $syncHashMatch.Groups[1].Value -eq $v3HashMatch.Groups[1].Value
if (-not $v3ParityPass) {
    throw "selected-MoE smoke did not report exact full-v2/full-v3 parity"
}
$expectedAsyncChainCalls = $FullV3EventSlots + 1
$v3AsyncParityPass =
    $smokeText -match "(?m)\bfull_provider_v3_async_chain_calls=$expectedAsyncChainCalls\b" -and
    $smokeText -match "(?m)\bfull_provider_v3_async_checked_outputs=$expectedAsyncChainCalls\b" -and
    $smokeText -match '(?m)\bfull_provider_v3_async_mismatches=0\b' -and
    $syncHashMatch.Success -and $v3AsyncHashMatch.Success -and
    $syncHashMatch.Groups[1].Value -eq $v3AsyncHashMatch.Groups[1].Value
if (-not $v3AsyncParityPass) {
    throw "selected-MoE smoke did not report exact $expectedAsyncChainCalls-call full-v3-async parity"
}
$expectedFullProviderHash = "c8b9f2290b8bbd3"
$expectedFullProviderHashPass =
    $syncHashMatch.Success -and
    $syncHashMatch.Groups[1].Value -eq $expectedFullProviderHash
if ($RequireExpectedFullProviderHash -ne 0 -and -not $expectedFullProviderHashPass) {
    throw "selected-MoE smoke full-provider hash did not match $expectedFullProviderHash"
}
$v3AsyncTotalMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_v3_async_chain_total_ms=([0-9]+(?:\.[0-9]+)?)\b'
)
$v3AsyncPerCallMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_v3_async_per_call_ms=([0-9]+(?:\.[0-9]+)?)\b'
)
if (-not $v3AsyncTotalMatch.Success -or -not $v3AsyncPerCallMatch.Success) {
    throw "selected-MoE smoke did not report full-v3-async chain timing"
}
$v3AsyncTotalMs = [double]::Parse(
    $v3AsyncTotalMatch.Groups[1].Value,
    [Globalization.CultureInfo]::InvariantCulture
)
$v3AsyncPerCallMs = [double]::Parse(
    $v3AsyncPerCallMatch.Groups[1].Value,
    [Globalization.CultureInfo]::InvariantCulture
)
$v3TimingMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_v3_reset_included_ms=([0-9]+(?:\.[0-9]+)?)\b'
)
if (-not $v3TimingMatch.Success) {
    throw "selected-MoE smoke did not report reset-included full-v3 timing"
}
$v3ResetIncludedMs = [double]::Parse(
    $v3TimingMatch.Groups[1].Value,
    [Globalization.CultureInfo]::InvariantCulture
)
$expectedProviderBackendMask =
    $(if ($NativeWmmaGate -ne 0) { 1 } else { 0 }) -bor
    $(if ($NativeWmmaDown -ne 0) { 2 } else { 0 }) -bor
    $(if ($TransposedRouter -ne 0) { 4 } else { 0 }) -bor
    $(if ($FullV3FusedCombine -ne 0) { 8 } else { 0 })
if ($smokeText -notmatch
    "(?m)\bprovider_backend_mask=$expectedProviderBackendMask\b" -or
    $smokeText -notmatch '(?m)\bprovider_backend_mask_pass=1\b') {
    throw "selected-MoE smoke did not confirm provider backend mask $expectedProviderBackendMask"
}

$runtimeFiles = @(
    "q8192_selected_moe_route_count.hsaco",
    "q8192_selected_moe_route_prefix_by_program.hsaco",
    "q8192_selected_moe_route_padded_prefix.hsaco",
    "q8192_selected_moe_route_scatter.hsaco",
    "q8192_selected_moe_gate_up_silu.hsaco",
    "q8192_selected_moe_down.hsaco",
    "qrt_triton_moe_q8192_provider.dll"
)
$artifacts = foreach ($name in $runtimeFiles) {
    $source = Join-Path $BuildDir $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "selected-MoE build output is missing: $source"
    }
    $destination = Join-Path $OutDir $name
    $sourceFullPath = [IO.Path]::GetFullPath($source)
    $destinationFullPath = [IO.Path]::GetFullPath($destination)
    if (-not [StringComparer]::OrdinalIgnoreCase.Equals(
            $sourceFullPath,
            $destinationFullPath
        )) {
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
    $item = Get-Item -LiteralPath $destination
    [ordered]@{
        file = $name
        bytes = $item.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
    }
}

$record = [ordered]@{
    schema_version = 1
    host = [Environment]::MachineName
    execution = "local_windows_process"
    repo_commit = (& git -C $repo rev-parse HEAD).Trim()
    dirty_tree = @(& git -C $repo status --porcelain).Count -ne 0
    command_file = $PSCommandPath
    offload_arch = $OffloadArch
    wsl_distribution = $WslDistribution
    triton_python = $TritonPython
    hipcc = $HipccPath
    aot_reused = $aotReused
    aot_source_dir = $resolvedReuseAotDir
    tile = [ordered]@{
        block_m = $BlockM
        block_n = $BlockN
        gate_block_n = $GateBlockN
        down_block_n = $DownBlockN
        gate_block_k = $GateBlockK
        down_block_k = $DownBlockK
        group_m = $GroupM
        num_warps = $NumWarps
        num_stages = $NumStages
        waves_per_eu = $WavesPerEu
        threads = $routeThreads
        route_num_warps = $NumWarps
        route_num_stages = $NumStages
        route_waves_per_eu = $WavesPerEu
        gate_num_warps = $GateNumWarps
        gate_num_stages = $GateNumStages
        gate_waves_per_eu = $GateWavesPerEu
        down_num_warps = $DownNumWarps
        down_num_stages = $DownNumStages
        down_waves_per_eu = $DownWavesPerEu
        vllm_silu_bf16_intermediate =
            ($VllmSiluBf16Intermediate -ne 0)
        route_threads = $routeThreads
        gate_threads = $gateThreads
        down_threads = $downThreads
        gate_dynamic_shared_bytes = [int]$gateMetadata.dynamic_shared_bytes
        down_dynamic_shared_bytes = [int]$downMetadata.dynamic_shared_bytes
    }
    build_dir = (Resolve-Path -LiteralPath $BuildDir).Path
    out_dir = (Resolve-Path -LiteralPath $OutDir).Path
    async_parity_pass = $asyncParityPass
    v3_parity_pass = $v3ParityPass
    v3_async_parity_pass = $v3AsyncParityPass
    router_debug_parity_pass = $routerDebugPass
    native_wmma_routed = ($NativeWmmaRouted -ne 0)
    native_wmma_gate = ($NativeWmmaGate -ne 0)
    native_wmma_down = ($NativeWmmaDown -ne 0)
    transposed_router = ($TransposedRouter -ne 0)
    router_threads = $RouterThreads
    router_token_tile = $RouterTokenTile
    full_v3_fused_combine = ($FullV3FusedCombine -ne 0)
    fused_combine_width = $FusedCombineWidth
    full_v3_event_slots = $FullV3EventSlots
    provider_backend_mask = $expectedProviderBackendMask
    expected_full_provider_hash = $expectedFullProviderHash
    expected_full_provider_hash_pass = $expectedFullProviderHashPass
    full_provider_v3_async_under_29ms = ($v3AsyncPerCallMs -le 29.0)
    v3_async_chain_total_ms = $v3AsyncTotalMs
    v3_async_per_call_ms = $v3AsyncPerCallMs
    v3_async_chain_calls = $expectedAsyncChainCalls
    v3_reset_included_ms = $v3ResetIncludedMs
    smoke = $smokeText
    artifacts = $artifacts
}
$json = $record | ConvertTo-Json -Depth 6
[IO.File]::WriteAllText(
    (Join-Path $OutDir "build-provenance.json"),
    $json + [Environment]::NewLine,
    $utf8
)
Write-Output $json
