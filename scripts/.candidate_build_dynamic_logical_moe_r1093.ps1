param(
    [string]$Repo = 'D:\projects\AIMA-dynamic-logical-r1',
    [ValidatePattern('^r[0-9]+$')]
    [string]$RunLabel = 'r1093',
    [ValidateRange(120, 1800)]
    [int]$TimeoutSeconds = 900,
    [switch]$NativeSharedSelected,
    [string]$ReuseAotDir = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$builder = Join-Path $Repo 'scripts\baiying_build_triton_moe_q8192.ps1'
$providerSource = Join-Path $Repo `
    'native\providers\triton_moe\qrt_triton_moe_q8192_provider.cpp'
$generatorSource = Join-Path $Repo `
    'native\generators\compile_q8192_triton_selected_moe.py'
$smokeSource = Join-Path $Repo `
    'native\providers\triton_moe\q8192_triton_selected_moe_smoke.cpp'
$outDir = Join-Path $Repo "build\dynamic-logical-moe-$RunLabel"
$childRecordPath = Join-Path $outDir 'build-provenance.json'
$recordPath = Join-Path $outDir 'qualification-provenance.json'
$metadataPath = Join-Path $outDir 'metadata.json'
$providerPath = Join-Path $outDir `
    'qrt_triton_moe_q8192_provider.dll'
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false
$requiredRuntimeEnvironment = [ordered]@{}
if ($NativeSharedSelected) {
    $requiredRuntimeEnvironment = [ordered]@{
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
if (-not $NativeSharedSelected -and
        -not [string]::IsNullOrWhiteSpace($ReuseAotDir)) {
    throw 'ReuseAotDir is qualified here only for NativeSharedSelected.'
}
if (-not [string]::IsNullOrWhiteSpace($ReuseAotDir)) {
    $ReuseAotDir = [IO.Path]::GetFullPath($ReuseAotDir)
    if (-not (Test-Path -LiteralPath $ReuseAotDir -PathType Container)) {
        throw "reused AOT directory not found: $ReuseAotDir"
    }
}

function Quote-NativeArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' `
        -replace '(\\+)$', '$1$1') + '"'
}

function Invoke-BoundedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$StdOutPath,
        [Parameter(Mandatory = $true)][string]$StdErrPath,
        [Parameter(Mandatory = $true)][int]$BoundSeconds,
        [System.Collections.IDictionary]$EnvironmentVariables = @{}
    )
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = @($Arguments | ForEach-Object {
        Quote-NativeArgument -Value $_
    }) -join ' '
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($entry in $EnvironmentVariables.GetEnumerator()) {
        $startInfo.EnvironmentVariables[[string]$entry.Key] = `
            [string]$entry.Value
    }
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    $watch = [Diagnostics.Stopwatch]::StartNew()
    [void]$process.Start()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $completed = $process.WaitForExit($BoundSeconds * 1000)
    if (-not $completed) {
        try { $process.Kill() } catch {}
        $process.WaitForExit()
    } else {
        $process.WaitForExit()
    }
    $watch.Stop()
    [IO.File]::WriteAllText(
        $StdOutPath,
        $stdoutTask.GetAwaiter().GetResult(),
        $utf8
    )
    [IO.File]::WriteAllText(
        $StdErrPath,
        $stderrTask.GetAwaiter().GetResult(),
        $utf8
    )
    $process.Refresh()
    return [ordered]@{
        completed = $completed
        timed_out = -not $completed
        exit_code = if ($completed) { [int]$process.ExitCode } else { 124 }
        wall_ms = [Math]::Round($watch.Elapsed.TotalMilliseconds, 6)
    }
}

if (Test-Path -LiteralPath $outDir) {
    throw "refusing to overwrite dynamic-logical MoE build: $outDir"
}
foreach ($inputPath in @(
        $builder,
        $providerSource,
        $generatorSource,
        $smokeSource
    )) {
    if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
        throw "dynamic-logical MoE build input is missing: $inputPath"
    }
}
[void](New-Item -ItemType Directory -Path $outDir)

$powershell = (Get-Command powershell.exe -ErrorAction Stop).Source
$arguments = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $builder,
    '-BuildDir', $outDir,
    '-OutDir', $outDir,
    '-Repetitions', '1',
    '-NativeWmmaGate', '1',
    '-NativeWmmaDown', '1',
    '-TransposedRouter', '1',
    '-RouterThreads', '256',
    '-RouterTokenTile', '8',
    '-FullV3FusedCombine', '1',
    '-FusedCombineWidth', '4',
    '-FullV3EventSlots', '16',
    '-RequireExpectedFullProviderHash', '0'
)
if (-not [string]::IsNullOrWhiteSpace($ReuseAotDir)) {
    $arguments += @('-ReuseAotDir', $ReuseAotDir)
}
if ($NativeSharedSelected) {
    $arguments += @(
        '-BlockM', '64',
        '-GroupM', '1',
        '-RoutedProjectionDebug', '0',
        '-BatchedHawkeye', '1',
        '-ExactShared', '0',
        '-ConditionalExactGate', '1',
        '-SortedConditionalExactGate', '1',
        '-RowMajorSortedConditionalExactGate', '1',
        '-ConditionalExactGateRows', '256',
        '-ConditionalExactDown', '1',
        '-ConditionalExactDownRows', '4',
        '-NativeFusedRouteLayout', '1',
        '-NativeWmmaLdsB', '1',
        '-NativeWmmaLdsBSplitGatePasses', '1',
        '-NativeWmmaLdsBSerialGateN32', '1',
        '-NativeWmmaLdsBSerialDownN32', '1',
        '-NativeWmmaLdsBSkipInactiveAStores', '1',
        '-NativeWmmaLdsBM64LoadThreads', '192',
        '-NativeWmmaLdsBM64FusedOverflow32', '1',
        '-NativeWmmaLosslessPalette', '1',
        '-NativeWmmaLosslessRowPalette', '1',
        '-NativeWmmaKStage', '32'
    )
}
$buildRun = Invoke-BoundedProcess -FilePath $powershell `
    -Arguments $arguments -WorkingDirectory $Repo `
    -StdOutPath (Join-Path $outDir 'build.stdout.json') `
    -StdErrPath (Join-Path $outDir 'build.stderr.log') `
    -BoundSeconds $TimeoutSeconds `
    -EnvironmentVariables $requiredRuntimeEnvironment
if (-not $buildRun.completed -or $buildRun.exit_code -ne 0) {
    throw "bounded dynamic-logical q8192 MoE build failed or timed out"
}

foreach ($outputPath in @(
        $childRecordPath,
        $metadataPath,
        $providerPath
    )) {
    if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
        throw "dynamic-logical q8192 MoE output is missing: $outputPath"
    }
}
$repoCommit = (& git -C $Repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $repoCommit -notmatch '^[0-9a-f]{40}$') {
    throw "could not resolve repository commit: $repoCommit"
}
$childRecord = Get-Content -Raw -LiteralPath $childRecordPath |
    ConvertFrom-Json
$expectedInputHashes = [ordered]@{
    provider_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $providerSource
    ).Hash.ToLowerInvariant()
    generator_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $generatorSource
    ).Hash.ToLowerInvariant()
    smoke_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $smokeSource
    ).Hash.ToLowerInvariant()
}
if ([string]$childRecord.host -ine [Environment]::MachineName -or
    [string]$childRecord.repo_commit -ne $repoCommit -or
    [string]$childRecord.command_file -ine $builder -or
    [string]$childRecord.command_file_sha256 -ne (
        Get-FileHash -Algorithm SHA256 -LiteralPath $builder
    ).Hash.ToLowerInvariant() -or
    [bool]$childRecord.aot_reused -ne `
        (-not [string]::IsNullOrWhiteSpace($ReuseAotDir)) -or
    @($childRecord.PSObject.Properties.Name) -notcontains `
        'expected_full_provider_hash_diagnostic_only' -or
    -not [bool]$childRecord.expected_full_provider_hash_diagnostic_only -or
    -not [bool]$childRecord.dynamic_logical_pass -or
    -not [bool]$childRecord.dynamic_logical_q8192_exact_pass -or
    [int]$childRecord.dynamic_logical_case_count -ne 32 -or
    @($childRecord.PSObject.Properties.Name) -notcontains `
        'dynamic_logical_timing_samples' -or
    [int]$childRecord.dynamic_logical_timing_samples -ne 3 -or
    @($childRecord.PSObject.Properties.Name) -notcontains `
        'dynamic_logical_timing_stat' -or
    [string]$childRecord.dynamic_logical_timing_stat -ne 'median' -or
    @($childRecord.PSObject.Properties.Name) -notcontains `
        'dynamic_logical_nonfinite' -or
    [int]$childRecord.dynamic_logical_nonfinite -ne 0 -or
    @($childRecord.PSObject.Properties.Name) -notcontains `
        'dynamic_logical_component_only' -or
    -not [bool]$childRecord.dynamic_logical_component_only -or
    @($childRecord.PSObject.Properties.Name) -notcontains `
        'inference_success_claimed' -or
    [bool]$childRecord.inference_success_claimed -or
    @($childRecord.dynamic_logical_cases).Count -ne 32 -or
    @($childRecord.PSObject.Properties.Name) -notcontains `
        'dynamic_logical_reset_included' -or
    [bool]$childRecord.dynamic_logical_reset_included -or
    [int]$childRecord.dynamic_logical_min_tokens -ne 2049 -or
    [int]$childRecord.dynamic_logical_max_tokens -ne 8192 -or
    [int]$childRecord.provider_backend_mask -ne 15) {
    throw 'dynamic-logical q8192 MoE child provenance failed qualification'
}
if ($NativeSharedSelected -and (
        [bool]$childRecord.exact_shared -or
        -not [bool]$childRecord.conditional_exact_gate -or
        -not [bool]$childRecord.sorted_conditional_exact_gate -or
        -not [bool]$childRecord.row_major_sorted_conditional_exact_gate -or
        [int]$childRecord.conditional_exact_gate_rows -ne 256 -or
        -not [bool]$childRecord.conditional_exact_down -or
        [int]$childRecord.conditional_exact_down_rows -ne 4 -or
        [int]$childRecord.tile.group_m -ne 1 -or
        -not [bool]$childRecord.batched_hawkeye -or
        -not [bool]$childRecord.native_wmma_gate -or
        -not [bool]$childRecord.native_wmma_down -or
        -not [bool]$childRecord.native_fused_route_layout -or
        -not [bool]$childRecord.native_wmma_lds_b -or
        [int]$childRecord.native_wmma_lds_b_m64_load_threads -ne 192 -or
        -not [bool]$childRecord.native_wmma_lds_b_m64_fused_overflow32 -or
        -not [bool]$childRecord.native_wmma_lossless_row_palette -or
        [int]$childRecord.native_wmma_k_stage -ne 32)) {
    throw 'native-shared selected-MoE child provenance differs from r1160'
}
foreach ($name in $expectedInputHashes.Keys) {
    if ([string]$childRecord.$name -ne [string]$expectedInputHashes[$name]) {
        throw "dynamic-logical q8192 MoE input hash differs: $name"
    }
}

$metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
$expectedKernelFiles = [ordered]@{
    route_count = 'q8192_selected_moe_route_count.hsaco'
    route_prefix_by_program = `
        'q8192_selected_moe_route_prefix_by_program.hsaco'
    route_padded_prefix = `
        'q8192_selected_moe_route_padded_prefix.hsaco'
    route_scatter = 'q8192_selected_moe_route_scatter.hsaco'
    gate_up_silu = 'q8192_selected_moe_gate_up_silu.hsaco'
    down = 'q8192_selected_moe_down.hsaco'
}
$metadataKernels = @($metadata.kernels)
$routeCount = @(
    $metadataKernels | Where-Object { $_.name -eq 'route_count' }
)
$routeScatter = @(
    $metadataKernels | Where-Object { $_.name -eq 'route_scatter' }
)
if ([int]$metadata.shape.tokens -ne 8192 -or
    [int]$metadata.shape.routes -ne 65536 -or
    [string]$metadata.target -ne 'gfx1151' -or
    $metadataKernels.Count -ne $expectedKernelFiles.Count -or
    $routeCount.Count -ne 1 -or $routeScatter.Count -ne 1 -or
    @($routeCount[0].abi) -notcontains 'logical_routes' -or
    @($routeScatter[0].abi) -notcontains 'logical_routes') {
    throw 'dynamic-logical q8192 MoE metadata lacks the logical-route ABI'
}
if ($NativeSharedSelected -and (
        [int]$metadata.shape.block_m -ne 64 -or
        [int]$metadata.shape.group_m -ne 1)) {
    throw 'native-shared base AOT metadata differs from block-M 64/group-M 1'
}
foreach ($expectedKernel in $expectedKernelFiles.GetEnumerator()) {
    $matches = @(
        $metadataKernels | Where-Object {
            [string]$_.name -ceq [string]$expectedKernel.Key
        }
    )
    if ($matches.Count -ne 1 -or
        [string]$matches[0].file -cne [string]$expectedKernel.Value) {
        throw "dynamic-logical q8192 MoE metadata differs for $($expectedKernel.Key)"
    }
}
$kernelPaths = @()
foreach ($kernel in $metadataKernels) {
    $kernelPath = Join-Path $outDir ([string]$kernel.file)
    if (-not (Test-Path -LiteralPath $kernelPath -PathType Leaf)) {
        throw "dynamic-logical q8192 MoE kernel is missing: $kernelPath"
    }
    $kernelItem = Get-Item -LiteralPath $kernelPath
    $kernelSha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $kernelPath
    ).Hash.ToLowerInvariant()
    if ([uint64]$kernel.bytes -ne [uint64]$kernelItem.Length -or
        [string]$kernel.sha256 -ne $kernelSha256) {
        throw "dynamic-logical q8192 MoE metadata hash differs: $($kernel.file)"
    }
    $kernelPaths += $kernelPath
}
$auxiliaryKernelNames = @()
if ($NativeSharedSelected) {
    $auxiliaryKernelNames = @(
        'q8192_triton_0626_row_major_sorted_conditional_exact_gate_rows256.hsaco',
        'q8192_triton_0626_zero_correction_gate_finalize.hsaco',
        'q8192_triton_0626_conditional_exact_down_rows4.hsaco'
    )
}
$auxiliaryKernelPaths = @()
foreach ($name in $auxiliaryKernelNames) {
    $path = Join-Path $outDir $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "native-shared auxiliary AOT is missing: $path"
    }
    $matches = @(
        $childRecord.artifacts | Where-Object {
            [string]$_.file -ceq $name
        }
    )
    $item = Get-Item -LiteralPath $path
    $sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $path
    ).Hash.ToLowerInvariant()
    if ($matches.Count -ne 1 -or
        [uint64]$matches[0].bytes -ne [uint64]$item.Length -or
        [string]$matches[0].sha256 -ne $sha256) {
        throw "native-shared auxiliary AOT provenance differs: $name"
    }
    $auxiliaryKernelPaths += $path
}
$runtimeArtifacts = @($providerPath, $metadataPath) + $kernelPaths + `
    $auxiliaryKernelPaths
$artifacts = @(
    foreach ($artifactPath in $runtimeArtifacts) {
        $item = Get-Item -LiteralPath $artifactPath
        [ordered]@{
            file = $item.Name
            bytes = [uint64]$item.Length
            sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $item.FullName
            ).Hash.ToLowerInvariant()
        }
    }
)
$expectedArtifactCount = 8 + $auxiliaryKernelNames.Count
if ($artifacts.Count -ne $expectedArtifactCount) {
    throw "dynamic-logical q8192 MoE artifact count is $($artifacts.Count), expected $expectedArtifactCount"
}

$record = [ordered]@{
    schema_version = 1
    record_type = 'qrt_dynamic_logical_q8192_moe_build'
    host = [Environment]::MachineName
    repo_commit = $repoCommit
    dirty_tree = @(& git -C $Repo status --porcelain).Count -ne 0
    run_label = $RunLabel
    command_file = $PSCommandPath
    command_file_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
    builder = $builder
    builder_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $builder).Hash.ToLowerInvariant()
    command = @($powershell) + $arguments
    root = $outDir
    timeout_seconds = $TimeoutSeconds
    native_shared_selected = [bool]$NativeSharedSelected
    aot_reused = -not [string]::IsNullOrWhiteSpace($ReuseAotDir)
    aot_source_dir = $ReuseAotDir
    base_aot_mode = $(
        if ([string]::IsNullOrWhiteSpace($ReuseAotDir)) {
            'generated_from_current_source'
        } else {
            'reused_qualified_directory'
        }
    )
    base_aot_block_m = [int]$metadata.shape.block_m
    base_aot_group_m = [int]$metadata.shape.group_m
    base_aot_dynamic_logical_abi = $true
    auxiliary_kernel_files = @($auxiliaryKernelNames)
    required_runtime_environment = @(
        foreach ($entry in $requiredRuntimeEnvironment.GetEnumerator()) {
            [ordered]@{
                name = [string]$entry.Key
                value = [string]$entry.Value
            }
        }
    )
    build = $buildRun
    child_provenance = $childRecordPath
    child_provenance_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $childRecordPath
    ).Hash.ToLowerInvariant()
    metadata = $metadataPath
    metadata_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $metadataPath
    ).Hash.ToLowerInvariant()
    provider = $providerPath
    provider_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $providerPath
    ).Hash.ToLowerInvariant()
    expected_full_provider_hash = `
        [string]$childRecord.expected_full_provider_hash
    expected_full_provider_hash_pass = `
        [bool]$childRecord.expected_full_provider_hash_pass
    expected_full_provider_hash_diagnostic_only = `
        [bool]$childRecord.expected_full_provider_hash_diagnostic_only
    dynamic_logical_pass = [bool]$childRecord.dynamic_logical_pass
    dynamic_logical_q8192_exact_pass = `
        [bool]$childRecord.dynamic_logical_q8192_exact_pass
    dynamic_logical_case_count = `
        [int]$childRecord.dynamic_logical_case_count
    dynamic_logical_timing_samples = `
        [int]$childRecord.dynamic_logical_timing_samples
    dynamic_logical_timing_stat = `
        [string]$childRecord.dynamic_logical_timing_stat
    dynamic_logical_nonfinite = `
        [int]$childRecord.dynamic_logical_nonfinite
    dynamic_logical_component_only = `
        [bool]$childRecord.dynamic_logical_component_only
    inference_success_claimed = `
        [bool]$childRecord.inference_success_claimed
    dynamic_logical_reset_included = `
        [bool]$childRecord.dynamic_logical_reset_included
    dynamic_logical_min_tokens = `
        [int]$childRecord.dynamic_logical_min_tokens
    dynamic_logical_max_tokens = `
        [int]$childRecord.dynamic_logical_max_tokens
    dynamic_logical_cases = @($childRecord.dynamic_logical_cases)
    provider_backend_mask = [int]$childRecord.provider_backend_mask
    provider_source_sha256 = $expectedInputHashes.provider_source_sha256
    generator_source_sha256 = $expectedInputHashes.generator_source_sha256
    smoke_source_sha256 = $expectedInputHashes.smoke_source_sha256
    artifacts = $artifacts
}
[IO.File]::WriteAllText(
    $recordPath,
    ($record | ConvertTo-Json -Depth 10) + "`n",
    $utf8
)
$record | ConvertTo-Json -Depth 10 -Compress
