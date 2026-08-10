param(
    [Parameter(Mandatory = $false)][string]$OutDir = "",
    [Parameter(Mandatory = $false)][string]$BaseKernelDir = "",
    [Parameter(Mandatory = $false)]
        [ValidateSet(
            "exact",
            "grouped",
            "grouped-bf16",
            "grouped-bf16-no-static",
            "exact-gate-grouped-down",
            "exact-gate-grouped-down-static-f32",
            "exact-gate-grouped-down-exact-shared",
            "grouped-gate-exact-down"
        )]
        [string]$Variant = "exact",
    [Parameter(Mandatory = $false)]
        [ValidateRange(1, 600)]
        [int]$CompileTimeoutSeconds = 300,
    [Parameter(Mandatory = $false)]
        [string]$RocmRoot = "C:\Program Files\AMD\ROCm\7.1",
    [Parameter(Mandatory = $false)]
        [ValidatePattern('^gfx[0-9a-f]+$')]
        [string]$OffloadArch = "gfx1151",
    [Parameter(Mandatory = $false)][string]$WslDistribution = "Ubuntu-24.04",
    [Parameter(Mandatory = $false)]
        [string]$TritonPython = "/opt/qwen36-vllm/bin/python"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false

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
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds
    )
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = ($Arguments | ForEach-Object {
        Quote-NativeArgument -Value $_
    }) -join " "
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $null = $process.Start()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $completed = $process.WaitForExit($TimeoutSeconds * 1000)
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

$generateBaseKernels = [string]::IsNullOrWhiteSpace($BaseKernelDir)
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repo "build\triton-moe-q1024-exact"
} elseif (-not [IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repo $OutDir
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
if ($generateBaseKernels) {
    $BaseKernelDir = Join-Path $OutDir "generated-q1024-base"
} elseif (-not [IO.Path]::IsPathRooted($BaseKernelDir)) {
    $BaseKernelDir = Join-Path $repo $BaseKernelDir
}
$BaseKernelDir = [IO.Path]::GetFullPath($BaseKernelDir)
$kernelDir = Join-Path $OutDir "moe-kernels"
New-Item -ItemType Directory -Force `
    -Path $OutDir, $kernelDir, $BaseKernelDir | Out-Null

$hipcc = Join-Path $RocmRoot "bin\hipcc.exe"
$llvmReadobj = Join-Path $RocmRoot "bin\llvm-readobj.exe"
$rocmInclude = Join-Path $RocmRoot "include"
$rocmLib = Join-Path $RocmRoot "lib"
$hipblasLtImport = Join-Path $rocmLib "libhipblaslt.dll.a"
$hipblasLtLink = Join-Path $OutDir "hipblaslt.lib"
$source = Join-Path `
    $repo `
    "native\providers\triton_moe\qrt_triton_moe_q8192_provider.cpp"
$generator = Join-Path `
    $repo `
    "native\generators\compile_q8192_triton_selected_moe.py"
$kernelSourceDir = Join-Path $repo "native\aot\gfx1151"
$providerDll = Join-Path $OutDir (
    "qrt_triton_moe_q1024_{0}_provider_slots64.dll" -f `
        ($Variant -replace '-', '_')
)
$exportsPath = Join-Path $OutDir "exports.txt"

$variantKernelNames = @(
    "q1024_selected_moe_route_scatter_metadata.json",
    "q1024_selected_moe_route_scatter.hsaco",
    "q1024_selected_moe_bf16_endpoint_metadata.json",
    "q1024_selected_moe_down_bf16_endpoint.hsaco",
    "q1024_triton_0626_exact_metadata.json",
    "q1024_triton_0626_exact_gate_up_silu_w8.hsaco",
    "q1024_triton_0626_exact_down_sum.hsaco",
    "q1024_triton_0626_exact_shared_gate_up_silu.hsaco",
    "q1024_triton_0626_exact_shared_gate_logit.hsaco",
    "q1024_triton_0626_exact_shared_down.hsaco"
)
$generatorRun = $null
if ($generateBaseKernels) {
    $wslGenerator = Convert-ToWslPath $generator
    $wslBaseKernelDir = Convert-ToWslPath $BaseKernelDir
    $generatorArguments = @(
        "-d", $WslDistribution,
        "--", $TritonPython, $wslGenerator,
        "--output-dir", $wslBaseKernelDir,
        "--metadata", "$wslBaseKernelDir/metadata.json",
        "--tokens", "1024"
    )
    $generatorRun = Invoke-BoundedProcess -FilePath "wsl.exe" `
        -Arguments $generatorArguments -WorkingDirectory $repo `
        -StdOutPath (Join-Path $OutDir "generate-base.stdout.txt") `
        -StdErrPath (Join-Path $OutDir "generate-base.stderr.txt") `
        -TimeoutSeconds $CompileTimeoutSeconds
    if (-not $generatorRun.completed -or $generatorRun.exit_code -ne 0) {
        throw "bounded q1024 base-kernel generation failed or timed out"
    }
}
$required = @(
    $hipcc,
    $llvmReadobj,
    $rocmInclude,
    $rocmLib,
    $hipblasLtImport,
    $source,
    $generator,
    $BaseKernelDir,
    (Join-Path $BaseKernelDir "metadata.json")
)
$required += $variantKernelNames | ForEach-Object {
    Join-Path $kernelSourceDir $_
}
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "required q1024 exact build input not found: $path"
    }
}

Copy-Item -Path (Join-Path $BaseKernelDir "*") `
    -Destination $kernelDir -Recurse -Force
foreach ($name in $variantKernelNames) {
    Copy-Item -LiteralPath (Join-Path $kernelSourceDir $name) `
        -Destination (Join-Path $kernelDir $name) -Force
}
Copy-Item -LiteralPath $hipblasLtImport `
    -Destination $hipblasLtLink -Force

$compiledMetadata = Get-Content -Raw `
    -LiteralPath (Join-Path $BaseKernelDir "metadata.json") |
    ConvertFrom-Json
$routeMetadata = $compiledMetadata.kernels |
    Where-Object { $_.name -eq "route_count" }
$gateMetadata = $compiledMetadata.kernels |
    Where-Object { $_.name -eq "gate_up_silu" }
$downMetadata = $compiledMetadata.kernels |
    Where-Object { $_.name -eq "down" }
if ($null -eq $routeMetadata -or $null -eq $gateMetadata -or
    $null -eq $downMetadata) {
    throw "q1024 metadata is missing route, gate/up, or down kernels"
}

$commonDefines = @(
    "-DQRT_TRITON_MOE_TOKENS=1024",
    "-DQRT_TRITON_MOE_KERNEL_TOKEN_LABEL=q1024",
    "-DQRT_TRITON_MOE_BLOCK_M=64",
    "-DQRT_TRITON_MOE_GATE_BLOCK_N=64",
    "-DQRT_TRITON_MOE_DOWN_BLOCK_N=64",
    "-DQRT_TRITON_MOE_GROUP_M=8",
    "-DQRT_TRITON_MOE_ROUTE_THREADS=$([int]$routeMetadata.threads)",
    "-DQRT_TRITON_MOE_GATE_THREADS=$([int]$gateMetadata.threads)",
    "-DQRT_TRITON_MOE_DOWN_THREADS=$([int]$downMetadata.threads)",
    "-DQRT_TRITON_MOE_GATE_SHARED_BYTES=$([int]$gateMetadata.dynamic_shared_bytes)",
    "-DQRT_TRITON_MOE_DOWN_SHARED_BYTES=$([int]$downMetadata.dynamic_shared_bytes)",
    "-DQRT_TRITON_MOE_ROUTER_THREADS=256",
    "-DQRT_TRITON_MOE_ROUTER_TOKEN_TILE=1",
    "-DQRT_TRITON_MOE_FULL_V3_EVENT_SLOTS=64",
    "-DQRT_TRITON_MOE_ROCBLAS_ROUTER=1",
    "-DQRT_TRITON_MOE_Q1024_EARLY_F32=1",
    "-DQRT_TRITON_MOE_ZERO_REQUEST_SCRATCH=1"
)
$variantDefines = switch ($Variant) {
    "exact" {
        @(
            "-DQRT_TRITON_MOE_Q1024_EXACT_ROUTED=1",
            "-DQRT_TRITON_MOE_Q1024_EXACT_SHARED=1"
        )
        break
    }
    "grouped" {
        @(
            "-DQRT_TRITON_MOE_Q1024_EXACT_ROUTED=0",
            "-DQRT_TRITON_MOE_Q1024_EXACT_SHARED=0"
        )
        break
    }
    "grouped-bf16" {
        @(
            "-DQRT_TRITON_MOE_Q1024_EXACT_ROUTED=0",
            "-DQRT_TRITON_MOE_Q1024_GROUPED_BF16_ENDPOINTS=1",
            "-DQRT_TRITON_MOE_Q1024_EXACT_SHARED=0",
            "-DQRT_TRITON_MOE_Q1024_STATIC_F32_COMBINE=1"
        )
        break
    }
    "grouped-bf16-no-static" {
        @(
            "-DQRT_TRITON_MOE_Q1024_EXACT_ROUTED=0",
            "-DQRT_TRITON_MOE_Q1024_GROUPED_BF16_ENDPOINTS=1",
            "-DQRT_TRITON_MOE_Q1024_EXACT_SHARED=0",
            "-DQRT_TRITON_MOE_Q1024_STATIC_F32_COMBINE=0"
        )
        break
    }
    "exact-gate-grouped-down" {
        @(
            "-DQRT_TRITON_MOE_Q1024_EXACT_ROUTED=0",
            "-DQRT_TRITON_MOE_Q1024_EXACT_GATE_GROUPED_DOWN=1",
            "-DQRT_TRITON_MOE_Q1024_GROUPED_BF16_ENDPOINTS=1",
            "-DQRT_TRITON_MOE_Q1024_EXACT_SHARED=0"
        )
        break
    }
    "exact-gate-grouped-down-static-f32" {
        @(
            "-DQRT_TRITON_MOE_Q1024_EXACT_ROUTED=0",
            "-DQRT_TRITON_MOE_Q1024_EXACT_GATE_GROUPED_DOWN=1",
            "-DQRT_TRITON_MOE_Q1024_GROUPED_BF16_ENDPOINTS=1",
            "-DQRT_TRITON_MOE_Q1024_EXACT_SHARED=0",
            "-DQRT_TRITON_MOE_Q1024_STATIC_F32_COMBINE=1"
        )
        break
    }
    "exact-gate-grouped-down-exact-shared" {
        @(
            "-DQRT_TRITON_MOE_Q1024_EXACT_ROUTED=0",
            "-DQRT_TRITON_MOE_Q1024_EXACT_GATE_GROUPED_DOWN=1",
            "-DQRT_TRITON_MOE_Q1024_GROUPED_BF16_ENDPOINTS=1",
            "-DQRT_TRITON_MOE_Q1024_EXACT_SHARED=1"
        )
        break
    }
    "grouped-gate-exact-down" {
        @(
            "-DQRT_TRITON_MOE_Q1024_EXACT_ROUTED=0",
            "-DQRT_TRITON_MOE_Q1024_GROUPED_GATE_EXACT_DOWN=1",
            "-DQRT_TRITON_MOE_Q1024_EXACT_SHARED=0"
        )
        break
    }
}
$defines = $commonDefines + $variantDefines
$arguments = @(
    "-std=c++17",
    "-O3",
    "--offload-arch=$OffloadArch"
) + $defines + @(
    "-I", $rocmInclude,
    "-L", $OutDir,
    "-shared",
    $source,
    "-o", $providerDll,
    "-lhipblaslt",
    "-lrocblas"
)
$compile = Invoke-BoundedProcess -FilePath $hipcc `
    -Arguments $arguments -WorkingDirectory $repo `
    -StdOutPath (Join-Path $OutDir "compile.stdout.txt") `
    -StdErrPath (Join-Path $OutDir "compile.stderr.txt") `
    -TimeoutSeconds $CompileTimeoutSeconds
if (-not $compile.completed -or $compile.exit_code -ne 0) {
    throw "bounded q1024 $Variant selected-MoE build failed or timed out"
}
if (-not (Test-Path -LiteralPath $providerDll -PathType Leaf)) {
    throw "q1024 $Variant build did not emit $providerDll"
}

$exports = & $llvmReadobj --coff-exports $providerDll
if ($LASTEXITCODE -ne 0) {
    throw "llvm-readobj failed for $providerDll"
}
$exportText = ($exports -join "`n") + "`n"
[IO.File]::WriteAllText($exportsPath, $exportText, $utf8)
$expectedExports = @(
    "qrt_triton_moe_q8192_prepare",
    "qrt_triton_moe_q8192_launch",
    "qrt_triton_moe_q8192_launch_full_v2_async",
    "qrt_triton_moe_q8192_last_error",
    "qrt_triton_moe_q8192_scratch_bytes"
)
foreach ($name in $expectedExports) {
    if ($exportText -notmatch [regex]::Escape($name)) {
        throw "q1024 $Variant provider is missing export $name"
    }
}

$runtimeArtifacts = @($providerDll)
$runtimeArtifacts += Get-ChildItem -LiteralPath $kernelDir -File |
    Sort-Object Name |
    Select-Object -ExpandProperty FullName
$artifactRecords = foreach ($path in $runtimeArtifacts) {
    $item = Get-Item -LiteralPath $path
    [ordered]@{
        path = $item.FullName
        bytes = $item.Length
        sha256 = (Get-FileHash -Algorithm SHA256 `
            -LiteralPath $path).Hash.ToLowerInvariant()
    }
}
$record = [ordered]@{
    schema_version = 1
    host = [Environment]::MachineName
    execution = "local_windows_process"
    remote_transport_used = $false
    repo_commit = (& git -C $repo rev-parse HEAD).Trim()
    dirty_tree = @(& git -C $repo status --porcelain).Count -ne 0
    command_file = $PSCommandPath
    variant = $Variant
    variant_defines = $variantDefines
    offload_arch = $OffloadArch
    compile_timeout_seconds = $CompileTimeoutSeconds
    base_kernels_generated = $generateBaseKernels
    base_kernel_generator = $generator
    base_kernel_generator_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $generator).Hash.ToLowerInvariant()
    base_kernel_generation = $generatorRun
    wsl_distribution = if ($generateBaseKernels) { $WslDistribution } else { $null }
    triton_python = if ($generateBaseKernels) { $TritonPython } else { $null }
    source_path = $source
    source_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $source).Hash.ToLowerInvariant()
    base_kernel_dir = $BaseKernelDir
    kernel_dir = $kernelDir
    provider_dll = $providerDll
    provider_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $providerDll).Hash.ToLowerInvariant()
    compile = $compile
    compile_arguments = $arguments
    expected_exports = $expectedExports
    artifacts = $artifactRecords
}
[IO.File]::WriteAllText(
    (Join-Path $OutDir "build-provenance.json"),
    ($record | ConvertTo-Json -Depth 8),
    $utf8
)
Write-Output $providerDll
