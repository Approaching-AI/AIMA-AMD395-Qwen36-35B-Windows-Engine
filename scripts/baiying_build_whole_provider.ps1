param(
    [Parameter(Mandatory = $false)][string]$OutDir = "",
    [Parameter(Mandatory = $false)]
        [ValidateRange(1, 600)]
        [int]$CompileTimeoutSeconds = 300,
    [Parameter(Mandatory = $false)]
        [string]$RocmRoot = "C:\Program Files\AMD\ROCm\7.1",
    [Parameter(Mandatory = $false)]
        [ValidatePattern('^gfx[0-9a-f]+$')]
        [string]$OffloadArch = "gfx1151"
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

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repo "build\whole-provider"
} elseif (-not [IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repo $OutDir
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$hipcc = Join-Path $RocmRoot "bin\hipcc.exe"
$clang = Join-Path $RocmRoot "bin\clang++.exe"
$rocmInclude = Join-Path $RocmRoot "include"
$rocmLib = Join-Path $RocmRoot "lib"
$hipblasltImportSource = Join-Path $rocmLib "libhipblaslt.dll.a"
$hipblasltImport = Join-Path $OutDir "hipblaslt.lib"
$source = Join-Path $repo "native\providers\whole_provider.cpp"
$hostSource = Join-Path $repo "native\providers\q1_moe_avx512bf16_host_provider.cpp"
$hostHeader = Join-Path $repo "native\providers\q1_moe_avx512bf16_host_provider.h"
$accumulatorHeader = Join-Path $repo (
    "native\providers\moe_accumulator\q1_moe_hawkeye_bf16_accumulator.h"
)
$qrtHeader = Join-Path $repo "native\src\qrt.h"
$hostObject = Join-Path $OutDir "q1_moe_avx512bf16_host_provider.obj"
$providerDll = Join-Path $OutDir "qrt_qwen36_whole_provider.dll"

foreach ($required in @(
        $hipcc,
        $clang,
        $rocmInclude,
        $rocmLib,
        $hipblasltImportSource,
        $source,
        $hostSource,
        $hostHeader,
        $accumulatorHeader,
        $qrtHeader
    )) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "required provider build input not found: $required"
    }
}
Copy-Item -LiteralPath $hipblasltImportSource `
    -Destination $hipblasltImport -Force

$hostArguments = @(
    "-std=c++17",
    "-O3",
    "-mavx512f",
    "-mavx512bf16",
    "-ffp-contract=off",
    "-c",
    $hostSource,
    "-o",
    $hostObject
)
$hostRun = Invoke-BoundedProcess -FilePath $clang `
    -Arguments $hostArguments -WorkingDirectory $repo `
    -StdOutPath (Join-Path $OutDir "compile-host.stdout.txt") `
    -StdErrPath (Join-Path $OutDir "compile-host.stderr.txt") `
    -TimeoutSeconds $CompileTimeoutSeconds
if (-not $hostRun.completed -or $hostRun.exit_code -ne 0) {
    throw "bounded AVX512BF16 host-provider build failed or timed out"
}

$providerArguments = @(
    "-DQRT_ENABLE_Q1_MOE_AVX512BF16_HOST_PROVIDER=1",
    "-DQRT_ENABLE_HIPBLASLT_RESIDENT_MATRIX_PROVIDER=1",
    "-DQRT_ENABLE_ROCBLAS_FULL_ATTENTION_BMM=1",
    "-DQRT_ENABLE_ROCBLAS_Q1_MOE_BATCHED=1",
    "-DQRT_ENABLE_ROCBLAS_Q1_MOE_ROUTER=1",
    "-DQRT_ENABLE_ROCBLAS_Q1_LINEAR_QKVZ=1",
    "-DQRT_ENABLE_ROCBLAS_Q1_LINEAR_OUT=1",
    "-DQRT_ENABLE_ROCBLAS_Q1_FULL_Q=1",
    "-DQRT_ENABLE_ROCBLAS_Q1_LM_HEAD=1",
    "-I", $rocmInclude,
    "-L", $OutDir,
    "-I", $rocmInclude,
    "-L", $rocmLib,
    "-std=c++17",
    "-O2",
    "--offload-arch=$OffloadArch",
    "-DQRT_STATIC=1",
    "-I", (Join-Path $repo "native\src"),
    "-I", (Join-Path $repo "native\providers"),
    "-shared",
    $source,
    "-o", $providerDll,
    "-lrocblas",
    "-lhipblaslt",
    "-Xlinker", $hostObject
)
$providerRun = Invoke-BoundedProcess -FilePath $hipcc `
    -Arguments $providerArguments -WorkingDirectory $repo `
    -StdOutPath (Join-Path $OutDir "compile-provider.stdout.txt") `
    -StdErrPath (Join-Path $OutDir "compile-provider.stderr.txt") `
    -TimeoutSeconds $CompileTimeoutSeconds
if (-not $providerRun.completed -or $providerRun.exit_code -ne 0) {
    throw "bounded whole-provider DLL build failed or timed out"
}
if (-not (Test-Path -LiteralPath $providerDll -PathType Leaf)) {
    throw "whole-provider build did not emit $providerDll"
}

$providerItem = Get-Item -LiteralPath $providerDll
$providerHash = (Get-FileHash -Algorithm SHA256 `
    -LiteralPath $providerDll).Hash.ToLowerInvariant()
$record = [ordered]@{
    schema_version = 1
    host = [Environment]::MachineName
    execution = "local_windows_process"
    repo_commit = (& git -C $repo rev-parse HEAD).Trim()
    dirty_tree = @(& git -C $repo status --porcelain).Count -ne 0
    command_file = $PSCommandPath
    offload_arch = $OffloadArch
    compile_timeout_seconds = $CompileTimeoutSeconds
    source_path = $source
    source_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $source).Hash.ToLowerInvariant()
    qrt_header_path = $qrtHeader
    qrt_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $qrtHeader).Hash.ToLowerInvariant()
    q1_moe_avx512bf16_host_provider_source_path = $hostSource
    q1_moe_avx512bf16_host_provider_source_sha256 = `
        (Get-FileHash -Algorithm SHA256 `
            -LiteralPath $hostSource).Hash.ToLowerInvariant()
    q1_moe_avx512bf16_host_provider_header_path = $hostHeader
    q1_moe_avx512bf16_host_provider_header_sha256 = `
        (Get-FileHash -Algorithm SHA256 `
            -LiteralPath $hostHeader).Hash.ToLowerInvariant()
    q1_moe_hawkeye_bf16_accumulator_header_path = $accumulatorHeader
    q1_moe_hawkeye_bf16_accumulator_header_sha256 = `
        (Get-FileHash -Algorithm SHA256 `
            -LiteralPath $accumulatorHeader).Hash.ToLowerInvariant()
    hipcc = $hipcc
    clang = $clang
    rocm_root = $RocmRoot
    hipblaslt_import_source = $hipblasltImportSource
    hipblaslt_import_source_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $hipblasltImportSource).Hash.ToLowerInvariant()
    host_compile = $hostRun
    provider_compile = $providerRun
    host_compile_arguments = $hostArguments
    provider_compile_arguments = $providerArguments
    artifacts = @(
        [ordered]@{
            name = "qrt_qwen36_whole_provider.dll"
            bytes = $providerItem.Length
            sha256 = $providerHash
        },
        [ordered]@{
            name = "q1_moe_avx512bf16_host_provider.obj"
            bytes = (Get-Item -LiteralPath $hostObject).Length
            sha256 = (Get-FileHash -Algorithm SHA256 `
                -LiteralPath $hostObject).Hash.ToLowerInvariant()
        }
    )
}
[IO.File]::WriteAllText(
    (Join-Path $OutDir "build-provenance.json"),
    ($record | ConvertTo-Json -Depth 8),
    $utf8
)
Write-Output $providerDll
