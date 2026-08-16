param(
    [Parameter(Mandatory = $true)]
        [ValidateSet(32, 64, 128, 256, 512, 1024, 2048, 4096)]
        [int]$Tokens,
    [Parameter(Mandatory = $false)][string]$OutDir = "",
    [Parameter(Mandatory = $false)]
        [string]$RocmRoot = "C:\Program Files\AMD\ROCm\7.1",
    [Parameter(Mandatory = $false)][string]$WslDistribution = "Ubuntu-24.04",
    [Parameter(Mandatory = $false)]
        [string]$TritonPython = "/opt/qwen36-vllm/bin/python",
    [Parameter(Mandatory = $false)]
        [ValidatePattern('^gfx[0-9a-f]+$')]
        [string]$OffloadArch = "gfx1151",
    [Parameter(Mandatory = $false)]
        [ValidateRange(1, 600)]
        [int]$CompileTimeoutSeconds = 300
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

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repo ("build\smooth-tail\q{0}" -f $Tokens)
} elseif (-not [IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repo $OutDir
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$source = Join-Path `
    $repo `
    "native\providers\triton_moe\qrt_triton_moe_q8192_provider.cpp"
$generator = Join-Path `
    $repo `
    "native\generators\compile_q8192_triton_selected_moe.py"
$hipcc = Join-Path $RocmRoot "bin\hipcc.exe"
$hipblasLtImport = Join-Path $RocmRoot "lib\libhipblaslt.dll.a"
foreach ($path in @($source, $generator, $hipcc, $hipblasLtImport)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "required smooth-tail q$Tokens build input missing: $path"
    }
}

$generatorArguments = @(
    "-d", $WslDistribution,
    "--", $TritonPython, (Convert-ToWslPath $generator),
    "--output-dir", (Convert-ToWslPath $OutDir),
    "--metadata", ((Convert-ToWslPath $OutDir) + "/metadata.json"),
    "--tokens", $Tokens.ToString()
)
$generatorRun = Invoke-BoundedProcess -FilePath "wsl.exe" `
    -Arguments $generatorArguments -WorkingDirectory $repo `
    -StdOutPath (Join-Path $OutDir "generate.stdout.txt") `
    -StdErrPath (Join-Path $OutDir "generate.stderr.txt") `
    -TimeoutSeconds $CompileTimeoutSeconds
if (-not $generatorRun.completed -or $generatorRun.exit_code -ne 0) {
    throw "bounded smooth-tail q$Tokens AOT generation failed or timed out"
}

$metadataPath = Join-Path $OutDir "metadata.json"
$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
$route = $metadata.kernels | Where-Object { $_.name -eq "route_count" }
$gate = $metadata.kernels | Where-Object { $_.name -eq "gate_up_silu" }
$down = $metadata.kernels | Where-Object { $_.name -eq "down" }
if ($null -eq $route -or $null -eq $gate -or $null -eq $down) {
    throw "smooth-tail q$Tokens metadata is incomplete"
}

$hipblasLtLink = Join-Path $OutDir "hipblaslt.lib"
Copy-Item -LiteralPath $hipblasLtImport -Destination $hipblasLtLink -Force
$providerName = "qrt_triton_moe_q{0}_fast_tail_provider.dll" -f $Tokens
$provider = Join-Path $OutDir $providerName
$compileArguments = @(
    "-std=c++17", "-O3", "--offload-arch=$OffloadArch",
    "-DQRT_TRITON_MOE_TOKENS=$Tokens",
    "-DQRT_TRITON_MOE_KERNEL_TOKEN_LABEL=q$Tokens",
    "-DQRT_TRITON_MOE_BLOCK_M=64",
    "-DQRT_TRITON_MOE_GATE_BLOCK_N=64",
    "-DQRT_TRITON_MOE_DOWN_BLOCK_N=64",
    "-DQRT_TRITON_MOE_GROUP_M=8",
    "-DQRT_TRITON_MOE_ROUTE_THREADS=$([int]$route.threads)",
    "-DQRT_TRITON_MOE_GATE_THREADS=$([int]$gate.threads)",
    "-DQRT_TRITON_MOE_DOWN_THREADS=$([int]$down.threads)",
    "-DQRT_TRITON_MOE_GATE_SHARED_BYTES=$([int]$gate.dynamic_shared_bytes)",
    "-DQRT_TRITON_MOE_DOWN_SHARED_BYTES=$([int]$down.dynamic_shared_bytes)",
    "-DQRT_TRITON_MOE_ROUTER_THREADS=256",
    "-DQRT_TRITON_MOE_ROUTER_TOKEN_TILE=8",
    "-DQRT_TRITON_MOE_FULL_V3_EVENT_SLOTS=16",
    "-DQRT_TRITON_MOE_NATIVE_WMMA_GATE=1",
    "-DQRT_TRITON_MOE_NATIVE_WMMA_DOWN=1",
    "-DQRT_TRITON_MOE_TRANSPOSED_ROUTER=1",
    "-DQRT_TRITON_MOE_FULL_V3_FUSED_COMBINE=1",
    "-DQRT_TRITON_MOE_FUSED_COMBINE_WIDTH=4",
    "-I", (Join-Path $RocmRoot "include"),
    "-L", $OutDir,
    "-shared", $source, "-o", $provider, "-lhipblaslt"
)
$compileRun = Invoke-BoundedProcess -FilePath $hipcc `
    -Arguments $compileArguments -WorkingDirectory $repo `
    -StdOutPath (Join-Path $OutDir "compile.stdout.txt") `
    -StdErrPath (Join-Path $OutDir "compile.stderr.txt") `
    -TimeoutSeconds $CompileTimeoutSeconds
if (-not $compileRun.completed -or $compileRun.exit_code -ne 0) {
    throw "bounded smooth-tail q$Tokens provider build failed or timed out"
}

$kernelFiles = @(
    Get-ChildItem -LiteralPath $OutDir -File |
        Where-Object { $_.Extension -eq ".hsaco" }
)
if ($kernelFiles.Count -ne 6) {
    throw "smooth-tail q$Tokens build produced $($kernelFiles.Count) kernels, expected 6"
}
if (-not (Test-Path -LiteralPath $provider -PathType Leaf)) {
    throw "smooth-tail q$Tokens provider output is missing: $provider"
}

$runtimeArtifacts = @($provider, $metadataPath) + @(
    $kernelFiles | Select-Object -ExpandProperty FullName
)
$artifacts = foreach ($path in $runtimeArtifacts | Sort-Object -Unique) {
    $item = Get-Item -LiteralPath $path
    [ordered]@{
        file = $item.Name
        bytes = $item.Length
        sha256 = (Get-FileHash -Algorithm SHA256 `
            -LiteralPath $item.FullName).Hash.ToLowerInvariant()
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
    tokens = $Tokens
    provider_backend_mask = 15
    source_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $source).Hash.ToLowerInvariant()
    generator = $generatorRun
    compile = $compileRun
    compile_arguments = $compileArguments
    artifacts = $artifacts
}
$json = $record | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText(
    (Join-Path $OutDir "build-provenance.json"),
    $json + [Environment]::NewLine,
    $utf8
)
Write-Output $json
