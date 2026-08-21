param(
    [string]$Repo = 'D:\projects\AIMA-dynamic-logical-r1',
    [Parameter(Mandatory = $true)]
    [string]$ProviderDll,
    [Parameter(Mandatory = $true)]
    [string]$KernelDir,
    [ValidatePattern('^r[0-9]+$')]
    [string]$RunLabel = 'r1094',
    [ValidateRange(3, 100)]
    [int]$Repetitions = 3,
    [ValidateRange(1, 1800)]
    [int]$CompileTimeoutSeconds = 600,
    [ValidateRange(60, 900)]
    [int]$CaseTimeoutSeconds = 300
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false
$source = Join-Path $Repo `
    'native\providers\gdn\q8192_aiter_fused_gdn_smoke.cpp'
$outDir = Join-Path $Repo "build\dynamic-logical-gdn-$RunLabel"
$smokeExe = Join-Path $outDir 'q8192_aiter_fused_gdn_smoke.exe'
$recordPath = Join-Path $outDir 'qualification-provenance.json'
$kernelNames = @(
    'q8192_aiter_fused_gdn.hsaco',
    'q262144_aiter_fused_gdn_bf16.hsaco',
    'q1024_seeded_aiter_fused_gdn.hsaco',
    'q32768_seeded_aiter_fused_gdn_bf16.hsaco'
)
$logicalTokens = @(
    2073, 2156, 2560, 3073, 4609, 6145, 2049, 2175,
    2176, 2177, 2559, 2561, 3071, 3072, 3583, 3584,
    3585, 4095, 4096, 4097, 4607, 4608, 6143, 6144,
    7167, 7168, 7169, 7679, 7680, 7681, 8191, 8192
)

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
    $startInfo.Arguments = @($Arguments | ForEach-Object {
        Quote-NativeArgument -Value $_
    }) -join ' '
    $startInfo.WorkingDirectory = $WorkingDirectory
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

if (Test-Path -LiteralPath $outDir) {
    throw "refusing to overwrite dynamic-logical GDN qualification: $outDir"
}
foreach ($leaf in @($source, $ProviderDll, $PSCommandPath)) {
    if (-not (Test-Path -LiteralPath $leaf -PathType Leaf)) {
        throw "dynamic-logical GDN qualification input is missing: $leaf"
    }
}
if (-not (Test-Path -LiteralPath $KernelDir -PathType Container)) {
    throw "dynamic-logical GDN kernel directory is missing: $KernelDir"
}
$resolvedProvider = (Resolve-Path -LiteralPath $ProviderDll).Path
$resolvedKernelDir = (Resolve-Path -LiteralPath $KernelDir).Path
$kernelPaths = @(
    foreach ($name in $kernelNames) {
        $path = Join-Path $resolvedKernelDir $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "dynamic-logical GDN kernel is missing: $path"
        }
        $path
    }
)
[void](New-Item -ItemType Directory -Path $outDir)
$hipcc = (Get-Command hipcc.exe -ErrorAction Stop).Source
$compileArguments = @(
    '-std=c++17', '-O3', '--offload-arch=gfx1151',
    $source,
    '-o', $smokeExe
)
$compileRun = Invoke-BoundedProcess -FilePath $hipcc `
    -Arguments $compileArguments -WorkingDirectory $Repo `
    -StdOutPath (Join-Path $outDir 'compile.stdout.log') `
    -StdErrPath (Join-Path $outDir 'compile.stderr.log') `
    -TimeoutSeconds $CompileTimeoutSeconds
if (-not $compileRun.completed -or $compileRun.exit_code -ne 0) {
    throw 'bounded dynamic-logical GDN smoke compile failed or timed out'
}

$modePattern = (
    '(?m)^q(?<q>[0-9]+)_aiter_fused_gdn_mode\s+' +
    'mode=(?<mode>decay|log_g)\s+tokens=(?<tokens>[0-9]+)\s+' +
    'provider_surface=(?<surface>dynamic|fixed)\b' +
    '[^\r\n]*\bprovider_async_one_sync_ms=' +
    '(?<ms>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\b' +
    '[^\r\n]*\bunmeasured_warmup=1\s+' +
    'external_reset_included=0\b[^\r\n]*' +
    '\basync_exact=1\b[^\r\n]*\bcomponent_only=1\s+' +
    'inference_success_claimed=0\s+close=1\s*$'
)
$cases = @()
foreach ($tokens in $logicalTokens) {
    $stdoutPath = Join-Path $outDir "q$tokens.stdout.log"
    $stderrPath = Join-Path $outDir "q$tokens.stderr.log"
    $run = Invoke-BoundedProcess -FilePath $smokeExe `
        -Arguments @(
            $resolvedKernelDir,
            $resolvedProvider,
            [string]$Repetitions,
            [string]$tokens
        ) `
        -WorkingDirectory $outDir `
        -StdOutPath $stdoutPath `
        -StdErrPath $stderrPath `
        -TimeoutSeconds $CaseTimeoutSeconds
    if (-not $run.completed -or $run.exit_code -ne 0) {
        throw "dynamic-logical GDN q$tokens smoke failed or timed out"
    }
    $text = Get-Content -Raw -LiteralPath $stdoutPath
    $matches = @([regex]::Matches($text, $modePattern))
    $expectedSurface = if ($tokens -eq 8192) { 'fixed' } else { 'dynamic' }
    $observedModes = @($matches | ForEach-Object {
        [string]$_.Groups['mode'].Value
    } | Sort-Object)
    if ($matches.Count -ne 2 -or
        ($observedModes -join ',') -ne 'decay,log_g' -or
        $text -notmatch (
            "(?m)^q$($tokens)_aiter_fused_gdn_smoke " +
            'status=pass\b[^\r\n]*\bcomponent_only=1\s+' +
            'inference_success_claimed=0\s*$'
        )) {
        throw "dynamic-logical GDN q$tokens output is incomplete"
    }
    $modeRecords = @()
    foreach ($match in $matches) {
        $milliseconds = [double]::Parse(
            $match.Groups['ms'].Value,
            [Globalization.CultureInfo]::InvariantCulture
        )
        if ([int]$match.Groups['q'].Value -ne $tokens -or
            [int]$match.Groups['tokens'].Value -ne $tokens -or
            [string]$match.Groups['surface'].Value -ne $expectedSurface -or
            [double]::IsNaN($milliseconds) -or
            [double]::IsInfinity($milliseconds) -or
            $milliseconds -le 0.0) {
            throw "dynamic-logical GDN q$tokens mode evidence is invalid"
        }
        $modeRecords += [ordered]@{
            mode = [string]$match.Groups['mode'].Value
            provider_surface = [string]$match.Groups['surface'].Value
            provider_async_one_sync_ms = $milliseconds
            tokens_per_second = [Math]::Round(
                [double]$tokens * 1000.0 / $milliseconds,
                6
            )
        }
    }
    $cases += [ordered]@{
        tokens = $tokens
        run = $run
        stdout = $stdoutPath
        stdout_sha256 = (Get-FileHash -Algorithm SHA256 `
            -LiteralPath $stdoutPath).Hash.ToLowerInvariant()
        stderr = $stderrPath
        stderr_sha256 = (Get-FileHash -Algorithm SHA256 `
            -LiteralPath $stderrPath).Hash.ToLowerInvariant()
        modes = $modeRecords
    }
}

$repoCommit = (& git -C $Repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $repoCommit -notmatch '^[0-9a-f]{40}$') {
    throw "could not resolve repository commit: $repoCommit"
}
$artifactPaths = @($resolvedProvider) + @($kernelPaths)
$artifacts = @(
    $artifactPaths | ForEach-Object {
        $item = Get-Item -LiteralPath $_
        [ordered]@{
            path = $item.FullName
            file = $item.Name
            bytes = [uint64]$item.Length
            sha256 = (Get-FileHash -Algorithm SHA256 `
                -LiteralPath $item.FullName).Hash.ToLowerInvariant()
        }
    }
)
$record = [ordered]@{
    schema_version = 1
    record_type = 'qrt_dynamic_logical_gdn_qualification'
    host = [Environment]::MachineName
    repo_commit = $repoCommit
    dirty_tree = @(& git -C $Repo status --porcelain).Count -ne 0
    run_label = $RunLabel
    command_file = $PSCommandPath
    command_file_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
    provider = $resolvedProvider
    provider_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $resolvedProvider).Hash.ToLowerInvariant()
    kernel_dir = $resolvedKernelDir
    smoke_source = $source
    smoke_source_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $source).Hash.ToLowerInvariant()
    smoke_executable = $smokeExe
    smoke_executable_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $smokeExe).Hash.ToLowerInvariant()
    hipcc = $hipcc
    compile_arguments = $compileArguments
    compile_timeout_seconds = $CompileTimeoutSeconds
    compile = $compileRun
    case_timeout_seconds = $CaseTimeoutSeconds
    repetitions = $Repetitions
    dynamic_logical_tokens = $logicalTokens
    dynamic_logical_case_count = $cases.Count
    dynamic_logical_mode_count = 2 * $cases.Count
    dynamic_logical_reset_included = $false
    unmeasured_warmup_per_mode = 1
    component_only = $true
    inference_success_claimed = $false
    cases = $cases
    artifacts = $artifacts
}
[IO.File]::WriteAllText(
    $recordPath,
    ($record | ConvertTo-Json -Depth 8) + "`n",
    $utf8
)
$record | ConvertTo-Json -Depth 8 -Compress
