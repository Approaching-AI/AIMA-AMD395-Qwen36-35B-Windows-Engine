param(
    [string]$Repo = 'D:\projects\AIMA-dynamic-logical-r1',
    [string]$CkRoot = 'D:\projects\composable-kernel-fdf4bb7',
    [ValidatePattern('^r[0-9]+$')]
    [string]$RunLabel = 'r1087',
    [ValidateRange(1, 1800)]
    [int]$CompileTimeoutSeconds = 600,
    [ValidateRange(60, 1800)]
    [int]$SmokeTimeoutSeconds = 900,
    [ValidateRange(1, 100)]
    [int]$SmokeRepetitions = 3
)

$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false
$sourceDir = Join-Path $Repo 'native\providers\ck_fmha'
$outDir = Join-Path $Repo "build\ck-fmha-q1-terminal-exact-$RunLabel"
$outPath = Join-Path $outDir 'qrt_ck_fmha_q1_terminal_exact.dll'
$directSmokeSource = Join-Path $sourceDir `
    'q8192_ck_fmha_direct_smoke.cpp'
$directSmokePath = Join-Path $outDir `
    'q8192_ck_fmha_direct_smoke.exe'
$ckInclude = Join-Path $CkRoot 'include'
$ckExample = Join-Path $CkRoot 'example\ck_tile\01_fmha'
$ckArchHeader = Join-Path $ckInclude 'ck_tile\core\arch\arch.hpp'
$providerSource = Join-Path $sourceDir 'qrt_ck_fmha_q8192_provider.cpp'
$apiSource = Join-Path $sourceDir 'fmha_fwd_api.cpp'
$instanceSource = Join-Path $sourceDir `
    'fmha_fwd_gfx1151_d256_bf16_f32out.cpp'
$accumulatorHeader = Join-Path $Repo `
    'native\providers\moe_accumulator\q1_moe_hawkeye_bf16_accumulator.h'

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
    throw "refusing to overwrite $RunLabel CK q1 terminal build: $outDir"
}
$hipcc = (Get-Command hipcc.exe -ErrorAction Stop).Source
$llvmReadobj = Join-Path (Split-Path -Parent $hipcc) 'llvm-readobj.exe'
foreach ($required in @(
        $ckArchHeader,
        $providerSource,
        $apiSource,
        $instanceSource,
        $directSmokeSource,
        $accumulatorHeader,
        $llvmReadobj
    )) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "r1087 CK build input not found: $required"
    }
}
$ckArchSource = Get-Content -Raw -LiteralPath $ckArchHeader
if ($ckArchSource -match 'struct\s+gfx115_t') {
    $ckArchType = 'ck_tile::gfx115_t'
} elseif ($ckArchSource -match 'struct\s+gfx11_t') {
    $ckArchType = 'ck_tile::gfx11_t'
} else {
    throw 'CK-Tile architecture type for gfx1151 was not found'
}
[void](New-Item -ItemType Directory -Path $outDir)
$arguments = @(
    '-std=c++17', '-O3', '--offload-arch=gfx1151',
    '-DCK_TILE_FMHA_FWD_FAST_EXP2=0',
    '-DQRT_CK_FMHA_VLLM_N32=1',
    '-DQRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL=1',
    "-DQRT_CK_ARCH_TYPE=$ckArchType",
    '-I', $ckInclude, '-I', $ckExample, '-shared',
    $providerSource,
    $apiSource,
    $instanceSource,
    '-o', $outPath
)
$compileRun = Invoke-BoundedProcess -FilePath $hipcc `
    -Arguments $arguments -WorkingDirectory $Repo `
    -StdOutPath (Join-Path $outDir 'compile.stdout.txt') `
    -StdErrPath (Join-Path $outDir 'compile.stderr.txt') `
    -TimeoutSeconds $CompileTimeoutSeconds
if (-not $compileRun.completed -or $compileRun.exit_code -ne 0) {
    throw "bounded $RunLabel CK q1 terminal build failed or timed out"
}
$smokeCompileArguments = @(
    '-std=c++17', '-O3', '--offload-arch=gfx1151',
    $directSmokeSource,
    '-o', $directSmokePath
)
$smokeCompileRun = Invoke-BoundedProcess -FilePath $hipcc `
    -Arguments $smokeCompileArguments -WorkingDirectory $Repo `
    -StdOutPath (Join-Path $outDir 'smoke-compile.stdout.txt') `
    -StdErrPath (Join-Path $outDir 'smoke-compile.stderr.txt') `
    -TimeoutSeconds $CompileTimeoutSeconds
if (-not $smokeCompileRun.completed -or
    $smokeCompileRun.exit_code -ne 0) {
    throw "bounded $RunLabel CK dynamic-logical smoke build failed or timed out"
}
$smokeStdoutPath = Join-Path $outDir 'dynamic-logical-smoke.stdout.log'
$smokeStderrPath = Join-Path $outDir 'dynamic-logical-smoke.stderr.log'
$smokeRun = Invoke-BoundedProcess -FilePath $directSmokePath `
    -Arguments @($outPath, [string]$SmokeRepetitions) `
    -WorkingDirectory $outDir `
    -StdOutPath $smokeStdoutPath `
    -StdErrPath $smokeStderrPath `
    -TimeoutSeconds $SmokeTimeoutSeconds
if (-not $smokeRun.completed -or $smokeRun.exit_code -ne 0) {
    throw "bounded $RunLabel CK dynamic-logical smoke failed or timed out"
}
$smokeText = Get-Content -Raw -LiteralPath $smokeStdoutPath
$dynamicLogicalMatches = [regex]::Matches(
    $smokeText,
    '(?m)^ck_fmha_dynamic_logical_case\s+' +
    'tokens=(?<tokens>[0-9]+)\b[^\r\n]*' +
    '\bcold_total_ms=(?<cold>[0-9]+(?:\.[0-9]+)?)\b' +
    '[^\r\n]*\bcold_submit_ms=(?<submit>[0-9]+(?:\.[0-9]+)?)\b' +
    '[^\r\n]*\bcold_device_completion_ms=' +
    '(?<device>[0-9]+(?:\.[0-9]+)?)\b' +
    '[^\r\n]*\bwarm_mean_ms=(?<warm>[0-9]+(?:\.[0-9]+)?)\b' +
    '[^\r\n]*\breset_included=0\b[^\r\n]*' +
    '\btiming_finite_positive=1\b[^\r\n]*' +
    '\btotal_nonfinite=0\b[^\r\n]*\bclose=1\s*$'
)
$expectedDynamicLogicalTokens = @(
    2073, 2156, 2560, 3073, 4609, 6145, 2049, 2175,
    2176, 2177, 2559, 2561, 3071, 3072, 3583, 3584,
    3585, 4095, 4096, 4097, 4607, 4608, 6143, 6144,
    7167, 7168, 7169, 7679, 7680, 7681, 8191, 8192
)
$observedDynamicLogicalTokens = @(
    $dynamicLogicalMatches | ForEach-Object {
        [int]$_.Groups['tokens'].Value
    }
)
$dynamicLogicalCases = @(
    foreach ($match in $dynamicLogicalMatches) {
        $coldMs = [double]::Parse(
            $match.Groups['cold'].Value,
            [Globalization.CultureInfo]::InvariantCulture
        )
        $submitMs = [double]::Parse(
            $match.Groups['submit'].Value,
            [Globalization.CultureInfo]::InvariantCulture
        )
        $deviceMs = [double]::Parse(
            $match.Groups['device'].Value,
            [Globalization.CultureInfo]::InvariantCulture
        )
        $warmMs = [double]::Parse(
            $match.Groups['warm'].Value,
            [Globalization.CultureInfo]::InvariantCulture
        )
        if ([double]::IsNaN($coldMs) -or
            [double]::IsInfinity($coldMs) -or $coldMs -le 0.0 -or
            [double]::IsNaN($submitMs) -or
            [double]::IsInfinity($submitMs) -or $submitMs -lt 0.0 -or
            [double]::IsNaN($deviceMs) -or
            [double]::IsInfinity($deviceMs) -or $deviceMs -lt 0.0 -or
            [double]::IsNaN($warmMs) -or
            [double]::IsInfinity($warmMs) -or $warmMs -le 0.0 -or
            $submitMs -gt $coldMs) {
            throw 'CK dynamic-logical smoke reported invalid timing'
        }
        $tokens = [int]$match.Groups['tokens'].Value
        [ordered]@{
            tokens = $tokens
            cold_total_ms = $coldMs
            cold_submit_ms = $submitMs
            cold_device_completion_ms = $deviceMs
            cold_tokens_per_second = [Math]::Round(
                [double]$tokens * 1000.0 / $coldMs,
                6
            )
            warm_mean_ms = $warmMs
            warm_tokens_per_second = [Math]::Round(
                [double]$tokens * 1000.0 / $warmMs,
                6
            )
            reset_included = $false
        }
    }
)
if ($dynamicLogicalMatches.Count -ne 32 -or
    ($observedDynamicLogicalTokens -join ',') -ne
        ($expectedDynamicLogicalTokens -join ',') -or
    $smokeText -notmatch (
        '(?m)^ck_fmha_dynamic_logical_smoke\s+' +
        'cases=32\s+all_close=1\s+' +
        'dynamic_logical_reset_included=0\s+' +
        'terminal_authority=product_gb10\s+' +
        'component_only=1\s+inference_success_claimed=0\s*$'
    )) {
    throw "$RunLabel CK dynamic-logical smoke output is incomplete"
}
$exports = & $llvmReadobj --coff-exports $outPath
if ($LASTEXITCODE -ne 0) {
    throw "llvm-readobj failed for $RunLabel CK q1 terminal DLL"
}
$exportText = ($exports -join "`n") + "`n"
$exportsPath = Join-Path $outDir 'exports.txt'
[IO.File]::WriteAllText($exportsPath, $exportText, $utf8)
$expectedExports = @(
    'qrt_ck_fmha_q8192_prepare',
    'qrt_ck_fmha_q8192_f32_launch',
    'qrt_ck_fmha_q1_kv8192_f32_launch',
    'qrt_ck_fmha_q1_dynamic_f32_launch',
    'qrt_ck_fmha_q8192_release'
)
foreach ($exportName in $expectedExports) {
    if ($exportText -notmatch [regex]::Escape($exportName)) {
        throw "$RunLabel CK q1 terminal DLL is missing export $exportName"
    }
}
$item = Get-Item -LiteralPath $outPath
$record = [ordered]@{
    schema_version = 1
    host = [Environment]::MachineName
    repo_commit = (& git -C $Repo rev-parse HEAD).Trim()
    dirty_tree = @(& git -C $Repo status --porcelain).Count -ne 0
    command_file = $PSCommandPath
    command_file_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
    run_label = $RunLabel
    path = $item.FullName
    bytes = $item.Length
    sha256 = (Get-FileHash -Algorithm SHA256 $outPath).Hash.ToLowerInvariant()
    provider_source_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $providerSource).Hash.ToLowerInvariant()
    api_source_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $apiSource).Hash.ToLowerInvariant()
    instance_source_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $instanceSource).Hash.ToLowerInvariant()
    direct_smoke_source_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $directSmokeSource).Hash.ToLowerInvariant()
    accumulator_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $accumulatorHeader).Hash.ToLowerInvariant()
    hipcc = $hipcc
    hipcc_arguments = $arguments
    compile_timeout_seconds = $CompileTimeoutSeconds
    compile = $compileRun
    smoke_compile_arguments = $smokeCompileArguments
    smoke_compile = $smokeCompileRun
    smoke_timeout_seconds = $SmokeTimeoutSeconds
    smoke_repetitions = $SmokeRepetitions
    smoke = $smokeRun
    smoke_stdout = $smokeStdoutPath
    smoke_stdout_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $smokeStdoutPath).Hash.ToLowerInvariant()
    smoke_stderr = $smokeStderrPath
    smoke_stderr_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $smokeStderrPath).Hash.ToLowerInvariant()
    dynamic_logical_tokens = $observedDynamicLogicalTokens
    dynamic_logical_cases = $dynamicLogicalCases
    dynamic_logical_case_count = $dynamicLogicalMatches.Count
    dynamic_logical_reset_included = $false
    dynamic_logical_component_only = $true
    inference_success_claimed = $false
    expected_exports = $expectedExports
    exports = $exportsPath
    ck_tile_n = 32
    blackwell_exact_terminal_rows = 16
    dedicated_q1_kv8192_f32_export = 1
    dedicated_q1_dynamic_f32_export = 1
    dynamic_q1_max_kv_tokens = 262144
    full_prefix_ck_launch_for_q1 = 0
}
[IO.File]::WriteAllText(
    (Join-Path $outDir 'build-provenance.json'),
    ($record | ConvertTo-Json -Depth 6) + "`n",
    $utf8
)
$record | ConvertTo-Json -Depth 6 -Compress
