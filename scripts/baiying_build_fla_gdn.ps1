param(
    [Parameter(Mandatory = $true)][string]$OutDir,
    [ValidateRange(30, 600)][int]$TimeoutSeconds = 240,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$WslDistribution = 'Ubuntu-24.04',
    [string]$TritonPython = '/opt/qwen36-vllm/bin/python',
    [string]$AotDir = ''
)

# Build only. Run individual probes separately through the guarded runner.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not [Environment]::MachineName.Equals('baiying', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'This native Windows builder must execute on baiying.'
}
$repo = Split-Path -Parent $PSScriptRoot
if (-not [IO.Path]::IsPathRooted($OutDir)) { $OutDir = Join-Path $repo $OutDir }
$OutDir = [IO.Path]::GetFullPath($OutDir)
if (Test-Path -LiteralPath $OutDir) { throw "Output already exists: $OutDir" }
$null = New-Item -ItemType Directory -Path $OutDir
$utf8 = New-Object Text.UTF8Encoding -ArgumentList $false
function Quote-Arg([string]$Value) {
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}
function Wsl-Path([string]$Value) {
    if ($Value -notmatch '^([A-Za-z]):\\(.*)$') { throw "Not a drive path: $Value" }
    return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2].Replace('\', '/')
}
$hipcc = (Get-Command hipcc.exe -ErrorAction Stop).Source
$vs = @(
    'D:\BuildTools\Common7\Tools\VsDevCmd.bat',
    'C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat',
    'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $vs) { throw 'VsDevCmd.bat was not found.' }
$generator = Join-Path $repo 'native\generators\compile_q8192_fla_chunk_gdn.py'
$provider = Join-Path $repo 'native\providers\gdn\qrt_fla_chunk_gdn_q8192_provider.cpp'
$smoke = Join-Path $repo 'native\providers\gdn\q64_fla_chunk_gdn_smoke.cpp'
$outputReplay = Join-Path $repo 'native\providers\gdn\fla_output_capture_replay.cpp'
$blackwellKkt = Join-Path $repo 'native\providers\gdn\blackwell_kkt.h'
$blackwellAccumulator = Join-Path $repo 'native\providers\moe_accumulator\q1_moe_hawkeye_bf16_accumulator.h'
if ($AotDir) {
    if (-not [IO.Path]::IsPathRooted($AotDir)) { $AotDir = Join-Path $repo $AotDir }
    $metaPath = Join-Path $AotDir 'metadata.json'
    $meta = [IO.File]::ReadAllText($metaPath) | ConvertFrom-Json
    if ($meta.target -ne 'gfx1151' -or $meta.source_sha256 -ne (Get-FileHash $generator -Algorithm SHA256).Hash.ToLowerInvariant()) {
        throw 'AOT target/source does not match this checkout.'
    }
    foreach ($kernel in $meta.kernels) {
        if ([IO.Path]::GetFileName($kernel.file) -ne $kernel.file) { throw 'Invalid AOT basename.' }
        $path = Join-Path $AotDir $kernel.file
        if ((Get-Item $path).Length -ne $kernel.bytes -or (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $kernel.sha256) {
            throw "AOT fingerprint mismatch: $path"
        }
        Copy-Item -LiteralPath $path -Destination (Join-Path $OutDir $kernel.file)
    }
    $specs = Join-Path $AotDir 'qrt_fla_gdn_kernel_specs.inc'
    if ((Get-FileHash $specs -Algorithm SHA256).Hash.ToLowerInvariant() -ne $meta.provider_specs_sha256) {
        throw 'AOT launch-header fingerprint mismatch.'
    }
    Copy-Item -LiteralPath $specs -Destination $OutDir
    Copy-Item -LiteralPath $metaPath -Destination $OutDir
}
$wslOut = Wsl-Path $OutDir
$wslGenerator = Wsl-Path $generator
$innerTimeout = [Math]::Min(150, $TimeoutSeconds - 15)
$batch = Join-Path $OutDir 'build-fla.bat'
$dll = Join-Path $OutDir 'qrt_fla_chunk_gdn_provider.dll'
$lines = @(
    '@echo off',
    "call $(Quote-Arg $vs) -arch=x64 -host_arch=x64",
    'if not "%errorlevel%"=="0" exit /b 21'
)
if (-not $AotDir) {
    # AOT is CPU-only. Do not let import-time driver discovery open a device,
    # and kill a compiler that ignores the first timeout signal inside WSL.
    # This host's WSL CLI treats quotes around -d as part of the registry name
    # when invoked from cmd.exe. The validated name needs no shell quoting.
    $lines += "wsl.exe -d $WslDistribution -- timeout --kill-after=5 $innerTimeout env HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MAX_JOBS=2 TRITON_CACHE_DIR=$wslOut/triton-cache $(Quote-Arg $TritonPython) $(Quote-Arg $wslGenerator) --output-dir $(Quote-Arg $wslOut) --metadata $(Quote-Arg ($wslOut + '/metadata.json'))"
    # WSL errors may be negative. 'if errorlevel 1' silently misses those.
    $lines += 'if not "%errorlevel%"=="0" exit /b 22'
}
$lines += @(
    "if not exist $(Quote-Arg (Join-Path $OutDir 'qrt_fla_gdn_kernel_specs.inc')) exit /b 26",
    "$(Quote-Arg $hipcc) -std=c++17 -O2 --offload-arch=gfx1151 -I$(Quote-Arg $OutDir) -shared $(Quote-Arg $provider) -o $(Quote-Arg $dll)",
    'if not "%errorlevel%"=="0" exit /b 23'
)
foreach ($tokens in @(64, 65, 7169)) {
    $exe = Join-Path $OutDir "q$tokens-fla-smoke.exe"
    $lines += "$(Quote-Arg $hipcc) -std=c++17 -O2 --offload-arch=gfx1151 -DQRT_FLA_GDN_SMOKE_TOKENS=$tokens $(Quote-Arg $smoke) -o $(Quote-Arg $exe)"
    $lines += 'if not "%errorlevel%"=="0" exit /b 24'
}
$replayExe = Join-Path $OutDir 'fla-output-capture-replay.exe'
$lines += "$(Quote-Arg $hipcc) -std=c++17 -O2 --offload-arch=gfx1151 $(Quote-Arg $outputReplay) -o $(Quote-Arg $replayExe)"
$lines += 'if not "%errorlevel%"=="0" exit /b 27'
[IO.File]::WriteAllText($batch, ($lines -join [Environment]::NewLine) + [Environment]::NewLine, $utf8)
$stdout = Join-Path $OutDir 'build.stdout.log'
$stderr = Join-Path $OutDir 'build.stderr.log'
$watch = [Diagnostics.Stopwatch]::StartNew()
$info = New-Object Diagnostics.ProcessStartInfo
$info.FileName = 'cmd.exe'
$info.Arguments = '/d /s /c ' + (Quote-Arg $batch)
$info.WorkingDirectory = $repo
$info.UseShellExecute = $false
$info.CreateNoWindow = $true
$info.RedirectStandardOutput = $true
$info.RedirectStandardError = $true
$process = New-Object Diagnostics.Process
$process.StartInfo = $info
$null = $process.Start()
$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()
$completed = $process.WaitForExit($TimeoutSeconds * 1000)
if (-not $completed) {
    $killer = Start-Process -FilePath 'taskkill.exe' -ArgumentList "/PID $($process.Id) /T /F" -NoNewWindow -PassThru
    $null = $killer.WaitForExit(5000)
    $null = $process.WaitForExit(5000)
    throw "Bounded FLA build timed out; inspect logs and process state before retry."
}
$process.Refresh()
$watch.Stop()
if (-not $stdoutTask.Wait(5000) -or -not $stderrTask.Wait(5000)) { throw 'Build output pipes did not close.' }
[IO.File]::WriteAllText($stdout, $stdoutTask.GetAwaiter().GetResult(), $utf8)
[IO.File]::WriteAllText($stderr, $stderrTask.GetAwaiter().GetResult(), $utf8)
if ($process.ExitCode -ne 0) { throw "FLA build exited $($process.ExitCode); see $stderr" }
$artifacts = @(Get-ChildItem -LiteralPath $OutDir -File | Where-Object {
    $_.Extension -in @('.hsaco', '.dll', '.exe', '.inc', '.json')
} | ForEach-Object {
    [ordered]@{file=$_.Name; bytes=$_.Length; sha256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$record = [ordered]@{
    schema_version=1; host=[Environment]::MachineName
    execution='local_windows_build_only'; repo_commit=(& git -C $repo rev-parse HEAD).Trim()
    dirty_tree=@(& git -C $repo status --porcelain).Count -ne 0
    command_file=$PSCommandPath; timeout_seconds=$TimeoutSeconds; wall_ms=$watch.Elapsed.TotalMilliseconds
    hipcc=$hipcc; wsl_distribution=$WslDistribution; triton_python=$TritonPython; precompiled_aot=$AotDir
    sources=@(@($generator, $provider, $smoke, $blackwellKkt, $blackwellAccumulator, $outputReplay) | ForEach-Object {
        [ordered]@{path=$_;sha256=(Get-FileHash $_ -Algorithm SHA256).Hash.ToLowerInvariant()}
    })
    artifacts=$artifacts; numerical_acceptance=$false
}
[IO.File]::WriteAllText((Join-Path $OutDir 'build-provenance.json'), ($record | ConvertTo-Json -Depth 6), $utf8)
Write-Output $dll
