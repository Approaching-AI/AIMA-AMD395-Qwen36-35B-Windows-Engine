param(
    [Parameter(Mandatory = $true)][string]$OutDir,
    [ValidateRange(30, 600)][int]$TimeoutSeconds = 240,
    [string]$WslDistribution = 'Ubuntu-24.04',
    [string]$TritonPython = '/opt/qwen36-vllm/bin/python'
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
$wslOut = Wsl-Path $OutDir
$wslGenerator = Wsl-Path $generator
$innerTimeout = [Math]::Min(150, $TimeoutSeconds - 15)
$batch = Join-Path $OutDir 'build-fla.bat'
$dll = Join-Path $OutDir 'qrt_fla_chunk_gdn_provider.dll'
$lines = @(
    '@echo off',
    "call $(Quote-Arg $vs) -arch=x64 -host_arch=x64",
    'if errorlevel 1 exit /b 21',
    "wsl.exe -d $(Quote-Arg $WslDistribution) -- timeout $innerTimeout env OMP_NUM_THREADS=2 MAX_JOBS=2 TRITON_CACHE_DIR=$wslOut/triton-cache $(Quote-Arg $TritonPython) $(Quote-Arg $wslGenerator) --output-dir $(Quote-Arg $wslOut) --metadata $(Quote-Arg ($wslOut + '/metadata.json'))",
    'if errorlevel 1 exit /b 22',
    "$(Quote-Arg $hipcc) -std=c++17 -O2 --offload-arch=gfx1151 -I$(Quote-Arg $OutDir) -shared $(Quote-Arg $provider) -o $(Quote-Arg $dll)",
    'if errorlevel 1 exit /b 23'
)
foreach ($tokens in @(64, 65, 7169)) {
    $exe = Join-Path $OutDir "q$tokens-fla-smoke.exe"
    $lines += "$(Quote-Arg $hipcc) -std=c++17 -O2 --offload-arch=gfx1151 -DQRT_FLA_GDN_SMOKE_TOKENS=$tokens $(Quote-Arg $smoke) -o $(Quote-Arg $exe)"
    $lines += 'if errorlevel 1 exit /b 24'
}
[IO.File]::WriteAllText($batch, ($lines -join [Environment]::NewLine) + [Environment]::NewLine, $utf8)
$stdout = Join-Path $OutDir 'build.stdout.log'
$stderr = Join-Path $OutDir 'build.stderr.log'
$watch = [Diagnostics.Stopwatch]::StartNew()
$process = Start-Process -FilePath 'cmd.exe' -ArgumentList ('/d /s /c ' + (Quote-Arg $batch)) -WorkingDirectory $repo -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$completed = $process.WaitForExit($TimeoutSeconds * 1000)
if (-not $completed) {
    $killer = Start-Process -FilePath 'taskkill.exe' -ArgumentList "/PID $($process.Id) /T /F" -NoNewWindow -PassThru
    $null = $killer.WaitForExit(5000)
    $null = $process.WaitForExit(5000)
    throw "Bounded FLA build timed out; inspect logs and process state before retry."
}
$process.Refresh()
$watch.Stop()
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
    hipcc=$hipcc; wsl_distribution=$WslDistribution; triton_python=$TritonPython
    sources=@(@($generator, $provider, $smoke) | ForEach-Object {
        [ordered]@{path=$_;sha256=(Get-FileHash $_ -Algorithm SHA256).Hash.ToLowerInvariant()}
    })
    artifacts=$artifacts; numerical_acceptance=$false
}
[IO.File]::WriteAllText((Join-Path $OutDir 'build-provenance.json'), ($record | ConvertTo-Json -Depth 6), $utf8)
Write-Output $dll
