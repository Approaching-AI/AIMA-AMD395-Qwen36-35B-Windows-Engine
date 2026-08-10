param(
    [Parameter(Mandatory = $false)][string]$OutDir = "",
    [Parameter(Mandatory = $false)]
        [ValidateRange(30, 1800)]
        [int]$TimeoutSeconds = 900,
    [Parameter(Mandatory = $false)][switch]$ValidateOnly,
    [Parameter(Mandatory = $false)][switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repo "build\qrt-server"
} elseif (-not [IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repo $OutDir
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Find-VsDevCmd {
    foreach ($candidate in @(
            "D:\BuildTools\Common7\Tools\VsDevCmd.bat",
            "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
            "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
        )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    throw "VsDevCmd.bat was not found"
}

function Quote-BatArg {
    param([Parameter(Mandatory = $true)][string]$Value)
    return [System.String]::Concat('"', $Value.Replace('"', '""'), '"')
}

$cargo = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
if (-not (Test-Path -LiteralPath $cargo -PathType Leaf)) {
    throw "cargo.exe was not found at $cargo"
}
$vsDevCmd = Find-VsDevCmd
$targetDir = Join-Path $OutDir "cargo-target"
$bat = Join-Path $OutDir "build-qrt-server.bat"
$stdoutPath = Join-Path $OutDir "build.stdout.txt"
$stderrPath = Join-Path $OutDir "build.stderr.txt"
$batLines = @(
    "@echo off",
    "call $(Quote-BatArg $vsDevCmd) -arch=x64 -host_arch=x64",
    "if errorlevel 1 exit /b 21",
    "set `"CARGO_TARGET_DIR=$targetDir`"",
    "cd /d $(Quote-BatArg $repo)",
    "$(Quote-BatArg $cargo) fmt --package qrt-server -- --check",
    "if errorlevel 1 exit /b 22"
)
if (-not $SkipTests) {
    $batLines += "$(Quote-BatArg $cargo) test -p qrt-server --no-fail-fast"
    $batLines += "if errorlevel 1 exit /b 23"
} else {
    $batLines += "$(Quote-BatArg $cargo) check -p qrt-server"
    $batLines += "if errorlevel 1 exit /b 24"
}
if (-not $ValidateOnly) {
    $batLines += "$(Quote-BatArg $cargo) build -p qrt-server --release"
    $batLines += "if errorlevel 1 exit /b 25"
}
[IO.File]::WriteAllText(
    $bat,
    ($batLines -join [Environment]::NewLine) + [Environment]::NewLine,
    $utf8
)

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = "cmd.exe"
$startInfo.Arguments = "/d /s /c $(Quote-BatArg $bat)"
$startInfo.WorkingDirectory = $repo
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
$stdout = $stdoutTask.GetAwaiter().GetResult()
$stderr = $stderrTask.GetAwaiter().GetResult()
[IO.File]::WriteAllText($stdoutPath, $stdout, $utf8)
[IO.File]::WriteAllText($stderrPath, $stderr, $utf8)
if (-not $completed) {
    throw "qrt server build timed out after $TimeoutSeconds seconds; see $stdoutPath and $stderrPath"
}
$process.Refresh()
if ($process.ExitCode -ne 0) {
    throw "qrt server build failed with exit code $($process.ExitCode); see $stdoutPath and $stderrPath"
}

$executable = $null
if (-not $ValidateOnly) {
    $builtExe = Join-Path $targetDir "release\qrt.exe"
    if (-not (Test-Path -LiteralPath $builtExe -PathType Leaf)) {
        throw "Cargo did not emit $builtExe"
    }
    $executable = Join-Path $OutDir "qrt.exe"
    Copy-Item -LiteralPath $builtExe -Destination $executable -Force
}
$record = [ordered]@{
    schema_version = 1
    host = [Environment]::MachineName
    execution = "local_windows_process"
    repo_commit = (& git -C $repo rev-parse HEAD).Trim()
    dirty_tree = @(& git -C $repo status --porcelain).Count -ne 0
    command_file = $PSCommandPath
    timeout_seconds = $TimeoutSeconds
    validate_only = [bool]$ValidateOnly
    tests_run = -not [bool]$SkipTests
    wall_ms = [Math]::Round($watch.Elapsed.TotalMilliseconds, 6)
    cargo = $cargo
    rustc_version = (& (Join-Path (Split-Path -Parent $cargo) "rustc.exe") --version).Trim()
    executable = if ($null -ne $executable) { $executable } else { $null }
    executable_sha256 = if ($null -ne $executable) {
        (Get-FileHash -Algorithm SHA256 -LiteralPath $executable).Hash.ToLowerInvariant()
    } else { $null }
}
$record | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $OutDir "build-provenance.json") -Encoding UTF8
if ($null -ne $executable) {
    Write-Output $executable
} else {
    Write-Output "validation passed"
}
