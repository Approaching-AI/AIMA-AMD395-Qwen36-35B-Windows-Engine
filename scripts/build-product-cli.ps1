param(
    [Parameter(Mandatory = $false)][string]$OutDir = "",
    [Parameter(Mandatory = $false)]
        [ValidateRange(30, 900)]
        [int]$TimeoutSeconds = 300,
    [Parameter(Mandatory = $false)]
        [ValidateRange(16777216, 536870912)]
        [long]$StackReserveBytes = 268435456,
    [Parameter(Mandatory = $false)][string]$VsDevCmdPath = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false

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

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repo "build\product-cli"
} elseif (-not [IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repo $OutDir
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
if ([string]::IsNullOrWhiteSpace($VsDevCmdPath)) {
    $VsDevCmdPath = Find-VsDevCmd
}
if (-not (Test-Path -LiteralPath $VsDevCmdPath -PathType Leaf)) {
    throw "VsDevCmd.bat was not found: $VsDevCmdPath"
}

$sourceDir = Join-Path $repo "native\src"
$sources = @(
    (Join-Path $sourceDir "qrt.c"),
    (Join-Path $sourceDir "qwen36_baseline.c"),
    (Join-Path $sourceDir "product_cli.c")
)
foreach ($source in $sources) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "product CLI source was not found: $source"
    }
}
$objects = @(
    (Join-Path $OutDir "qrt.obj"),
    (Join-Path $OutDir "qwen36_baseline.obj"),
    (Join-Path $OutDir "product_cli.obj")
)
$executable = Join-Path $OutDir "qrt-product.exe"
$commandFile = Join-Path $OutDir "build-product-cli.bat"
$stdoutPath = Join-Path $OutDir "build.stdout.txt"
$stderrPath = Join-Path $OutDir "build.stderr.txt"

$batLines = @(
    "@echo off",
    "call $(Quote-BatArg $VsDevCmdPath) -arch=x64 -host_arch=x64",
    "if errorlevel 1 exit /b 21"
)
for ($index = 0; $index -lt $sources.Count; ++$index) {
    $batLines += (
        "cl /nologo /std:c11 /O2 /W4 /D_CRT_SECURE_NO_WARNINGS /c " +
        "/I$(Quote-BatArg $sourceDir) $(Quote-BatArg $sources[$index]) " +
        "/Fo$(Quote-BatArg $objects[$index])"
    )
    $batLines += "if errorlevel 1 exit /b $($index + 22)"
}
$quotedObjects = @($objects | ForEach-Object { Quote-BatArg $_ }) -join " "
$batLines += (
    "link /nologo $quotedObjects /OUT:$(Quote-BatArg $executable) " +
    "/STACK:$StackReserveBytes"
)
$batLines += "if errorlevel 1 exit /b 25"
[IO.File]::WriteAllText(
    $commandFile,
    ($batLines -join [Environment]::NewLine) + [Environment]::NewLine,
    $utf8
)

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = "cmd.exe"
$startInfo.Arguments = "/d /s /c $(Quote-BatArg $commandFile)"
$startInfo.WorkingDirectory = $repo
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
}
$process.WaitForExit()
$watch.Stop()
[IO.File]::WriteAllText(
    $stdoutPath,
    $stdoutTask.GetAwaiter().GetResult(),
    $utf8
)
[IO.File]::WriteAllText(
    $stderrPath,
    $stderrTask.GetAwaiter().GetResult(),
    $utf8
)
if (-not $completed) {
    throw "product CLI build timed out after $TimeoutSeconds seconds"
}
$process.Refresh()
if ($process.ExitCode -ne 0) {
    throw "product CLI build failed with exit code $($process.ExitCode); see $stderrPath"
}
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "product CLI build did not emit $executable"
}

$sourceRecords = foreach ($source in @($sources) + @(
        (Join-Path $sourceDir "qrt.h"),
        (Join-Path $sourceDir "qrt_prefix_logit.h"),
        (Join-Path $sourceDir "qrt_prefix_checkpoint.h"),
        (Join-Path $sourceDir "qwen36_baseline.h")
    )) {
    [ordered]@{
        path = $source
        sha256 = (Get-FileHash -Algorithm SHA256 `
            -LiteralPath $source).Hash.ToLowerInvariant()
    }
}
$record = [ordered]@{
    schema_version = 1
    host = [Environment]::MachineName
    repo_commit = (& git -C $repo rev-parse HEAD).Trim()
    dirty_tree = @(& git -C $repo status --porcelain).Count -ne 0
    command_file = $commandFile
    command_file_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $commandFile).Hash.ToLowerInvariant()
    build_script = $PSCommandPath
    build_script_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
    timeout_seconds = $TimeoutSeconds
    stack_reserve_bytes = $StackReserveBytes
    wall_ms = [Math]::Round($watch.Elapsed.TotalMilliseconds, 6)
    sources = @($sourceRecords)
    executable = $executable
    executable_bytes = (Get-Item -LiteralPath $executable).Length
    executable_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $executable).Hash.ToLowerInvariant()
    compiler_stdout_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $stdoutPath).Hash.ToLowerInvariant()
    compiler_stderr_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath $stderrPath).Hash.ToLowerInvariant()
}
[IO.File]::WriteAllText(
    (Join-Path $OutDir "build-provenance.json"),
    ($record | ConvertTo-Json -Depth 5) + [Environment]::NewLine,
    $utf8
)
Write-Output $executable
