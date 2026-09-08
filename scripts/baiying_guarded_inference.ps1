param(
    [Parameter(Mandatory = $true)][string]$SpecPath,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [ValidateRange(1, 900)][int]$TimeoutSeconds = 90,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProgressPreference = 'SilentlyContinue'
if (-not [Environment]::MachineName.Equals('baiying', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'This acceptance runner must execute on baiying.'
}
$spec = [IO.File]::ReadAllText($SpecPath) | ConvertFrom-Json
foreach ($path in @($spec.executable, $spec.working_directory)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing input: $path" }
}
if (Test-Path -LiteralPath $OutDir) { throw "Output already exists: $OutDir" }
$OutDir = [IO.Path]::GetFullPath($OutDir)
$null = New-Item -ItemType Directory -Path $OutDir
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false
function Write-Record([string]$Name, $Value) {
    [IO.File]::WriteAllText((Join-Path $OutDir $Name),
        (($Value | ConvertTo-Json -Depth 16) + "`n"), $utf8)
}
function Quote-Argument([string]$Value) {
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}
function Get-EngineProcesses {
    return @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -match '^(qrt.*|hipcc|clang.*|cl|link|lld-link|rustc|cargo|ollama|llama.*|vllm|sglang)$'
    } | Select-Object Id, ProcessName)
}

# OS memory sampling does not call the GPU driver or block on WMI per poll.
# KILL_ON_JOB_CLOSE also stops descendants if the supervisor exits unexpectedly.
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class QrtRunGuard {
    [StructLayout(LayoutKind.Sequential)] public struct Memory {
        public uint Length, Load;
        public ulong TotalPhysical, AvailablePhysical, TotalCommit, AvailableCommit;
        public ulong TotalVirtual, AvailableVirtual, AvailableExtendedVirtual;
    }
    [StructLayout(LayoutKind.Sequential)] struct BasicLimit {
        public long ProcessTime, JobTime;
        public uint Flags;
        public UIntPtr MinWorkingSet, MaxWorkingSet;
        public uint ActiveProcesses;
        public UIntPtr Affinity;
        public uint Priority, Scheduling;
    }
    [StructLayout(LayoutKind.Sequential)] struct IoCounters {
        public ulong ReadOperations, WriteOperations, OtherOperations;
        public ulong ReadBytes, WriteBytes, OtherBytes;
    }
    [StructLayout(LayoutKind.Sequential)] struct ExtendedLimit {
        public BasicLimit Basic;
        public IoCounters Io;
        public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool GlobalMemoryStatusEx(ref Memory value);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr security, string name);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetInformationJobObject(IntPtr job, int kind, ref ExtendedLimit value, uint size);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError=true)] public static extern bool TerminateJobObject(IntPtr job, uint exitCode);
    [DllImport("kernel32.dll", SetLastError=true)] public static extern bool CloseHandle(IntPtr handle);
    public static Memory Snapshot() {
        var value = new Memory(); value.Length = (uint)Marshal.SizeOf(value);
        if (!GlobalMemoryStatusEx(ref value)) throw new Win32Exception();
        return value;
    }
    public static IntPtr Create() {
        var job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) throw new Win32Exception();
        var value = new ExtendedLimit(); value.Basic.Flags = 0x2000;
        if (!SetInformationJobObject(job, 9, ref value, (uint)Marshal.SizeOf(value))) {
            var error = new Win32Exception(); CloseHandle(job); throw error;
        }
        return job;
    }
    public static void Assign(IntPtr job, IntPtr process) {
        if (!AssignProcessToJobObject(job, process)) throw new Win32Exception();
    }
}
'@

$mutex = New-Object Threading.Mutex($false, 'Global\QRT_Baiying_GPU_Experiment')
$locked = $false
$job = [IntPtr]::Zero
$process = $null
$resultCode = 2
try {
    try { $locked = $mutex.WaitOne(0) }
    catch [Threading.AbandonedMutexException] { $locked = $true }
    if (-not $locked) { throw 'Another guarded experiment holds the GPU lock.' }
    $beforeMemory = [QrtRunGuard]::Snapshot()
    $beforeProcesses = @(Get-EngineProcesses)
    $os = Get-CimInstance Win32_OperatingSystem -OperationTimeoutSec 5
    $boot = $os.LastBootUpTime.ToString('o')
    $gpus = @(Get-CimInstance Win32_VideoController -OperationTimeoutSec 5 |
        Where-Object { $_.Name -match 'AMD|Radeon' } | Select-Object Name, Status, DriverVersion)
    $checks = [ordered]@{
        no_engine_or_compiler = $beforeProcesses.Count -eq 0
        host_memory_reserve = $beforeMemory.AvailablePhysical -ge 8GB
        commit_reserve = $beforeMemory.AvailableCommit -ge 20GB
        amd_gpu_ok = $gpus.Count -gt 0 -and @($gpus | Where-Object { $_.Status -ne 'OK' }).Count -eq 0
        output_disk_reserve = (New-Object IO.DriveInfo([IO.Path]::GetPathRoot($OutDir))).AvailableFreeSpace -ge 10GB
    }
    $inputHashes = [ordered]@{}
    for ($index = 0; $index -lt $spec.arguments.Count; $index++) {
        if ($spec.arguments[$index] -in @('--provider-dll', '--env-file', '--tokens', '--expected-output')) {
            $inputPath = [string]$spec.arguments[$index + 1]
            $inputHashes[[string]$spec.arguments[$index]] = [ordered]@{
                path = $inputPath
                sha256 = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
    }
    if ($spec.PSObject.Properties.Name -contains 'repo_commit') {
        $actualCommit = (& git -C $spec.working_directory rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0) { throw 'Cannot verify source commit.' }
        $checks['source_commit_exact'] = $actualCommit -eq $spec.repo_commit
    }
    $preflight = [ordered]@{
        host = [Environment]::MachineName
        utc = [DateTime]::UtcNow.ToString('o')
        command_file = $PSCommandPath
        command_file_sha256 = (Get-FileHash $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        spec = $spec
        executable_sha256 = (Get-FileHash $spec.executable -Algorithm SHA256).Hash.ToLowerInvariant()
        input_fingerprints = $inputHashes
        boot = $boot
        memory = $beforeMemory
        gpus = $gpus
        processes = $beforeProcesses
        checks = $checks
        pass = -not ($checks.Values -contains $false)
    }
    Write-Record 'preflight.json' $preflight
    if (-not $preflight.pass) { throw 'Host preflight failed; no process was launched.' }
    if ($PreflightOnly) { $preflight | ConvertTo-Json -Depth 8 -Compress; exit 0 }

    $stdoutPath = Join-Path $OutDir 'product.stdout.jsonl'
    $stderrPath = Join-Path $OutDir 'product.stderr.log'
    $telemetryPath = Join-Path $OutDir 'telemetry.jsonl'
    $telemetry = New-Object IO.StreamWriter($telemetryPath, $false, $utf8)
    $telemetry.AutoFlush = $true
    $job = [QrtRunGuard]::Create()
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $reason = 'completed'
    try {
        $argumentLine = @($spec.arguments | ForEach-Object { Quote-Argument ([string]$_) }) -join ' '
        $process = Start-Process -FilePath $spec.executable -ArgumentList $argumentLine `
            -WorkingDirectory $spec.working_directory -PassThru -NoNewWindow `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        try { [QrtRunGuard]::Assign($job, $process.Handle) }
        catch { if (-not $process.HasExited) { $process.Kill() }; throw }
        Write-Record 'launch.json' ([ordered]@{
            pid = $process.Id; utc = [DateTime]::UtcNow.ToString('o')
            timeout_seconds = $TimeoutSeconds; job_kill_on_close = $true
        })
        while (-not $process.WaitForExit(250)) {
            $memory = [QrtRunGuard]::Snapshot()
            $process.Refresh()
            $telemetry.WriteLine(([ordered]@{
                elapsed_ms = [Math]::Round($watch.Elapsed.TotalMilliseconds)
                available_physical = $memory.AvailablePhysical
                available_commit = $memory.AvailableCommit
                process_private_bytes = $process.PrivateMemorySize64
            } | ConvertTo-Json -Compress))
            if ($watch.Elapsed.TotalSeconds -ge $TimeoutSeconds) { $reason = 'timeout' }
            elseif ($memory.AvailablePhysical -lt 8GB) { $reason = 'host_memory_reserve' }
            elseif ($memory.AvailableCommit -lt 20GB) { $reason = 'commit_reserve' }
            elseif ((Get-Item $stdoutPath).Length + (Get-Item $stderrPath).Length -gt 64MB) {
                $reason = 'log_size_limit'
            }
            if ($reason -ne 'completed') {
                if (-not [QrtRunGuard]::TerminateJobObject($job, 124)) { throw 'Unable to terminate owned job.' }
                break
            }
        }
        if (-not $process.WaitForExit(5000)) { throw 'Owned process did not exit within the cleanup bound.' }
    } finally { $telemetry.Dispose() }
    $watch.Stop()
    $process.Refresh()
    # Close even after normal completion to remove any inherited descendants.
    $null = [QrtRunGuard]::CloseHandle($job)
    $job = [IntPtr]::Zero
    $summaries = @(Get-Content -LiteralPath $stdoutPath | Where-Object { $_ -match '^\{"type":"summary"' })
    $summary = if ($summaries.Count -gt 0) { $summaries[-1] | ConvertFrom-Json } else { $null }
    $afterProcesses = @(Get-EngineProcesses)
    $afterBoot = (Get-CimInstance Win32_OperatingSystem -OperationTimeoutSec 5).LastBootUpTime.ToString('o')
    $afterMemory = [QrtRunGuard]::Snapshot()
    $afterGpus = @(Get-CimInstance Win32_VideoController -OperationTimeoutSec 5 |
        Where-Object { $_.Name -match 'AMD|Radeon' } | Select-Object Name, Status, DriverVersion)
    $hostChecks = [ordered]@{
        same_boot = $boot -eq $afterBoot
        no_engine_process = $afterProcesses.Count -eq 0
        host_memory_reserve = $afterMemory.AvailablePhysical -ge 8GB
        commit_reserve = $afterMemory.AvailableCommit -ge 20GB
        amd_gpu_ok = $afterGpus.Count -gt 0 -and @($afterGpus | Where-Object { $_.Status -ne 'OK' }).Count -eq 0
    }
    $cliPass = $reason -eq 'completed' -and $process.ExitCode -eq 0 -and
        $null -ne $summary -and $summary.status -eq 'pass'
    $record = [ordered]@{
        schema_version = 1; host = [Environment]::MachineName
        command_file = $PSCommandPath; spec = $spec
        preflight = $preflight; reason = $reason; exit_code = $process.ExitCode
        wall_ms = [Math]::Round($watch.Elapsed.TotalMilliseconds, 3)
        summary_present = $null -ne $summary; cli_pass = $cliPass; summary = $summary
        after_memory = $afterMemory; after_gpus = $afterGpus; after_processes = $afterProcesses
        host_checks = $hostChecks; host_checks_pass = -not ($hostChecks.Values -contains $false)
        numerical_acceptance_requires_external_oracle = $true
    }
    Write-Record 'run-record.json' $record
    $record | ConvertTo-Json -Depth 16 -Compress
    $resultCode = if ($cliPass -and $record.host_checks_pass) { 0 } else { 20 }
} catch {
    Write-Record 'supervisor-error.json' ([ordered]@{
        utc = [DateTime]::UtcNow.ToString('o'); error = $_.Exception.Message
        launched = $null -ne $process; numerical_correctness_claimed = $false
    })
    Write-Warning $_.Exception.Message
} finally {
    if ($job -ne [IntPtr]::Zero) { $null = [QrtRunGuard]::CloseHandle($job) }
    if ($null -ne $process) { $process.Dispose() }
    if ($locked) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
exit $resultCode
