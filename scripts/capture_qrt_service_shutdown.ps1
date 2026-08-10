[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$StateFile,

    [Parameter(Mandatory = $true)]
    [string]$ReadyEvidenceFile,

    [Parameter(Mandatory = $true)]
    [string]$OutputFile,

    [ValidateRange(1, 300)]
    [int]$WaitSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not [Environment]::MachineName.Equals(
        "baiying",
        [StringComparison]::OrdinalIgnoreCase
    )) {
    throw "the qrt shutdown audit must run locally on baiying"
}

function Read-JsonObject {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $bytes = [IO.File]::ReadAllBytes($resolved)
    $value = [Text.Encoding]::UTF8.GetString($bytes) | ConvertFrom-Json
    if ($null -eq $value) {
        throw "$Label is empty: $resolved"
    }
    return [ordered]@{
        path = $resolved
        bytes = $bytes
        value = $value
        sha256 = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

function Require-Equal {
    param(
        [Parameter(Mandatory = $true)]$Actual,
        [Parameter(Mandatory = $true)]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if ([string]$Actual -cne [string]$Expected) {
        throw "$Label differs: actual=$Actual expected=$Expected"
    }
}

$stateSnapshot = Read-JsonObject -Path $StateFile -Label "service state"
$readySnapshot = Read-JsonObject -Path $ReadyEvidenceFile -Label "ready evidence"
$state = $stateSnapshot.value
$ready = $readySnapshot.value

if ([string]$ready.record_type -cne "qrt_live_service_runtime_evidence") {
    throw "ready evidence has the wrong record type: $($ready.record_type)"
}
if (-not [string]::Equals(
        [string]$ready.host,
        [Environment]::MachineName,
        [StringComparison]::OrdinalIgnoreCase
    )) {
    throw "ready evidence belongs to another host: $($ready.host)"
}
if ([string]$ready.execution -cne "local_windows_process") {
    throw "ready evidence did not use a local Windows process"
}
$readyState = $ready.service_state.payload
$readyProcess = $ready.service_process
if ($null -eq $readyState -or $null -eq $readyProcess) {
    throw "ready evidence is missing service state or process identity"
}
if ([string]$readyState.status -cne "ready") {
    throw "ready evidence did not capture a ready service"
}

$serviceProcessId = [uint32]$readyProcess.pid
Require-Equal $readyState.pid $serviceProcessId "ready state/process PID"
Require-Equal $state.pid $serviceProcessId "stopped state PID"
if ([string]$readyProcess.name -ine "qrt.exe") {
    throw "ready process was not qrt.exe: $($readyProcess.name)"
}
if (-not [string]::Equals(
        [string]$ready.service_state.path,
        [string]$stateSnapshot.path,
        [StringComparison]::OrdinalIgnoreCase
    )) {
    throw "ready evidence and shutdown audit use different state files"
}
$readyExecutable = (Resolve-Path -LiteralPath $readyProcess.executable_path).Path
$readyExecutableSha256 = (
    Get-FileHash -LiteralPath $readyExecutable -Algorithm SHA256
).Hash.ToLowerInvariant()
Require-Equal `
    $readyExecutableSha256 `
    $readyProcess.executable_sha256 `
    "ready executable SHA-256"
$readyListeners = @($ready.listener)
$readyAddress = [Uri]("http://" + [string]$readyState.address)
$matchingReadyListeners = @(
    $readyListeners | Where-Object {
        [uint32]$_.owning_process -eq $serviceProcessId -and
        [uint16]$_.local_port -eq $readyAddress.Port
    }
)
if ($matchingReadyListeners.Count -lt 1) {
    throw "ready evidence did not bind PID $serviceProcessId to port $($readyAddress.Port)"
}
foreach ($field in @(
        "address",
        "model",
        "model_path",
        "provider_dll",
        "max_model_len",
        "repo_commit",
        "host",
        "started_unix_seconds",
        "ready_unix_seconds"
    )) {
    Require-Equal $state.$field $readyState.$field "service field $field"
}
if ([string]$state.status -cne "stopped") {
    throw "service state is not stopped: $($state.status)"
}
if ($null -eq $state.stopped_unix_seconds) {
    throw "service state has no stopped timestamp"
}
if ([uint64]$state.stopped_unix_seconds -lt [uint64]$state.ready_unix_seconds) {
    throw "service stopped timestamp precedes readiness"
}

$address = [Uri]("http://" + [string]$state.address)
$deadline = [DateTime]::UtcNow.AddSeconds($WaitSeconds)
do {
    $processMatches = @(
        Get-CimInstance Win32_Process -Filter "ProcessId=$serviceProcessId"
    )
    $listeners = @(
        Get-NetTCPConnection `
            -State Listen `
            -LocalPort $address.Port `
            -ErrorAction SilentlyContinue
    )
    if ($processMatches.Count -eq 0 -and $listeners.Count -eq 0) {
        break
    }
    if ([DateTime]::UtcNow -ge $deadline) {
        $liveProcesses = $processMatches |
            Select-Object ProcessId, Name, ExecutablePath, CreationDate
        $liveListeners = $listeners |
            Select-Object LocalAddress, LocalPort, OwningProcess, State
        throw (
            "service shutdown was incomplete after $WaitSeconds seconds: " +
            "processes=$($liveProcesses | ConvertTo-Json -Compress) " +
            "listeners=$($liveListeners | ConvertTo-Json -Compress)"
        )
    }
    Start-Sleep -Milliseconds 200
} while ($true)

$payload = [ordered]@{
    schema_version = 1
    record_type = "qrt_stopped_service_runtime_evidence"
    captured_utc = [DateTime]::UtcNow.ToString("o")
    host = [Environment]::MachineName
    execution = "local_windows_process"
    command_file = $PSCommandPath
    command_file_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    wait_seconds = $WaitSeconds
    ready_evidence = [ordered]@{
        path = $readySnapshot.path
        bytes = [uint64]$readySnapshot.bytes.Length
        sha256 = $readySnapshot.sha256
        service_process = $readyProcess
    }
    stopped_state = [ordered]@{
        path = $stateSnapshot.path
        bytes = [uint64]$stateSnapshot.bytes.Length
        sha256 = $stateSnapshot.sha256
        payload = $state
    }
    assertions = [ordered]@{
        same_service_instance = $true
        terminal_state = "stopped"
        stopped_timestamp_present = $true
        process_absent = $true
        listener_absent = $true
        pid = $serviceProcessId
        address = [string]$state.address
        model = [string]$state.model
    }
}

$outputParent = Split-Path -Parent $OutputFile
if ([string]::IsNullOrWhiteSpace($outputParent)) {
    $outputParent = (Get-Location).Path
}
[IO.Directory]::CreateDirectory($outputParent) | Out-Null
$resolvedOutputParent = (Resolve-Path -LiteralPath $outputParent).Path
$outputLeaf = Split-Path -Leaf $OutputFile
if ([string]::IsNullOrWhiteSpace($outputLeaf)) {
    throw "output file has no leaf name"
}
$resolvedOutput = Join-Path $resolvedOutputParent $outputLeaf
if (Test-Path -LiteralPath $resolvedOutput) {
    throw "refusing to overwrite shutdown evidence: $resolvedOutput"
}
$json = $payload | ConvertTo-Json -Depth 20
$temporaryOutput = "$resolvedOutput.tmp-$([Guid]::NewGuid().ToString('N'))"
try {
    [IO.File]::WriteAllText(
        $temporaryOutput,
        $json + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporaryOutput -Destination $resolvedOutput
}
finally {
    if (Test-Path -LiteralPath $temporaryOutput -PathType Leaf) {
        Remove-Item -LiteralPath $temporaryOutput
    }
}

$json
