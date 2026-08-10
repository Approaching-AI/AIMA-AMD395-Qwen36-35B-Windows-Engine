[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$StateFile,

    [Parameter(Mandatory = $true)]
    [string]$OutputFile,

    [string[]]$RequireCommandFragment = @(),

    [uint32[]]$RelatedProcessId = @(),

    [string[]]$ArtifactPath = @(),

    [string[]]$LoadedModuleRoot = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-StringSha256 {
    param([Parameter(Mandatory = $true)][string]$Value)

    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha256.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
    }
}

function Protect-CommandLine {
    param([Parameter(Mandatory = $true)][string]$CommandLine)

    return [regex]::Replace(
        $CommandLine,
        '(?i)(--api-key(?:=|\s+))(?:(?:"[^"]*")|(?:''[^'']*'')|\S+)',
        '$1[REDACTED]'
    )
}

function Get-ProcessEvidence {
    param([Parameter(Mandatory = $true)][uint32]$ProcessIdentifier)

    $matches = @(
        Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessIdentifier"
    )
    if ($matches.Count -ne 1) {
        throw "expected one live process for PID $ProcessIdentifier, found $($matches.Count)"
    }
    $item = $matches[0]
    if ([string]::IsNullOrWhiteSpace([string]$item.CommandLine)) {
        throw "process $ProcessIdentifier has no readable command line"
    }
    if ([string]::IsNullOrWhiteSpace([string]$item.ExecutablePath)) {
        throw "process $ProcessIdentifier has no readable executable path"
    }
    $resolvedExecutable = (Resolve-Path -LiteralPath $item.ExecutablePath).Path
    $rawCommandLine = [string]$item.CommandLine
    $protectedCommandLine = Protect-CommandLine $rawCommandLine
    return [ordered]@{
        pid = [uint32]$item.ProcessId
        name = [string]$item.Name
        executable_path = $resolvedExecutable
        executable_sha256 = (Get-FileHash -LiteralPath $resolvedExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
        creation_time_utc = $item.CreationDate.ToUniversalTime().ToString("o")
        command_line = $protectedCommandLine
        command_line_sha256 = Get-StringSha256 $protectedCommandLine
        api_key_redacted = $protectedCommandLine -cne $rawCommandLine
    }
}

$resolvedState = (Resolve-Path -LiteralPath $StateFile).Path
$stateBytes = [IO.File]::ReadAllBytes($resolvedState)
$state = [Text.Encoding]::UTF8.GetString($stateBytes) | ConvertFrom-Json
if ($state.status -ne "ready") {
    throw "service state is not ready: $($state.status)"
}
$serviceProcessId = [uint32]$state.pid
$serviceProcess = Get-ProcessEvidence -ProcessIdentifier $serviceProcessId
if ($serviceProcess.name -ine "qrt.exe") {
    throw "state PID $serviceProcessId is $($serviceProcess.name), not qrt.exe"
}

foreach ($fragment in $RequireCommandFragment) {
    if ([string]::IsNullOrWhiteSpace($fragment)) {
        throw "required command fragment cannot be empty"
    }
    if ($serviceProcess.command_line.IndexOf(
            $fragment,
            [StringComparison]::OrdinalIgnoreCase
        ) -lt 0) {
        throw "service command line is missing required fragment: $fragment"
    }
}

$address = [Uri]("http://" + [string]$state.address)
$listeners = @(
    Get-NetTCPConnection -State Listen -LocalPort $address.Port |
        Where-Object { [uint32]$_.OwningProcess -eq $serviceProcessId }
)
if ($listeners.Count -lt 1) {
    throw "PID $serviceProcessId does not own a listener on port $($address.Port)"
}

$relatedProcesses = @(
    foreach ($relatedId in $RelatedProcessId) {
        if ($relatedId -eq $serviceProcessId) {
            throw "related PID must differ from the service PID"
        }
        Get-ProcessEvidence -ProcessIdentifier $relatedId
    }
)

$resolvedModuleRoots = @(
    foreach ($root in $LoadedModuleRoot) {
        $resolvedRoot = (Resolve-Path -LiteralPath $root).Path
        if (-not (Get-Item -LiteralPath $resolvedRoot).PSIsContainer) {
            throw "loaded-module root must be a directory: $resolvedRoot"
        }
        $resolvedRoot.TrimEnd('\')
    }
)
$loadedModules = @()
if ($resolvedModuleRoots.Count -ne 0) {
    $loadedModules = @(
        Get-Process -Id $serviceProcessId -Module |
            Where-Object {
                $modulePath = [string]$_.FileName
                foreach ($root in $resolvedModuleRoots) {
                    if ($modulePath.Equals($root, [StringComparison]::OrdinalIgnoreCase) -or
                        $modulePath.StartsWith(
                            $root + '\',
                            [StringComparison]::OrdinalIgnoreCase
                        )) {
                        return $true
                    }
                }
                return $false
            } |
            Sort-Object FileName -Unique |
            ForEach-Object {
                $moduleItem = Get-Item -LiteralPath $_.FileName
                [ordered]@{
                    name = [string]$_.ModuleName
                    path = $moduleItem.FullName
                    bytes = [uint64]$moduleItem.Length
                    sha256 = (Get-FileHash -LiteralPath $moduleItem.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                }
            }
    )
    if ($loadedModules.Count -eq 0) {
        throw "no loaded modules matched the requested roots"
    }
}

$artifacts = @(
    foreach ($path in $ArtifactPath) {
        $resolvedArtifact = (Resolve-Path -LiteralPath $path).Path
        $artifactItem = Get-Item -LiteralPath $resolvedArtifact
        if (-not $artifactItem.PSIsContainer) {
            [ordered]@{
                path = $resolvedArtifact
                bytes = [uint64]$artifactItem.Length
                sha256 = (Get-FileHash -LiteralPath $resolvedArtifact -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
        else {
            throw "artifact path must be a file: $resolvedArtifact"
        }
    }
)

$payload = [ordered]@{
    schema_version = 1
    record_type = "qrt_live_service_runtime_evidence"
    captured_utc = [DateTime]::UtcNow.ToString("o")
    host = [Environment]::MachineName
    execution = "local_windows_process"
    command_file = $PSCommandPath
    command_file_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    service_state = [ordered]@{
        path = $resolvedState
        sha256 = (Get-FileHash -LiteralPath $resolvedState -Algorithm SHA256).Hash.ToLowerInvariant()
        payload = $state
    }
    service_process = $serviceProcess
    required_command_fragments = @($RequireCommandFragment)
    listener = @(
        foreach ($listener in $listeners) {
            [ordered]@{
                local_address = [string]$listener.LocalAddress
                local_port = [uint16]$listener.LocalPort
                owning_process = [uint32]$listener.OwningProcess
            }
        }
    )
    related_processes = $relatedProcesses
    loaded_module_roots = $resolvedModuleRoots
    loaded_modules = $loadedModules
    artifacts = $artifacts
}

$outputParent = Split-Path -Parent $OutputFile
if ([string]::IsNullOrWhiteSpace($outputParent)) {
    $outputParent = (Get-Location).Path
}
[IO.Directory]::CreateDirectory($outputParent) | Out-Null
$resolvedOutputParent = (Resolve-Path -LiteralPath $outputParent).Path
$outputLeaf = Split-Path -Leaf $OutputFile
$resolvedOutput = Join-Path $resolvedOutputParent $outputLeaf
if (Test-Path -LiteralPath $resolvedOutput) {
    throw "refusing to overwrite runtime evidence: $resolvedOutput"
}
$json = $payload | ConvertTo-Json -Depth 16
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
    if (Test-Path -LiteralPath $temporaryOutput) {
        Remove-Item -LiteralPath $temporaryOutput
    }
}

$json
