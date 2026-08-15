param(
    [Parameter(Mandatory = $true)][string]$RuntimeDir,
    [Parameter(Mandatory = $false)][string]$OutDir = "",
    [Parameter(Mandatory = $false)][string]$Version = "1.0.1"
)

$ErrorActionPreference = "Stop"

function Get-PortableRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    $separator = [IO.Path]::DirectorySeparatorChar
    $baseFull = [IO.Path]::GetFullPath($BasePath).TrimEnd(
        [char[]]@('\', '/')
    ) + $separator
    $targetFull = [IO.Path]::GetFullPath($TargetPath)
    if (-not $targetFull.StartsWith(
            $baseFull, [StringComparison]::OrdinalIgnoreCase
        )) {
        throw "path is outside the expected base directory: $targetFull"
    }
    return $targetFull.Substring($baseFull.Length).Replace('\', '/')
}

$repo = Split-Path -Parent $PSScriptRoot
$RuntimeDir = [IO.Path]::GetFullPath($RuntimeDir)
if (-not (Test-Path -LiteralPath $RuntimeDir -PathType Container)) {
    throw "runtime directory not found: $RuntimeDir"
}
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repo "dist"
} elseif (-not [IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repo $OutDir
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$manifestPath = Join-Path $RuntimeDir "runtime-manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "runtime manifest not found: $manifestPath"
}
$runtimeManifest = Get-Content -Raw -LiteralPath $manifestPath |
    ConvertFrom-Json
if ($runtimeManifest.dirty_tree -ne $false) {
    throw "refusing to package a runtime built from a dirty worktree"
}
if ($runtimeManifest.offload_arch -ne "gfx1151") {
    throw "runtime manifest does not target gfx1151"
}
if ($runtimeManifest.composable_kernel_tree_sha256 -notmatch '^[0-9a-f]{64}$' -or
    [int]$runtimeManifest.composable_kernel_source_file_count -lt 1) {
    throw "runtime manifest has no reproducible Composable Kernel source fingerprint"
}

$baseName = "AIMA-AMD395-Qwen36-35B-Windows-Engine-v$Version"
$stage = Join-Path $OutDir $baseName
$archive = Join-Path $OutDir "$baseName.zip"
$archiveHash = Join-Path $OutDir "$baseName.zip.sha256"
foreach ($destination in @($stage, $archive, $archiveHash)) {
    if (Test-Path -LiteralPath $destination) {
        throw "refusing to overwrite existing package output: $destination"
    }
}

$requiredFiles = @(
    "engine\qrt.exe",
    "whole-provider\qrt_qwen36_whole_provider.dll",
    "q1024-moe\qrt_triton_moe_q1024_exact_provider_slots64.dll",
    "q8192-moe\qrt_triton_moe_q8192_provider.dll",
    "ck-fmha\qrt_ck_fmha_continuous_long.dll",
    "aiter-gdn\qrt_aiter_fused_gdn_q8192_provider.dll",
    "runtime.env",
    "runtime-manifest.json"
)
foreach ($relative in $requiredFiles) {
    $path = Join-Path $RuntimeDir $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "runtime package input is missing: $path"
    }
}
foreach ($relative in @(
        "q1024-moe\moe-kernels", "q8192-moe", "aiter-gdn", "aot\gfx1151"
    )) {
    $path = Join-Path $RuntimeDir $relative
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "runtime kernel directory is missing: $path"
    }
}

$runtimeArtifactRecords = @($runtimeManifest.artifacts)
if ($runtimeArtifactRecords.Count -eq 0) {
    throw "runtime manifest has no artifacts"
}
$runtimeArtifactPaths = @()
$validatedRuntimeArtifacts = foreach ($record in $runtimeArtifactRecords) {
    $relative = ([string]$record.path).Replace('\', '/')
    $parts = $relative.Split('/')
    if ([string]::IsNullOrWhiteSpace($relative) -or
        [IO.Path]::IsPathRooted($relative) -or $parts -contains "..") {
        throw "runtime manifest contains unsafe artifact path: $relative"
    }
    $source = [IO.Path]::GetFullPath((Join-Path $RuntimeDir $relative))
    $runtimePrefix = $RuntimeDir.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $source.StartsWith($runtimePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "runtime manifest artifact escapes the runtime directory: $relative"
    }
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "runtime manifest artifact is missing: $source"
    }
    $item = Get-Item -LiteralPath $source
    $sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
    if ($item.Length -ne [long]$record.bytes -or $sha256 -ne [string]$record.sha256) {
        throw "runtime manifest artifact hash/size mismatch: $relative"
    }
    $runtimeArtifactPaths += $relative
    [ordered]@{ relative = $relative; source = $source }
}
foreach ($relative in $requiredFiles | Where-Object { $_ -ne "runtime-manifest.json" }) {
    $normalized = $relative.Replace('\', '/')
    if ($runtimeArtifactPaths -notcontains $normalized) {
        throw "required runtime file is absent from the manifest: $normalized"
    }
}
$kernelMinimums = @(
    [ordered]@{ prefix = "q1024-moe/moe-kernels/"; count = 1 },
    [ordered]@{ prefix = "q8192-moe/"; count = 6 },
    [ordered]@{ prefix = "aiter-gdn/"; count = 4 },
    [ordered]@{ prefix = "aot/gfx1151/"; count = 20 }
)
foreach ($minimum in $kernelMinimums) {
    $count = @(
        $runtimeArtifactPaths | Where-Object {
            $_.StartsWith($minimum.prefix, [StringComparison]::OrdinalIgnoreCase) -and
            $_.EndsWith(".hsaco", [StringComparison]::OrdinalIgnoreCase)
        }
    ).Count
    if ($count -lt $minimum.count) {
        throw "runtime manifest has only $count HSACO files under $($minimum.prefix)"
    }
}

New-Item -ItemType Directory -Path $stage | Out-Null
foreach ($artifact in $validatedRuntimeArtifacts) {
    $destination = Join-Path $stage $artifact.relative
    $destinationDirectory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
    Copy-Item -LiteralPath $artifact.source -Destination $destination
}
Copy-Item -LiteralPath $manifestPath -Destination $stage
foreach ($name in @(
        "README.md", "README.zh-CN.md", "LICENSE", "NOTICE",
        "THIRD_PARTY_NOTICES.md", "SECURITY.md"
    )) {
    Copy-Item -LiteralPath (Join-Path $repo $name) -Destination $stage
}
Copy-Item -LiteralPath (Join-Path $repo "third_party") `
    -Destination (Join-Path $stage "third_party") -Recurse

$fileRecords = foreach ($file in Get-ChildItem -LiteralPath $stage `
        -Recurse -File | Sort-Object FullName) {
    $relative = Get-PortableRelativePath -BasePath $stage `
        -TargetPath $file.FullName
    [ordered]@{
        path = $relative
        bytes = $file.Length
        sha256 = (Get-FileHash -Algorithm SHA256 `
            -LiteralPath $file.FullName).Hash.ToLowerInvariant()
    }
}
$releaseManifest = [ordered]@{
    schema_version = 1
    project = "AIMA-AMD395-Qwen36-35B-Windows-Engine"
    version = $Version
    target = "windows-x86_64-gfx1151"
    source_commit = $runtimeManifest.repo_commit
    files = @($fileRecords)
}
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false
[IO.File]::WriteAllText(
    (Join-Path $stage "FILE-SHA256SUMS.json"),
    ($releaseManifest | ConvertTo-Json -Depth 6) + "`n",
    $utf8
)

Add-Type -AssemblyName System.IO.Compression
$archiveStream = [IO.File]::Open($archive, [IO.FileMode]::CreateNew)
try {
    $zip = New-Object -TypeName System.IO.Compression.ZipArchive `
        -ArgumentList @(
            $archiveStream,
            [IO.Compression.ZipArchiveMode]::Create,
            $false
        )
    try {
        foreach ($file in Get-ChildItem -LiteralPath $stage `
                -Recurse -File | Sort-Object FullName) {
            $relative = Get-PortableRelativePath -BasePath $stage `
                -TargetPath $file.FullName
            $entryName = "$baseName/$relative"
            $entry = $zip.CreateEntry(
                $entryName, [IO.Compression.CompressionLevel]::Optimal
            )
            $entryStream = $entry.Open()
            $sourceStream = [IO.File]::OpenRead($file.FullName)
            try {
                $sourceStream.CopyTo($entryStream)
            } finally {
                $sourceStream.Dispose()
                $entryStream.Dispose()
            }
        }
    } finally {
        $zip.Dispose()
    }
} finally {
    $archiveStream.Dispose()
}
$sha256 = (Get-FileHash -Algorithm SHA256 `
    -LiteralPath $archive).Hash.ToLowerInvariant()
[IO.File]::WriteAllText(
    $archiveHash,
    "$sha256  $baseName.zip`n",
    [Text.Encoding]::ASCII
)

[ordered]@{
    archive = $archive
    bytes = (Get-Item -LiteralPath $archive).Length
    sha256 = $sha256
    checksum_file = $archiveHash
    source_commit = $runtimeManifest.repo_commit
} | ConvertTo-Json -Depth 4
