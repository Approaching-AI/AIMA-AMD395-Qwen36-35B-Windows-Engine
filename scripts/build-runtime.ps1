param(
    [Parameter(Mandatory = $false)][string]$OutDir = "",
    [Parameter(Mandatory = $true)][string]$CkRoot,
    [Parameter(Mandatory = $false)]
        [string]$RocmRoot = "C:\Program Files\AMD\ROCm\7.1",
    [Parameter(Mandatory = $false)][string]$WslDistribution = "Ubuntu-24.04",
    [Parameter(Mandatory = $false)]
        [string]$TritonPython = "/opt/qwen36-vllm/bin/python",
    [Parameter(Mandatory = $false)]
        [ValidatePattern('^gfx[0-9a-f]+$')]
        [string]$OffloadArch = "gfx1151"
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
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repo "build\runtime"
} elseif (-not [IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repo $OutDir
}
$OutDir = [IO.Path]::GetFullPath($OutDir)
$CkRoot = [IO.Path]::GetFullPath($CkRoot)
if (-not (Test-Path -LiteralPath $CkRoot -PathType Container)) {
    throw "Composable Kernel checkout not found: $CkRoot"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$wholeDir = Join-Path $OutDir "whole-provider"
$moeDir = Join-Path $OutDir "q1024-moe"
$q8192MoeDir = Join-Path $OutDir "q8192-moe"
$ckDir = Join-Path $OutDir "ck-fmha"
$gdnDir = Join-Path $OutDir "aiter-gdn"
$engineDir = Join-Path $OutDir "engine"
$baseAotDir = Join-Path $OutDir ("aot\" + $OffloadArch)
New-Item -ItemType Directory -Force -Path $ckDir | Out-Null
New-Item -ItemType Directory -Force -Path $baseAotDir | Out-Null

$runtimeProfileSource = Join-Path $repo "engine\runtime.env"
if (-not (Test-Path -LiteralPath $runtimeProfileSource -PathType Leaf)) {
    throw "runtime profile not found: $runtimeProfileSource"
}
$acceptedQ8192AotDir = Join-Path $repo "native\aot\gfx1151"
if (-not (Test-Path -LiteralPath $acceptedQ8192AotDir -PathType Container)) {
    throw "accepted q8192 AOT directory not found: $acceptedQ8192AotDir"
}
$baseAotSourceFiles = @(
    Get-ChildItem -LiteralPath $acceptedQ8192AotDir -File |
        Where-Object { $_.Extension -in @(".hsaco", ".json") } |
        Sort-Object Name
)
if (@($baseAotSourceFiles | Where-Object { $_.Extension -eq ".hsaco" }).Count -lt 20) {
    throw "accepted AOT inventory is incomplete: $acceptedQ8192AotDir"
}
foreach ($file in $baseAotSourceFiles) {
    Copy-Item -LiteralPath $file.FullName -Destination $baseAotDir -Force
}

& (Join-Path $PSScriptRoot "baiying_build_whole_provider.ps1") `
    -OutDir $wholeDir -RocmRoot $RocmRoot -OffloadArch $OffloadArch
if ($LASTEXITCODE -ne 0) { throw "whole-provider build failed" }

& (Join-Path $PSScriptRoot "baiying_build_triton_moe_q1024_exact.ps1") `
    -OutDir $moeDir -Variant exact -RocmRoot $RocmRoot `
    -OffloadArch $OffloadArch -WslDistribution $WslDistribution `
    -TritonPython $TritonPython
if ($LASTEXITCODE -ne 0) { throw "q1024 selected-MoE build failed" }

& (Join-Path $PSScriptRoot "baiying_build_triton_moe_q8192.ps1") `
    -BuildDir $q8192MoeDir -OutDir $q8192MoeDir `
    -HipccPath (Join-Path $RocmRoot "bin\hipcc.exe") `
    -OffloadArch $OffloadArch -WslDistribution $WslDistribution `
    -TritonPython $TritonPython -Repetitions 1 `
    -NativeWmmaGate 1 -NativeWmmaDown 1 -TransposedRouter 1 `
    -RouterThreads 256 -RouterTokenTile 8 `
    -FullV3FusedCombine 1 -FusedCombineWidth 4 `
    -FullV3EventSlots 16 -ReuseAotDir $acceptedQ8192AotDir `
    -RequireExpectedFullProviderHash 1
if ($LASTEXITCODE -ne 0) { throw "q8192 selected-MoE build failed" }

& (Join-Path $PSScriptRoot "baiying_build_ck_fmha_q8192.ps1") `
    -CkRoot $CkRoot `
    -OutPath (Join-Path $ckDir "qrt_ck_fmha_continuous_long.dll") `
    -OffloadArch $OffloadArch `
    -HipccPath (Join-Path $RocmRoot "bin\hipcc.exe")
if ($LASTEXITCODE -ne 0) { throw "CK FMHA provider build failed" }

& (Join-Path $PSScriptRoot "baiying_build_aiter_fused_gdn_q8192.ps1") `
    -BuildDir $gdnDir -OffloadArch $OffloadArch `
    -HipccPath (Join-Path $RocmRoot "bin\hipcc.exe") `
    -WslDistribution $WslDistribution -TritonPython $TritonPython
if ($LASTEXITCODE -ne 0) { throw "AITER GDN provider build failed" }

& (Join-Path $PSScriptRoot "baiying_build_qrt_server.ps1") `
    -OutDir $engineDir
if ($LASTEXITCODE -ne 0) { throw "qrt engine build failed" }

$runtimeProfile = Join-Path $OutDir "runtime.env"
Copy-Item -LiteralPath $runtimeProfileSource `
    -Destination $runtimeProfile -Force

$runtimeFiles = @(
    (Join-Path $engineDir "qrt.exe"),
    (Join-Path $wholeDir "qrt_qwen36_whole_provider.dll"),
    (Join-Path $moeDir "qrt_triton_moe_q1024_exact_provider_slots64.dll"),
    (Join-Path $q8192MoeDir "qrt_triton_moe_q8192_provider.dll"),
    (Join-Path $ckDir "qrt_ck_fmha_continuous_long.dll"),
    (Join-Path $gdnDir "qrt_aiter_fused_gdn_q8192_provider.dll"),
    $runtimeProfile
)
foreach ($path in $runtimeFiles) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "runtime build output is missing: $path"
    }
}
$runtimeKernelFiles = @(
    Get-ChildItem -LiteralPath (Join-Path $moeDir "moe-kernels") -File |
        Where-Object { $_.Extension -in @(".hsaco", ".json") } |
        Select-Object -ExpandProperty FullName
    Get-ChildItem -LiteralPath $q8192MoeDir -File |
        Where-Object {
            $_.Extension -eq ".hsaco" -or $_.Name -eq "metadata.json"
        } |
        Select-Object -ExpandProperty FullName
    Get-ChildItem -LiteralPath $gdnDir -File |
        Where-Object { $_.Extension -in @(".hsaco", ".json") } |
        Select-Object -ExpandProperty FullName
    Get-ChildItem -LiteralPath $baseAotDir -File |
        Where-Object { $_.Extension -in @(".hsaco", ".json") } |
        Select-Object -ExpandProperty FullName
)
$runtimeArtifactFiles = @($runtimeFiles + $runtimeKernelFiles | Sort-Object -Unique)

$artifacts = foreach ($path in $runtimeArtifactFiles) {
    $item = Get-Item -LiteralPath $path
    [ordered]@{
        path = Get-PortableRelativePath -BasePath $OutDir `
            -TargetPath $item.FullName
        bytes = $item.Length
        sha256 = (Get-FileHash -Algorithm SHA256 `
            -LiteralPath $item.FullName).Hash.ToLowerInvariant()
    }
}
$ckCommitOutput = & git -C $CkRoot rev-parse HEAD 2>$null
$ckCommit = if ($LASTEXITCODE -eq 0) { ($ckCommitOutput | Out-String).Trim() } else { $null }
$ckDirty = if ($null -ne $ckCommit) {
    @(& git -C $CkRoot status --porcelain).Count -ne 0
} else {
    $null
}
$ckSourceFiles = @(Get-ChildItem -LiteralPath $CkRoot -Recurse -File | Sort-Object FullName)
$ckTreeLines = foreach ($file in $ckSourceFiles) {
    $relative = Get-PortableRelativePath -BasePath $CkRoot `
        -TargetPath $file.FullName
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
    "$relative`t$hash`n"
}
$ckTreeBytes = [Text.Encoding]::UTF8.GetBytes(($ckTreeLines -join ""))
$ckHasher = [Security.Cryptography.SHA256]::Create()
try {
    $ckTreeDigest = $ckHasher.ComputeHash($ckTreeBytes)
} finally {
    $ckHasher.Dispose()
}
$ckTreeSha256 = ($ckTreeDigest | ForEach-Object { $_.ToString("x2") }) -join ""
$record = [ordered]@{
    schema_version = 1
    host = [Environment]::MachineName
    repo_commit = (& git -C $repo rev-parse HEAD).Trim()
    dirty_tree = @(& git -C $repo status --porcelain).Count -ne 0
    offload_arch = $OffloadArch
    rocm_sdk = Split-Path -Leaf $RocmRoot
    composable_kernel_commit = $ckCommit
    composable_kernel_dirty_tree = $ckDirty
    composable_kernel_source_file_count = $ckSourceFiles.Count
    composable_kernel_tree_sha256 = $ckTreeSha256
    layout = [ordered]@{
        engine = "engine"
        whole_provider = "whole-provider"
        arbitrary_moe_provider = "q1024-moe"
        arbitrary_moe_kernel_dir = "q1024-moe/moe-kernels"
        q8192_moe_provider = "q8192-moe/qrt_triton_moe_q8192_provider.dll"
        q8192_moe_kernel_dir = "q8192-moe"
        ck_fmha_provider = "ck-fmha"
        aiter_gdn_provider = "aiter-gdn"
        base_aot = "aot/$OffloadArch"
        runtime_profile = "runtime.env"
    }
    artifacts = @($artifacts)
}
$manifestPath = Join-Path $OutDir "runtime-manifest.json"
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false
[IO.File]::WriteAllText(
    $manifestPath,
    ($record | ConvertTo-Json -Depth 6) + [Environment]::NewLine,
    $utf8
)
Write-Output $manifestPath
