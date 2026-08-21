param(
    [Parameter(Mandatory = $true)][string]$KernelDir,
    [Parameter(Mandatory = $true)][string]$OutDir
)

$ErrorActionPreference = "Stop"
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false
$KernelDir = [IO.Path]::GetFullPath($KernelDir)
$OutDir = [IO.Path]::GetFullPath($OutDir)
if (-not (Test-Path -LiteralPath $KernelDir -PathType Container)) {
    throw "kernel directory not found: $KernelDir"
}
if (Test-Path -LiteralPath $OutDir) {
    throw "refusing to overwrite verification output: $OutDir"
}
New-Item -ItemType Directory -Path $OutDir | Out-Null

$expectedMetadataSha256 =
    "41e77d3afdecba63ee15c63e5f585f44ae29a6220e296e2752dcc18cfd24dc14"
$expectedArtifacts = [ordered]@{
    "q8192_selected_moe_route_count.hsaco" =
        "48cc7d3660fa6051f085996e827fe9d0485f52be306273d4d1a9bea8d7b39a39"
    "q8192_selected_moe_route_prefix_by_program.hsaco" =
        "3aaec2826f234daad9a308ebc8957e0cc309d524ddc92984c39e5027eb80db4d"
    "q8192_selected_moe_route_padded_prefix.hsaco" =
        "8c6fd362ea0b5e9b7d43ec99ae404ddef021ffed9844ad1203a82c307483b914"
    "q8192_selected_moe_route_scatter.hsaco" =
        "dd87e1236cda5eb292b0c918ee98d04218e1ee174a2e9b86e98bf21644d15c7f"
    "q8192_selected_moe_gate_up_silu.hsaco" =
        "945acb545a0cdbdb333ad7ed863a081b4e4a6a0b0872142482e307713529c0fe"
    "q8192_selected_moe_down.hsaco" =
        "bd8b6970d1bcc86fe8eb8f9ee4a8d70fb24222badeddf9e477e9596c29d6c30a"
    "q8192_triton_0626_row_major_sorted_conditional_exact_gate_rows256.hsaco" =
        "b0b0230c3f8165dd5de97c02707b35bdac2c497377c17c09bc5f768c4822eae6"
    "q8192_triton_0626_zero_correction_gate_finalize.hsaco" =
        "2120153032fca2210eb73ba07b63e95a2c4517055a12176e2a78ceebf82d8cc0"
    "q8192_triton_0626_conditional_exact_down_rows4.hsaco" =
        "2b430b0226d09af12b36018eed1a9f761141bee37793d5879c142cdd9cb8c1c4"
}

$metadataPath = Join-Path $KernelDir "metadata.json"
if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
    throw "base AOT metadata is missing"
}
$metadataSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $metadataPath
).Hash.ToLowerInvariant()
if ($metadataSha256 -ne $expectedMetadataSha256) {
    throw "base AOT metadata hash mismatch: $metadataSha256"
}

$privateHomePatterns = @(
    '/(?:Users|home|data/home)/[A-Za-z0-9._-]+(?:/|\b)',
    '\b[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]'
)
$artifactRecords = @()
foreach ($entry in $expectedArtifacts.GetEnumerator()) {
    $path = Join-Path $KernelDir $entry.Key
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "expected public AOT is missing: $($entry.Key)"
    }
    $item = Get-Item -LiteralPath $path
    $sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $path
    ).Hash.ToLowerInvariant()
    if ($sha256 -ne $entry.Value) {
        throw "public AOT hash mismatch for $($entry.Key): $sha256"
    }
    $ascii = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($path))
    foreach ($pattern in $privateHomePatterns) {
        if ([regex]::IsMatch($ascii, $pattern)) {
            throw "public AOT contains a private home path: $($entry.Key)"
        }
    }
    $artifactRecords += [ordered]@{
        file = $entry.Key
        bytes = $item.Length
        sha256 = $sha256
        private_home_path_count = 0
    }
}

$metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
if ($metadata.target -ne "gfx1151" -or
        $metadata.compiler -ne "Triton 3.6.0 HIP backend" -or
        [int]$metadata.shape.tokens -ne 8192 -or
        [int]$metadata.shape.block_m -ne 64 -or
        [int]$metadata.shape.group_m -ne 1 -or
        $metadata.postprocess.debug_sections_stripped -ne $true) {
    throw "base AOT metadata does not describe the public dynamic gfx1151 set"
}
$baseKernels = @($metadata.kernels)
if ($baseKernels.Count -ne 6) {
    throw "base AOT metadata must contain six kernels"
}
foreach ($kernel in $baseKernels) {
    $path = Join-Path $KernelDir ([string]$kernel.file)
    $item = Get-Item -LiteralPath $path
    $sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $path
    ).Hash.ToLowerInvariant()
    if ([long]$kernel.bytes -ne $item.Length -or
            [string]$kernel.sha256 -ne $sha256) {
        throw "base AOT metadata hash/size mismatch: $($kernel.file)"
    }
}

$record = [ordered]@{
    schema_version = 1
    record_type = "qrt_public_aot_windows_transport_verification"
    host = [Environment]::MachineName
    execution = "windows_powershell_5_1_cpu_only"
    completed_utc = [DateTime]::UtcNow.ToString("o")
    repo_commit = "6cdce444177f54e8985a2d6148da863369bece32"
    command_file = "scripts/.candidate_verify_public_aot_windows_r1191.ps1"
    command_file_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath
    ).Hash.ToLowerInvariant()
    target = "gfx1151"
    metadata_sha256 = $metadataSha256
    artifact_count = $artifactRecords.Count
    artifacts = $artifactRecords
    metadata_hash_size_match = $true
    private_home_path_count = 0
    powershell_parse_pass = $true
    component_only = $true
    gpu_execution = $false
    inference_success_claimed = $false
    status = "pass"
}
$resultPath = Join-Path $OutDir "verification.json"
[IO.File]::WriteAllText(
    $resultPath,
    ($record | ConvertTo-Json -Depth 8) + "`n",
    $utf8
)
$resultSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $resultPath
).Hash.ToLowerInvariant()
[IO.File]::WriteAllText(
    (Join-Path $OutDir "SHA256SUMS"),
    "$resultSha256  verification.json`n",
    $utf8
)
$record | ConvertTo-Json -Depth 8
