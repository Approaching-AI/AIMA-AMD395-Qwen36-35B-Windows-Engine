param(
    [Parameter(Mandatory = $true)][string]$ProviderBuildDir,
    [string]$ModelPath = 'D:\models\Qwen3.6-35B-A3B',
    [string]$OutDir = ''
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
if (-not [Environment]::MachineName.Equals('baiying', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Run this bounded GPU replay locally on baiying'
}
$repo = Split-Path -Parent $PSScriptRoot
$commit = (& git -C $repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or @(& git -C $repo status --porcelain).Count -ne 0) {
    throw 'Replay provenance requires a clean committed checkout'
}
function Verify-File([string]$Path, [string]$Sha) {
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha) {
        throw ('Changed replay input: ' + $Path)
    }
}
$ProviderBuildDir = [IO.Path]::GetFullPath($ProviderBuildDir)
$metadataPath = Join-Path $ProviderBuildDir 'build-provenance.json'
$metadata = [IO.File]::ReadAllText($metadataPath) | ConvertFrom-Json
if ($metadata.repo_commit -ne $commit -or $metadata.dirty_tree -or
    -not $metadata.attention_admission_replay -or $metadata.hawkeye_replay_lanes -ne 4 -or
    -not $metadata.provider_compile.completed -or $metadata.provider_compile.exit_code -ne 0) {
    throw 'Build the committed replay with -AttentionAdmissionReplay -HawkeyeReplayLanes 4'
}
Verify-File $metadata.source_path $metadata.source_sha256
Verify-File $metadata.compile_source_path $metadata.compile_source_sha256
$executable = Join-Path $ProviderBuildDir 'qrt-attention-admission-replay.exe'
$artifact = @($metadata.artifacts | Where-Object {$_.name -eq 'qrt-attention-admission-replay.exe'})
if ($artifact.Count -ne 1) { throw 'Replay executable is absent from build provenance' }
Verify-File $executable $artifact[0].sha256
$fixtureRoot = Join-Path $repo 'tests\fixtures\prefix32k-attention-output'
$manifestPath = Join-Path $fixtureRoot 'manifest.json'
$manifest = [IO.File]::ReadAllText($manifestPath) | ConvertFrom-Json
$gated = Join-Path $fixtureRoot 'gated-bf16.bin'
$expected = Join-Path $fixtureRoot 'expected-bf16.bin'
Verify-File $gated $manifest.files.'gated-bf16.bin'.sha256
Verify-File $expected $manifest.files.'expected-bf16.bin'.sha256
Verify-File (Join-Path $ModelPath 'model.safetensors.index.json') $manifest.weights.model_index_sha256
if ($manifest.weights.bytes -ne 16777216 -or $manifest.weights.dtype -ne 'bf16' -or
    $manifest.weights.shard -notmatch '^model-[0-9]+-of-[0-9]+\.safetensors$') {
    throw 'Unexpected bounded weight fixture'
}
if ([string]::IsNullOrWhiteSpace($OutDir)) { $OutDir = Join-Path $repo 'build\attention-admission-result' }
$OutDir = [IO.Path]::GetFullPath($OutDir)
if (Test-Path $OutDir) { throw 'Refusing to overwrite replay evidence' }
$null = New-Item -ItemType Directory -Path $OutDir
$weightsPath = Join-Path $OutDir 'o-projection-bf16.bin'
$handle = [IO.File]::OpenRead((Join-Path $ModelPath $manifest.weights.shard))
try {
    $null = $handle.Seek([long]$manifest.weights.offset, [IO.SeekOrigin]::Begin)
    $reader = [IO.BinaryReader]::new($handle)
    $weights = $reader.ReadBytes([int]$manifest.weights.bytes)
    if ($weights.Length -ne $manifest.weights.bytes) { throw 'Truncated model weight span' }
    [IO.File]::WriteAllBytes($weightsPath, $weights)
} finally { $handle.Dispose() }
Verify-File $weightsPath $manifest.weights.sha256
$spec = [ordered]@{
    executable = $executable
    arguments = @($gated, $weightsPath, $expected)
    working_directory = $repo
    repo_commit = $commit
    command_file = $PSCommandPath
    command_file_sha256 = (Get-FileHash $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    executable_sha256 = $artifact[0].sha256
    build_metadata_sha256 = (Get-FileHash $metadataPath -Algorithm SHA256).Hash.ToLowerInvariant()
    fixture_manifest_sha256 = (Get-FileHash $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    reference = $manifest
    model = $ModelPath
    model_loaded = $false
    inference_acceptance = $false
    purpose = 'Reproduce a1000ppb candidate miss and verify10000ppb on three original GB10 rows; the8192-row execution slab repeats those inputs and is not a full original model transaction'
}
$specPath = Join-Path $OutDir 'spec.json'
[IO.File]::WriteAllText($specPath, ($spec | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
$env:PATH = (Join-Path $metadata.rocm_root 'bin') + ';' + $env:PATH
$runDir = Join-Path $OutDir 'run'
& (Join-Path $repo 'scripts\baiying_guarded_inference.ps1') -SpecPath $specPath -OutDir $runDir -TimeoutSeconds 60 | Out-Null
$run = [IO.File]::ReadAllText((Join-Path $runDir 'run-record.json')) | ConvertFrom-Json
$run | Select-Object reason, exit_code, wall_ms, host_checks_pass | ConvertTo-Json
Get-Content (Join-Path $runDir 'product.stdout.jsonl')
if ($run.reason -ne 'completed' -or $run.exit_code -ne 0 -or -not $run.host_checks_pass) {
    throw 'Original attention projection admission regression failed'
}
