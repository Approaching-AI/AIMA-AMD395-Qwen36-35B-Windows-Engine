param(
    [Parameter(Mandatory=$true)][string]$SourceManifest,
    [Parameter(Mandatory=$true)][ValidateSet('build','probe')][string]$Phase
)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
if(-not [Environment]::MachineName.Equals('baiying',[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Requires native baiying'
}
$m=Get-Content -LiteralPath $SourceManifest -Raw|ConvertFrom-Json
$utf8=[Text.UTF8Encoding]::new($false)
function Verify-File([string]$Path,[string]$Hash) {
    if((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Hash) {
        throw ('Changed input: '+$Path)
    }
}
function Verify-Owner([string]$Path) {
    $r=Get-Content -LiteralPath $Path -Raw|ConvertFrom-Json
    if(-not $r.host.Equals('baiying',[StringComparison]::OrdinalIgnoreCase) -or
       -not $r.host_checks_pass -or @($r.after_processes).Count -ne 0) { throw 'Prior owner is not clean' }
    foreach($entry in $r.host_checks.PSObject.Properties) { if(-not $entry.Value){throw 'Host check failed'} }
    return $r
}
Verify-File $PSCommandPath $m.command_file_sha256
if([IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)) -ne [IO.Path]::GetFullPath($m.repo)) {
    throw 'Command checkout differs'
}
Verify-File $m.prior_owner_record $m.prior_owner_record_sha256
$null=Verify-Owner $m.prior_owner_record
foreach($entry in $m.source_inputs) { Verify-File (Join-Path $m.repo $entry.path) $entry.sha256 }
foreach($entry in $m.runtime_inputs) { Verify-File $entry.path $entry.sha256 }
$env:PATH=$m.rocm_root+'/bin;'+$env:PATH
foreach($entry in @(Get-ChildItem Env:|Where-Object{$_.Name -match '^(QRT_|AIMA_|HIPBLASLT_|TENSILE_)'})) {
    [Environment]::SetEnvironmentVariable($entry.Name,$null,'Process')
}
$manifestHash=(Get-FileHash $SourceManifest -Algorithm SHA256).Hash.ToLowerInvariant()
$spec=[ordered]@{
    working_directory=$m.repo;repo_commit=$m.source_commit;source_inputs=$m.source_inputs
    runtime_inputs=$m.runtime_inputs;source_manifest_sha256=$manifestHash
    command_file=$PSCommandPath;command_file_sha256=$m.command_file_sha256
    model='none; exact rational matrix control';model_loaded=$false
    inference_acceptance=$false;performance_acceptance=$false
}
if($Phase -eq 'build') {
    $out=$m.build_run_directory;$timeout=240
    $spec.executable=$m.python_executable
    $spec.arguments=@((Join-Path $m.repo 'tools/build_linux_core_gemm_probe.py'),'--out',$m.build_directory,'--rocm',$m.rocm_root)
} else {
    $buildRecord=Verify-Owner ($m.build_run_directory+'/run-record.json')
    if($buildRecord.exit_code -ne 0 -or $buildRecord.reason -ne 'completed' -or
       $buildRecord.spec.source_manifest_sha256 -ne $manifestHash) { throw 'Build binding differs' }
    $b=Get-Content -LiteralPath ($m.build_directory+'/build-provenance.json') -Raw|ConvertFrom-Json
    if(-not $b.completed -or $b.source_commit -ne $m.source_commit) { throw 'Build incomplete' }
    Verify-File $b.artifact.path $b.artifact.sha256
    $out=$m.run_directory;$timeout=420
    $spec.executable=$b.artifact.path;$spec.executable_sha256=$b.artifact.sha256;$spec.arguments=@()
}
if((Test-Path $out) -or (Test-Path ($out+'.json'))) { throw 'Evidence already exists' }
$null=New-Item -ItemType Directory -Path (Join-Path $m.repo 'build') -Force
[IO.File]::WriteAllText(($out+'.json'),($spec|ConvertTo-Json -Depth 12),$utf8)
& (Join-Path $m.repo 'scripts/baiying_guarded_inference.ps1') -SpecPath ($out+'.json') -OutDir $out -TimeoutSeconds $timeout|Out-Null
$r=Verify-Owner ($out+'/run-record.json')
if($r.reason -ne 'completed' -or $r.exit_code -ne 0 -or -not $r.cli_pass) { throw 'Bounded component phase failed' }
[ordered]@{phase=$Phase;completed=$true;source_commit=$m.source_commit;inference_acceptance=$false}|ConvertTo-Json -Compress
