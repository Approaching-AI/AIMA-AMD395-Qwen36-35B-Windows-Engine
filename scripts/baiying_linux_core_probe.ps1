param(
    [Parameter(Mandatory=$true)][string]$SourceManifest,
    [Parameter(Mandatory=$true)][ValidateSet('build','product')][string]$Phase
)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
if(-not [Environment]::MachineName.Equals('baiying',[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Requires the local baiying execution host'
}
$utf8=[Text.UTF8Encoding]::new($false)
$m=[IO.File]::ReadAllText($SourceManifest)|ConvertFrom-Json
function Verify-File([string]$Path,[string]$Sha) {
    if((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha) {
        throw ('Changed input: '+$Path)
    }
}
function Verify-Owner([string]$Path) {
    $r=[IO.File]::ReadAllText($Path)|ConvertFrom-Json
    if(-not $r.host.Equals('baiying',[StringComparison]::OrdinalIgnoreCase) -or
       -not $r.host_checks_pass -or @($r.after_processes).Count -ne 0) {
        throw 'Prior owner did not leave a clean host'
    }
    foreach($entry in $r.host_checks.PSObject.Properties) {
        if(-not $entry.Value) { throw 'Prior owner host check failed' }
    }
    return $r
}
Verify-File $PSCommandPath $m.command_file_sha256
Verify-File $m.prior_owner_record $m.prior_owner_record_sha256
$null=Verify-Owner $m.prior_owner_record
$repo=[IO.Path]::GetFullPath($m.repo)
if([IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)) -ne $repo) {
    throw 'Command does not belong to the planned checkout'
}
foreach($entry in $m.build_source_inputs) { Verify-File (Join-Path $repo $entry.path) $entry.sha256 }
foreach($entry in $m.guard_inputs) { Verify-File (Join-Path $repo $entry.path) $entry.sha256 }
$guard=Join-Path $repo 'scripts/baiying_guarded_inference.ps1'
$env:PATH=$m.rocm_root+'/bin;'+$env:PATH
$manifestSha=(Get-FileHash -LiteralPath $SourceManifest -Algorithm SHA256).Hash.ToLowerInvariant()
if($Phase -eq 'build') {
    $out=$m.build_run_directory
    $specPath=$out+'.json'
    if((Test-Path $out) -or (Test-Path $specPath) -or (Test-Path $m.build_directory)) {
        throw 'Build evidence already exists'
    }
    $spec=[ordered]@{
        executable=$m.python_executable
        arguments=@((Join-Path $repo 'tools/build_linux_core_windows.py'),'--out',$m.build_directory,
                    '--rocm',$m.rocm_root,'--timeout-seconds','1500')
        working_directory=$repo;repo_commit=$m.source_commit;source_inputs=$m.build_source_inputs
        command_file=$PSCommandPath;command_file_sha256=$m.command_file_sha256
        source_manifest_sha256=$manifestSha;model='none; native compilation and host I/O contracts'
        model_loaded=$false;inference_acceptance=$false;performance_acceptance=$false
    }
    if($m.current_text_decode -eq $true) {
        $spec.arguments+=@('--current-text-decode')
    }
    if($m.gb10_convolution -eq $true) {
        $spec.arguments+=@('--gb10-convolution')
    }
    if($m.gb10_gdn -eq $true) {
        $spec.arguments+=@('--gb10-gdn')
    }
    $null=New-Item -ItemType Directory -Path (Join-Path $repo 'build') -Force
    [IO.File]::WriteAllText($specPath,($spec|ConvertTo-Json -Depth 12),$utf8)
    & $guard -SpecPath $specPath -OutDir $out -TimeoutSeconds 1740|Out-Null
    $r=Verify-Owner (Join-Path $out 'run-record.json')
    if($r.reason -ne 'completed' -or $r.exit_code -ne 0) { throw 'Build did not complete successfully' }
    $metadata=Join-Path $m.build_directory 'build-provenance.json'
    $b=[IO.File]::ReadAllText($metadata)|ConvertFrom-Json
    if(-not $b.completed -or $b.dirty_tree -or $b.repo_commit -ne $m.source_commit -or
       $b.upstream_commit -ne $m.upstream_commit) { throw 'Build metadata differs from source plan' }
    foreach($entry in $b.artifacts) { Verify-File $entry.path $entry.sha256 }
    [ordered]@{phase=$Phase;completed=$true;source_commit=$m.source_commit;
        metadata_sha256=(Get-FileHash $metadata -Algorithm SHA256).Hash.ToLowerInvariant();
        inference_acceptance=$false}|ConvertTo-Json -Compress
    exit 0
}

$buildRecord=Join-Path $m.build_run_directory 'run-record.json'
$bRun=Verify-Owner $buildRecord
if($bRun.reason -ne 'completed' -or $bRun.exit_code -ne 0 -or
   $bRun.spec.source_manifest_sha256 -ne $manifestSha) { throw 'Build/source manifest binding differs' }
$metadata=Join-Path $m.build_directory 'build-provenance.json'
$b=[IO.File]::ReadAllText($metadata)|ConvertFrom-Json
if(-not $b.completed -or $b.dirty_tree -or $b.repo_commit -ne $m.source_commit) {
    throw 'Native build is not complete'
}
foreach($entry in $b.artifacts) { Verify-File $entry.path $entry.sha256 }
Verify-File $m.ck_provider.path $m.ck_provider.sha256
Verify-File $m.vision_image.path $m.vision_image.sha256
Verify-File $m.prompt.path $m.prompt.sha256
foreach($entry in $m.environment_files) { Verify-File $entry.path $entry.sha256 }
foreach($entry in @(Get-ChildItem Env:|Where-Object{$_.Name -match '^(QRT_|AIMA_)'})) {
    [Environment]::SetEnvironmentVariable($entry.Name,$null,'Process')
}
foreach($entry in $m.environment.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable($entry.Name,[string]$entry.Value,'Process')
}
$out=$m.run_directory
$specPath=$out+'.json'
if((Test-Path $out) -or (Test-Path $specPath)) { throw 'Product evidence already exists' }
$spec=[ordered]@{}
foreach($entry in $m.run_spec.PSObject.Properties) { $spec[$entry.Name]=$entry.Value }
$artifacts=@($b.artifacts|Where-Object{
    [IO.Path]::GetFullPath($_.path) -eq [IO.Path]::GetFullPath($spec.executable)
})
if($artifacts.Count -ne 1) { throw 'Planned executable is absent from build artifacts' }
$spec.executable_sha256=$artifacts[0].sha256
$spec.build_metadata_sha256=(Get-FileHash $metadata -Algorithm SHA256).Hash.ToLowerInvariant()
$spec.build_run_sha256=(Get-FileHash $buildRecord -Algorithm SHA256).Hash.ToLowerInvariant()
$spec.source_manifest_sha256=$manifestSha
$spec.environment=$m.environment
$spec.inference_acceptance=$false
$spec.performance_acceptance=$false
[IO.File]::WriteAllText($specPath,($spec|ConvertTo-Json -Depth 12),$utf8)
& $guard -SpecPath $specPath -OutDir $out -TimeoutSeconds 600|Out-Null
$r=Verify-Owner (Join-Path $out 'run-record.json')
foreach($entry in $b.artifacts) { Verify-File $entry.path $entry.sha256 }
foreach($entry in $m.environment_files) { Verify-File $entry.path $entry.sha256 }
Verify-File $m.prompt.path $m.prompt.sha256
Verify-File $m.ck_provider.path $m.ck_provider.sha256
if($r.reason -ne 'completed' -or $r.exit_code -ne 0) { throw 'Product process failed or timed out' }
[ordered]@{phase=$Phase;completed=$true;source_commit=$m.source_commit;
    correctness_requires_external_gb10_comparison=$true;performance_acceptance=$false}|ConvertTo-Json -Compress
exit 0
