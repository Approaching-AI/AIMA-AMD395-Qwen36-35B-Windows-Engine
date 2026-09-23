from pathlib import Path
import base64,hashlib,json,socket,subprocess,sys

B=Path(__file__).resolve().parent;D=B/'dense-selected-explicit-windows-r1'
assert socket.gethostname().split('.')[0].lower()!='baiying'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
EXPECTED_MANIFEST='eb016da03377684388e3a1f79a71bf6500a550ec630d9059f8857e43680aca17'
assert sha(D/'manifest.json')==EXPECTED_MANIFEST
m=read(D/'manifest.json');assert sha(D/'worker.py')==m['worker_sha256']
entries=[*m['source_inputs'],*m['images'].values()]
for e in entries:assert (D/e['file']).stat().st_size==e['bytes'] and sha(D/e['file'])==e['sha256']
inputs=list({e['input']['file']:e['input'] for e in m['projections']}.values())
for e in inputs:assert (D/'inputs'/e['file']).stat().st_size==e['bytes'] and sha(D/'inputs'/e['file'])==e['sha256']
prior_path=B/m['active_owner_local_record'];ready=prior_path.exists()
if ready:
 prior=read(prior_path);ready=prior['host_checks_pass'] and all(prior['host_checks'].values()) and not prior['after_processes']
if sys.argv[1:]==['--check']:
 print(json.dumps(dict(prepared_manifest_verified=True,full256k_owner_complete_and_clean=bool(ready),remote_calls=0,executed=False)));sys.exit(0)
assert not sys.argv[1:] and ready,'Active full256k owner has no completed clean host record; zero remote calls.'
remote=m['remote_directory'];opts=['-o','BatchMode=yes','-o','ConnectTimeout=10']
quote=lambda s:"'"+str(s).replace("'","''")+"'"
wrapper="$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue';[Console]::InputEncoding=[Text.UTF8Encoding]::new($false);[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);$s=[Console]::In.ReadToEnd();& ([ScriptBlock]::Create($s))"
remote_command='powershell.exe -NoProfile -EncodedCommand '+base64.b64encode(wrapper.encode('utf-16le')).decode()
def ps(script,timeout=30):return subprocess.run(['ssh',*opts,'baiying',remote_command],input=script,capture_output=True,text=True,errors='replace',timeout=timeout)
def save(name,value):
 with (D/name).open('x')as f:json.dump(value,f,indent=2);f.write('\n')
def record(r):return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
verify="function Verify([string]$p,[string]$s){if((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() -ne $s){throw ('Changed file '+$p)}}\n"
admission="if(-not [Environment]::MachineName.Equals('baiying',[StringComparison]::OrdinalIgnoreCase)){throw 'Unexpected host'}\n"+verify
admission+='Verify '+quote(m['active_owner_remote_record'])+' '+quote(sha(prior_path))+'\n'
admission+='$prior=Get-Content -Raw -LiteralPath '+quote(m['active_owner_remote_record'])+'|ConvertFrom-Json;if(-not $prior.host_checks_pass -or @($prior.after_processes).Count -ne 0){throw "Previous owner was not clean"}\n'
setup=admission+'if(Test-Path '+quote(remote)+"){throw 'Component directory exists'};$null=New-Item -ItemType Directory -Path "+quote(remote)+' -Force\n'
r=ps(setup);save('setup.json',dict(command_script=setup,**record(r)));r.check_returncode()
for folder in ['source','inputs']:
 subprocess.run(['scp',*opts,'-r',str(D/folder),'baiying:'+remote+'/'],capture_output=True,text=True,check=True,timeout=120)
for local in [D/'worker.py',D/'manifest.json',*[D/e['file']for e in m['images'].values()]]:
 subprocess.run(['scp',*opts,str(local),'baiying:'+remote+'/'+local.name],capture_output=True,text=True,check=True,timeout=45)
spec=dict(executable=m['python_executable'],arguments=[remote+'/worker.py',remote],working_directory=m['execution_checkout'],
 repo_commit=m['execution_checkout_commit'],candidate_base_commit=m['candidate_base_commit'],source_manifest_sha256=sha(D/'manifest.json'),
 command_file=remote+'/dispatch.ps1',model=m['model'],model_loaded=False,actual_model_weight_tensors_loaded=True,
 purpose='Full q8192 original-layer QKV/Z/OUT; current selected replay versus explicit-layout unsigned32 AOT; complete coverage plus unchanged WMMA selector; component only',
 inference_acceptance=False,performance_acceptance=False)
script=admission+'Verify '+quote(m['guard_file'])+' '+quote(m['guard_sha256'])+'\n'
script+='Verify '+quote(remote+'/manifest.json')+' '+quote(sha(D/'manifest.json'))+'\n'
script+='Verify '+quote(remote+'/worker.py')+' '+quote(m['worker_sha256'])+'\n'
script+='$env:PATH='+quote(m['rocm_root']+'/bin;')+'+$env:PATH\n'
script+='$spec=@\'\n'+json.dumps(spec)+'\n\'@\n[IO.File]::WriteAllText('+quote(remote+'/spec.json')+',$spec,[Text.UTF8Encoding]::new($false))\n'
script+='& '+quote(m['guard_file'])+' -SpecPath '+quote(remote+'/spec.json')+' -OutDir '+quote(remote+'/run')+' -TimeoutSeconds 900|Out-Null\n'
script+='$r=Get-Content -Raw -LiteralPath '+quote(remote+'/run/run-record.json')+'|ConvertFrom-Json;if($r.reason -ne "completed" -or $r.exit_code -notin @(0,6) -or -not $r.host_checks_pass -or @($r.after_processes).Count -ne 0){throw "Component host execution failed"};$r.host_checks|ConvertTo-Json -Compress\n'
(D/'dispatch.ps1').write_text(script)
subprocess.run(['scp',*opts,str(D/'dispatch.ps1'),'baiying:'+remote+'/dispatch.ps1'],capture_output=True,text=True,check=True,timeout=30)
r=ps('& '+quote(remote+'/dispatch.ps1'),960);save('dispatch.json',dict(command_file=remote+'/dispatch.ps1',command_file_sha256=sha(D/'dispatch.ps1'),**record(r)))
collection=[]
for folder,names in [('run',['run-record.json','preflight.json','launch.json','product.stdout.jsonl','product.stderr.log','telemetry.jsonl']),
 ('build',['build-record.json','compiler.stdout.log','compiler.stderr.log']),('outputs',['result.json'])]:
 (D/folder).mkdir()
 for name in names:
  t=subprocess.run(['scp',*opts,'baiying:'+remote+'/'+folder+'/'+name,str(D/folder/name)],capture_output=True,text=True,timeout=40)
  collection.append(dict(file=folder+'/'+name,**record(t)))
save('collection.json',collection)
r.check_returncode();assert all(c['returncode']==0 for c in collection)
run=read(D/'run/run-record.json');report=read(D/'outputs/result.json')
assert run['host_checks_pass'] and all(run['host_checks'].values()) and not run['after_processes']
assert report['manifest_sha256']==sha(D/'manifest.json') and report['cleanup_pass'] and report['source_files_unchanged']
print(json.dumps(dict(host=report['host'],candidate_base_commit=report['candidate_base_commit'],all_components_match=report['all_components_match'],
 native_exit_code=run['exit_code'],result_sha256=sha(D/'outputs/result.json'),run_sha256=sha(D/'run/run-record.json'),inference_acceptance=False)),flush=True)
