from pathlib import Path
import sys
sys.path.insert(0,'/Users/jiawei-macmini/projects/AIMA-persistent-gdn-candidate/tools')
from check_linux_core_q8192 import source_input_inventory
import base64,hashlib,json,socket,subprocess,tarfile,traceback,shutil
B=Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0].lower()!='baiying'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
stage=read(B/'linux-core-windows-persistent-gdn-stage-r1.json')
assert sha(B/stage['dispatcher'])==stage['dispatcher_sha256']
assert sha(B/stage['manifest'])==stage['manifest_sha256']
assert sha(B/stage['bundle'])==stage['bundle_sha256']
assert sha(B/'qualify_persistent_gdn_components_r1.py')==stage['qualifier_sha256']
assert sha(B/'qualify_explicit_q8192_components_r1.py')==stage['other_qualifier_sha256']
plan=read(B/stage['manifest'])
# No tensor archives or derived slices are downloaded. Native run files and
# binary are individually bounded to32MiB; reserve128MiB for normal evidence.
assert shutil.disk_usage(B).free > (128 << 20)
D=B/'linux-core-windows-persistent-gdn-r1';D.mkdir()
opts=['-o','BatchMode=yes','-o','ConnectTimeout=10']
def record(name,value):
 with (D/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def quote(value):return "'"+str(value).replace("'","''")+"'"
wrapper="$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue';[Console]::InputEncoding=[Text.UTF8Encoding]::new($false);[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);$s=[Console]::In.ReadToEnd();& ([ScriptBlock]::Create($s))"
remote_command='powershell.exe -NoProfile -EncodedCommand '+base64.b64encode(wrapper.encode('utf-16le')).decode()
ps_calls=0
def ps(script,timeout=30):
 global ps_calls
 ps_calls+=1
 local=D/('dispatch-'+str(ps_calls)+'.ps1')
 local.write_text("$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue';[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)\n"+script)
 remote='P:/projects/persistent-gdn-dispatch-r1-'+str(ps_calls)+'.ps1'
 subprocess.run(['scp',*opts,str(local),'baiying:'+remote],capture_output=True,text=True,check=True,timeout=30)
 return subprocess.run(['ssh',*opts,'baiying','powershell.exe -NoProfile -File '+remote],capture_output=True,text=True,errors='replace',timeout=timeout)
def result(r):return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def scp(remote,local,download=True):
 if download:
  size=ps('(Get-Item -LiteralPath '+quote(remote)+').Length',30);size.check_returncode()
  assert 0<=int(size.stdout.strip())<=(32<<20)
 source,target=('baiying:'+remote,str(local)) if download else (str(local),'baiying:'+remote)
 subprocess.run(['scp',*opts,source,target],capture_output=True,text=True,timeout=90,check=True)
def copy_run(remote,folder):
 folder.mkdir()
 for name in ('run-record.json','preflight.json','launch.json','product.stdout.jsonl','product.stderr.log','telemetry.jsonl'):
  scp(remote+'/'+name,folder/name)
 return read(folder/'run-record.json')
def clean(r):
 assert r['host'].lower()=='baiying' and r['host_checks_pass'] and not r['after_processes']
 assert all(r['host_checks'].values())
try:
 from qualify_persistent_gdn_components_r1 import qualify
 selection=qualify(B,Path('/Users/jiawei-macmini/projects/AIMA-persistent-gdn-candidate'),**plan['component_selection_options'])
 assert selection==plan['native_component_selection'] and selection['ready']
 assert selection['candidate_commit']==plan['source_commit']
 remote_bundle='P:/projects/'+stage['bundle'];remote_plan='P:/projects/'+stage['manifest']
 scp(remote_bundle,B/stage['bundle'],False);scp(remote_plan,B/stage['manifest'],False)
 script=r'''
function Run-Git([string[]]$Arguments) {
 $i=New-Object Diagnostics.ProcessStartInfo;$i.FileName='git.exe';$i.UseShellExecute=$false
 $i.RedirectStandardOutput=$true;$i.RedirectStandardError=$true
 $i.Arguments=($Arguments|ForEach-Object{'"'+$_.Replace('"','\"')+'"'}) -join ' '
 $p=New-Object Diagnostics.Process;$p.StartInfo=$i;$null=$p.Start()
 $out=$p.StandardOutput.ReadToEndAsync();$err=$p.StandardError.ReadToEndAsync()
 if(-not $p.WaitForExit(30000)){try{$p.Kill()}catch{};throw 'Git timeout'}
 $s=$out.GetAwaiter().GetResult();$e=$err.GetAwaiter().GetResult()
 if($p.ExitCode -ne 0){throw ('Git failed: '+$e)};return $s.Trim()
}
'''
 script+='\n$repo='+quote(plan['repo'])+';$seed='+quote(stage['seed_repo'])+';$bundle='+quote(remote_bundle)+'\n'
 script+="if((Get-FileHash $bundle -Algorithm SHA256).Hash.ToLowerInvariant() -ne "+quote(stage['bundle_sha256'])+"){throw 'Bundle changed'}\n"
 script+="if(Test-Path $repo){throw 'Experiment checkout already exists'}\n"
 script+="$null=Run-Git -Arguments @('-C',$seed,'fetch',$bundle,'HEAD')\n"
 script+="$null=Run-Git -Arguments @('-C',$seed,'worktree','add','--no-checkout','--detach',$repo,"+quote(stage['source_commit'])+")\n"
 script+="$null=Run-Git -Arguments @('-C',$repo,'sparse-checkout','init','--cone')\n"
 script+="$null=Run-Git -Arguments @('-C',$repo,'sparse-checkout','set','native','third_party','tools','scripts','contracts')\n"
 script+="$null=Run-Git -Arguments @('-C',$repo,'read-tree','-mu','HEAD')\n"
 script+="if((Run-Git -Arguments @('-C',$repo,'rev-parse','HEAD')) -ne "+quote(stage['source_commit'])+" -or (Run-Git -Arguments @('-C',$repo,'status','--porcelain'))){throw 'Checkout differs'}\n"
 r=ps(script,150);record('stage-dispatch.json',dict(command_script=script,**result(r)));r.check_returncode()
 for phase,timeout in [('build',1860),('product',720)]:
  command='& '+quote(plan['command_file'])+' -SourceManifest '+quote(remote_plan)+' -Phase '+phase+'\n'
  print(json.dumps(dict(status='dispatching',phase=phase,source=stage['source_commit'])),flush=True)
  r=ps(command,timeout);record(phase+'-dispatch.json',dict(command_script=command,**result(r)))
  run=copy_run(plan['build_run_directory'] if phase=='build' else plan['run_directory'],D/phase)
  clean(run)
  if phase=='build':
   scp(plan['build_directory']+'/build-provenance.json',D/'build-provenance.json')
   build=read(D/'build-provenance.json')
   for x in build['commands']:
    if x['exit_code'] or x['timed_out']:
     for stream in ('stdout','stderr'):scp(plan['build_directory']+'/'+x['label']+'.'+stream+'.txt',D/(x['label']+'.'+stream+'.txt'))
   assert '-DQRT_SM121_DPP_REDUCTION=1' in build['compile_flags'] and '-DQRT_SM121_COMPACT_NORMALIZE=1' in build['compile_flags']
   assert build['completed'] and source_input_inventory(build['source_inputs'])==source_input_inventory(plan['build_source_inputs'])
   assert build['optional_adaptations']['current_text_decode']['prefill_changed'] is False
   assert '--current-text-decode' in run['spec']['arguments']
   assert '--gb10-prefill-projections' in run['spec']['arguments']
   assert 'gb10_prefill_projections' in build['optional_adaptations']
   assert '--gb10-normalization' in run['spec']['arguments']
   assert '--gb10-moe' in run['spec']['arguments']
   assert '--gb10-projections' in run['spec']['arguments']
   assert '--gb10-gdn' in run['spec']['arguments']
   assert '--gb10-convolution' in run['spec']['arguments']
   assert build['optional_adaptations']['gb10_gdn']['decode_beta']=='FP32'
   native=build['optional_adaptations']['gb10_gdn']['native_prefill_opt_in']
   assert native['additional_scratch_bytes']==100663296 and native['reused_conversion_scratch_bytes']==5242880
   assert native['embedded_image_bytes']==743344 and native['aot_launches_per_linear_layer']==390
   assert native['embedded_include_sha256']=='d9542180bdc7db0adbd3f9ca165c73107b377c27ebb6a16979e4382cdd479eb7'
   fused=native['persistent']
   assert fused['setting']=='AIMA_PORT_NATIVE_GDN_PERSISTENT=1' and fused['default']=='0'
   assert fused['requires']=='AIMA_PORT_NATIVE_GDN_PREFILL=1' and fused['aot_launches_per_linear_layer']==7
   assert fused['images']==6 and fused['embedded_image_bytes']==1545744
   assert fused['additional_scratch_bytes_over_native_prefill']==0 and fused['reused_conversion_scratch_bytes']==2097152
   assert fused['block_shared_bytes']==6144
   assert fused['embedded_include_sha256']=='76b163a8774c2dc2b86c7756ba04c5509dceb64e9ff622cd916afda1ea1e4eff'
   assert build['optional_adaptations']['gb10_convolution']['prefill_changed'] is True
   assert all(x['exit_code']==0 and not x['timed_out'] for x in build['commands'])
   assert build['coff_verification']['all_image_bytes_and_symbols_match'] and build['coff_verification']['images']==72
   assert len([x for x in build['commands'] if x['label'].startswith('compile-')])==61
   binary=next(x for x in build['artifacts'] if x['path'].replace('\\','/').lower()==plan['run_spec']['executable'].lower())
   scp(binary['path'].replace('\\','/'),D/'qrt-linux-core-q8192-probe.exe')
   assert sha(D/'qrt-linux-core-q8192-probe.exe')==binary['sha256']
  r.check_returncode();assert run['reason']=='completed' and run['exit_code']==0

 events=[json.loads(line) for line in (D/'product/product.stdout.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
 P=Path('/Users/jiawei-macmini/projects/AIMA-persistent-gdn-candidate')
 oracle=next(c for c in read(P/'contracts/gb10_cold_token_matrix_20260911_oracle.json')['cases'] if c['name']=='q8192-out512')
 actual=events[-1];expected=oracle['expected']['output_token_ids']
 differences=[dict(index=i,actual=a,expected=e)for i,(a,e)in enumerate(zip(actual['output_token_ids'],expected))if a!=e]
 command=['python3.12',str(P/'tools/check_linux_core_q8192.py'),'--run',str(D/'product'),'--build-metadata',str(D/'build-provenance.json'),
  '--build-run',str(D/'build/run-record.json'),'--plan',str(B/stage['manifest']),'--output',str(D/'gb10-qualified.json')]
 checked=subprocess.run(command,capture_output=True,text=True,timeout=30)
 record('observer.json',dict(command=command,returncode=checked.returncode,stdout=checked.stdout,stderr=checked.stderr,checker_sha256=sha(P/'tools/check_linux_core_q8192.py')))
 stderr=(D/'product/product.stderr.log').read_text(encoding='utf-8-sig')
 markers=[json.loads(line)for line in stderr.splitlines()if line.startswith('{')]
 find=lambda name:[row for row in markers if row.get('event')==name]
 env=plan['environment'];activation={}
 terminal=find('terminal_prefill_attention')+find('terminal_prefill_moe')
 activation['terminal_route']=terminal==[dict(event='terminal_prefill_attention',layer=39,queries=1,kv_tokens=8192,projection_rows=1),dict(event='terminal_prefill_moe',layer=39,rows=1,carrier_row=8191)]
 gdn=find('native_gdn_prefill')
 activation['gdn']=gdn==([dict(event='native_gdn_prefill',layer=i,tokens=8192,stages=6,aot_launches=7,chunk_tokens=64,
  original_preparation=True,ordered_integer_accumulator=True,cold=True,persistent_recurrence=True)for i in range(40)if i%4!=3]if env['AIMA_PORT_NATIVE_GDN_PREFILL']==env['AIMA_PORT_NATIVE_GDN_PERSISTENT']=='1'else[])
 attention=find('ordered_attention_prefill')
 activation['attention']=attention==([dict(event='ordered_attention_prefill',layer=i,queries=8192,kv_tokens=8192,aot_launches=193)
  for i in range(3,39,4)]if env['AIMA_PORT_ORDERED_ATTENTION_PREFILL']=='1'else[])
 moe=find('native_moe_prefill')
 activation['native_moe']=([row['layer']for row in moe]==list(range(39)) and all(row['tokens']==8192 and row['expert_projections']==2 for row in moe)) if env['AIMA_PORT_NATIVE_MOE_PREFILL']=='1'else not moe
 for family,setting in [('prefill','AIMA_PORT_PREFILL_ORDERED_REPLAY'),('routed','AIMA_PORT_ROUTED_ORDERED_REPLAY')]:
  rows=find(family+'_ordered_replay_summary');batch=int(env[setting])
  activation[family+'_replay']=bool(len(rows)==1 and rows[0]['batch']==batch and rows[0]['projections']>0 and rows[0]['launches']>0
   and all(rows[0][key]for key in('raw_bf16_operands','device_synchronized','warmup_excluded')))if batch else not rows
 activation['unprofiled']=not find('prefill_projection_profile') and not any(line.startswith('SM121_COMPLETED_STAGE_PROFILE ')for line in stderr.splitlines())
 activation['original_model_runtime']=actual.get('oracle_tensor_reads')==0 and 'observation_directory'not in events[0]
 record('activation.json',dict(checks=activation,all_pass=all(activation.values()),markers=markers,
  environment={k:v for k,v in env.items()if k.startswith('AIMA_PORT_')and any(x in k for x in('REPLAY','PREFILL'))},
  stderr_sha256=sha(D/'product/product.stderr.log'),inference_acceptance=False,performance_acceptance=False))
 record('result.json',dict(actual=actual,gb10_boundary_pass=checked.returncode==0,candidate_route_observed=all(activation.values()),
  continuation_mismatches=len(differences),actual_output_tokens=len(actual['output_token_ids']),expected_output_tokens=len(expected),
  first_difference=differences[:1],source_manifest_sha256=stage['manifest_sha256'],run_sha256=sha(D/'product/run-record.json'),
  performance_acceptance=False,release_qualified=False))
 print(json.dumps(dict(status='product_complete',gb10_boundary_pass=checked.returncode==0,candidate_route_observed=all(activation.values()),
  mismatches=len(differences),first_difference=differences[:1],load_ms=actual['command_to_ready_ms'],ttft_ms=actual['ttft_ms'],tpot_ms=actual['tpot_ms'])),flush=True)
except BaseException as error:
 record('controller-error.json',dict(error=str(error),traceback=traceback.format_exc(),inference_acceptance=False))
 raise
