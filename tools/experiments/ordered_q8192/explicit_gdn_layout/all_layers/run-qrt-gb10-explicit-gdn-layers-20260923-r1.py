import subprocess,socket,json,time,hashlib,tarfile,shutil
from pathlib import Path
assert socket.gethostname()=="aitopatom-66c4"
name="qrt-gb10-explicit-gdn-layers-20260923-r1";root=Path("/home/qujing")/name;archive=root.with_suffix(".tar")
assert not root.exists()
assert shutil.disk_usage(root.parent).free > 4*(1<<30)
assert hashlib.sha256(archive.read_bytes()).hexdigest()=="262745dc7c2160366a62f034ceaba174527517fbf9525a50a2cf0655c0be6b17"
assert subprocess.check_output(["docker","inspect","--format","{{.State.Running}}","qwen36-lm-head-capture"],text=True,timeout=10).strip()=="false"
root.mkdir()
with tarfile.open(archive) as t:t.extractall(root,filter="data")
command=['docker', 'run', '--name', 'qrt-gb10-explicit-gdn-layers-20260923-r1', '--hostname', 'aitopatom-66c4', '--gpus', 'all', '--network', 'none', '--memory', '108g', '--shm-size', '8g', '--cpus', '8', '-e', 'PYTHONPATH=/work/scripts', '-e', 'VLLM_ALLOW_LONG_MAX_MODEL_LEN=1', '-e', 'VLLM_WORKER_MULTIPROC_METHOD=spawn', '-v', '/home/qujing/qrt-gb10-explicit-gdn-layers-20260923-r1:/work', '-v', '/mnt/data/models/Qwen3.6-35B-A3B:/models:ro', '-v', '/home/qujing/qrt-gb10-explicit-gdn-layers-20260923-r1/frozen-autotune/2CJTOSHYLMSPKXL7RUUHQ4LLQALA7NZQI4QTHJQXPRCMHXIYAM2A:/tmp/torchinductor_root/triton/0/2CJTOSHYLMSPKXL7RUUHQ4LLQALA7NZQI4QTHJQXPRCMHXIYAM2A:ro', '-v', '/home/qujing/qrt-gb10-explicit-gdn-layers-20260923-r1/frozen-autotune/VPMREUOOCIQFFM6F2BD52MW5DBLQX5MZYNP3NPO4UDGLHEDVJVMA:/tmp/torchinductor_root/triton/0/VPMREUOOCIQFFM6F2BD52MW5DBLQX5MZYNP3NPO4UDGLHEDVJVMA:ro', '-v', '/home/qujing/qrt-gb10-explicit-gdn-layers-20260923-r1/frozen-autotune/42KM3SNV53RK5Y26DFKIXJEYDZCQTCZFTX4FAQ4QEZFO6Y6BAYFQ:/tmp/torchinductor_root/triton/0/42KM3SNV53RK5Y26DFKIXJEYDZCQTCZFTX4FAQ4QEZFO6Y6BAYFQ:ro', '-v', '/home/qujing/qrt-gb10-explicit-gdn-layers-20260923-r1/frozen-autotune/IB2KXOFEBUNQLGXRBO4ASRKBFUBVKHFDUETYFVNLYJUB5KHB3DWA:/tmp/torchinductor_root/triton/0/IB2KXOFEBUNQLGXRBO4ASRKBFUBVKHFDUETYFVNLYJUB5KHB3DWA:ro', '-v', '/home/qujing/qrt-gb10-explicit-gdn-layers-20260923-r1/frozen-autotune/FQEL5X3XJ2UCDIC5NKUXRW3EGK2GKQ22I6WJMORTF3C67DNK7VJQ:/tmp/torchinductor_root/triton/0/FQEL5X3XJ2UCDIC5NKUXRW3EGK2GKQ22I6WJMORTF3C67DNK7VJQ:ro', '-v', '/home/qujing/qrt-gb10-explicit-gdn-layers-20260923-r1/frozen-autotune/MLDITCPF2J4K5FFLCBFKNAOPPXKDUDXR5GXFBFC3EWL45CST37QA:/tmp/torchinductor_root/triton/0/MLDITCPF2J4K5FFLCBFKNAOPPXKDUDXR5GXFBFC3EWL45CST37QA:ro', '-v', '/home/qujing/qrt-gb10-explicit-gdn-layers-20260923-r1/frozen-autotune/OWQNSEVIE4AD2W4XQTOVELAPLYMNYRTD6QSBEJ6WPSVYOWSE3UOQ:/tmp/torchinductor_root/triton/0/OWQNSEVIE4AD2W4XQTOVELAPLYMNYRTD6QSBEJ6WPSVYOWSE3UOQ:ro', '-v', '/home/qujing/qrt-gb10-explicit-gdn-layers-20260923-r1/frozen-autotune/B5RPHOQP75ILD4MSVARVLZXQKTG77NPCDWVFXCXRH7VNQBYQU2IA:/tmp/torchinductor_root/triton/0/B5RPHOQP75ILD4MSVARVLZXQKTG77NPCDWVFXCXRH7VNQBYQU2IA:ro', '-v', '/home/qujing/qrt-gb10-explicit-gdn-layers-20260923-r1/frozen-inductor/193163c3593ce105d150614c04cf633c380ad33243d68253a5c88646814c063c.best_config:/tmp/torchinductor_root/6d/193163c3593ce105d150614c04cf633c380ad33243d68253a5c88646814c063c.best_config:ro', '-v', '/home/qujing/qrt-native-gdn-ordered-pipeline-20260923-r1:/table:ro', '-w', '/work', '--entrypoint', 'timeout', 'sha256:822f5c399ce6a08c0583302e0d82ff3938158b73977b6e1fbdf00ed72735a30d', '--signal=TERM', '--kill-after=10', '1830', 'python3', 'scripts/capture_gb10_ordered_gdn_matrix.py', '--source-commit', '3fcbdafb10a14e94c15bd94a1f821f180bc95373', '--oracle-q7169', 'contracts/arbitrary_q7169_gb10_oracle.json', '--oracle-q8192', 'contracts/hprefill_q8192_gb10_oracle.json', '--output-dir', 'capture', '--expected-host', 'aitopatom-66c4', '--timeout-seconds', '1800', '--execute']
inputs=json.loads((root/'source-inputs.json').read_text())
assert inputs['source_commit']=='3fcbdafb10a14e94c15bd94a1f821f180bc95373'
for relative,expected in inputs['files'].items():
 assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected
profile=json.loads((root/'frozen-autotune-profile.json').read_text())
for entry in profile['records']:
 assert hashlib.sha256((root/entry['stage_path']).read_bytes()).hexdigest()==entry['sha256']
inductor_profile=json.loads((root/'frozen-inductor-profile.json').read_text())
assert hashlib.file_digest((root/inductor_profile['stage_path']).open('rb'),'sha256').hexdigest()==inductor_profile['sha256']
def available():
 for line in Path("/proc/meminfo").read_text().splitlines():
  if line.startswith("MemAvailable:"):return int(line.split()[1])*1024
# The original CPU weight-hash pass populated the shared model's page cache.
# Advise eviction only for those immutable model files; never change model data
# or global VM settings. CUDA UMA reports physical free pages rather than Linux
# MemAvailable, so retain before/after records instead of changing model limits.
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).strip()
import os
model=Path('/mnt/data/models/Qwen3.6-35B-A3B');index_path=model/'model.safetensors.index.json'
assert hashlib.file_digest(index_path.open('rb'),'sha256').hexdigest()=='41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83'
index=json.loads(index_path.read_text());shards=sorted(set(index['weight_map'].values()));assert len(shards)==26
hint=dict(operation='posix_fadvise_DONTNEED_original_model_files',model=str(model),model_data_modified=False,global_vm_settings_changed=False,meminfo_before=Path('/proc/meminfo').read_text(),files=[])
for file in shards:
 assert Path(file).name==file and file.endswith('.safetensors')
 path=model/file;before=path.stat()
 with path.open('rb')as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 hint['files'].append(dict(file=file,bytes=after.st_size,mtime_ns=after.st_mtime_ns))
hint['meminfo_after']=Path('/proc/meminfo').read_text();(root/'model-page-cache-hint.json').write_text(json.dumps(hint,indent=2)+'\n')
assert available()>110*(1<<30)

assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).strip()
start=time.monotonic();minimum=available();reason="completed"
with (root/"container.log").open("w") as log:
 process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
 while process.poll() is None:
  minimum=min(minimum,available())
  if time.monotonic()-start>1842 or minimum<8*(1<<30):
   reason="deadline" if time.monotonic()-start>1842 else "host_memory_reserve"
   subprocess.run(["docker","kill",name],capture_output=True,timeout=15)
   break
  try:process.wait(timeout=1)
  except subprocess.TimeoutExpired:pass
 process.wait(timeout=20)
gpu_after=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).strip()
# Persist any unexpected GPU cleanup result before qualification.
status=subprocess.check_output(["docker","inspect","--format","{{.State.Running}} {{.State.ExitCode}}",name],text=True,timeout=10).strip()
record=dict(host=socket.gethostname(),command=command,command_file=__file__,command_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_commit="3fcbdafb10a14e94c15bd94a1f821f180bc95373",returncode=process.returncode,reason=reason,wall_seconds=time.monotonic()-start,minimum_host_available_bytes=minimum,container_status=status,archive_sha256="262745dc7c2160366a62f034ceaba174527517fbf9525a50a2cf0655c0be6b17")
record['gpu_processes_after']=gpu_after
record['host_guard_pass']=not gpu_after and minimum>=8*(1<<30)
record['frozen_autotune_profile_sha256']=hashlib.sha256((root/'frozen-autotune-profile.json').read_bytes()).hexdigest()
(root/'dispatch-core.json').write_text(json.dumps(record,indent=2)+'\n')
try:
 diff=subprocess.check_output(['docker','diff',name],text=True,timeout=45)
 (root/'docker-diff.txt').write_text(diff)
 record['unexpected_new_autotune_cache_files']=[line[2:] for line in diff.splitlines() if line.endswith('.autotune.json')]
 record['frozen_autotune_cache_no_misses']=not record['unexpected_new_autotune_cache_files']
except subprocess.TimeoutExpired:
 record['postprocessing_error']='docker diff exceeded 45 seconds; durable dispatch-core.json preserved'
 record['unexpected_new_autotune_cache_files']=None
 record['frozen_autotune_cache_no_misses']=None
import io
record['frozen_inductor_profile_sha256']=hashlib.file_digest((root/'frozen-inductor-profile.json').open('rb'),'sha256').hexdigest()
record['frozen_inductor_cache_preserved']=False
record['original_inductor_kernel_source_preserved']=False
try:
 observed={}
 for field,path in [('cache',inductor_profile['cache_path']),('kernel',inductor_profile['kernel_path'])]:
  raw=subprocess.check_output(['docker','cp',name+':'+path,'-'],timeout=15)
  assert len(raw)<(1<<20)
  with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
   members=[m for m in tar.getmembers() if m.isfile()];assert len(members)==1 and members[0].size<(1<<19)
   data=tar.extractfile(members[0]).read()
  observed[field]=dict(path=path,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
 record['inductor_observation']=observed
 record['frozen_inductor_cache_preserved']=observed['cache']['sha256']==inductor_profile['sha256']
 record['original_inductor_kernel_source_preserved']=observed['kernel']['sha256']==inductor_profile['kernel_sha256']
except Exception as error:record['inductor_observation_error']=repr(error)
capture_path=root/'capture/capture.json'

record['candidate_base_commit']=inputs['candidate_base_commit']
record['original_token_matrix_qualified']=False
record['all_original_gdn_results_match']=False
if capture_path.exists():
 captured=json.loads(capture_path.read_text())
 frozen=json.loads((root/'contracts/gb10_cold_token_matrix_20260911_oracle.json').read_text())
 frozen={c['name']:c for c in frozen['cases']}
 checks=[]
 for actual in captured['cases']:
  expected=frozen[actual['name']]
  checks.append(dict(name=actual['name'],prompt_matches=actual['prompt']['u32le_sha256']==expected['prompt']['u32le_sha256'],
   all_outputs_match=actual['output_token_ids']==expected['expected']['output_token_ids'],
   first_logit_matches=abs(actual['first_token_raw_logit']-expected['expected']['first_token_raw_logit'])<=0.125))
 record['token_checks']=checks
 record['original_token_matrix_qualified']=bool(captured['completed'] and captured['controls_qualified'] and
  [c['name'] for c in checks]==['q7169-out32','q8192-out32','q8192-out512'] and
  all(all(v for k,v in c.items() if k!='name') for c in checks))
 product=next((c for c in captured['cases'] if c['name']=='q8192-out512'),None)
 if product is not None:
  report=product['worker']['ordered_gdn']
  record['all_original_gdn_results_match']=bool(report['layers_complete'] and report['all_original_gdn_results_match']
   and report['exp2_table_unchanged'] and report['pending_calls']==0)
record['operator_comparison_qualified']=bool(record['original_token_matrix_qualified'] and
 record['all_original_gdn_results_match'] and record['frozen_autotune_cache_no_misses'] and
 record['host_guard_pass'] and record['frozen_inductor_cache_preserved'] and record['original_inductor_kernel_source_preserved'])
record['inference_acceptance']=False
record['performance_acceptance']=False
(root/'dispatch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
for line in (root/'container.log').read_text(errors='replace').splitlines()[-18:]:print(line[:1500])
selected=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.log','.txt','.py','.bin') and p.stat().st_size<4*(1<<20)]
assert len(selected)<200 and sum(p.stat().st_size for p in selected)<16*(1<<20)
download={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in selected}
(root/'compact-download-manifest.json').write_text(json.dumps(download,indent=2)+'\n')
selected.append(root/'compact-download-manifest.json')
compact=root/'compact-download.tar.gz'
with tarfile.open(compact,'x:gz',compresslevel=1) as tar:
 for p in selected:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
print(json.dumps(dict(compact_archive_bytes=compact.stat().st_size,
 compact_archive_sha256=hashlib.file_digest(compact.open('rb'),'sha256').hexdigest(),
 selected_files=len(selected),selected_bytes=sum(p.stat().st_size for p in selected),
 collection_pass=True,operator_comparison_qualified=record['operator_comparison_qualified'])),flush=True)
raise SystemExit(process.returncode or (0 if record['operator_comparison_qualified'] else 1))
