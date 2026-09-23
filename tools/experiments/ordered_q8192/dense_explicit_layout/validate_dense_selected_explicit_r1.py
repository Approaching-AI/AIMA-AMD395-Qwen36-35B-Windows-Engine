from pathlib import Path
import hashlib, json, socket, subprocess, sys

B = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0].lower() != 'baiying'
P = Path('/Users/jiawei-macmini/projects/AIMA-ordered-q8192-candidate')
O = B / 'dense-selected-explicit-r1'
owner = B / 'qrt-gb10-explicit-gdn-layers-20260923-r1/dispatch.json'
ready = owner.exists()
if ready:
    prior = json.loads(owner.read_text())
    ready = not prior['gpu_processes_after'] and prior['host_guard_pass'] and prior['container_status'].startswith('false ')
if sys.argv[1:] == ['--check']:
    print(json.dumps(dict(previous_reference_complete_and_clean=ready, remote_calls=0, executed=False)))
    raise SystemExit(0)
assert not sys.argv[1:] and ready, 'Current reference model owner must complete; zero remote calls.'
O.mkdir()
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
sources = {'kernel.py': B / 'dense_selected_explicit_r1.py', 'worker.py': B / 'dense_selected_explicit_worker_r1.py'}
for path in sources.values():
    compile(path.read_bytes(), str(path), 'exec')
reference_path = B / 'qrt-gb10-q8192-first64-20260922-r1/capture/q8192-out512.json'
reference = json.loads(reference_path.read_text())
assert reference['control_pass'] and reference['full_matrix_case_pass']
entries = {v['label']: v for v in reference['worker']['runtime_boundaries']['files'].values() if v['transaction'] == 0}
manifest = dict(source_hashes={k: sha(p) for k, p in sources.items()},
    candidate_base_commit=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True, timeout=10).strip(),
    candidate_outside_runtime=True, reference_case_sha256=sha(reference_path),
    original_prompt=reference['prompt'], original_outputs=reference['output_token_ids'],
    original_first_logit=reference['first_token_raw_logit'],
    original_transaction=reference['worker']['runtime_boundaries']['transactions'][0],
    integer_arithmetic_source_sha256=sha(P / 'native/providers/gdn/integer_u.py'),
    projections=[dict(name='qkv-k2048', input=entries['layer-00-input-rmsnorm'],
        output=entries['linear-00-qkv'], weight='model.language_model.layers.0.linear_attn.in_proj_qkv.weight'),
        dict(name='linear-out-k4096', input=entries['linear-00-gated'],
        output=entries['linear-00-output-projection'], weight='model.language_model.layers.0.linear_attn.out_proj.weight')],
    product_performance_acceptance=False, note='71 actual q8192 rows; full original output widths, shuffled selected indices, empty/partial/multiple queues and both physical weight layouts.')
(O / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
remote = '/home/qujing/qrt-dense-selected-explicit-20260923-r1'
opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
subprocess.run(['ssh', *opts, 'gb10-4t', 'python3 -'], input='from pathlib import Path\nPath(' + repr(remote) + ').mkdir()\n',
    text=True, capture_output=True, check=True, timeout=25)
for name, path in sources.items():
    subprocess.run(['scp', *opts, str(path), 'gb10-4t:' + remote + '/' + name], capture_output=True, text=True, check=True, timeout=30)
subprocess.run(['scp', *opts, str(O / 'manifest.json'), 'gb10-4t:' + remote + '/manifest.json'], capture_output=True, text=True, check=True, timeout=30)
driver = 'remote=' + repr(remote) + '\n' + r'''
from pathlib import Path
import hashlib,json,socket,subprocess,time
p=Path(remote);name=p.name
probe=['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']
before=subprocess.check_output(probe,text=True,timeout=15).strip();assert not before,before
memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
assert memory['MemAvailable']>12*(1<<30)
command=['docker','run','--rm','--name',name,'--gpus','all','--network','none','--memory','8g','--shm-size','1g','--cpus','4',
 '-v',str(p)+':/work','-v','/home/qujing/qrt-gb10-q8192-first64-20260922-r1:/reference:ro',
 '-v','/mnt/data/models/Qwen3.6-35B-A3B:/models:ro','--entrypoint','timeout',
 'sha256:822f5c399ce6a08c0583302e0d82ff3938158b73977b6e1fbdf00ed72735a30d','--signal=TERM','--kill-after=5','420','python3','/work/worker.py']
started=time.monotonic()
try:result=subprocess.run(command,capture_output=True,text=True,timeout=440)
finally:subprocess.run(['docker','stop','-t','2',name],capture_output=True,text=True,timeout=15)
after=subprocess.check_output(probe,text=True,timeout=15).strip()
record=dict(host=socket.gethostname(),command_file=str(p/'worker.py'),command=command,returncode=result.returncode,
 stdout=result.stdout,stderr=result.stderr,wall_seconds=time.monotonic()-started,available_before=memory['MemAvailable'],
 gpu_processes_before=before,gpu_processes_after=after,inference_acceptance=False,performance_acceptance=False)
(p/'dispatch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));assert not after,after
'''
(O / 'driver.py').write_text(driver)
run = subprocess.run(['ssh', *opts, 'gb10-4t', 'python3 -'], input=driver, capture_output=True, text=True, timeout=470)
(O / 'controller.json').write_text(json.dumps(dict(returncode=run.returncode, stdout=run.stdout, stderr=run.stderr,
    driver_sha256=sha(O / 'driver.py')), indent=2) + '\n')
run.check_returncode()
dispatch = json.loads(run.stdout)
(O / 'dispatch.json').write_text(json.dumps(dispatch, indent=2) + '\n')
print(dispatch['stdout'], flush=True)
print(dispatch['stderr'][-9000:], flush=True)
download = subprocess.run(['scp', *opts, 'gb10-4t:' + remote + '/output/result.json', str(O)],
    capture_output=True, text=True, timeout=30)
download.check_returncode()
report = json.loads((O / 'result.json').read_text())
assert report['manifest_sha256'] == sha(O / 'manifest.json')
for image in report['compiled'].values():
    subprocess.run(['scp', *opts, 'gb10-4t:' + remote + '/output/' + image['file'], str(O)],
        capture_output=True, text=True, check=True, timeout=30)
    assert sha(O / image['file']) == image['sha256']
    for kind in ('amdgcn','ttgir'):
        name=Path(image['file']).with_suffix('.'+kind).name
        subprocess.run(['scp',*opts,'gb10-4t:'+remote+'/output/'+name,str(O)],capture_output=True,text=True,check=True,timeout=30)
print(json.dumps(dict(result_sha256=sha(O / 'result.json'), returncode=dispatch['returncode'],
    all_original_values_match=report['all_original_bf16_values_match'], values=report['total_bf16_values'],
    amd_compiled=len(report['compiled']), amd_gpu_executed=False)), flush=True)
assert dispatch['returncode'] == 0 and report['all_original_bf16_values_match']
