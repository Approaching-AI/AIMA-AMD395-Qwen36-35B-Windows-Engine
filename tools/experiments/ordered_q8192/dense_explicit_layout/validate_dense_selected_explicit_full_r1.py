from pathlib import Path
import hashlib, json, socket, subprocess, sys

B = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0].lower() != 'baiying'
P = Path('/Users/jiawei-macmini/projects/AIMA-ordered-q8192-candidate')
reference_name = 'qrt-gb10-dense-full-q8192-20260923-r1'
reference_root = B / reference_name
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
ready = (reference_root / 'dispatch.json').exists()
if ready:
    dispatch = read(reference_root / 'dispatch.json')
    ready = dispatch['reference_boundary_qualified'] and dispatch['host_guard_pass'] and not dispatch['gpu_processes_after']
if sys.argv[1:] == ['--check']:
    print(json.dumps(dict(original_full_reference_ready=bool(ready), remote_calls=0)))
    raise SystemExit(0)
assert not sys.argv[1:] and ready
O = B / 'dense-selected-explicit-full-r1'
O.mkdir()
reference_path = reference_root / 'capture/q8192-out512.json'
reference = read(reference_path)
assert reference['control_pass'] and reference['worker']['dense_capture']['complete']
entries = reference['worker']['dense_capture']['files']
projections = []
for name, x, y, weight, layout in [('qkv', 'input', 'qkv', 'in_proj_qkv', 'transposed'),
        ('z', 'input', 'z', 'in_proj_z', 'transposed'), ('linear-out', 'gated', 'out', 'out_proj', 'contiguous')]:
    projections.append(dict(name=name, input=entries[x], output=entries[y],
        weight='model.language_model.layers.0.linear_attn.' + weight + '.weight', layout=layout))
sources = {'kernel.py': B / 'dense_selected_explicit_r1.py', 'worker.py': B / 'dense_selected_u32_full_worker_r1.py',
           'small-component.json': B / 'dense-selected-explicit-r1/result.json'}
for path in sources.values():
    if path.suffix == '.py':
        compile(path.read_bytes(), str(path), 'exec')
manifest = dict(source_hashes={name: sha(path) for name, path in sources.items()},
    candidate_base_commit=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True, timeout=10).strip(),
    candidate_outside_runtime=True, reference_case_sha256=sha(reference_path),
    reference_dispatch_sha256=sha(reference_root / 'dispatch.json'),
    original_prompt=reference['prompt'], original_outputs=reference['output_token_ids'],
    original_first_logit=reference['first_token_raw_logit'], projections=projections,
    original_full_reference_qualified=True, product_performance_acceptance=False)
(O / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
remote = '/home/qujing/qrt-dense-selected-explicit-full-20260923-r1'
opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
subprocess.run(['ssh', *opts, 'gb10-4t', 'python3 -'], input='from pathlib import Path\nPath(' + repr(remote) + ').mkdir()\n',
    text=True, capture_output=True, check=True, timeout=25)
for name, path in sources.items():
    subprocess.run(['scp', *opts, str(path), 'gb10-4t:' + remote + '/' + name], capture_output=True, text=True, check=True, timeout=30)
subprocess.run(['scp', *opts, str(O / 'manifest.json'), 'gb10-4t:' + remote + '/manifest.json'], capture_output=True, text=True, check=True, timeout=30)
driver = 'remote=' + repr(remote) + '\nreference_name=' + repr(reference_name) + '\n' + r'''
from pathlib import Path
import json,socket,subprocess,time
p=Path(remote);name=p.name
probe=['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']
before=subprocess.check_output(probe,text=True,timeout=15).strip();assert not before,before
memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
assert memory['MemAvailable']>12*(1<<30)
command=['docker','run','--rm','--name',name,'--gpus','all','--network','none','--memory','8g','--shm-size','1g','--cpus','4',
 '-v',str(p)+':/work','-v','/home/qujing/'+reference_name+':/reference:ro',
 '-v','/mnt/data/models/Qwen3.6-35B-A3B:/models:ro','--entrypoint','timeout',
 'sha256:822f5c399ce6a08c0583302e0d82ff3938158b73977b6e1fbdf00ed72735a30d','--signal=TERM','--kill-after=5','420','python3','/work/worker.py']
started=time.monotonic()
try:
 with (p/'container.log').open('w')as stream:
  result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,text=True,timeout=440)
finally:subprocess.run(['docker','stop','-t','2',name],capture_output=True,text=True,timeout=15)
after=subprocess.check_output(probe,text=True,timeout=15).strip()
record=dict(host=socket.gethostname(),command_file=str(p/'worker.py'),command=command,returncode=result.returncode,
 wall_seconds=time.monotonic()-started,available_before=memory['MemAvailable'],
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
for remote_file in ('container.log', 'output/result.json'):
    r = subprocess.run(['scp', *opts, 'gb10-4t:' + remote + '/' + remote_file, str(O / Path(remote_file).name)],
        capture_output=True, text=True, timeout=45)
    r.check_returncode()
report = read(O / 'result.json')
assert report['manifest_sha256'] == sha(O / 'manifest.json')
assert report['total_bf16_values'] == 352321536
print((O / 'container.log').read_text()[-10000:], flush=True)
print(json.dumps(dict(result_sha256=sha(O / 'result.json'), returncode=dispatch['returncode'],
    all_original_values_match=report['all_original_bf16_values_match'], values=report['total_bf16_values'],
    amd_gpu_executed=False)), flush=True)
assert dispatch['returncode'] == 0 and report['all_original_bf16_values_match']
