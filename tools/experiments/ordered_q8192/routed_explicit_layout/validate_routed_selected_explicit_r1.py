"""Bounded full-q8192 routed-expert comparisons on original model operands."""
from pathlib import Path
import hashlib
import json
import shutil
import socket
import subprocess
import sys

B = Path(__file__).resolve().parent
W = Path('/Users/jiawei-macmini/projects/AIMA-ordered-q8192-candidate')
O = B / 'routed-selected-explicit-r1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
assert socket.gethostname().split('.')[0].lower() != 'baiying'
owner = read(B / 'attention-explicit-layout-r2/dispatch.json')
assert owner['returncode'] == 0 and not owner['gpu_processes_after']
if sys.argv[1:] == ['--check']:
    print(json.dumps(dict(previous_reference_complete_and_clean=True, remote_calls=0, executed=False)))
    sys.exit(0)
assert not sys.argv[1:]
reference_name = 'qrt-gb10-routed-full-q8192-20260923-r1'
ref = B / reference_name
analysis = ref / 'reference-analysis.json'
assert sha(analysis) == '392c0313d3374f5eb344c96ffc48e73486106304268ca42732bb709116347938'
reference = read(analysis)
assert reference['reference_qualified'] and reference['original_576_tokens_and_full_first_logits_match']
assert reference['all_downloaded_inputs_verified'] and reference['input_output_hashes_equal_all_layer_reference']
old = read(B / 'routed-selected-u32-windows-r1/manifest.json')
source = read(B / 'routed-explicit-source-equivalence-r1.json')
assert source['kernel_arithmetic_and_addressing_ast_unchanged_after_removing_only_explicit_layouts']
assert source['candidate_sha256'] == sha(B / 'routed_selected_explicit_r1.py')
assert subprocess.check_output(['git', '-C', str(W), 'rev-parse', 'HEAD'], text=True, timeout=10).strip() == '2e7cf269b6885055c60c2b5c13c195744644bf3e'
sources = {'routed_explicit.py': B / 'routed_selected_explicit_r1.py',
    'worker.py': B / 'routed_selected_explicit_worker_r1.py',
    'explicit_dense.py': B / 'dense_selected_explicit_r1.py',
    'routed_selected.py': W / 'native/providers/ordered_q8192/routed_selected.py',
    'dense_selected.py': W / 'native/providers/ordered_q8192/dense_selected.py',
    'reference-analysis.json': analysis,
    'routed-explicit-source-equivalence-r1.json': B / 'routed-explicit-source-equivalence-r1.json'}
for p in sources.values():
    if p.suffix == '.py':
        compile(p.read_bytes(), str(p), 'exec')
for e in old['source_inputs']:
    assert sha(sources[Path(e['file']).name]) == e['sha256']
O.mkdir()
(O / 'source').mkdir()
for name, path in sources.items():
    shutil.copyfile(path, O / 'source' / name)
manifest = dict(source_hashes={name: sha(path) for name, path in sources.items()},
    candidate_base_commit='2e7cf269b6885055c60c2b5c13c195744644bf3e', candidate_outside_runtime=True,
    arithmetic_source_equivalence=source, original_reference=old['original_reference'],
    original_source_commit=reference['source_commit'], original_model_reference='/mnt/data/models/Qwen3.6-35B-A3B',
    model_index_sha256=old['model_index_sha256'], projections=old['projections'],
    previous_reference_dispatch_sha256=sha(B / 'attention-explicit-layout-r2/dispatch.json'),
    existing_control_methods='Pipeline.compare and Pipeline.malformed copied verbatim from the qualified full q8192 observer.',
    inference_acceptance=False, performance_acceptance=False, release_qualified=False)
(O / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
remote = '/home/qujing/qrt-routed-selected-explicit-20260923-r1'
opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
subprocess.run(['ssh', *opts, 'gb10-4t', 'python3 -'],
    input='from pathlib import Path\nPath(' + repr(remote) + ').mkdir()\n',
    capture_output=True, text=True, check=True, timeout=25)
for name, p in sources.items():
    subprocess.run(['scp', *opts, str(p), 'gb10-4t:' + remote + '/' + name],
                   capture_output=True, text=True, check=True, timeout=40)
subprocess.run(['scp', *opts, str(O / 'manifest.json'), 'gb10-4t:' + remote + '/manifest.json'],
               capture_output=True, text=True, check=True, timeout=30)
driver = 'remote=' + repr(remote) + '\nreference_name=' + repr(reference_name) + '\n' + r'''
from pathlib import Path
import json,socket,subprocess,time
p=Path(remote);name=p.name
probe=['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']
before=subprocess.check_output(probe,text=True,timeout=15).strip();assert not before,before
mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
assert mem['MemAvailable']>16*(1<<30)
command=['docker','run','--rm','--name',name,'--gpus','all','--network','none','--memory','10g','--shm-size','1g','--cpus','4',
 '-v',str(p)+':/work','-v','/home/qujing/'+reference_name+':/reference:ro',
 '-v','/mnt/data/models/Qwen3.6-35B-A3B:/models:ro',
 '--entrypoint','timeout','sha256:822f5c399ce6a08c0583302e0d82ff3938158b73977b6e1fbdf00ed72735a30d',
 '--signal=TERM','--kill-after=5','600','python3','/work/worker.py']
started=time.monotonic();returncode=None;timed_out=False
try:
 with (p/'container.log').open('x')as file:
  r=subprocess.run(command,stdout=file,stderr=subprocess.STDOUT,text=True,timeout=620);returncode=r.returncode
except subprocess.TimeoutExpired:
 timed_out=True;returncode=124
finally:subprocess.run(['docker','stop','-t','2',name],capture_output=True,text=True,timeout=15)
after=subprocess.check_output(probe,text=True,timeout=15).strip()
record=dict(host=socket.gethostname(),command_file=str(p/'worker.py'),command=command,returncode=returncode,
 timed_out=timed_out,wall_seconds=time.monotonic()-started,available_before=mem['MemAvailable'],
 gpu_processes_before=before,gpu_processes_after=after,inference_acceptance=False,performance_acceptance=False)
(p/'dispatch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));assert not after,after
'''
(O / 'driver.py').write_text(driver)
result = subprocess.run(['ssh', *opts, 'gb10-4t', 'python3 -'], input=driver, capture_output=True, text=True, timeout=650)
(O / 'controller.json').write_text(json.dumps(dict(returncode=result.returncode, stdout=result.stdout,
    stderr=result.stderr, driver_sha256=sha(O / 'driver.py')), indent=2) + '\n')
result.check_returncode()
dispatch = json.loads(result.stdout)
(O / 'dispatch.json').write_text(json.dumps(dispatch, indent=2) + '\n')
collection = []
for source, dest in [('container.log', 'container.log'), ('output/compilation.json', 'compilation.json'),
                     ('output/result.json', 'result.json')]:
    r = subprocess.run(['scp', *opts, 'gb10-4t:' + remote + '/' + source, str(O / dest)],
                       capture_output=True, text=True, timeout=40)
    collection.append(dict(file=dest, returncode=r.returncode, stderr=r.stderr))
(O / 'collection.json').write_text(json.dumps(collection, indent=2) + '\n')
print((O / 'container.log').read_text()[-9000:], flush=True)
if (O / 'compilation.json').exists():
    for name, e in read(O / 'compilation.json').items():
        for extension, expected in [('hsaco', e['sha256']), ('amdgcn', e['amdgcn_sha256']), ('ttgir', e['ttgir_sha256'])]:
            filename = name + '.' + extension
            subprocess.run(['scp', *opts, 'gb10-4t:' + remote + '/output/' + filename, str(O / filename)],
                           capture_output=True, text=True, check=True, timeout=40)
            assert sha(O / filename) == expected
assert all(e['returncode'] == 0 for e in collection), 'Incomplete experiment reports; original logs preserved.'
report = read(O / 'result.json')
assert report['manifest_sha256'] == sha(O / 'manifest.json')
print(json.dumps(dict(result_sha256=sha(O / 'result.json'), returncode=dispatch['returncode'],
    all_components_match=report['all_components_match'], full_bf16_values=report['full_bf16_values'],
    sparse_bf16_values=report['sparse_bf16_values'], amd_gpu_executed=False)), flush=True)
assert dispatch['returncode'] == 0 and report['all_components_match']
