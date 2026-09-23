"""Run full-shape attention controls on the available correctness reference."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import socket
import subprocess
import sys

B = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0].lower() != 'baiying'
W = Path('/Users/jiawei-macmini/projects/AIMA-ordered-q8192-candidate')
G = Path('/Users/jiawei-macmini/projects/AIMA-explicit-gdn-layout-candidate')
O = B / 'attention-explicit-layout-r1'
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
read = lambda p: json.loads(p.read_text())
owner = read(B / 'dense-selected-explicit-full-r1/dispatch.json')
assert owner['returncode'] == 0 and not owner['gpu_processes_after']
if sys.argv[1:] == ['--check']:
    print(json.dumps(dict(previous_reference_complete_and_clean=True, remote_calls=0, executed=False)))
    sys.exit(0)
assert not sys.argv[1:]
reference_name = 'qrt-gb10-attention-full-q8192-20260923-r1'
ref = B / reference_name
analysis = ref / 'reference-analysis.json'
assert sha(analysis) == '1ea3c9ef4a79e5879d30a629551fb0593b48ed162cff30be3e3de23d51ce33aa'
assert read(analysis)['reference_qualified'] and read(analysis)['original_576_tokens_match']
assert read(ref / 'dispatch.json')['host_guard_pass'] and not read(ref / 'dispatch.json')['gpu_processes_after']
assert subprocess.check_output(['git', '-C', str(W), 'rev-parse', 'HEAD'], text=True, timeout=10).strip() == '2e7cf269b6885055c60c2b5c13c195744644bf3e'
assert subprocess.check_output(['git', '-C', str(G), 'rev-parse', 'HEAD'], text=True, timeout=10).strip() == 'b07bf58e5140ea6e256c477f4aacfd39fc89695d'
sources = {
    'attention_explicit.py': B / 'attention_explicit_layout_r1.py',
    'worker.py': B / 'attention_explicit_layout_worker_r1.py',
    'explicit_group16.py': G / 'native/providers/gdn_explicit_layout/group16.py',
    'explicit_exp2.py': G / 'native/providers/gdn_explicit_layout/exp2.py',
    'explicit_dense.py': B / 'dense_selected_explicit_r1.py',
    'attention.py': W / 'native/providers/ordered_q8192/attention.py',
    'integer_u.py': W / 'native/providers/gdn/integer_u.py',
    'ordered_pipeline.py': W / 'native/providers/gdn/ordered_pipeline.py',
    'dense_selected.py': W / 'native/providers/ordered_q8192/dense_selected.py',
    'original-reference.json': analysis,
    'prior-component.json': B / 'attention-ordered-u32-r2/result.json',
    'rcp.bin': B / 'qrt-rcp-table-20260911-r2/sm121-attention-rcp-delta-i8.bin',
}
for path in sources.values():
    if path.suffix == '.py':
        compile(path.read_bytes(), str(path), 'exec')
# Verify helpers that only gained explicit coordinate layouts retain their AST.
class StripLayouts(ast.NodeTransformer):
    def visit_AnnAssign(self, node):
        if isinstance(node.annotation, ast.Attribute) and node.annotation.attr == 'constexpr':
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        node.keywords = [k for k in node.keywords if k.arg != 'layout']
        return node

def body(path, name):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    return ast.dump(StripLayouts().visit(node), include_attributes=False)

unchanged = ['_sum32_original', 'probability_kernel', '_reciprocal', 'selected_pv_kernel']
for name in unchanged:
    assert body(sources['attention_explicit.py'], name) == body(sources['attention.py'], name), name
assert body(sources['explicit_dense.py'], '_pair16') == body(sources['dense_selected.py'], '_pair16')
assert body(sources['explicit_group16.py'], '_group16') == body(sources['integer_u.py'], '_group16')
O.mkdir()
(O / 'source').mkdir()
for name, path in sources.items():
    if name != 'rcp.bin':
        shutil.copyfile(path, O / 'source' / name)
manifest = dict(source_hashes={name: sha(path) for name, path in sources.items()},
    candidate_base_commit='2e7cf269b6885055c60c2b5c13c195744644bf3e',
    reused_explicit_gdn_source_commit='b07bf58e5140ea6e256c477f4aacfd39fc89695d',
    candidate_outside_runtime=True, reference_dispatch_sha256=sha(ref / 'dispatch.json'),
    reference_case_sha256=sha(ref / 'capture/q8192-out512.json'), original_reference=read(analysis),
    query_starts=list(range(0, 8192, 128)), full_context_tokens=8192, query_slab=128,
    arithmetic_ast_unchanged_after_removing_only_explicit_layouts=unchanged + ['_pair16', '_group16'],
    qk_and_tiled_pv_changes='Separate coordinate layouts for left/right operands and output; ordered group16 arithmetic unchanged.',
    tables=dict(exp2=dict(bytes=183174448, sha256='f490940df2bd80421159a96424c3e922330b7ca120d5ae7b629a973b9183730b'),
                rcp=dict(bytes=8388640, sha256=sha(sources['rcp.bin']))),
    inference_acceptance=False, performance_acceptance=False, release_qualified=False)
(O / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
remote = '/home/qujing/qrt-attention-explicit-layout-20260923-r1'
opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
subprocess.run(['ssh', *opts, 'gb10-4t', 'python3 -'],
    input='from pathlib import Path\nPath(' + repr(remote) + ').mkdir()\n', capture_output=True,
    text=True, check=True, timeout=25)
for name, path in sources.items():
    subprocess.run(['scp', *opts, str(path), 'gb10-4t:' + remote + '/' + name], capture_output=True,
                   text=True, check=True, timeout=40)
subprocess.run(['scp', *opts, str(O / 'manifest.json'), 'gb10-4t:' + remote + '/manifest.json'],
               capture_output=True, text=True, check=True, timeout=30)
driver = 'remote=' + repr(remote) + '\nreference_name=' + repr(reference_name) + '\n' + r'''
from pathlib import Path
import json,socket,subprocess,time
p=Path(remote);name=p.name
probe=['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']
before=subprocess.check_output(probe,text=True,timeout=15).strip();assert not before,before
mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
assert mem['MemAvailable']>12*(1<<30)
command=['docker','run','--rm','--name',name,'--gpus','all','--network','none','--memory','8g','--shm-size','1g','--cpus','4',
 '-v',str(p)+':/work','-v','/home/qujing/'+reference_name+':/reference:ro',
 '-v','/home/qujing/qrt-native-gdn-ordered-pipeline-20260923-r1:/table:ro',
 '--entrypoint','timeout','sha256:822f5c399ce6a08c0583302e0d82ff3938158b73977b6e1fbdf00ed72735a30d',
 '--signal=TERM','--kill-after=5','600','python3','/work/worker.py']
started=time.monotonic();returncode=None;timed_out=False
try:
 with (p/'container.log').open('x')as file:
  r=subprocess.run(command,stdout=file,stderr=subprocess.STDOUT,text=True,timeout=620);returncode=r.returncode
except subprocess.TimeoutExpired:
 timed_out=True;returncode=124
finally:
 stopped=subprocess.run(['docker','stop','-t','2',name],capture_output=True,text=True,timeout=15)
after=subprocess.check_output(probe,text=True,timeout=15).strip()
record=dict(host=socket.gethostname(),command_file=str(p/'worker.py'),command=command,returncode=returncode,
 timed_out=timed_out,wall_seconds=time.monotonic()-started,available_before=mem['MemAvailable'],
 gpu_processes_before=before,gpu_processes_after=after,inference_acceptance=False,performance_acceptance=False)
(p/'dispatch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));assert not after,after
'''
(O / 'driver.py').write_text(driver)
result = subprocess.run(['ssh', *opts, 'gb10-4t', 'python3 -'], input=driver,
                        capture_output=True, text=True, timeout=650)
(O / 'controller.json').write_text(json.dumps(dict(returncode=result.returncode, stdout=result.stdout,
    stderr=result.stderr, driver_sha256=sha(O / 'driver.py')), indent=2) + '\n')
result.check_returncode()
dispatch = json.loads(result.stdout)
(O / 'dispatch.json').write_text(json.dumps(dispatch, indent=2) + '\n')
collection = []
for remote_name, local_name in [('container.log', 'container.log'), ('output/compilation.json', 'compilation.json'),
                               ('output/result.json', 'result.json')]:
    r = subprocess.run(['scp', *opts, 'gb10-4t:' + remote + '/' + remote_name, str(O / local_name)],
                       capture_output=True, text=True, timeout=40)
    collection.append(dict(file=local_name, returncode=r.returncode, stderr=r.stderr))
(O / 'collection.json').write_text(json.dumps(collection, indent=2) + '\n')
print((O / 'container.log').read_text()[-9000:], flush=True)
if (O / 'compilation.json').exists():
    compiled = read(O / 'compilation.json')
    for name, e in compiled.items():
        for extension, expected in [('hsaco', e['sha256']), ('amdgcn', e['amdgcn_sha256']), ('ttgir', e['ttgir_sha256'])]:
            filename = name + '.' + extension
            subprocess.run(['scp', *opts, 'gb10-4t:' + remote + '/output/' + filename, str(O / filename)],
                           capture_output=True, text=True, check=True, timeout=40)
            assert sha(O / filename) == expected
assert all(x['returncode'] == 0 for x in collection), 'Reference experiment did not produce every report; logs preserved.'
report = read(O / 'result.json')
assert report['manifest_sha256'] == sha(O / 'manifest.json')
print(json.dumps(dict(result_sha256=sha(O / 'result.json'), returncode=dispatch['returncode'],
    all_components_match=report['all_components_match'], qk_fp32_values=report['qk_fp32_values'],
    context_bf16_values=report['context_bf16_values'], amd_gpu_executed=False)), flush=True)
assert dispatch['returncode'] == 0 and report['all_components_match']
