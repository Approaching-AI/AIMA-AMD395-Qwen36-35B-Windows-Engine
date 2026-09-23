"""Prepare the native product manifest only after actual gfx1151 comparisons pass."""
from pathlib import Path
import hashlib
import json
import socket
import subprocess
import sys
from qualify_native_gdn_explicit_r1 import COMMIT, qualify

B = Path(__file__).resolve().parent
P = Path('/Users/jiawei-macmini/projects/AIMA-explicit-gdn-layout-candidate')
assert socket.gethostname().split('.')[0].lower() != 'baiying'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
selection = qualify(B, P)
if sys.argv[1:] == ['--check']:
    print(json.dumps(dict(component_numerical_gate_ready=selection['ready'],
                          selection=selection, remote_calls=0, executed=False)))
    sys.exit(0)
assert not sys.argv[1:] and selection['ready'], 'Actual repeated AMD comparisons must pass first; zero remote calls.'
old = read(B / 'linux-core-windows-ordered-gdn-u64-source-r1.json')
repo = 'P:/projects/AIMA-public-linux-core-explicit-gdn-20260923-r1'
plan = json.loads(json.dumps(old).replace(old['repo'], repo))
plan['source_commit'] = plan['run_spec']['repo_commit'] = COMMIT
prior = B / 'native-gdn-layout-controls-windows-r2/run/run-record.json'
numerical = B / 'native-gdn-layout-controls-windows-r2/outputs/result.json'
plan['prior_owner_record'] = 'P:/projects/native-gdn-layout-controls-20260923-r2/run/run-record.json'
plan['prior_owner_record_sha256'] = sha(prior)
paths = {x['path'] for x in old['build_source_inputs']}
paths.update(p.relative_to(P).as_posix() for p in (P / 'native/linux_core_port').glob('*') if p.is_file())
paths.update('native/providers/gdn_explicit_layout/' + name for name in ('ordered_pipeline.py', 'group16.py', 'exp2.py'))
paths.add('tools/compile_linux_core_gdn_explicit.py')
plan['build_source_inputs'] = [dict(path=p, bytes=(P / p).stat().st_size, sha256=sha(P / p)) for p in sorted(paths)]
assert plan['environment'] == old['environment']
assert plan['environment']['AIMA_PORT_NATIVE_GDN_PREFILL'] == '1'
plan['local_gdn_host_contract_sha256'] = sha(B / 'linux-core-gdn-explicit-host-r1/result.json')
plan['local_gdn_preparation_sha256'] = sha(B / 'linux-core-gdn-explicit-preparation-r1/prepare.json')
plan['native_component_selection'] = selection
plan['ordered_gdn_trial'].update(candidate_source_commit=COMMIT,
    native_component_result_sha256=sha(numerical), native_component_run_sha256=sha(prior),
    preparation_evidence_sha256=sha(P / 'benchmarks/correctness/ordered-gdn-explicit-runtime-preparation-20260923.json'),
    embedded_image_bytes=743344, explicit_layout=True, native_control_source_commit=old['source_commit'])
manifest = B / 'linux-core-windows-explicit-gdn-source-r1.json'
with manifest.open('x') as f:
    f.write(json.dumps(plan, indent=2) + '\n')
dispatcher = B / 'dispatch_linux_core_explicit_gdn_r1.py'
compile(dispatcher.read_bytes(), str(dispatcher), 'exec')
bundle = B / 'linux-core-windows-explicit-gdn-source-r1.bundle'
assert not bundle.exists()
subprocess.run(['git', '-C', str(P), 'bundle', 'create', str(bundle), old['source_commit'] + '..HEAD'],
               check=True, timeout=60)
stage = dict(source_commit=COMMIT, seed_repo=old['repo'], manifest=manifest.name,
             manifest_sha256=sha(manifest), dispatcher=dispatcher.name, dispatcher_sha256=sha(dispatcher),
             qualifier_sha256=sha(B / 'qualify_native_gdn_explicit_r1.py'),
             bundle=bundle.name, bundle_sha256=sha(bundle))
with (B / 'linux-core-windows-explicit-gdn-stage-r1.json').open('x') as f:
    f.write(json.dumps(stage, indent=2) + '\n')
print(json.dumps(stage))
