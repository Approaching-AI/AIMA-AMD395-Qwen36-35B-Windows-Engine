"""Use the unchanged native dense harness with the explicit-layout images."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

B = Path(__file__).resolve().parent
old = B / 'dense-selected-u32-windows-r1'
target = B / 'dense-selected-explicit-windows-r1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
small_dir = B / 'dense-selected-explicit-r1'
full_dir = B / 'dense-selected-explicit-full-r1'
small, full = read(small_dir / 'result.json'), read(full_dir / 'result.json')
assert small['all_original_bf16_values_match'] and full['all_original_bf16_values_match']
assert full['total_bf16_values'] == 352321536 and small['total_bf16_values'] == 4362240
assert full['source_hashes']['kernel.py'] == small['source_hashes']['kernel.py'] == sha(B / 'dense_selected_explicit_r1.py')
for directory in (small_dir, full_dir):
    dispatch = read(directory / 'dispatch.json')
    assert dispatch['returncode'] == 0 and not dispatch['gpu_processes_after']
plan = read(old / 'manifest.json')
target.mkdir()
(target / 'inputs').mkdir()
for item in plan['source_inputs']:
    source = old / item['file']
    assert sha(source) == item['sha256'] and source.stat().st_size == item['bytes']
    destination = target / item['file']
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
shutil.copyfile(B / 'dense_selected_explicit_r1.py', target / 'source/candidate.py')
shutil.copyfile(old / 'worker.py', target / 'worker.py')
assert sha(target / 'worker.py') == plan['worker_sha256']
inputs = {x['input']['file']: x['input'] for x in plan['projections']}
for name, item in inputs.items():
    path = old / 'inputs' / name
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    os.link(path, target / 'inputs' / name)
images = {}
for batch, item in small['compiled'].items():
    path = small_dir / item['file']
    assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
    assert item['metadata']['shared'] == item['layout_conversion_count'] == item['shared_memory_instructions'] == 0
    os.link(path, target / path.name)
    images[batch] = dict(file=path.name, bytes=item['bytes'], sha256=item['sha256'],
        symbol=item['metadata']['name'], num_warps=item['metadata']['num_warps'], shared_bytes=0,
        original_compilation_record=item, numerical_proof_sha256=sha(full_dir / 'result.json'))
assert full['sources'] == read(B / 'dense-selected-u32-full-r1/result.json')['sources']
plan['baseline']['source_base_commit'] = plan['candidate_base_commit']
plan.update(candidate_base_commit='2e7cf269b6885055c60c2b5c13c195744644bf3e',
    candidate_source_changes_relative_to_base_commit=True, candidate_variant='explicit_layout',
    previous_native_plan_sha256=sha(old / 'manifest.json'),
    previous_actual_native_report_sha256=sha(old / 'outputs/result.json'),
    arithmetic_source_equivalence_sha256=sha(B / 'dense-explicit-source-equivalence-r1.json'),
    active_owner_local_record='q1-cache-capture-prefix256k-long-prefix262144-suffix1024-out512-mode1-r1/run-record.json',
    active_owner_remote_record='P:/projects/AIMA-public-q1-cache-capture-20260923-r1/build/q1-cache-capture-prefix256k-long-prefix262144-suffix1024-out512-mode1-r1/run-record.json',
    remote_directory='D:/projects/dense-selected-explicit-20260923-r1', images=images,
    source_inputs=[dict(file=p.relative_to(target).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                   for p in sorted((target / 'source').rglob('*')) if p.is_file()],
    native_harness_and_baseline_unchanged=True, staged_on_baiying=False, executed=False)
plan['original_reference']['candidate_cuda_numerical_report_sha256'] = sha(full_dir / 'result.json')
(target / 'manifest.json').write_text(json.dumps(plan, indent=2) + '\n')
script = (B / 'dispatch_dense_selected_u32_windows_r1.py').read_text()
script = script.replace('dense-selected-u32-windows-r1', target.name).replace(
    '8915b50c1dcd5b1e1b66c5da39a6340fcb3488fc1d0e199ec4661ede505d5a65', sha(target / 'manifest.json'))
script = script.replace('current selected replay versus unsigned32 AOT', 'current selected replay versus explicit-layout unsigned32 AOT')
dispatcher = B / 'dispatch_dense_selected_explicit_windows_r1.py'
compile(script, str(dispatcher), 'exec')
dispatcher.write_text(script)
print(json.dumps(dict(manifest_sha256=sha(target / 'manifest.json'), images=3,
    source_files=len(plan['source_inputs']), native_cases=54,
    dispatcher_sha256=sha(dispatcher), native_executed=False)))
