"""Prepare paired original/explicit routed trials using unchanged model files."""
from pathlib import Path
import ast
import hashlib
import json
import os
import shutil

B = Path(__file__).resolve().parent
OLD = B / 'routed-selected-u32-windows-r1'
REFERENCE = B / 'routed-selected-explicit-r1'
TARGET = B / 'routed-selected-explicit-windows-r1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
reference = read(REFERENCE / 'result.json')
assert sha(REFERENCE / 'result.json') == 'c0985474f34e6b1eb610cb0a3cecac97485fe63fed37cce012ca22f7ece29779'
dispatch = read(REFERENCE / 'dispatch.json')
assert dispatch['returncode'] == 0 and not dispatch['gpu_processes_after']
assert reference['all_components_match'] and reference['source_files_unchanged']
assert reference['full_bf16_values'] == 603979776 and reference['sparse_bf16_values'] == 24558
assert len(reference['malformed_controls']) == 15 and all(r['pass'] for r in reference['malformed_controls'])
old = read(OLD / 'manifest.json')
old_result = read(OLD / 'outputs/result.json')
old_run = read(OLD / 'run/run-record.json')
assert old_result['all_components_match'] and old_result['cleanup_pass']
assert old_run['exit_code'] == 0 and old_run['host_checks_pass'] and not old_run['after_processes']
worker = B / 'routed_selected_explicit_windows_worker_r1.py'
tree = ast.parse(worker.read_text())
byte_constants = {x.value for x in ast.walk(tree) if isinstance(x, ast.Constant) and isinstance(x.value, bytes)}
assert {b'\xa5', b'\xa5\xa5', b'\0\0'} <= byte_constants
TARGET.mkdir()
(TARGET / 'source').mkdir()
for e in old['source_inputs']:
    p = OLD / e['file']
    assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
    shutil.copyfile(p, TARGET / e['file'])
for name in ['routed_explicit.py', 'explicit_dense.py']:
    p = REFERENCE / 'source' / name
    assert sha(p) == reference['source_hashes'][name]
    shutil.copyfile(p, TARGET / 'source' / name)
images = {}
for name, e in old['images'].items():
    p = OLD / e['file']
    assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
    filename = 'baseline-' + p.name
    os.link(p, TARGET / filename)
    images['baseline.' + name] = dict(e, file=filename,
        baseline_native_result_sha256=sha(OLD / 'outputs/result.json'))
for name, e in reference['compiled'].items():
    p = REFERENCE / e['file']
    assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
    assert e['layout_conversion_count'] == 0
    assert e['resources']['.vgpr_spill_count'] == e['resources']['.private_segment_fixed_size'] == ['0']
    assert e['metadata']['shared'] == 16
    filename = 'explicit-' + p.name
    os.link(p, TARGET / filename)
    images['explicit.' + name] = dict(file=filename, bytes=e['bytes'], sha256=e['sha256'],
        symbol=e['metadata']['name'], num_warps=e['metadata']['num_warps'], shared_bytes=e['metadata']['shared'],
        compilation_record=e, cuda_numerical_report_sha256=sha(REFERENCE / 'result.json'))
inputs = {}
for name, e in old['inputs'].items():
    p = OLD / 'inputs' / e['file']
    assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
    inputs[name] = dict(e, path=old['remote_directory'] + '/inputs/' + e['file'])
plan = dict(old)
projections = []
for previous, row in zip(old['projections'], reference['sources']):
    assert previous['name'] == row['name'] and previous['weight'] == row['weight']
    assert previous['output'] == row['original_output'] and previous['input'] == row['input']
    projections.append(dict(previous, input=inputs[previous['name']]))
shutil.copyfile(worker, TARGET / 'worker.py')
owner = read(B / 'runtime-attention-images-windows-r1/manifest.json')
plan.update(candidate_base_commit=reference['candidate_base_commit'], candidate_variant='explicit_layout',
    candidate_changes_relative_to_base_commit=True, images=images, inputs=inputs, projections=projections,
    worker_sha256=sha(TARGET / 'worker.py'), remote_directory='D:/projects/routed-selected-explicit-20260923-r1',
    active_owner_local_record=owner['active_owner_local_record'], active_owner_remote_record=owner['active_owner_remote_record'],
    native_timeout_seconds=900, cuda_numerical_report_sha256=sha(REFERENCE / 'result.json'),
    baseline_manifest_sha256=sha(OLD / 'manifest.json'), baseline_native_result_sha256=sha(OLD / 'outputs/result.json'),
    local_inputs_directory='routed-selected-u32-windows-r1/inputs',
    source_inputs=[dict(file=p.relative_to(TARGET).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                   for p in sorted((TARGET / 'source').iterdir())],
    expected_full_shape_comparisons=24, expected_queue_controls=17,
    malformed_input_controls=15, partial_shuffled_controls=2,
    exact_previous_baseline_images_unchanged=True, same_run_baseline_and_explicit_interleaved=True,
    executed=False, staged_on_baiying=False, inference_acceptance=False, performance_acceptance=False)
plan.pop('cuda_compile_report_sha256')
(TARGET / 'manifest.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(dict(manifest_sha256=sha(TARGET / 'manifest.json'), worker_sha256=plan['worker_sha256'],
    images=len(images), sources=len(plan['source_inputs']), full_shape_comparisons=24, queue_controls=17,
    native_executed=False)))
