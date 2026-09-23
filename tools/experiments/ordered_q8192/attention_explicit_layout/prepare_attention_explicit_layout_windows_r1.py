"""Prepare paired full-q8192 old/explicit attention measurements, without dispatch."""
from pathlib import Path
import hashlib
import json
import os
import shutil

B = Path(__file__).resolve().parent
OLD = B / 'attention-ordered-u32-windows-r1'
REFERENCE = B / 'attention-explicit-layout-r2'
TARGET = B / 'attention-explicit-layout-windows-r1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
reference = read(REFERENCE / 'result.json')
assert sha(REFERENCE / 'result.json') == 'e78a9f09b0fae55c1bd369a98ba1de0aa8b5b75bf23bf0b0cb06ae102bb2a6ac'
dispatch = read(REFERENCE / 'dispatch.json')
assert dispatch['returncode'] == 0 and not dispatch['gpu_processes_after']
assert reference['all_components_match'] and reference['sources_unchanged'] and reference['guards_pass']
assert reference['qk_fp32_values'] == 1073741824 and reference['context_bf16_values'] == 100663296
old = read(OLD / 'manifest.json')
old_run = read(OLD / 'run/run-record.json')
old_result = read(OLD / 'outputs/result.json')
assert old_result['all_components_match'] and old_result['cleanup_pass']
assert old_run['exit_code'] == 0 and old_run['host_checks_pass'] and not old_run['after_processes']
TARGET.mkdir()
(TARGET / 'source').mkdir()
for e in old['source_inputs']:
    p = OLD / e['file']
    assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
    shutil.copyfile(p, TARGET / e['file'])
for name in ['attention_explicit.py', 'explicit_group16.py', 'explicit_exp2.py', 'explicit_dense.py']:
    p = REFERENCE / 'source' / name
    assert sha(p) == reference['source_hashes'][name]
    shutil.copyfile(p, TARGET / 'source' / name)
images = {}
for name, e in old['images'].items():
    p = OLD / e['file']
    assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
    filename = 'baseline-' + p.name
    os.link(p, TARGET / filename)
    images['pack' if name == 'pack' else 'baseline.' + name] = dict(e, file=filename,
        baseline_native_result_sha256=sha(OLD / 'outputs/result.json'))
for name, e in reference['compiled'].items():
    p = REFERENCE / e['file']
    assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
    assert e['layout_conversion_count'] == 0
    assert e['resources']['.vgpr_spill_count'] == e['resources']['.private_segment_fixed_size'] == ['0']
    if name != 'pv':
        assert e['metadata']['shared'] == e['shared_memory_instructions'] == 0
    filename = 'explicit-' + p.name
    os.link(p, TARGET / filename)
    images['explicit.' + name] = dict(file=filename, bytes=e['bytes'], sha256=e['sha256'],
        symbol=e['metadata']['name'], num_warps=e['metadata']['num_warps'], shared_bytes=e['metadata']['shared'],
        compilation_record=e, cuda_numerical_report_sha256=sha(REFERENCE / 'result.json'))
cases = []
for original, row in zip(old['cases'], reference['records']):
    assert original['query_start'] == row['query_start'] and original['queries'] == 128
    assert original['qk_original_mma']['sha256'] == row['qk']['expected']['sha256']
    assert original['context_original']['sha256'] == row['pv']['selected32']['expected']['sha256']
    assert row['probability']['bit_mismatches'] == row['scales']['bit_mismatches'] == 0
    cases.append(dict(original, probability_sha256=row['probability']['expected']['sha256'],
                      scales_sha256=row['scales']['expected']['sha256']))
assert len(cases) == 64
inputs = {}
for name, e in old['inputs'].items():
    p = OLD / 'inputs' / e['file']
    assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
    inputs[name] = dict(e, path=old['remote_directory'] + '/inputs/' + e['file'])
shutil.copyfile(B / 'attention_explicit_layout_windows_worker_r1.py', TARGET / 'worker.py')
plan = dict(old)
owner = read(B / 'runtime-attention-images-windows-r1/manifest.json')
plan.update(candidate_base_commit=reference['candidate_base_commit'], candidate_variant='explicit_layout',
    candidate_changes_relative_to_base_commit=True, images=images, inputs=inputs, cases=cases,
    worker_sha256=sha(TARGET / 'worker.py'), remote_directory='D:/projects/attention-explicit-layout-20260923-r1',
    active_owner_local_record=owner['active_owner_local_record'], active_owner_remote_record=owner['active_owner_remote_record'],
    native_timeout_seconds=300, cuda_numerical_report_sha256=sha(REFERENCE / 'result.json'),
    baseline_manifest_sha256=sha(OLD / 'manifest.json'), baseline_native_result_sha256=sha(OLD / 'outputs/result.json'),
    local_inputs_directory='attention-ordered-u32-windows-r1/inputs',
    source_inputs=[dict(file=p.relative_to(TARGET).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                   for p in sorted((TARGET / 'source').iterdir())],
    variants=dict(baseline=dict(qk='baseline.qk', probability='baseline.probability', pv='baseline.pv', pv_kind='selected'),
        explicit_selected=dict(qk='explicit.qk', probability='explicit.probability', pv='explicit.pv', pv_kind='selected'),
        explicit_tiled8x8=dict(qk='explicit.qk', probability='explicit.probability', pv='explicit.tiled8x8', pv_kind='tiled', pv_bn=8),
        explicit_tiled8x16=dict(qk='explicit.qk', probability='explicit.probability', pv='explicit.tiled8x16', pv_kind='tiled', pv_bn=16)),
    expected_full_q8192_passes=8, expected_slab_comparisons=512, expected_selected_queue_controls=4,
    expected_context_host_only=True, exact_previous_baseline_images_unchanged=True,
    executed=False, staged_on_baiying=False, inference_acceptance=False, performance_acceptance=False)
(TARGET / 'manifest.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(dict(manifest_sha256=sha(TARGET / 'manifest.json'), worker_sha256=plan['worker_sha256'],
    images=len(images), sources=len(plan['source_inputs']), full_q8192_passes=8, slabs=512, native_executed=False)))
