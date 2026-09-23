"""Require actual repeated gfx1151 results before selecting the explicit images."""
from pathlib import Path
import hashlib
import json
import subprocess

COMMIT = 'b07bf58e5140ea6e256c477f4aacfd39fc89695d'
PLAN_SHA = '1ff1ce5417880a0f3a1c64f08d0adfe63ef2febfa9596167afe414b92e326dce'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()


def validate_case(actual, planned):
    for key in ('name', 'tokens', 'kernel_set', 'candidate_source', 'reference'):
        assert actual[key] == planned[key], key
    assert actual['all_values_match'] and actual['inputs_unchanged'] and actual['guards_pass']
    rows = {r['surface']: r for r in actual['records']}
    assert len(rows) == len(actual['records']) == 8 and set(rows) == set(planned['expected'])
    for surface, row in rows.items():
        expected = planned['expected'][surface]
        assert row['bit_exact'] and row['sha256'] == row['expected_sha256'] == expected['sha256']
        assert row['bytes'] == expected['bytes'] and row['reference'] == expected
    assert len(actual['checkpoints']) == len(planned['checkpoints']) == (planned['tokens'] + 63) // 64
    for index, (row, expected) in enumerate(zip(actual['checkpoints'], planned['checkpoints'])):
        assert row['bit_exact']
        assert row['chunk'] == expected.get('chunk', index) == index
        assert row['first_position'] == expected.get('first_position', 64 * index) == 64 * index
        assert row['tokens'] == min(64, planned['tokens'] - 64 * index)
        assert row['incoming_bf16_sha256'] == row['expected_bf16_sha256'] == expected['sha256']


def qualify(base, checkout):
    root = base / 'native-gdn-layout-controls-windows-r2'
    assert sha(root / 'manifest.json') == PLAN_SHA
    plan = read(root / 'manifest.json')
    assert subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
                                   text=True, timeout=15).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain'],
                                       text=True, timeout=15).strip()
    compiled = read(checkout / 'native/linux_core_port/gb10_gdn_ordered_compile.json')
    for file, expected in compiled['source_files'].items():
        assert sha(checkout / file) == expected
    variant = plan['candidate_source_variants']['explicit_layout']
    assert {Path(f).name: h for f, h in compiled['source_files'].items()} == variant['source_files']
    assert sha(checkout / 'native/linux_core_port/gb10_gdn_ordered_images.inc') == compiled['embedded_image_sha256']
    for name, image in compiled['compiled'].items():
        planned = plan['images']['explicit_layout/' + name]
        assert image['sha256'] == planned['sha256'] and image['bytes'] == planned['bytes']
        assert sha(root / planned['file']) == planned['sha256']
    required = [root / 'outputs/result.json', root / 'run/run-record.json']
    if not all(p.is_file() for p in required):
        return dict(ready=False, reason='Actual native trial and clean completion record are not yet present',
                    source_commit=COMMIT, prepared_images_verified=True, remote_calls=0)
    report, run = map(read, required)
    assert run['host'].lower() == report['host'].lower() == 'baiying'
    assert run['reason'] == 'completed' and run['exit_code'] in (0, 6)
    assert run['host_checks_pass'] and all(run['host_checks'].values()) and not run['after_processes']
    assert report['cleanup_pass'] and report['guards_pass'] and report['exp2_table_unchanged']
    assert run['spec']['source_manifest_sha256'] == report['manifest_sha256'] == PLAN_SHA
    assert report['worker_sha256'] == plan['worker_sha256'] == sha(root / 'worker.py')
    for key in ('images', 'candidate_source_variants', 'execution_checkout', 'execution_checkout_commit', 'source_model_reference'):
        assert report[key] == plan[key]
    assert report['source_commit'] == plan['candidate_source_commit']
    assert len(report['cases']) == len(plan['cases']) == 24
    assert [(c['name'], c['tokens'], c['kernel_set']) for c in report['cases']] == [
        (c['name'], c['tokens'], c['kernel_set']) for c in plan['cases']]
    passed = []
    for actual, planned in zip(report['cases'], plan['cases']):
        if planned['kernel_set'] in ('u64', 'explicit_layout'):
            validate_case(actual, planned)
            local_case = root / 'outputs' / actual['name'] / 'result.json'
            assert read(local_case) == actual
            if actual['tokens'] == 64:
                image = root / 'outputs' / actual['name'] / 'w.bin'
                assert image.stat().st_size == planned['expected']['w']['bytes']
                assert sha(image) == planned['expected']['w']['sha256']
            passed.append(actual['name'])
    assert len(passed) == 10
    selected = [c['name'] for c in report['cases'] if c['kernel_set'] == 'explicit_layout']
    assert selected == [f'first64-explicit_layout-repeat{i}' for i in range(4)] + ['continuous7169-explicit_layout']
    return dict(ready=True, kind='explicit_image_selection_from_complete_24_case_native_trial',
                source_commit=COMMIT, native_overall_trial_pass=report['all_components_match'],
                native_exit_code=run['exit_code'], selected_kernel_set='explicit_layout',
                selected_cases=selected, interleaved_u64_control_cases=[c for c in passed if c not in selected],
                all_selected_values_bit_exact=True, all_selected_checkpoints_bit_exact=True,
                source_and_embedded_images_match_executed_explicit_layout=True,
                result_sha256=sha(required[0]), run_sha256=sha(required[1]),
                other_case_results=[dict(name=c['name'], passed=c['all_values_match'])
                                    for c in report['cases'] if c['kernel_set'] not in ('u64', 'explicit_layout')],
                full_product_correctness_still_required=True, inference_acceptance=False,
                performance_acceptance=False)
