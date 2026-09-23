"""Select only numerically qualified images for a subsequent q8192 product run.

This checks recorded component evidence. It executes no remote command and
does not treat component timings as product performance acceptance.
"""
from pathlib import Path
import hashlib
import json
import subprocess

COMMIT = 'c85259254b805788dc80680caad37685708c98a8'
PLANS = {
    'dense': ('dense-selected-explicit-windows-r1', 'eb016da03377684388e3a1f79a71bf6500a550ec630d9059f8857e43680aca17'),
    'attention': ('attention-explicit-layout-windows-r1', 'c9c6166d37626ff9dada5da4d3bea09dba8da8ebe5a6a348de885084c8175a00'),
    'routed': ('routed-selected-explicit-windows-r1', '82d3aaa6e4667f28b7a7b78968353fe9988835a15abc3bf5acfd674f7e15564b'),
}
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()


def load_component(base, family):
    directory, expected = PLANS[family]
    root = base / directory
    assert sha(root / 'manifest.json') == expected
    plan = read(root / 'manifest.json')
    assert sha(root / 'worker.py') == plan['worker_sha256']
    for entry in [*plan['images'].values(), *plan['source_inputs']]:
        path = root / entry['file']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
    if not all((root / path).is_file() for path in ('outputs/result.json', 'run/run-record.json')):
        return plan, None, dict(ready=False, reason='Actual native result and clean completion are pending',
                                family=family, manifest_sha256=expected)
    result, run = read(root / 'outputs/result.json'), read(root / 'run/run-record.json')
    assert run['host'].lower() == result['host'].lower() == 'baiying'
    assert run['reason'] == 'completed' and run['exit_code'] in (0, 6)
    assert run['host_checks_pass'] and all(run['host_checks'].values()) and not run['after_processes']
    assert result['cleanup_pass'] and result['source_files_unchanged']
    assert result['manifest_sha256'] == run['spec']['source_manifest_sha256'] == expected
    assert result['worker_sha256'] == plan['worker_sha256']
    assert result['images'] == plan['images'] and result['original_reference'] == plan['original_reference']
    assert result['candidate_base_commit'] == plan['candidate_base_commit']
    assert result['execution_checkout_commit'] == plan['execution_checkout_commit']
    return plan, result, dict(ready=True, family=family, manifest_sha256=expected,
        result_sha256=sha(root / 'outputs/result.json'), run_sha256=sha(root / 'run/run-record.json'),
        entire_trial_pass=result['all_components_match'], exit_code=run['exit_code'],
        inference_acceptance=False, performance_acceptance=False)


def validate_dense(plan, result, batch):
    assert batch in (32, 64, 128) and len(result['records']) == 54
    planned = {p['name']: p for p in plan['projections']}
    assert len(result['sources']) == len(planned) == 3
    for source in result['sources']:
        expected = planned[source['name']]
        assert source['input'] == expected['input'] and source['output'] == expected['output']
        assert source['weight']['sha256'] == expected['weight_tensor']['sha256']
        assert source['inputs_unchanged'] and source['guards_pass']
        assert {p['mode'] for p in source['phases']} == {'selected', 'complete'}
        assert all(p['queues_unchanged'] for p in source['phases'])
    methods = {'baseline', f'u32-{batch}'}
    selected = [r for r in result['records'] if r['method'] in methods]
    expected_keys = {(name, mode, repeat, method) for name in planned for mode in ('selected', 'complete')
                     for repeat in range(3 if mode == 'selected' else 1) for method in methods}
    keys = [(r['projection'], r['queue_mode'], r['repeat'], r['method']) for r in selected]
    assert len(keys) == len(set(keys)) == len(expected_keys) == 24 and set(keys) == expected_keys
    for row in selected:
        expected = planned[row['projection']]
        assert row['all_original_values_match']
        assert row['output_sha256'] == row['original_output_sha256'] == expected['output']['sha256']
        assert row['layout'] == expected['layout'] and row['tokens'] == 8192
        assert row['output_bf16_values'] * 2 == expected['output']['bytes']
    return selected


def validate_attention(plan, result):
    assert len(result['records']) == 512 and len(result['passes']) == 8
    assert result['inputs_tables_queues_debug_unchanged'] and result['packed_value_unchanged'] and result['guards_pass']
    variants = {'baseline', 'explicit_selected'}
    selected = [r for r in result['records'] if r['variant'] in variants]
    keys = [(r['variant'], r['repeat'], r['query_start']) for r in selected]
    expected_keys = {(variant, repeat, start) for variant in variants for repeat in range(2) for start in range(0, 8192, 128)}
    assert len(keys) == len(set(keys)) == len(expected_keys) == 256 and set(keys) == expected_keys
    cases = {r['query_start']: r for r in plan['cases']}
    for row in selected:
        case = cases[row['query_start']]
        assert row['queries'] == 128 and row['qk_fp32_values'] == 128 * 16 * 8192
        expected = {'qk': case['qk_original_mma']['sha256'], 'context': case['context_original']['sha256'],
                    'probability': case['probability_sha256'], 'scales': case['scales_sha256']}
        for name, digest in expected.items():
            assert row[name + '_bit_exact'] and row[name + '_sha256'] == row['expected_' + name + '_sha256'] == digest
    passes = [r for r in result['passes'] if r['variant'] in variants]
    assert len(passes) == 4 and {(r['variant'], r['repeat']) for r in passes} == {(v, i) for v in variants for i in range(2)}
    for row in passes:
        assert row['whole_context_bit_exact'] and row['guards_pass']
        assert row['whole_context_sha256'] == row['expected_context_sha256'] == plan['expected_context_sha256']
    queues = result['selected_queue_controls']
    assert len(queues) == 4 and {(r['query_start'], r['selected_count']) for r in queues} == {(q, n) for q in (0, 8064) for n in (0, 137)}
    assert all(all(r[k] for k in ('whole_output_including_unselected_poison_matches', 'input_surfaces_match_original',
        'queues_unchanged', 'guards_pass', 'expected_values_from_original_hash_qualified_same_run_baseline')) for r in queues)
    return passes


def validate_routed(plan, result, batch):
    assert batch in (32, 64, 128) and len(result['records']) == 24
    planned = {p['name']: p for p in plan['projections']}
    assert len(result['sources']) == len(planned) == 2
    for source in result['sources']:
        expected = planned[source['name']]
        assert source['input'] == expected['input'] and source['output'] == expected['output']
        assert source['weight']['sha256'] == expected['weight']['sha256']
        assert source['inputs_queues_debug_unchanged'] and source['guards_pass']
    selected = [r for r in result['records'] if r['batch'] == batch]
    keys = [(r['projection'], r['variant'], r['repeat']) for r in selected]
    expected_keys = {(name, variant, repeat) for name in planned for variant in ('baseline', 'explicit') for repeat in range(2)}
    assert len(keys) == len(set(keys)) == len(expected_keys) == 8 and set(keys) == expected_keys
    for row in selected:
        assert row['original_bit_exact'] and row['invalid_flags'] == 0 and row['guards_pass']
        assert row['output_sha256'] == row['original_output_sha256'] == planned[row['projection']]['output']['sha256']
        assert row['bf16_values'] * 2 == planned[row['projection']]['output']['bytes']
        assert row['logical_tokens'] == 8192 and row['logical_routes'] == 65536 and row['empty_first_and_last_window']
    controls = result['queue_controls']
    assert len(controls) == 17 and len({(r['projection'], r['case']) for r in controls}) == 17
    assert all(r['pass'] and r['actual_flags'] == r['expected_flags'] and r['output_sha256'] == r['expected_output_sha256']
        and all(r[k] for k in ('output_and_unselected_poison_match', 'invalid_expert_zero', 'inputs_unchanged', 'guards_pass')) for r in controls)
    assert {(r['projection'], r['case']) for r in controls if r['case'] == 'partial_shuffled'} == {(name, 'partial_shuffled') for name in planned}
    return selected


def qualify(base, checkout, dense_batch=64, routed_batch=64, attention=True, gdn=True):
    assert subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True, timeout=15).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain'], text=True, timeout=15).strip()
    composition = read(checkout / 'native/linux_core_port/explicit_q8192_compile.json')
    for name, expected in composition['source_files'].items():
        assert sha(checkout / name) == expected
    for image in composition['embedded_includes'].values():
        assert sha(checkout / 'native/linux_core_port' / image['file']) == image['sha256']
    required = [('dense', dense_batch), ('routed', routed_batch), ('attention', attention)]
    selections = {}
    for family, option in required:
        if not option:
            continue
        plan, result, entry = load_component(base, family)
        assert composition['exact_pending_native_plan_hashes'][family] == entry['manifest_sha256']
        if result is not None:
            rows = validate_attention(plan, result) if family == 'attention' else (
                validate_dense(plan, result, option) if family == 'dense' else validate_routed(plan, result, option))
            entry.update(selected_option=option, selected_and_baseline_comparisons=len(rows), selected_values_match_original=True)
        selections[family] = entry
    if gdn:
        from qualify_native_gdn_explicit_r1 import qualify as qualify_gdn
        source = checkout.parent / 'AIMA-explicit-gdn-layout-candidate'
        for name, expected in composition['gdn_files'].items():
            assert sha(checkout / name) == sha(source / name) == expected
        selections['gdn'] = qualify_gdn(base, source)
    assert selections
    return dict(ready=all(x['ready'] for x in selections.values()), candidate_commit=COMMIT,
        selections=selections, environment=dict(AIMA_PORT_PREFILL_ORDERED_REPLAY=str(dense_batch),
            AIMA_PORT_ROUTED_ORDERED_REPLAY=str(routed_batch), AIMA_PORT_ORDERED_ATTENTION_PREFILL=str(int(attention)),
            AIMA_PORT_NATIVE_GDN_PREFILL=str(int(gdn)), AIMA_PORT_NATIVE_MOE_PREFILL=str(int(bool(routed_batch)))),
        full_product_correctness_still_required=True, remote_calls=0, inference_acceptance=False, performance_acceptance=False)


if __name__ == '__main__':
    base = Path(__file__).resolve().parent
    print(json.dumps(qualify(base, base.parent.parent.parent / 'AIMA-explicit-q8192-candidate'), indent=2))
