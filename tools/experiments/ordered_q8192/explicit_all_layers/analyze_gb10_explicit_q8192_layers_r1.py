"""Verify the completed original-model comparison without executing a model."""
from pathlib import Path
import hashlib
import json
import subprocess

B = Path(__file__).resolve().parent
E = Path('/Users/jiawei-macmini/projects/AIMA-explicit-q8192-candidate')
NAME = 'qrt-gb10-explicit-q8192-layers-20260923-r1'
D = B / NAME
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
stage = read(B / (NAME + '-stage.json'))
dispatch = read(D / 'dispatch.json')
manifest = read(D / 'source-inputs.json')
transport = read(B / (NAME + '-transport.json'))
assert dispatch['returncode'] == transport['returncode'] == 0
assert dispatch['reason'] == 'completed' and dispatch['container_status'] == 'false 0'
assert not dispatch['gpu_processes_after']
assert dispatch['minimum_host_available_bytes'] >= 8 * (1 << 30)
for key in ('host_guard_pass', 'frozen_autotune_cache_no_misses',
            'frozen_inductor_cache_preserved', 'original_inductor_kernel_source_preserved',
            'original_token_matrix_qualified', 'operator_comparison_qualified',
            'all_original_dense_results_match', 'all_original_attention_results_match',
            'all_original_routed_results_match'):
    assert dispatch[key], key
download = read(D / 'compact-download-manifest.json')
for relative, item in download.items():
    path = D / relative
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], relative
assert sha(D / 'source-inputs.json') == stage['source_inputs_sha256']
assert sha(B / stage['script']) == stage['script_sha256'] == dispatch['command_file_sha256']
assert sha(B / stage['dispatcher']) == stage['dispatcher_sha256']
assert sha(B / stage['archive']) == stage['archive_sha256'] == dispatch['archive_sha256']
for relative, digest in manifest['files'].items():
    assert sha(D / relative) == digest, relative
assert manifest['candidate_base_commit'] == manifest['composition_commit'] == 'c85259254b805788dc80680caad37685708c98a8'
assert not manifest['model_compute_sources_modified'] and not manifest['candidate_outputs_fed_to_model']
for filename in ('attention_explicit.py', 'explicit_dense.py', 'routed_explicit.py',
                 'explicit_exp2.py', 'explicit_group16.py', 'pack_value.py'):
    raw = subprocess.check_output(['git', '-C', str(E), 'show',
        manifest['composition_commit'] + ':native/providers/explicit_q8192/' + filename], timeout=15)
    assert hashlib.sha256(raw).hexdigest() == manifest['files']['candidate/' + filename], filename

capture = read(D / 'capture/capture.json')
assert capture['completed'] and capture['controls_qualified']
cases = capture['cases']
assert [c['name'] for c in cases] == ['q7169-out32', 'q8192-out32', 'q8192-out512']
frozen = {c['name']: c for c in read(D / 'contracts/gb10_cold_token_matrix_20260911_oracle.json')['cases']}
prior_dirs = {family: B / ('qrt-gb10-ordered-' + family + '-layers-20260923-' +
                          ('r2' if family == 'attention' else 'r1'))
              for family in ('dense', 'attention', 'routed')}
token_controls = []
for case in cases:
    expected = frozen[case['name']]
    assert case['control_pass'] and case['prompt']['u32le_sha256'] == expected['prompt']['u32le_sha256']
    assert case['output_token_ids'] == expected['expected']['output_token_ids']
    assert abs(case['first_token_raw_logit'] - expected['expected']['first_token_raw_logit']) <= 0.125
    assert case['worker']['complete'] and not case['worker']['observer_modifies_output']
    logits = D / 'capture' / case['name'] / case['worker']['file']
    assert sha(logits) == case['worker']['sha256'] and logits.stat().st_size == case['worker']['bytes']
    for family, prior in prior_dirs.items():
        old = next(c for c in read(prior / 'capture/capture.json')['cases'] if c['name'] == case['name'])
        assert case['worker']['sha256'] == old['worker']['sha256'], (case['name'], family)
    if case['name'] != 'q8192-out512':
        assert not any(k.startswith('ordered_') for k in case['worker'])
    token_controls.append(dict(name=case['name'], prompt_sha256=case['prompt']['u32le_sha256'],
        output_tokens=len(case['output_token_ids']), all_original_tokens_match=True,
        first_token=case['first_token_id'], first_logit=case['first_token_raw_logit'],
        first_logit_error=abs(case['first_token_raw_logit'] - expected['expected']['first_token_raw_logit']),
        original_full_logits_sha256=case['worker']['sha256'], prior_full_logits_match=True))
assert sum(c['output_tokens'] for c in token_controls) == 576
product = cases[-1]
summaries = {}
old_rows = {}
for family, prior in prior_dirs.items():
    key = 'layers' if family == 'attention' else 'projections'
    summary = read(D / ('capture/q8192-out512/ordered-' + family + '-all-layers.json'))
    assert summary == product['worker']['ordered_' + family]
    assert summary['pending_calls'] == 0 and summary['original_outputs_returned_unchanged']
    assert not summary['candidate_outputs_fed_to_model'] and summary['all_original_' + family + '_results_match']
    assert summary['layers_complete' if family == 'attention' else 'projections_complete']
    summaries[family] = summary
    old = read(prior / ('capture/q8192-out512/ordered-' + family + '-all-layers.json'))
    old_rows[family] = {r['layer' if family == 'attention' else 'key']: r for r in old[key]}

shapes = {'gdn_qkvz': (12288, 2048), 'gdn_out': (2048, 4096), 'attention_qkv': (9216, 2048),
          'attention_out': (2048, 4096), 'shared_gate_up': (1024, 2048), 'shared_down': (2048, 512)}
dense_expected = ['%02d-%s' % (layer, label) for layer in range(40)
    for label in ((['attention_qkv', 'attention_out'] if layer % 4 == 3 else ['gdn_qkvz', 'gdn_out'])
                  + ['shared_gate_up', 'shared_down'])]
dense = summaries['dense']
assert dense['expected_projections'] == dense_expected
assert [r['key'] for r in dense['projections']] == dense_expected
dense_values = 0
for row in dense['projections']:
    assert row == read(D / ('capture/q8192-out512/ordered-dense-' + row['key'] + '.json'))
    assert all(row[k] for k in ('inputs_unchanged', 'original_output_unchanged', 'all_values_match'))
    assert not row['candidate_outputs_fed_to_model']
    old = old_rows['dense'][row['key']]
    assert row['inputs'] == old['inputs'] and row['comparison']['original'] == old['comparison']['original']
    c = row['comparison']; n, k = shapes[row['projection']]
    assert (c['tokens'], c['n'], c['k'], c['batch']) == (8192, n, k, 64)
    assert [v['shape'] for v in row['inputs']] == [[8192, k], [n, k]]
    assert all(v['dtype'] == 'torch.bfloat16' for v in row['inputs'])
    assert c['layout'] == ('transposed' if row['projection'] in ('gdn_qkvz', 'attention_qkv') else 'contiguous')
    assert all(c[k] for k in ('guards_pass', 'queues_unchanged', 'weight_view_unchanged', 'debug_unchanged'))
    assert c['actual'] == c['original'] and c['actual']['shape'] == [8192, n]
    assert c['bf16_mismatches'] == 0 and c['queued_cells'] == 8192 * n
    assert c['windows'] == (8192 * n + (1 << 20) - 1) // (1 << 20)
    dense_values += 8192 * n
assert dense_values == dense['bf16_values'] == 5452595200

attention = summaries['attention']
assert attention['tables_unchanged']
assert [r['layer'] for r in attention['layers']] == list(range(3, 40, 4))
qk_values = context_values = 0
for row in attention['layers']:
    assert row == read(D / ('capture/q8192-out512/ordered-attention-layer-%02d.json' % row['layer']))
    assert all(row[k] for k in ('inputs_unchanged', 'original_output_unchanged', 'all_values_match'))
    old = old_rows['attention'][row['layer']]
    assert row['inputs'] == old['inputs'] and row['original_kernel'] == old['original_kernel']
    assert row['comparison']['original'] == old['comparison']['original']
    assert [v['shape'] for v in row['inputs']] == [[8192, 4096], [8192, 512], [8192, 512]]
    assert all(v['dtype'] == 'torch.bfloat16' for v in row['inputs'])
    c = row['comparison']
    assert all(c[k] for k in ('guards_pass', 'packed_value_matches', 'queues_unchanged', 'debug_unchanged'))
    assert c['actual'] == c['original'] and c['actual']['shape'] == [8192, 4096]
    assert c['qk_bit_mismatches'] == c['context_bit_mismatches'] == 0
    assert [s['query_start'] for s in c['slabs']] == list(range(0, 8192, 128))
    for slab in c['slabs']:
        assert slab['queries'] == 128 and slab['qk_fp32_values'] == 128 * 16 * 8192
        assert slab['context_bf16_values'] == 128 * 4096
        assert slab['qk_bit_mismatches'] == slab['context_bit_mismatches'] == 0
    for key in ('qk_fp32_values', 'context_bf16_values'):
        assert sum(s[key] for s in c['slabs']) == c[key]
    assert row['original_kernel']['original_run_returned_unchanged']
    qk_values += c['qk_fp32_values']; context_values += c['context_bf16_values']
assert qk_values == attention['qk_fp32_values'] == 10737418240
assert context_values == attention['context_bf16_values'] == 335544320

routed = summaries['routed']
expected = ['%02d-%s' % (i, kind) for i in range(40) for kind in ('gate-up', 'down')]
assert [r['key'] for r in routed['projections']] == expected
assert sorted(routed['expected_projections']) == sorted(expected)
dense_rows = {r['key']: r for r in dense['projections']}
full = sparse = malformed = 0
for row in routed['projections']:
    assert row == read(D / ('capture/q8192-out512/ordered-routed-' + row['key'] + '.json'))
    assert all(row[k] for k in ('all_values_match', 'inputs_unchanged', 'original_output_unchanged',
                               'original_router_and_sorted_dispatch_agree'))
    assert not row['candidate_outputs_fed_to_model']
    old = old_rows['routed'][row['key']]
    for key in ('inputs', 'original_output', 'original_config', 'padded_routes'):
        assert row[key] == old[key], (row['key'], key)
    n, k = (2048, 512) if row['down'] else (1024, 2048)
    assert row['inputs'][1]['shape'] == [256, n, k]
    assert row['inputs'][2]['shape'] == row['inputs'][3]['shape'] == [8192, 8]
    if not row['down']:
        assert dense_rows['%02d-shared_gate_up' % row['layer']]['inputs'][0] == row['inputs'][0]
    for c in row['comparisons']:
        assert c['bf16_mismatches'] == c['invalid_flags'] == 0 and c['actual'] == c['original']
        assert all(c[key] for key in ('guards_pass', 'queues_unchanged', 'debug_unchanged'))
        assert c['logical_routes'] == 65536 and (c['n'], c['k']) == (n, k)
        if c['sparse']:
            assert c['compared_values'] == 4093 and c['unselected_outputs_unchanged']
            sparse += c['compared_values']
        else:
            assert c['batch'] == 64 and c['compared_values'] == 65536 * n
            assert c['actual'] == row['original_output']; full += c['compared_values']
    for c in row['malformed_controls']:
        assert c['pass'] and c['actual_flags'] == c['expected_flags']
        assert all(c[k] for k in ('only_declared_output_touched', 'inputs_unchanged', 'guards_pass'))
        malformed += 1
    if row['layer'] == 0:
        assert [c['batch'] for c in row['comparisons']] == [64, 32, 64, 128]
    else:
        assert len(row['comparisons']) == 1 and not row['malformed_controls']
assert full == routed['full_bf16_values'] == 8053063680
assert sparse == routed['sparse_bf16_values'] == 24558
assert malformed == routed['malformed_controls'] == 15

report = dict(schema=1, kind='original_model_all_layer_explicit_q8192_cuda_comparison',
    authority_host='gb10-4t', runtime_host=dispatch['host'], model='/mnt/data/models/Qwen3.6-35B-A3B',
    candidate_commit=manifest['composition_commit'], original_source_commit=manifest['source_commit'],
    dependency_source_commit=manifest['dependency_source_commit'], command=dispatch['command'],
    command_file=dispatch['command_file'], command_file_sha256=dispatch['command_file_sha256'],
    source_inputs_sha256=sha(D / 'source-inputs.json'), dispatch_sha256=sha(D / 'dispatch.json'),
    capture_sha256=sha(D / 'capture/capture.json'), token_controls=token_controls,
    values=dict(dense_bf16=dense_values, attention_qk_fp32=qk_values,
                attention_context_bf16=context_values, routed_full_bf16=full, routed_sparse_bf16=sparse),
    dense_projections=160, attention_layers=10, attention_slabs=640, routed_projections=80,
    malformed_routed_controls=malformed, all_original_values_bit_exact=True,
    prior_original_operands_and_outputs_match=True, original_576_tokens_match=True,
    full_first_logits_match_all_three_prior_qualified_captures=True,
    full_first_logits_are_additional_diagnostic=True,
    prior_summary_sha256={family: sha(prior / ('capture/q8192-out512/ordered-' + family + '-all-layers.json'))
                         for family, prior in prior_dirs.items()},
    summary_sha256={family: sha(D / ('capture/q8192-out512/ordered-' + family + '-all-layers.json'))
                    for family in prior_dirs}, verified_download_files=len(download),
    frozen_caches_preserved=True, cleanup_pass=True, candidate_outputs_fed_to_model=False,
    candidate_amd_gpu_executed=False, inference_acceptance=False,
    performance_acceptance=False, release_qualified=False)
with (D / 'operator-analysis.json').open('x') as f:
    json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
print(json.dumps(dict(report='operator-analysis.json', sha256=sha(D / 'operator-analysis.json'),
                     values=report['values'], verified_download_files=len(download), pass_=True)))
