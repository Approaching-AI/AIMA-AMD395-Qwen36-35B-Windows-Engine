from pathlib import Path
import hashlib, json, subprocess, sys

B = Path(__file__).resolve().parent
P = B.parents[1]
G = Path('/Users/jiawei-macmini/projects/AIMA-explicit-gdn-layout-candidate')
name = 'qrt-gb10-compact-gdn-layers-20260923-r1'
D = B / name
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
read = lambda p: json.loads(p.read_text())
if sys.argv[1:] == ['--check']:
    print(json.dumps(dict(complete=(D / 'dispatch.json').exists(), remote_calls=0)))
    raise SystemExit(0)
assert not sys.argv[1:]
stage = read(B / (name + '-stage.json'))
dispatch = read(D / 'dispatch.json')
transport = read(B / (name + '-transport.json'))
assert transport['returncode'] == dispatch['returncode'] == 0
assert dispatch['reason'] == 'completed' and dispatch['container_status'] == 'false 0'
for key in ('host_guard_pass', 'frozen_autotune_cache_no_misses',
            'frozen_inductor_cache_preserved', 'original_inductor_kernel_source_preserved',
            'original_token_matrix_qualified', 'all_original_gdn_results_match', 'operator_comparison_qualified'):
    assert dispatch[key], key
assert not dispatch['gpu_processes_after'] and dispatch['minimum_host_available_bytes'] >= 8 * (1 << 30)
manifest = read(D / 'source-inputs.json')
assert sha(D / 'source-inputs.json') == stage['source_inputs_sha256']
for relative, digest in manifest['files'].items():
    assert sha(D / relative) == digest, relative
base = manifest['candidate_base_commit']
for file in ('ordered_pipeline.py', 'group16.py', 'exp2.py', 'persistent.py'):
    original = subprocess.check_output(['git', '-C', str(P), 'show', base + ':tools/experiments/ordered_q8192/persistent_gdn/compact/' + file], timeout=15)
    assert hashlib.sha256(original).hexdigest() == manifest['files']['candidate/explicit_layout/' + file]
for file in ('ordered_pipeline.py', 'ordered_inverse.py', 'integer_u.py'):
    original = subprocess.check_output(['git', '-C', str(G), 'show', 'e15c6cd1a0e91243ebd330fa55ac24348603e151:native/providers/gdn/' + file], timeout=15)
    assert hashlib.sha256(original).hexdigest() == manifest['files']['candidate/u64/' + file]
download = read(D/'compact-download-manifest.json')
for relative, entry in download.items():
    path=D/relative
    assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'], relative
assert sha(B/stage['script'])==stage['script_sha256']==dispatch['command_file_sha256']
assert sha(B/stage['archive'])==stage['archive_sha256']==dispatch['archive_sha256']
assert sha(B/stage['dispatcher'])==stage['dispatcher_sha256']
capture = read(D / 'capture/capture.json')
cases = capture['cases']
assert [c['name'] for c in cases] == ['q7169-out32', 'q8192-out32', 'q8192-out512']
assert all(c['control_pass'] for c in cases)
assert sum(len(c['output_token_ids']) for c in cases) == 576
frozen={c['name']:c for c in read(D/'contracts/gb10_cold_token_matrix_20260911_oracle.json')['cases']}
prior_capture=read(B/'qrt-gb10-explicit-gdn-layers-20260923-r1/capture/capture.json')
for case, prior in zip(cases,prior_capture['cases']):
    expected=frozen[case['name']]
    assert case['prompt']['u32le_sha256']==expected['prompt']['u32le_sha256']
    assert case['output_token_ids']==expected['expected']['output_token_ids']
    assert abs(case['first_token_raw_logit']-expected['expected']['first_token_raw_logit'])<=0.125
    assert case['worker']['sha256']==prior['worker']['sha256']
    assert sha(D/'capture'/case['name']/case['worker']['file'])==case['worker']['sha256']
product = cases[-1]
assert product['prompt']['u32le_sha256'] == 'dda20edc609f935f34d3d41ca4a84ffefa66726676756fe26ff3bef4fbff0b96'
assert product['first_token_id'] == 144 and product['first_token_raw_logit'] == 10.375
summary = product['worker']['ordered_gdn']
assert summary == read(D / 'capture/q8192-out512/ordered-summary.json')
assert summary['all_original_gdn_results_match'] and summary['exp2_table_unchanged']
assert summary['pending_calls'] == 0 and not summary['candidate_outputs_fed_to_model']
rows = summary['layers']
assert [r['layer'] for r in rows] == [i for i in range(40) if i % 4 != 3]
counts = dict(core_bf16_values=0, final_state_fp32_values=0, cumsum_fp32_values=0)
for row in rows:
    assert row == read(D / ('capture/q8192-out512/ordered-layer-%02d.json' % row['layer']))
    assert row['tokens'] == 8192 and row['chunks'] == 128
    assert all(row[k] for k in ('copied_inputs_unchanged', 'original_inputs_unchanged', 'original_outputs_unchanged'))
    assert not row['candidate_outputs_fed_to_model']
    cumsum = row['preparation']['cumsum']
    assert cumsum['bit_mismatches'] == 0 and cumsum['actual'] == cumsum['original']
    counts['cumsum_fp32_values'] += cumsum['values']
    assert [v['variant'] for v in row['variants']] == ['persistent']
    for variant in row['variants']:
        assert variant['guards_pass']
        for surface, key in (('core', 'core_bf16_values'), ('final_state', 'final_state_fp32_values')):
            value = variant[surface]
            assert value['bit_mismatches'] == 0 and value['actual'] == value['original']
            assert value['original'] == row['original_outputs'][0 if surface == 'core' else 1]
            counts[key] += value['values']
assert counts == dict(core_bf16_values=1006632960, final_state_fp32_values=15728640, cumsum_fp32_values=7864320)
prior_summary=read(B/'qrt-gb10-explicit-gdn-layers-20260923-r1/capture/q8192-out512/ordered-summary.json')
for row, prior in zip(rows,prior_summary['layers']):
    assert row['layer']==prior['layer']
    for key in ('original_inputs','original_outputs','preparation'):
        assert row[key]==prior[key], (row['layer'],key)
old = read(B / 'qrt-gb10-q8192-first64-20260922-r1/capture/q8192-out512.json')
result = dict(kind='original_q8192_all_linear_layers_compact_persistent_gdn_comparison',
    authority_host='gb10-4t', runtime_host=dispatch['host'],
    model='/mnt/data/models/Qwen3.6-35B-A3B', candidate_base_commit=base,
    command=dispatch['command'], command_file=dispatch['command_file'],
    command_file_sha256=dispatch['command_file_sha256'],
    source_inputs_sha256=sha(D / 'source-inputs.json'),
    capture_sha256=sha(D / 'capture/capture.json'), dispatch_sha256=sha(D / 'dispatch.json'),
    verified_download_files=len(download), original_operands_and_outputs_match_previous_capture=True,
    original_model_source_commit=manifest['source_commit'], dependency_source_commit=manifest['dependency_source_commit'],
    layers=30, chunks_per_layer=128, variants=['persistent'], compared_values=counts,
    all_original_values_bit_exact=True, original_576_tokens_match=True,
    first_token=144, first_logit=10.375,
    full_first_logits_match_prior_qualified_reference=product['worker']['sha256'] == old['worker']['sha256'],
    full_first_logits_are_additional_diagnostic=True,
    candidate_outputs_fed_to_model=False, original_inputs_unchanged=True, original_outputs_unchanged=True,
    guards_pass=True, table_unchanged=True, frozen_caches_preserved=True, cleanup_pass=True,
    candidate_amd_gpu_executed=False, candidate_inference_acceptance=False, performance_acceptance=False,
    remaining_scope='Native gfx1151 execution and complete real Windows q8192 continuation/performance acceptance.')
path = D / 'operator-analysis.json'
with path.open('x') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(report=path.name, sha256=sha(path), **counts,
                     all_original_values_bit_exact=True, original_576_tokens_match=True)))
