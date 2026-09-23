"""Publish full-q8192 CUDA evidence without claiming an AMD speedup."""
from pathlib import Path
import hashlib
import json
import shutil

B = Path(__file__).resolve().parent
ROOT = B.parent.parent
PUBLIC = ROOT.parent / 'AIMA-public-r1191-candidate'
D = B / 'attention-explicit-layout-r2'
NATIVE = B / 'attention-explicit-layout-windows-r1'
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
read = lambda p: json.loads(p.read_text())


def main():
    result, manifest, dispatch = (read(D / name) for name in ('result.json', 'manifest.json', 'dispatch.json'))
    assert result['manifest_sha256'] == sha(D / 'manifest.json')
    assert sha(D / 'result.json') == 'e78a9f09b0fae55c1bd369a98ba1de0aa8b5b75bf23bf0b0cb06ae102bb2a6ac'
    assert result['all_components_match'] and result['inputs_and_tables_unchanged'] and result['sources_unchanged']
    assert dispatch['returncode'] == 0 and not dispatch['gpu_processes_after']
    assert all(e['returncode'] == 0 for e in read(D / 'collection.json'))
    plan = read(NATIVE / 'manifest.json')
    assert sha(NATIVE / 'manifest.json') == 'c9c6166d37626ff9dada5da4d3bea09dba8da8ebe5a6a348de885084c8175a00'
    assert sha(NATIVE / 'worker.py') == plan['worker_sha256']
    rows = []
    for r in result['records']:
        assert all(r[k]['bit_mismatches'] == 0 for k in ('qk', 'probability', 'scales'))
        assert all(x['bit_mismatches'] == 0 for x in r['pv'].values())
        rows.append(dict(query_start=r['query_start'],
            qk_candidate_sha256=r['qk']['actual']['sha256'], qk_original_sha256=r['qk']['expected']['sha256'],
            probability_candidate_sha256=r['probability']['actual']['sha256'],
            probability_original_sha256=r['probability']['expected']['sha256'],
            scales_candidate_sha256=r['scales']['actual']['sha256'], scales_original_sha256=r['scales']['expected']['sha256'],
            context_candidate_sha256={name: x['actual']['sha256'] for name, x in r['pv'].items()},
            context_original_sha256=r['pv']['selected32']['expected']['sha256'],
            guards_pass=r['guards_pass'], probability_and_scales_unchanged=r['probability_and_scales_unchanged'],
            unused_debug_unchanged=r['unused_debug_unchanged']))
    archived = {}
    base = Path('tools/experiments/ordered_q8192/attention_explicit_layout')
    files = {name: B / name for name in ['attention_explicit_layout_r1.py', 'attention_explicit_layout_r2.py',
        'attention_explicit_layout_worker_r1.py', 'validate_attention_explicit_layout_r1.py',
        'validate_attention_explicit_layout_r2.py', 'attention_explicit_layout_windows_worker_r1.py',
        'prepare_attention_explicit_layout_windows_r1.py', 'dispatch_attention_explicit_layout_windows_r1.py', Path(__file__).name]}
    files.update({name: D / 'source' / name for name in ['explicit_group16.py', 'explicit_exp2.py', 'explicit_dense.py']})
    for name, source in files.items():
        compile(source.read_bytes(), str(source), 'exec')
        path = base / name
        for repo in (ROOT, PUBLIC):
            (repo / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, repo / path)
        archived[path.as_posix()] = dict(bytes=source.stat().st_size, sha256=sha(source))
    failure = read(B / 'attention-explicit-layout-r1/dispatch.json')
    assert failure['returncode'] == 1 and not failure['gpu_processes_after']
    old = read(B / 'attention-ordered-u32-windows-r1/outputs/result.json')
    assert old['all_components_match'] and old['cleanup_pass']
    report = dict(schema=1, classification='explicit_layout_full_q8192_attention_cuda_controls_amd_trial_pending',
        candidate_base_commit=result['candidate_base_commit'], candidate_changes_relative_to_base_commit=True,
        reused_explicit_gdn_source_commit=manifest['reused_explicit_gdn_source_commit'],
        candidate_source_sha256=manifest['source_hashes']['attention_explicit.py'],
        source_hashes=result['source_hashes'],
        arithmetic_ast_controls=manifest['arithmetic_ast_unchanged_after_normalizing_explicit_layouts_and_equivalent_broadcast'],
        qk_and_tiled_pv_changes=manifest['qk_and_tiled_pv_changes'],
        original_reference=result['original_reference'],
        reference_execution=dict(host=result['host'], device=result['device'], command=dispatch['command'],
            command_file=dispatch['command_file'], model_reference=result['original_reference']['model'],
            original_model_loaded=False, actual_original_operands=True,
            dispatch_sha256=sha(D / 'dispatch.json'), result_sha256=sha(D / 'result.json'),
            manifest_sha256=sha(D / 'manifest.json'), wall_seconds=dispatch['wall_seconds'],
            process_timeout_seconds=600, returncode=dispatch['returncode'], gpu_processes_after=dispatch['gpu_processes_after']),
        comparisons={k: result[k] for k in ['qk_fp32_values', 'probability_bf16_values', 'scale_fp32_values',
            'context_bf16_values', 'all_components_match', 'inputs_and_tables_unchanged', 'sources_unchanged', 'guards_pass',
            'complete_context_sha256', 'original_context_sha256', 'selected_queue_controls']},
        comparison_scope='All64 query slabs of one full q8192 original layer; stored QK includes causal -inf and probability/scales include deliberately untouched future-buffer poison.',
        rows=rows, compiled_images=result['compiled'],
        prior_frontend_failure=dict(dispatch_sha256=sha(B / 'attention-explicit-layout-r1/dispatch.json'),
            log_sha256=sha(B / 'attention-explicit-layout-r1/container.log'), returncode=1, gpu_processes_after='',
            reason='Pinned Gluon lacks broadcast_to; r2 uses the equivalent first result of broadcast(index, probability).',
            numerical_failure=False, later_full_numerical_pass=True),
        previous_actual_amd_control=dict(result_sha256=sha(B / 'attention-ordered-u32-windows-r1/outputs/result.json'),
            source_commit=old['candidate_base_commit'], passes=old['passes'], inference_acceptance=False, performance_acceptance=False),
        pending_native_trial=dict(manifest_sha256=sha(NATIVE / 'manifest.json'), worker_sha256=plan['worker_sha256'],
            images={name: {k: e[k] for k in ['file', 'bytes', 'sha256', 'symbol', 'shared_bytes']} for name, e in plan['images'].items()},
            variants=plan['variants'], full_q8192_passes=8, slab_comparisons=512, partial_queue_controls=4,
            process_timeout_seconds=300, source_files=plan['source_inputs'],
            original_probabilities_and_scales_are_expected_hashes_only=True,
            native_baseline_images_unchanged=True, staged_on_baiying=False, executed=False),
        exact_reproduction_sources=archived,
        decision='Prepare paired native measurement while the original full256k capture owns baiying. Leave the optional runtime unchanged.',
        limitations=['CUDA comparison and gfx1151 compilation do not establish AMD correctness or performance.',
            'No real-model candidate token loop has run for these images.',
            'Selected PV retains16 shared bytes for its output-index range reduction; the other four images have no shared-memory instructions.',
            'Removing layout conversion is not itself evidence of a speedup; register usage and actual full-shape timing remain relevant.',
            'Original oracle tokens, logit tolerance and mission thresholds remain unchanged.'],
        native_execution=False, inference_acceptance=False, performance_acceptance=False, release_qualified=False)
    path = Path('benchmarks/correctness/ordered-attention-explicit-layout-controls-20260923.json')
    raw = json.dumps(report, indent=2) + '\n'
    for repo in (ROOT, PUBLIC):
        (repo / path).write_text(raw)
    print(json.dumps(dict(report_sha256=sha(ROOT / path), bytes=len(raw.encode()),
        archived_sources=len(archived), native_execution=False)))


if __name__ == '__main__':
    main()
