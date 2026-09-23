"""Record original-operand CUDA comparisons and the unexecuted AMD trial."""
from pathlib import Path
import hashlib
import json
import shutil

B = Path(__file__).resolve().parent
ROOT = B.parent.parent
PUBLIC = ROOT.parent / 'AIMA-public-r1191-candidate'
D = B / 'routed-selected-explicit-r1'
NATIVE = B / 'routed-selected-explicit-windows-r1'
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
read = lambda p: json.loads(p.read_text())


def main():
    result, manifest, dispatch = (read(D / name) for name in ('result.json', 'manifest.json', 'dispatch.json'))
    assert result['manifest_sha256'] == sha(D / 'manifest.json')
    assert sha(D / 'result.json') == 'c0985474f34e6b1eb610cb0a3cecac97485fe63fed37cce012ca22f7ece29779'
    assert result['all_components_match'] and result['source_files_unchanged']
    assert dispatch['returncode'] == 0 and not dispatch['gpu_processes_after']
    assert all(e['returncode'] == 0 for e in read(D / 'collection.json'))
    assert result['full_bf16_values'] == 603979776 and result['sparse_bf16_values'] == 24558
    assert len(result['records']) == 12 and len(result['malformed_controls']) == 15
    assert all(r['bf16_mismatches'] == 0 and r['guards_pass'] for r in result['records'])
    assert all(r['pass'] for r in result['malformed_controls'])
    assert all(r['baseline_matches_original'] and r['expected_output_unchanged']
               and r['inputs_unchanged'] and r['guards_pass'] for r in result['sources'])
    for name, expected in manifest['source_hashes'].items():
        assert sha(D / 'source' / name) == expected
    for name, e in result['compiled'].items():
        for suffix, key in [('hsaco', 'sha256'), ('amdgcn', 'amdgcn_sha256'), ('ttgir', 'ttgir_sha256')]:
            assert sha(D / (name + '.' + suffix)) == e[key]
        assert e['layout_conversion_count'] == 0 and e['metadata']['shared'] == 16
        assert e['resources']['.vgpr_spill_count'] == e['resources']['.private_segment_fixed_size'] == ['0']
    plan = read(NATIVE / 'manifest.json')
    assert sha(NATIVE / 'manifest.json') == '82d3aaa6e4667f28b7a7b78968353fe9988835a15abc3bf5acfd674f7e15564b'
    assert sha(NATIVE / 'worker.py') == plan['worker_sha256']
    assert plan['expected_full_shape_comparisons'] == 24 and plan['expected_queue_controls'] == 17
    assert not plan['executed'] and not plan['staged_on_baiying']
    archived = {}
    base = Path('tools/experiments/ordered_q8192/routed_explicit_layout')
    files = {name: B / name for name in [
        'routed_selected_explicit_r1.py', 'routed_selected_explicit_worker_r1.py',
        'validate_routed_selected_explicit_r1.py', 'routed_selected_explicit_windows_worker_r1.py',
        'prepare_routed_selected_explicit_windows_r1.py', 'dispatch_routed_selected_explicit_windows_r1.py',
        'routed-explicit-source-equivalence-r1.json', Path(__file__).name]}
    files.update({name: D / 'source' / name for name in ['explicit_dense.py', 'dense_selected.py', 'routed_selected.py']})
    for name, source in files.items():
        if source.suffix == '.py':
            compile(source.read_bytes(), str(source), 'exec')
        else:
            read(source)
        path = base / name
        for repo in (ROOT, PUBLIC):
            (repo / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, repo / path)
        archived[path.as_posix()] = dict(bytes=source.stat().st_size, sha256=sha(source))
    old = read(B / 'routed-selected-u32-windows-r1/outputs/result.json')
    old_run = read(B / 'routed-selected-u32-windows-r1/run/run-record.json')
    assert old['all_components_match'] and old['cleanup_pass']
    assert old_run['exit_code'] == 0 and old_run['host_checks_pass'] and not old_run['after_processes']
    report = dict(schema=1, classification='explicit_layout_full_q8192_routed_cuda_controls_amd_trial_pending',
        candidate_base_commit=result['candidate_base_commit'], candidate_changes_relative_to_base_commit=True,
        candidate_source_sha256=manifest['source_hashes']['routed_explicit.py'], source_hashes=result['source_hashes'],
        arithmetic_ast_controls=manifest['arithmetic_source_equivalence'],
        original_reference=result['original_reference'], original_source_commit=manifest['original_source_commit'],
        reference_execution=dict(host=dispatch['host'], container_host=result['host'], device=result['device'],
            command=dispatch['command'], command_file=dispatch['command_file'],
            model_reference=manifest['original_model_reference'], model_index_sha256=manifest['model_index_sha256'],
            actual_weight_tensors_loaded=True, original_model_engine_loaded=False,
            original_model_inputs_and_output_hashes=True, candidate_outputs_fed_to_original_model=False,
            dispatch_sha256=sha(D / 'dispatch.json'), result_sha256=sha(D / 'result.json'),
            manifest_sha256=sha(D / 'manifest.json'), wall_seconds=dispatch['wall_seconds'],
            process_timeout_seconds=600, returncode=dispatch['returncode'], gpu_processes_after=dispatch['gpu_processes_after']),
        comparison_scope='Original full q8192 first-layer gate-up and down projections, three tile sizes each, plus sparse queues and invalid-input controls.',
        baseline_qualification='For each projection the unchanged baseline first reproduces the complete original-model output SHA256; that full baseline tensor then supplies candidate per-element comparisons.',
        comparisons={k: result[k] for k in ['full_bf16_values', 'sparse_bf16_values', 'all_components_match',
            'source_files_unchanged', 'records', 'malformed_controls', 'sources']},
        compiled_images=result['compiled'],
        previous_actual_amd_control=dict(result_sha256=sha(B / 'routed-selected-u32-windows-r1/outputs/result.json'),
            run_record_sha256=sha(B / 'routed-selected-u32-windows-r1/run/run-record.json'),
            source_commit=result['candidate_base_commit'], all_components_match=True, cleanup_pass=True,
            inference_acceptance=False, performance_acceptance=False),
        pending_native_trial=dict(manifest_sha256=sha(NATIVE / 'manifest.json'), worker_sha256=plan['worker_sha256'],
            model_reference=plan['model'], source_files=plan['source_inputs'],
            images={name: {k: e[k] for k in ['file', 'bytes', 'sha256', 'symbol', 'shared_bytes']} for name, e in plan['images'].items()},
            full_q8192_comparisons=24, malformed_queue_controls=15, partial_shuffled_queue_controls=2,
            partial_expected_values_from_same_run_hash_qualified_baseline=True,
            original_baseline_images_unchanged=True, baseline_and_candidate_interleaved=True,
            reverse_order_passes=True, process_timeout_seconds=900, staged_on_baiying=False, executed=False),
        exact_reproduction_sources=archived,
        decision='Run paired native controls after the active full256k capture completes cleanly; retain the current runtime until correctness and product timing qualify a replacement.',
        limitations=['CUDA numerical checks and gfx1151 compilation do not establish AMD correctness or performance.',
            'These are original full-shape operands for one routed layer, not a new real-model candidate token loop.',
            'All six images retain 16 shared bytes for invalid-queue flag reduction; explicit arithmetic layouts alone do not establish a speedup.',
            'Register allocation and paired native full-shape timing remain to be measured.',
            'Original oracle tokens, logit tolerance and mission thresholds remain unchanged.'],
        native_execution=False, inference_acceptance=False, performance_acceptance=False, release_qualified=False)
    path = Path('benchmarks/correctness/ordered-routed-explicit-layout-controls-20260923.json')
    raw = json.dumps(report, indent=2) + '\n'
    for repo in (ROOT, PUBLIC):
        (repo / path).write_text(raw)
    print(json.dumps(dict(report_sha256=sha(ROOT / path), bytes=len(raw.encode()),
        archived_sources=len(archived), native_execution=False)))


if __name__ == '__main__':
    main()
