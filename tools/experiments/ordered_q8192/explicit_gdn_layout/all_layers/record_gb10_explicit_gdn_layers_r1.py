from pathlib import Path
import hashlib
import json
import shutil

B = Path(__file__).resolve().parent
R = B.parents[1]
P = Path('/Users/jiawei-macmini/projects/AIMA-public-r1191-candidate')
name = 'qrt-gb10-explicit-gdn-layers-20260923-r1'
D = B / name
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
analysis = read(D / 'operator-analysis.json')
assert analysis['all_original_values_bit_exact'] and analysis['original_576_tokens_match']
assert analysis['full_first_logits_match_prior_qualified_reference']
summary = read(D / 'capture/q8192-out512/ordered-summary.json')
prior = read(B / 'qrt-gb10-ordered-gdn-layers-20260923-r1/capture/q8192-out512/ordered-summary.json')
layers = []
for row, old in zip(summary['layers'], prior['layers']):
    assert row['layer'] == old['layer']
    for key in ('original_inputs', 'original_outputs', 'preparation'):
        assert row[key] == old[key], (row['layer'], key)
    layers.append(dict(layer=row['layer'], tokens=8192, chunks=128,
        original_inputs=row['original_inputs'], original_outputs=row['original_outputs'],
        original_inputs_and_outputs_match_previous_qualified_capture=True,
        variants=[dict(variant=v['variant'], core_values=v['core']['values'],
            core_bit_mismatches=v['core']['bit_mismatches'],
            core_sha256=v['core']['actual']['sha256'],
            final_state_values=v['final_state']['values'],
            final_state_bit_mismatches=v['final_state']['bit_mismatches'],
            final_state_sha256=v['final_state']['actual']['sha256'],
            guards_pass=v['guards_pass']) for v in row['variants']],
        candidate_outputs_fed_to_model=False))
assert len(layers) == 30
dispatch = read(D / 'dispatch.json')
stage = read(B / (name + '-stage.json'))
manifest = read(D / 'source-inputs.json')
archive_rel = Path('tools/experiments/ordered_q8192/explicit_gdn_layout/all_layers')
sources = {
    'observer.py': D / 'scripts/capture_ordered_gdn_layers.py',
    'source-inputs.json': D / 'source-inputs.json',
    'prepare_gb10_explicit_gdn_layers_r1.py': B / 'prepare_gb10_explicit_gdn_layers_r1.py',
    'run-' + name + '.py': B / ('run-' + name + '.py'),
    'analyze_gb10_explicit_gdn_layers_r1.py': B / 'analyze_gb10_explicit_gdn_layers_r1.py',
    'record_gb10_explicit_gdn_layers_r1.py': Path(__file__),
}
reproducers = {}
for file, source in sources.items():
    reproducers[str(archive_rel / file)] = dict(bytes=source.stat().st_size, sha256=sha(source))
    for repo in (R, P):
        target = repo / archive_rel / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
report = dict(schema=1, classification='original_model_all_layer_explicit_gdn_cuda_comparison',
    candidate_source_commit='b07bf58e5140ea6e256c477f4aacfd39fc89695d',
    original_source_commit=manifest['source_commit'], dependency_source_commit=manifest['dependency_source_commit'],
    analysis=analysis, analysis_sha256=sha(D / 'operator-analysis.json'),
    candidate_only_values=dict(core_bf16=1006632960, final_state_fp32=15728640),
    u64_control_values=dict(core_bf16=1006632960, final_state_fp32=15728640),
    shared_cumsum_fp32_values=7864320,
    original_576_output_token_ids_preserved=True,
    original_all_layer_inputs_and_outputs_match_prior_qualified_capture=True,
    previous_all_layer_summary_sha256=sha(B / 'qrt-gb10-ordered-gdn-layers-20260923-r1/capture/q8192-out512/ordered-summary.json'),
    layers=layers, stage=stage, stage_sha256=sha(B / (name + '-stage.json')),
    source_manifest=manifest, exact_reproduction_sources=reproducers,
    native_product_helpers=dict(
        qualifier_sha256=sha(B / 'qualify_native_gdn_explicit_r1.py'),
        stage_generator_sha256=sha(B / 'stage_linux_core_explicit_gdn_r1.py'),
        dispatcher_sha256=sha(B / 'dispatch_linux_core_explicit_gdn_r1.py'),
        host_controls=read(B / 'gdn-explicit-qualifier-host-controls-r1.json'),
        native_run_performed=False),
    completion=dict(returncode=dispatch['returncode'], reason=dispatch['reason'],
        wall_seconds=dispatch['wall_seconds'], container_status=dispatch['container_status'],
        minimum_host_available_bytes=dispatch['minimum_host_available_bytes'],
        host_guard_pass=dispatch['host_guard_pass'], gpu_processes_after=dispatch['gpu_processes_after'],
        original_token_matrix_qualified=dispatch['original_token_matrix_qualified'],
        frozen_autotune_cache_no_misses=dispatch['frozen_autotune_cache_no_misses'],
        frozen_inductor_cache_preserved=dispatch['frozen_inductor_cache_preserved'],
        original_inductor_kernel_source_preserved=dispatch['original_inductor_kernel_source_preserved']),
    download=dict(manifest_sha256=sha(D / 'compact-download-manifest.json'),
        verified_members=len(read(D / 'compact-download-manifest.json')),
        supplemental_source_binding=read(D / 'source-binding-completion-r1.json'),
        supplemental_source_binding_sha256=sha(D / 'source-binding-completion-r1.json'),
        note='Initial local analysis required the omitted .best_config file; it was retrieved unchanged and analysis rerun without native or model re-execution.'),
    limitations=['CUDA operator comparisons; no new gfx1151 arithmetic or product run',
                 'Candidate results never entered model computation',
                 'Observer wall times include hashes, copies and JIT; they are not inference-performance evidence'],
    inference_acceptance=False, performance_acceptance=False, release_qualified=False)
relative = Path('benchmarks/correctness/ordered-gdn-explicit-all-layers-20260923.json')
payload = json.dumps(report, indent=2) + '\n'
for repo in (R, P):
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
print(json.dumps(dict(report=str(relative), bytes=len(payload), sha256=sha(R / relative),
    candidate_core_values=1006632960, candidate_state_values=15728640,
    original_576_outputs=True, original_inputs_match_previous_capture=True)))
