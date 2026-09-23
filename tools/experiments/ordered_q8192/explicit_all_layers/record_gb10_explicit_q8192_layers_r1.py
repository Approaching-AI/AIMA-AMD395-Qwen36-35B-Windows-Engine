"""Publish compact, source-bound completed GB10 comparisons in both repositories."""
from pathlib import Path
import hashlib
import json
import shutil

B = Path(__file__).resolve().parent
R = B.parents[1]
P = Path('/Users/jiawei-macmini/projects/AIMA-public-r1191-candidate')
NAME = 'qrt-gb10-explicit-q8192-layers-20260923-r1'
D = B / NAME
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
analysis = read(D / 'operator-analysis.json')
assert analysis['all_original_values_bit_exact'] and analysis['original_576_tokens_match']
assert analysis['prior_original_operands_and_outputs_match'] and analysis['cleanup_pass']
dispatch = read(D / 'dispatch.json')
manifest = read(D / 'source-inputs.json')
archive = Path('tools/experiments/ordered_q8192/explicit_all_layers')
sources = {name: B / name for name in (
    'prepare_explicit_q8192_all_layers_r1.py', 'dispatch_gb10_explicit_q8192_layers_r1.py',
    'run-' + NAME + '.py', 'analyze_gb10_explicit_q8192_layers_r1.py',
    'record_gb10_explicit_q8192_layers_r1.py',
    'explicit-q8192-all-layer-composition-control-r1.json',
    'explicit-q8192-all-layer-observer-equivalence-r1.json')}
sources.update({name: D / 'scripts' / name for name in (
    'capture_explicit_q8192_layers.py', 'capture_explicit_dense_layers.py',
    'capture_explicit_attention_layers.py', 'capture_explicit_routed_layers.py',
    'capture_gb10_explicit_q8192_matrix.py')})
sources['source-inputs.json'] = D / 'source-inputs.json'
sources['stage.json'] = B / (NAME + '-stage.json')
reproduction = {}
for name, source in sources.items():
    relative = archive / name
    reproduction[str(relative)] = dict(bytes=source.stat().st_size, sha256=sha(source))
    for repo in (R, P):
        target = repo / relative
        assert not target.exists(), target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

families = {}
for family in ('dense', 'attention', 'routed'):
    summary = read(D / ('capture/q8192-out512/ordered-' + family + '-all-layers.json'))
    rows = summary['layers' if family == 'attention' else 'projections']
    compact = []
    for row in rows:
        key = '%02d' % row['layer'] if family == 'attention' else row['key']
        filename = 'ordered-' + family + ('-layer-' if family == 'attention' else '-') + key + '.json'
        original = row['original_output'] if family == 'routed' else row['comparison']['original']
        item = dict(key=key, layer=row['layer'], input_sha256=[x['sha256'] for x in row['inputs']],
            original_output_sha256=original['sha256'], complete_row_sha256=sha(D / 'capture/q8192-out512' / filename),
            all_values_match=True, original_operands_and_outputs_match_previous_capture=True,
            inputs_outputs_queues_and_guards_unchanged=True)
        if family == 'dense':
            item.update(projection=row['projection'], bf16_values=row['comparison']['queued_cells'], mismatches=0)
        elif family == 'attention':
            item.update(qk_fp32_values=row['comparison']['qk_fp32_values'], qk_mismatches=0,
                context_bf16_values=row['comparison']['context_bf16_values'], context_mismatches=0,
                slabs=64, original_kernel_hash=row['original_kernel']['kernel_hash'])
        else:
            item.update(full_bf16_values=sum(c['compared_values'] for c in row['comparisons'] if not c['sparse']),
                sparse_bf16_values=sum(c['compared_values'] for c in row['comparisons'] if c['sparse']),
                malformed_controls=len(row['malformed_controls']), mismatches=0)
        compact.append(item)
    families[family] = dict(summary_sha256=analysis['summary_sha256'][family], rows=compact)

report = dict(schema=1, classification='original_model_all_layer_explicit_q8192_cuda_comparison',
    candidate_source_commit=manifest['composition_commit'], original_source_commit=manifest['source_commit'],
    analysis=analysis, analysis_sha256=sha(D / 'operator-analysis.json'),
    families=families, source_manifest_sha256=sha(D / 'source-inputs.json'),
    source_archive_sha256=dispatch['archive_sha256'], exact_reproduction_sources=reproduction,
    observer_composition_control=read(B / 'explicit-q8192-all-layer-composition-control-r1.json'),
    observer_method_equivalence=read(B / 'explicit-q8192-all-layer-observer-equivalence-r1.json'),
    completion={key: dispatch[key] for key in ('returncode', 'reason', 'wall_seconds', 'container_status',
        'minimum_host_available_bytes', 'host_guard_pass', 'gpu_processes_after',
        'original_token_matrix_qualified', 'frozen_autotune_cache_no_misses',
        'frozen_inductor_cache_preserved', 'original_inductor_kernel_source_preserved')},
    download=dict(verified_files=analysis['verified_download_files'],
        manifest_sha256=sha(D / 'compact-download-manifest.json'),
        reclamation=read(B / (NAME + '-download-archive-reclamation.json')),
        complete_evidence_directory='/home/qujing/' + NAME),
    selected_variants=dict(dense='BM64', routed='BM64 full, BM32/BM64/BM128 sparse controls',
        attention='explicit QK/probability/selected PV with unchanged value pack'),
    related_preparation='benchmarks/correctness/explicit-q8192-runtime-preparation-20260923.json',
    limitations=['Candidate results were compared with original model values and never fed to the model.',
        'The complete-layer run uses CUDA on GB10; native gfx1151 arithmetic remains untested for these images.',
        'Diagnostic tiled PV variants are outside this all-layer comparison and current runtime binding.',
        'Observer wall times include hashing, copying, and compilation; no inference performance is accepted.'],
    inference_acceptance=False, performance_acceptance=False, release_qualified=False)
relative = Path('benchmarks/correctness/explicit-q8192-all-layers-20260923.json')
payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
for repo in (R, P):
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(payload)
print(json.dumps(dict(report=str(relative), bytes=len(payload), sha256=sha(R / relative),
                     original_576_outputs=True, original_all_layer_values_match=True)))
