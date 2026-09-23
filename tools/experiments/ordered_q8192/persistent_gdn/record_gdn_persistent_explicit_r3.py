from pathlib import Path
import hashlib
import json
import shutil

B = Path(__file__).resolve().parent
R = B.parents[1]
P = Path('/Users/jiawei-macmini/projects/AIMA-public-r1191-candidate')
D = B/'qrt-gdn-persistent-explicit-20260923-r3'
C = D/'collected'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
manifest = read(D/'manifest.json')
download = read(C/'download-manifest.json')
for name, entry in download.items():
    p = C/name
    assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256']
for name, digest in manifest['source_hashes'].items():
    assert sha(D/name) == sha(C/name) == digest
dispatch = read(C/'dispatch.json')
assert [x['action'] for x in dispatch['actions']] == ['compile', 'pipeline']
assert all(x['returncode'] == 0 and x['cleanup_pass'] for x in dispatch['actions'])
assert dispatch['actions'][1]['gpu_processes_after'] == ''
compile_report = read(C/'aot/result.json')
numeric = read(C/'numerical/result.json')
assert numeric['all_original_values_match'] and numeric['inputs_state_scores_unchanged'] and numeric['guards_pass']
assert [r['name'] for r in numeric['records']] == ['original-q7169-capture', 'original-q7169-runtime',
    'original-first64-runtime', 'nonzero-seeded65-runtime']
for row in numeric['records']:
    assert row['original_values_match'] and row['guards_pass']
    assert all(c['bit_mismatches'] == 0 and c['actual_sha256'] == c['original_sha256'] for c in row['comparisons'].values())
for row in compile_report['images']:
    p = C/'aot'/row['file']
    assert sha(p) == row['sha256'] and p.stat().st_size == row['bytes']
    assert row['resources']['.vgpr_spill_count'] == ['0']
    assert row['resources']['.private_segment_fixed_size'] == ['0']
    assert row['shared_memory_bytes'] == 6144
failures = []
sources = {name: D/name for name in [*manifest['source_hashes'], 'manifest.json']}
sources.update({'dispatch_gdn_persistent_explicit_r3.py':B/'dispatch_gdn_persistent_explicit_r3.py',
                'record_gdn_persistent_explicit_r3.py':Path(__file__)})
for revision in (1, 2):
    prior = B/('qrt-gdn-persistent-explicit-20260923-r%d' % revision)
    old = read(prior/'collected/dispatch.json')
    assert len(old['actions']) == 1 and old['actions'][0]['returncode'] == 1
    assert old['actions'][0]['cleanup_pass'] and not old['actions'][0]['gpu_used']
    failures.append(dict(revision=revision, manifest_sha256=sha(prior/'manifest.json'),
        kernel_sha256=sha(prior/'persistent.py'), dispatch=old,
        stderr_sha256=sha(prior/'collected/compile.stderr.log'),
        reason=manifest['preceding_compile_failure' if revision == 1 else 'preceding_compile_failure_r2']['reason']))
    sources['persistent-failed-r%d.py' % revision] = prior/'persistent.py'
archive = Path('tools/experiments/ordered_q8192/persistent_gdn')
repro = {}
for name, source in sources.items():
    relative = archive/name
    repro[str(relative)] = dict(bytes=source.stat().st_size, sha256=sha(source))
    for repo in (R, P):
        target = repo/relative
        assert not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
report = dict(schema=1, classification='explicit_shared_memory_persistent_gdn_recurrence_experiment',
    candidate_base_commit=manifest['candidate_source_base_commit'],
    original_reference=manifest['reference_model'], source_manifest_sha256=sha(D/'manifest.json'),
    source_archive_sha256=dispatch['source_archive_sha256'], numerical=numeric,
    numerical_sha256=sha(C/'numerical/result.json'), offline_compile=compile_report,
    offline_compile_sha256=sha(C/'aot/result.json'), dispatch=dispatch,
    dispatch_sha256=sha(C/'dispatch.json'), preserved_compile_failures=failures,
    verified_download_files=len(download), download_manifest_sha256=sha(C/'download-manifest.json'),
    exact_reproduction_sources=repro,
    structural_change=dict(block_ownership='one head and eight value rows; all K128 and all chunks',
        blocks=512, block_threads=128, shared_bytes_per_block=6144,
        no_grid_barrier=True, explicit_barriers_between_semantic_phases=True,
        q8192_recurrence_launches_before=384, q8192_recurrence_launches_candidate=1,
        prior_fp32_state_preserved_until_all_outputs_complete=True,
        k16_order_bf16_rounding_and_fp32_fma_unchanged=True),
    limitations=['No runtime binding or native gfx1151 execution yet.',
        'Complete q8192 original-model all-layer comparison remains pending.',
        'CUDA diagnostic times include JIT compilation and are not retained inference performance.',
        'Fewer launches are a structural fact; native and full-model timing still determine selection.'],
    inference_acceptance=False, performance_acceptance=False, release_qualified=False)
relative = Path('benchmarks/correctness/persistent-gdn-explicit-controls-20260923.json')
payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
for repo in (R, P):
    with (repo/relative).open('x') as f: f.write(payload)
print(json.dumps(dict(report=str(relative), bytes=len(payload), sha256=sha(R/relative),
    original_q7169_values_match=True, native_execution=False)))
