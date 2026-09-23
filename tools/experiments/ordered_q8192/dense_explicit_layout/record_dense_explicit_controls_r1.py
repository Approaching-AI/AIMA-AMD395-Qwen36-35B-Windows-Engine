from pathlib import Path
import hashlib
import json
import shutil

B = Path(__file__).resolve().parent
R = B.parents[1]
P = Path('/Users/jiawei-macmini/projects/AIMA-public-r1191-candidate')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
small = B / 'dense-selected-explicit-r1'
full = B / 'dense-selected-explicit-full-r1'
native = B / 'dense-selected-explicit-windows-r1'
a, b = read(small / 'result.json'), read(full / 'result.json')
assert a['all_original_bf16_values_match'] and b['all_original_bf16_values_match']
assert a['total_bf16_values'] == 4362240 and b['total_bf16_values'] == 352321536
for item in (a, b):
    assert item['source_hashes']['kernel.py'] == sha(B / 'dense_selected_explicit_r1.py')
for directory in (small, full):
    d = read(directory / 'dispatch.json')
    assert d['returncode'] == 0 and not d['gpu_processes_after']
archives = {}
destination = Path('tools/experiments/ordered_q8192/dense_explicit_layout')
files = ['dense_selected_explicit_r1.py', 'dense_selected_explicit_worker_r1.py',
         'validate_dense_selected_explicit_r1.py', 'validate_dense_selected_explicit_full_r1.py',
         'dense_selected_u32_full_worker_r1.py', 'prepare_dense_selected_explicit_windows_r1.py',
         'dispatch_dense_selected_explicit_windows_r1.py', 'dense-explicit-source-equivalence-r1.json',
         'record_dense_explicit_controls_r1.py']
for name in files:
    source = B / name
    archives[str(destination / name)] = dict(bytes=source.stat().st_size, sha256=sha(source))
    for repo in (R, P):
        target = repo / destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
plan = read(native / 'manifest.json')
assert sha(native / 'worker.py') == plan['worker_sha256']
assert sha(native / 'worker.py') == sha(B / 'dense-selected-u32-windows-r1/worker.py')
rows = {}
for size, item in a['compiled'].items():
    image = small / item['file']
    assert sha(image) == item['sha256'] == plan['images'][size]['sha256']
    assert item['metadata']['shared'] == item['layout_conversion_count'] == item['shared_memory_instructions'] == 0
    rows[size] = item
report = dict(schema=1, classification='explicit_layout_dense_cuda_controls_native_pending',
    candidate_base_commit='2e7cf269b6885055c60c2b5c13c195744644bf3e',
    candidate_changes_relative_to_base_commit=True,
    candidate_sha256=sha(B / 'dense_selected_explicit_r1.py'),
    arithmetic_source_equivalence=read(B / 'dense-explicit-source-equivalence-r1.json'),
    algorithm='Same selected pairs and ascending K16 FP32 carry; explicit pair/K16 lane layout',
    lane_layout=dict(size_per_thread=[1,1], threads_per_warp=[2,16], warps_per_block=[4,1], order=[1,0]),
    original_reference=dict(host='gb10-4t', actual_host='aitopatom-66c4',
        source_commit='3fcbdafb10a14e94c15bd94a1f821f180bc95373',
        model='/mnt/data/models/Qwen3.6-35B-A3B',
        full_reference_analysis_sha256=sha(B / 'qrt-gb10-dense-full-q8192-20260923-r1/reference-analysis.json'),
        full_reference_dispatch_sha256=b['reference_dispatch_sha256'],
        q8192_case_sha256=b['reference_case_sha256'],
        original_output_tokens=512, first_token=144, first_logit=10.375),
    small_case=dict(manifest={key: read(small / 'manifest.json')[key] for key in
        ('source_hashes', 'candidate_base_commit', 'reference_case_sha256', 'original_prompt', 'projections')},
        manifest_sha256=sha(small / 'manifest.json'),
        dispatch=read(small / 'dispatch.json'), dispatch_sha256=sha(small / 'dispatch.json'),
        result_sha256=sha(small / 'result.json'), compared_bf16_values=4362240, records=a['records'],
        scope='71 real rows, both physical weight layouts, shuffled selected indices, empty/partial/multiple queues'),
    full_q8192=dict(manifest_sha256=sha(full / 'manifest.json'),
        dispatch=read(full / 'dispatch.json'), dispatch_sha256=sha(full / 'dispatch.json'),
        result_sha256=sha(full / 'result.json'), compared_bf16_values=352321536,
        source_tensors=b['sources'], records=b['records'], all_original_values_match=True),
    compiled_images=rows,
    pending_native_trial=dict(manifest_sha256=sha(native / 'manifest.json'),
        native_timeout_seconds=900, cases=54, full_q8192_projections=3,
        selected_and_complete_queues=True, unchanged_baseline_harness=True,
        native_worker_sha256=plan['worker_sha256'],
        dispatcher_sha256=sha(B / 'dispatch_dense_selected_explicit_windows_r1.py'),
        active_owner=plan['active_owner_local_record'], remote_directory=plan['remote_directory'],
        prepared_only=True, staged_on_baiying=False, executed=False),
    exact_reproduction_sources=archives,
    decision='Measure on gfx1151 before optional runtime integration; previous implicit-layout u32 was slower than baseline',
    limitations=['CUDA arithmetic agreement does not establish AMD correctness or speed',
                 'Register usage increases with tile size: 64, 117, 229 VGPRs; zero LDS or spills',
                 'No model engine runs the candidate and no Windows product timing is available'],
    native_execution=False, inference_acceptance=False, performance_acceptance=False, release_qualified=False)
relative = Path('benchmarks/correctness/ordered-dense-explicit-layout-controls-20260923.json')
payload = json.dumps(report, indent=2, default=str) + '\n'
for repo in (R, P):
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload)
print(json.dumps(dict(report=str(relative), sha256=sha(R / relative), bytes=len(payload),
    full_q8192_values=352321536, bit_mismatches=0, native_cases_pending=54)))
