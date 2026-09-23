from pathlib import Path
import hashlib
import json
import shutil
import socket
import subprocess

B = Path(__file__).resolve().parent
R = B.parent.parent
G = Path('/Users/jiawei-macmini/projects/AIMA-explicit-gdn-layout-candidate')
U = Path('/Users/jiawei-macmini/projects/AIMA-ordered-gdn-u64-candidate')
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
read = lambda p: json.loads(p.read_text())
D = B / 'qrt-gdn-explicit-runtime-rebuild-20260923-r1'
current_dir = B / 'linux-core-gdn-explicit-preparation-r1'
control_dir = B / 'linux-core-gdn-u64-control-preparation-r2'
current, control = read(current_dir / 'prepare.json'), read(control_dir / 'prepare.json')
matching = ['upstream_revision', 'upstream_inventory_sha256', 'imported_files',
            'imported_bytes', 'overlays', 'images', 'image_bytes']
assert all(current[k] == control[k] for k in matching)
normalize = lambda x, repo, out: x.replace(str(repo), '<checkout>').replace(str(out), '<preparation>')
assert [normalize(p, G, current_dir) for p in current['sources']] == [
    normalize(p, U, control_dir) for p in control['sources']]
generated = {}
for name in ('aot_registry.cpp', 'decode_registry.cpp', 'prefill_registry.cpp', 'frozen_text_registry.cpp', 'aot_images.S'):
    a, b = current_dir / name, control_dir / name
    assert normalize(a.read_text(), G, current_dir) == normalize(b.read_text(), U, control_dir)
    generated[name] = dict(candidate_sha256=sha(a), control_sha256=sha(b),
                           content_matches_after_only_checkout_path_normalization=True)

source_files = ['native/linux_core_port/gb10_gdn_ordered_images.inc',
                'native/linux_core_port/gb10_gdn_ordered_compile.json',
                'tools/compile_linux_core_gdn_explicit.py', 'tools/prepare_linux_core_windows.py',
                'tools/test_linux_core_gdn.py',
                *['native/providers/gdn_explicit_layout/' + f for f in ('ordered_pipeline.py', 'group16.py', 'exp2.py')]]
selected = read(G / source_files[1])
plan_file = B / 'native-gdn-layout-controls-windows-r2/manifest.json'
plan = read(plan_file)
for name, item in selected['compiled'].items():
    planned = plan['images']['explicit_layout/' + name]
    assert item['sha256'] == planned['sha256'] and item['bytes'] == planned['bytes']
    assert sha(B / 'linux-core-gdn-explicit-asset-import-r1' / item['file']) == item['sha256']
assert sum(x['bytes'] for x in selected['compiled'].values()) == 743344
comparison = read(D / 'image-comparison-r2.json')
assert comparison['executable_layout_and_content_match']
host = read(B / 'linux-core-gdn-explicit-host-r1/result.json')
dispatch = read(D / 'collected/dispatch.json')
archive_rel = Path('tools/experiments/ordered_q8192/explicit_gdn_layout/runtime_preparation')
archived = {}
for name in ('prepare_gdn_explicit_runtime_assets_r1.py', 'rebuild_gdn_explicit_runtime_r1.py',
             'compare_gdn_explicit_rebuild_r2.py', 'record_gdn_explicit_runtime_r1.py'):
    src = B / name
    archived[str(archive_rel / name)] = dict(bytes=src.stat().st_size, sha256=sha(src))
    for repo in (R, G):
        dst = repo / archive_rel / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

report = dict(
    schema=1, classification='optional_explicit_layout_gdn_runtime_prepared_native_pending',
    controller_host=socket.gethostname(), candidate_base_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=G, text=True, timeout=10).strip(),
    candidate_changes_relative_to_base_commit=True,
    source_files={f: dict(bytes=(G / f).stat().st_size, sha256=sha(G / f)) for f in source_files},
    source_snapshot_commit='759b496e', exact_reproduction_scripts=archived,
    selected_runtime_images={k: dict(file=v['file'], bytes=v['bytes'], sha256=v['sha256'],
        name=v['metadata']['name'], shared=v['metadata']['shared'], warps=v['metadata']['num_warps'],
        resources=v['resources']) for k, v in selected['compiled'].items()},
    native_trial=dict(plan_sha256=sha(plan_file), cases=len(plan['cases']),
                      all_eight_runtime_images_identical=True, executed=False),
    upstream_reference='benchmarks/correctness/ordered-gdn-explicit-layout-controls-20260923.json',
    upstream_reference_sha256=sha(R / 'benchmarks/correctness/ordered-gdn-explicit-layout-controls-20260923.json'),
    runtime_option='AIMA_PORT_NATIVE_GDN_PREFILL=1', default_enabled=False,
    scope='cold q8192 opt-in prototype; original input preparation, chunk64 cumsum and decode bindings',
    dependencies=dict(binary_image_bytes=743344, previous_u64_image_bytes=738352,
                      additional_binary_bytes=4992, generated_include_bytes=4555326,
                      new_runtime_libraries=0, new_device_allocation_bytes=0,
                      existing_matrix_scratch_bytes=100663296, reused_conversion_scratch_bytes=5242880,
                      compiler='Existing pinned Triton 3.6.0, including Gluon; CPU-only build'),
    host_validation=dict(command=['python3.12', 'tools/test_linux_core_gdn.py', '--out',
                                  str(B / 'linux-core-gdn-explicit-host-r1')],
                         report_sha256=sha(B / 'linux-core-gdn-explicit-host-r1/result.json'), **host),
    build_preparation=dict(command=['python3.12', 'tools/prepare_linux_core_windows.py', '--out', str(current_dir),
                            '--windows-rectangular-ck', '--current-text-decode', '--gb10-convolution', '--gb10-gdn',
                            '--gb10-projections', '--gb10-prefill-projections', '--gb10-normalization', '--gb10-moe'],
                           candidate_report_sha256=sha(current_dir / 'prepare.json'),
                           control_report_sha256=sha(control_dir / 'prepare.json'),
                           matching_fields=matching, compilation_units=len(current['sources']),
                           compilation_units_match=True, generated=generated,
                           imported_files=current['imported_files'], imported_bytes=current['imported_bytes'],
                           images=len(current['images']), image_bytes=current['image_bytes'],
                           overlays=len(current['overlays']),
                           native_prefill_opt_in=current['optional_adaptations']['gb10_gdn']['native_prefill_opt_in']),
    cpu_rebuild=dict(host='aitopatom-66c4', manifest_sha256=sha(D / 'manifest.json'),
                     download_manifest_sha256=sha(D / 'collected/download-manifest.json'),
                     compilation_report_sha256=sha(D / 'collected/aot/result.json'),
                     dispatch_sha256=sha(D / 'collected/dispatch.json'), dispatch=dispatch,
                     comparison_report_sha256=sha(D / 'image-comparison-r2.json'), comparison=comparison,
                     original_failed_comparison=read(D / 'image-comparison.json'),
                     original_failure='SHT_NOBITS incorrectly read from file; corrected postprocessing only, without replacing images'),
    limitations=['No new AMD arithmetic execution or Windows build',
                 'No new model output, TTFT, load-time or retained-performance result',
                 'Seven explicit kernels have zero LDS/spills, but higher VGPR use may reduce occupancy',
                 'Underlying ordinary-u32 AMD failure cause is not established'],
    windows_build_qualified=False, amd_native_qualified=False,
    inference_acceptance=False, performance_acceptance=False, release_qualified=False)
target = Path('benchmarks/correctness/ordered-gdn-explicit-runtime-preparation-20260923.json')
payload = json.dumps(report, indent=2) + '\n'
for repo in (R, G):
    path = repo / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
for name in ('ordered-gdn-explicit-layout-controls-20260923.json',
             'ordered-gdn-internal-reduction-controls-20260923.json',
             'ordered-gdn-u64-native-product-20260923.json'):
    shutil.copyfile(R / 'benchmarks/correctness' / name, G / 'benchmarks/correctness' / name)
print(json.dumps(dict(report=str(target), sha256=sha(R / target), bytes=len(payload),
                      images=8, compilation_units=60, unchanged_overlays=17, native_executed=False)))
