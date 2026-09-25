"""Record a composed candidate without promoting pending AMD experiments."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

B = Path(__file__).resolve().parent
ROOT = B.parent.parent
E = ROOT.parent / 'AIMA-explicit-q8192-candidate'
G = ROOT.parent / 'AIMA-explicit-gdn-layout-candidate'
W = ROOT.parent / 'AIMA-ordered-q8192-candidate'
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
read = lambda p: json.loads(p.read_text())
manifest = read(E / 'native/linux_core_port/explicit_q8192_compile.json')
hosts = []
for name in ('explicit-gdn-host-r1', 'explicit-prefill-host-r1', 'explicit-attention-host-r1'):
    path = E / 'build' / name / 'result.json'
    data = read(path)
    inputs = data['inputs']
    if isinstance(inputs, list):
        inputs = {e['path']: e['sha256'] for e in inputs}
    assert all(sha(E / name) == expected for name, expected in inputs.items())
    hosts.append(dict(file=f'build/{name}/result.json', sha256=sha(path), inputs=inputs,
        result=data['result'], build_command=data['build_command'], sources_match=True, host_only=True))
build_path = E / 'build/explicit-q8192-prepare-r1/prepare.json'
build = read(build_path)
old = read(W / 'build/ordered-q8192-preparation-20260923-r1/prepare.json')
assert build['images'] == old['images'] and build['overlays'] == old['overlays']
assert build['imported_files'] == 327 and build['imported_bytes'] == 29415573
assert len(build['sources']) == 61 and len(build['overlays']) == 17 and len(build['images']) == 72
attention = dict(build['optional_ordered_attention_prefill'])
assert attention.pop('embedded_include_sha256') == manifest['embedded_includes']['attention']['sha256']
previous_attention = dict(old['optional_ordered_attention_prefill'])
previous_attention.pop('embedded_include_sha256')
assert attention == previous_attention
assert build['optional_adaptations']['gb10_gdn']['native_prefill_opt_in']['embedded_image_bytes'] == 743344
unchanged = ['native/linux_core_port/gb10_gdn.hip.cpp', 'native/linux_core_port/gb10_gdn.h',
    'native/linux_core_port/gb10_ordered_attention.cpp', 'native/linux_core_port/gb10_ordered_attention.h',
    'native/linux_core_port/gb10_prefill_projection.hip.cpp', 'native/linux_core_port/gb10_prefill_projection.h']
for name in unchanged:
    assert sha(E / name) == sha(W / name)
rebuild_dir = B / 'qrt-explicit-q8192-runtime-rebuild-20260923-r1'
rebuild = read(rebuild_dir / 'image-comparison-r1.json')
assert rebuild['executable_layout_and_content_match'] and rebuild['source_files_match'] and rebuild['code_mutation_detected']
assert len(rebuild['rows']) == 15
dispatch = read(rebuild_dir / 'collected/dispatch.json')
assert dispatch['returncode'] == 0 and dispatch['cleanup_pass'] and not dispatch['gpu_used']
sources = {}
for name in ('prepare_explicit_q8192_runtime_r1.py', 'rebuild_explicit_q8192_runtime_r1.py',
             'compare_explicit_q8192_rebuild_r1.py', 'compare_gdn_explicit_rebuild_r2.py', Path(__file__).name):
    source = B / name
    compile(source.read_bytes(), str(source), 'exec')
    relative = Path('tools/experiments/ordered_q8192/explicit_composition') / name
    (E / relative).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, E / relative)
    sources[relative.as_posix()] = dict(bytes=source.stat().st_size, sha256=sha(source))
record = dict(schema=1, classification='explicit_layout_optional_runtime_prepared_pending_native_and_product_qualification',
    base_commit=manifest['source_base_commit'], gdn_source_commit=manifest['gdn_source_commit'],
    controller_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, timeout=10).strip(),
    compiled_manifest_sha256=sha(E / 'native/linux_core_port/explicit_q8192_compile.json'),
    runtime_launch_sources_unchanged={name: sha(E / name) for name in unchanged},
    selected_q8192_images=13, selected_q8192_binary_bytes=521376, selected_gdn_images=8, selected_gdn_binary_bytes=743344,
    exact_pending_native_plan_hashes=manifest['exact_pending_native_plan_hashes'],
    gdn_pending_native_plan_sha256='1ff1ce5417880a0f3a1c64f08d0adfe63ef2febfa9596167afe414b92e326dce',
    host_checks=hosts, build_preparation=dict(file=str(build_path), sha256=sha(build_path), imported_files=327,
        imported_bytes=29415573, existing_images=72, compilation_units=61, overlays=17,
        original_imports_and_generated_overlays_unchanged=True, optional_adaptations=build['optional_adaptations'],
        optional_ordered_attention_prefill=build['optional_ordered_attention_prefill']),
    offline_rebuild=dict(report_sha256=sha(rebuild_dir / 'image-comparison-r1.json'),
        dispatch_sha256=sha(rebuild_dir / 'collected/dispatch.json'), host=dispatch['host'], command=dispatch['command'],
        wall_seconds=dispatch['seconds'], images=15, executable_layout_and_content_match=True,
        code_mutation_detected=True, ignored_loaded_bytes=rebuild['ignored_loaded_bytes'],
        gpu_used=False, selected_images_changed=False),
    prior_gdn_rebuild=dict(file='benchmarks/correctness/ordered-gdn-explicit-runtime-preparation-20260923.json',
        sha256=sha(ROOT / 'benchmarks/correctness/ordered-gdn-explicit-runtime-preparation-20260923.json'),
        gdn_compiler_and_source_and_embedded_images_unchanged_from_b07bf58=True),
    reproduction_sources=sources, defaults_unchanged_and_optional_settings_disabled=True,
    runtime_dependencies_added=[], native_gpu_execution=False, native_windows_build=False,
    inference_acceptance=False, performance_acceptance=False, release_qualified=False)
out = E / 'native/linux_core_port/explicit_q8192_preparation.json'
with out.open('x') as file:
    json.dump(record, file, indent=2)
    file.write('\n')
print(json.dumps(dict(record_sha256=sha(out), bytes=out.stat().st_size, native_execution=False)))
