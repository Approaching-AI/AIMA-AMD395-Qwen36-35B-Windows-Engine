"""Compose exact pending component images in an isolated optional runtime."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys

B = Path(__file__).resolve().parent
ROOT = B.parent.parent
E = ROOT.parent / 'AIMA-explicit-q8192-candidate'
G = ROOT.parent / 'AIMA-explicit-gdn-layout-candidate'
D = B / 'explicit-q8192-runtime-assets-r1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
assert subprocess.check_output(['git', '-C', str(E), 'rev-parse', 'HEAD'], text=True, timeout=10).strip() == '2e7cf269b6885055c60c2b5c13c195744644bf3e'
assert subprocess.check_output(['git', '-C', str(G), 'rev-parse', 'HEAD'], text=True, timeout=10).strip() == 'b07bf58e5140ea6e256c477f4aacfd39fc89695d'
resume = sys.argv[1:] == ['--resume-source-copy']
assert not sys.argv[1:] or resume
if resume:
    assert D.is_dir() and not list(D.iterdir())
else:
    D.mkdir()
source_dir = E / 'native/providers/explicit_q8192'
source_dir.mkdir(exist_ok=resume)
source_files = {
    'attention_explicit.py': B / 'attention-explicit-layout-r2/source/attention_explicit.py',
    'explicit_dense.py': B / 'routed-selected-explicit-r1/source/explicit_dense.py',
    'routed_explicit.py': B / 'routed-selected-explicit-r1/source/routed_explicit.py',
    'explicit_group16.py': B / 'attention-explicit-layout-r2/source/explicit_group16.py',
    'explicit_exp2.py': B / 'attention-explicit-layout-r2/source/explicit_exp2.py',
    'pack_value.py': E / 'native/providers/ordered_q8192/pack_value.py',
}
for name, source in source_files.items():
    compile(source.read_bytes(), str(source), 'exec')
    if resume:
        assert sha(source) == sha(source_dir / name)
    else:
        shutil.copyfile(source, source_dir / name)
compiled = {}
plans = {}


def import_image(name, entry, path, native):
    assert sha(path) == entry['sha256'] == native['sha256']
    assert path.stat().st_size == entry['bytes'] == native['bytes']
    assert entry['metadata']['name'] == native['symbol']
    assert entry['metadata']['shared'] == native['shared_bytes']
    assert entry['metadata']['num_warps'] == native['num_warps'] == 4
    target = D / (name + '.hsaco')
    target.hardlink_to(path)
    compiled[name] = dict(entry, file=target.name)


dense = read(B / 'dense-selected-explicit-full-r1/result.json')
assert dense['all_original_bf16_values_match']
plan = read(B / 'dense-selected-explicit-windows-r1/manifest.json')
plans['dense'] = sha(B / 'dense-selected-explicit-windows-r1/manifest.json')
for batch in (32, 64, 128):
    entry = dense['compiled'][str(batch)]
    native = plan['images'][str(batch)]
    import_image(f'dense{batch}', entry, B / 'dense-selected-explicit-windows-r1' / native['file'], native)
routed = read(B / 'routed-selected-explicit-r1/result.json')
assert routed['all_components_match']
plan = read(B / 'routed-selected-explicit-windows-r1/manifest.json')
plans['routed'] = sha(B / 'routed-selected-explicit-windows-r1/manifest.json')
for name, entry in routed['compiled'].items():
    entry = dict(entry, signature=dict(X='*bf16', W='*bf16', RouteIds='*i32', RouteWeights='*fp32', Counts='*i32',
        Indices='*i32', Output='*bf16', Debug='*fp32', Invalid='*i32', Routes='i32'),
        constants=dict(DOWN=name.startswith('down'), BM=int(name.split('bm')[1]), WINDOW=4 << 20, CAPTURE_F32=False))
    import_image(name, entry, B / 'routed-selected-explicit-r1' / entry['file'], plan['images']['explicit.' + name])
attention = read(B / 'attention-explicit-layout-r2/result.json')
assert attention['all_components_match']
plan = read(B / 'attention-explicit-layout-windows-r1/manifest.json')
plans['attention'] = sha(B / 'attention-explicit-layout-windows-r1/manifest.json')
for name, entry in attention['compiled'].items():
    import_image(name, entry, B / 'attention-explicit-layout-r2' / entry['file'], plan['images']['explicit.' + name])
pack = plan['images']['pack']
entry = dict(file=pack['file'], bytes=pack['bytes'], sha256=pack['sha256'],
    metadata=dict(name=pack['symbol'], shared=pack['shared_bytes'], num_warps=pack['num_warps']),
    signature=dict(Source='*bf16', Packed='*bf16', T='i32'), constants={},
    options=dict(num_warps=4, num_stages=2, enable_fp_fusion=False))
import_image('pack', entry, B / 'attention-explicit-layout-windows-r1' / pack['file'], pack)
module_path = E / 'tools/compile_linux_core_q8192_explicit.py'
spec = importlib.util.spec_from_file_location('q8192_explicit_compiler', module_path)
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)
includes = compiler.embed(D, compiled)
for path in includes.values():
    shutil.copyfile(path, E / 'native/linux_core_port' / path.name)
gdn_files = ['native/linux_core_port/gb10_gdn_ordered_images.inc',
    'native/linux_core_port/gb10_gdn_ordered_compile.json', 'tools/compile_linux_core_gdn_explicit.py',
    *['native/providers/gdn_explicit_layout/' + name for name in ('ordered_pipeline.py', 'group16.py', 'exp2.py')]]
for name in gdn_files:
    target = E / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(G / name, target)
for name in ['native/linux_core_port/gb10_gdn.hip.cpp', 'native/linux_core_port/gb10_gdn.h',
             'native/linux_core_port/gb10_gdn_host_contract_test.cpp', 'native/providers/gdn/ordered_inverse.py']:
    assert sha(E / name) == sha(G / name)


def replace(relative, old, new):
    path = E / relative
    text = path.read_text()
    assert text.count(old) == 1, (relative, old, text.count(old))
    path.write_text(text.replace(old, new))


replace('tools/test_linux_core_gdn.py', '(shared==64||shared==128||shared==512)', 'shared==0')
replace('tools/test_linux_core_gdn.py', '(shared==64||shared==128||shared==1024)', 'shared==16')
replace('tools/test_linux_core_gdn.py', '?1024u:256u', '?1024u:0u')
for old, new in [('qk.shared==256', 'qk.shared==0'), ('prob.shared==16', 'prob.shared==0'), ('pv.shared==64', 'pv.shared==16')]:
    replace('native/linux_core_port/gb10_ordered_attention_host_test.cpp', old, new)
replace('tools/prepare_linux_core_windows.py',
    'core="original chunk64 cumsum and eight embedded original-order integer-accumulator GDN images",',
    'core="original chunk64 cumsum and inverse; seven explicit-layout original-order unsigned32 GDN images",')
replace('tools/prepare_linux_core_windows.py',
    'embedded_image_bytes=651736, aot_launches_per_linear_layer=390,',
    'embedded_image_bytes=sum(item["bytes"] for item in json.loads(\n'
    '                (ROOT / "native/linux_core_port/gb10_gdn_ordered_compile.json").read_text())["compiled"].values()),\n'
    '            aot_launches_per_linear_layer=390,')
gdn = read(G / 'native/linux_core_port/gb10_gdn_ordered_compile.json')
record = dict(schema=1, source_base_commit='2e7cf269b6885055c60c2b5c13c195744644bf3e',
    source_files={str((source_dir / name).relative_to(E)): sha(source_dir / name) for name in source_files},
    selected_images=compiled, embedded_includes={name: dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)) for name, path in includes.items()},
    exact_pending_native_plan_hashes=plans, gdn_source_commit='b07bf58e5140ea6e256c477f4aacfd39fc89695d',
    gdn_files={name: sha(E / name) for name in gdn_files},
    gdn_images={name: {k: entry[k] for k in ('bytes', 'sha256')} for name, entry in gdn['compiled'].items()},
    runtime_launch_bindings_unchanged=True, scalar_arithmetic_sources_unchanged=True,
    all_images_identical_to_pending_component_trials=True, settings_default_disabled=True,
    diagnostic_tiled_pv_embedded=False, compiler_script_sha256=sha(module_path),
    importer_sha256=sha(Path(__file__)), native_execution=False,
    inference_acceptance=False, performance_acceptance=False, release_qualified=False)
(E / 'native/linux_core_port/explicit_q8192_compile.json').write_text(json.dumps(record, indent=2) + '\n')
(D / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(images=len(compiled), embedded_q8192_images=13, gdn_images=8,
    selected_images_bound_to_pending_trials=True, native_execution=False)))
