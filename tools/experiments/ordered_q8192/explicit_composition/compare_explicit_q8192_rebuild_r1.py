"""Bind an offline rebuild to the exact images awaiting native qualification."""
from pathlib import Path
import hashlib
import importlib.util
import json
import struct

B = Path(__file__).resolve().parent
E = B.parent.parent.parent / 'AIMA-explicit-q8192-candidate'
D = B / 'qrt-explicit-q8192-runtime-rebuild-20260923-r1'
C = D / 'collected'
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
read = lambda p: json.loads(p.read_text())
spec = importlib.util.spec_from_file_location('elf_controls', B / 'compare_gdn_explicit_rebuild_r2.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
inspect = module.inspect
selected = read(E / 'native/linux_core_port/explicit_q8192_compile.json')
rebuilt = read(C / 'aot/result.json')
dispatch = read(C / 'dispatch.json')
assert dispatch['returncode'] == 0 and dispatch['cleanup_pass'] and not dispatch['gpu_used']
assert selected['source_files'] == rebuilt['source_files']
assert set(selected['selected_images']) == set(rebuilt['compiled'])
rows = []
for name, entry in rebuilt['compiled'].items():
    old = selected['selected_images'][name]
    for key in ('signature', 'constants', 'options'):
        assert entry[key] == old[key], (name, key)
    for key in ('name', 'shared', 'num_warps'):
        assert entry['metadata'][key] == old['metadata'][key], (name, key)
    a = C / 'aot' / entry['file']
    b = B / 'explicit-q8192-runtime-assets-r1' / old['file']
    assert sha(a) == entry['sha256'] and sha(b) == old['sha256']
    left, right = inspect(a.read_bytes()), inspect(b.read_bytes())
    rows.append(dict(name=name, rebuilt_sha256=sha(a), selected_sha256=sha(b),
        all_file_bytes_match=sha(a) == sha(b), rebuilt=left, selected=right,
        executable_layout_and_content_match=left == right))
original = (C / 'aot/qk.hsaco').read_bytes()
h = struct.unpack_from('<16sHHIQQQIHHHHHH', original)
code_mutation_detected = False
for i in range(h[12]):
    s = struct.unpack_from('<IIQQQQIIQQ', original, h[6] + i * h[11])
    if s[2] & 4 and s[2] & 2:
        variant = bytearray(original)
        variant[s[4]] ^= 1
        code_mutation_detected = inspect(variant) != inspect(original)
        break
assert code_mutation_detected
report = dict(schema=1, selected_manifest_sha256=sha(E / 'native/linux_core_port/explicit_q8192_compile.json'),
    rebuilt_report_sha256=sha(C / 'aot/result.json'), dispatch_sha256=sha(C / 'dispatch.json'),
    compiler_sha256=sha(E / 'tools/compile_linux_core_q8192_explicit.py'),
    comparison_script_sha256=sha(Path(__file__)), elf_comparator_sha256=sha(B / 'compare_gdn_explicit_rebuild_r2.py'),
    source_files_match=True, signatures_constants_launches_match=True,
    code_mutation_detected=code_mutation_detected, rows=rows,
    executable_layout_and_content_match=all(r['executable_layout_and_content_match'] for r in rows),
    ignored_loaded_bytes='Only eight-byte ELF e_shoff at file offset 40; zero-fill section offsets do not define file data.',
    selected_images_changed=False, gpu_executed=False, inference_acceptance=False, performance_acceptance=False)
path = D / 'image-comparison-r1.json'
with path.open('x') as out:
    json.dump(report, out, indent=2)
    out.write('\n')
print(json.dumps(dict(report_sha256=sha(path), images=len(rows),
    executable_layout_and_content_match=report['executable_layout_and_content_match'],
    mismatches=[r['name'] for r in rows if not r['executable_layout_and_content_match']])))
assert report['executable_layout_and_content_match']
