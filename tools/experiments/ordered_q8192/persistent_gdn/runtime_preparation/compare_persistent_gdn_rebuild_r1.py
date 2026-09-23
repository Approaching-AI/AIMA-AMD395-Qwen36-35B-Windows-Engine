"""Compare CPU-rebuilt executable content against the exact native trial images."""
from pathlib import Path
import hashlib
import json
import struct
from compare_gdn_explicit_rebuild_r2 import inspect

B=Path(__file__).resolve().parent
F=B.parent.parent.parent/'AIMA-persistent-gdn-candidate'
D=B/'qrt-persistent-gdn-runtime-rebuild-20260923-r1'
C=D/'collected'
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
selected=json.loads((F/'native/linux_core_port/gb10_gdn_persistent_compile.json').read_text())
rebuilt=json.loads((C/'aot/result.json').read_text())
assert rebuilt['source_files']==selected['source_files']
assert rebuilt['compiler_sha256']==selected['compiler_sha256']==sha(F/'tools/compile_linux_core_gdn_persistent.py')
rows=[]
for name,old in selected['compiled'].items():
    entry=rebuilt['compiled'][name]
    for key in ('signature','constants','options'):
        assert entry[key]==old[key],(name,key)
    for key in ('name','shared','num_warps','warp_size'):
        assert entry['metadata'][key]==old['metadata'][key],(name,key)
    original=B/'persistent-gdn-runtime-assets-r1'/old['file']
    replacement=C/'aot'/entry['file']
    assert sha(original)==old['sha256'] and sha(replacement)==entry['sha256']
    left,right=inspect(original.read_bytes()),inspect(replacement.read_bytes())
    rows.append(dict(name=name,selected_sha256=sha(original),rebuilt_sha256=sha(replacement),
        all_file_bytes_match=sha(original)==sha(replacement),selected=left,rebuilt=right,
        executable_layout_and_content_match=left==right))
# A changed instruction must still fail the comparison that allows debug paths.
raw=(C/'aot/recurrence.hsaco').read_bytes();header=struct.unpack_from('<16sHHIQQQIHHHHHH',raw)
for i in range(header[12]):
    section=struct.unpack_from('<IIQQQQIIQQ',raw,header[6]+i*header[11])
    if section[2]&4 and section[2]&2:
        variant=bytearray(raw);variant[section[4]]^=1
        assert inspect(variant)!=inspect(raw)
        break
else: raise AssertionError('No executable section')
report=dict(rows=rows,source_files_match=True,signatures_constants_launches_match=True,
    executable_layout_and_content_match=all(row['executable_layout_and_content_match']for row in rows),
    code_mutation_detected=True,ignored_loaded_bytes='Only eight-byte ELF e_shoff, at file offset 40',
    dispatch_sha256=sha(C/'dispatch.json'),selected_compile_sha256=sha(F/'native/linux_core_port/gb10_gdn_persistent_compile.json'),
    comparison_script_sha256=sha(Path(__file__)),elf_reader_sha256=sha(B/'compare_gdn_explicit_rebuild_r2.py'),
    cpu_only=True,product_images_changed=False,native_execution=False,inference_acceptance=False,performance_acceptance=False)
path=D/'image-comparison.json'
with path.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(report=str(path),sha256=sha(path),images=len(rows),
    executable_layout_and_content_match=report['executable_layout_and_content_match'],
    full_file_matches=sum(row['all_file_bytes_match']for row in rows))))
assert report['executable_layout_and_content_match']
