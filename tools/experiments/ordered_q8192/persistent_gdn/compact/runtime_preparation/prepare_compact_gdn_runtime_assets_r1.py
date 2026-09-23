"""Bind the compact recurrence to the unchanged six-launch native ABI."""
from pathlib import Path
import copy,hashlib,importlib.util,json,shutil
B=Path(__file__).resolve().parent
F=B.parent.parent.parent/'AIMA-persistent-gdn-candidate'
H=F.parent/'AIMA-compact-gdn-candidate'
C=B/'qrt-gdn-persistent-compact-20260923-r1/collected'
N=B/'native-gdn-compact-windows-r1'
D=B/'compact-gdn-runtime-assets-r1';D.mkdir()
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
read=lambda p:json.loads(p.read_text())
source=C/'persistent.py'
assert sha(source)=='68a15d8d08542ff95c9be6f7124ec3b0a32a09b70692d9d0f51b2849a8b8ed1b'
shutil.copyfile(source,H/'native/providers/gdn_persistent/persistent.py')
compiled=copy.deepcopy(read(F/'native/linux_core_port/gb10_gdn_persistent_compile.json'))
compact=next(x for x in read(C/'aot/result.json')['images']if x['name']=='persistent-runtime')
assert compact['sha256']=='d74aa694ebc68bb0c48ffeffd7f2c0a65742f3ef1bac28c8ee89931d23ddb820'
compact=copy.deepcopy(compact);compact['file']='recurrence.hsaco';compiled['compiled']['recurrence']=compact
plan=read(N/'manifest.json')
for name,item in compiled['compiled'].items():
    original=plan['images']['persistent-runtime'if name=='recurrence'else'u64/'+name]
    assert item['sha256']==original['sha256'] and item['bytes']==original['bytes']
    path=N/original['file'];assert sha(path)==item['sha256'];shutil.copyfile(path,D/item['file'])
compiler=H/'tools/compile_linux_core_gdn_persistent.py'
assert sha(compiler)==compiled['compiler_sha256']
spec=importlib.util.spec_from_file_location('compact_embed',compiler)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
include=module.embed(D,compiled['compiled'])
shutil.copyfile(include,H/'native/linux_core_port/gb10_gdn_persistent_images.inc')
compiled.update(source_base_commit='01d1418b28f55339f28c29b6b47821c29626c984',
    recurrence_source_commit='d9958548527efeae90bc893c4ed3c0ce56c9cd71',
    source_files={name:sha(H/name)for name in compiled['source_files']},
    native_plan_sha256=sha(N/'manifest.json'),embedded_include_sha256=sha(include),embedded_include_bytes=include.stat().st_size,
    parent_compile_manifest_sha256=sha(F/'native/linux_core_port/gb10_gdn_persistent_compile.json'),
    native_execution=False,inference_acceptance=False,performance_acceptance=False)
payload=json.dumps(compiled,indent=2)+'\n'
(D/'preparation.json').write_text(payload)
(H/'native/linux_core_port/gb10_gdn_persistent_compile.json').write_text(payload)
print(json.dumps(dict(images=6,image_bytes=sum(x['bytes']for x in compiled['compiled'].values()),
    include_sha256=sha(include),compile_manifest_sha256=sha(D/'preparation.json'),native_executed=False)))
