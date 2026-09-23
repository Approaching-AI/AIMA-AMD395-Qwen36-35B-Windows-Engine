"""Re-embed the six exact pending native images without changing compiled bytes."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil
B=Path(__file__).resolve().parent
F=B.parent.parent.parent/'AIMA-persistent-gdn-candidate'
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
plan=B/'native-gdn-persistent-windows-r1/manifest.json'
compiled=json.loads((F/'native/linux_core_port/gb10_gdn_persistent_compile.json').read_text())
assert sha(plan)==compiled['native_plan_sha256']=='11db30bb44542a931e243b5aa1eaa7b7100861bce943cd0745ad3ae248c2cc3b'
manifest=json.loads(plan.read_text())
compiler=F/'tools/compile_linux_core_gdn_persistent.py'
assert sha(compiler)==compiled['compiler_sha256']
for name,expected in compiled['source_files'].items():assert sha(F/name)==expected
for name,item in compiled['compiled'].items():
    source=manifest['images']['persistent-runtime'if name=='recurrence'else'u64/'+name]
    for key in ('bytes','sha256','signature','constants','options'):assert item[key]==source[key],(name,key)
    assert item['metadata']['name']==source['symbol']
    assert item['metadata']['num_warps']==source['num_warps'] and item['metadata']['shared']==source['shared_bytes']
    path=plan.parent/source['file'];assert sha(path)==source['sha256'] and path.stat().st_size==source['bytes']
    shutil.copyfile(path,out/item['file'])
spec=importlib.util.spec_from_file_location('persistent_embed',compiler)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
include=module.embed(out,compiled['compiled'])
assert sha(include)==compiled['embedded_include_sha256']==sha(F/'native/linux_core_port/gb10_gdn_persistent_images.inc')
report=dict(images=6,image_bytes=sum(x['bytes']for x in compiled['compiled'].values()),
    plan_sha256=sha(plan),compiler_sha256=sha(compiler),include_sha256=sha(include),
    exact_trial_image_bytes_preserved=True,native_execution=False,inference_acceptance=False,performance_acceptance=False)
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
