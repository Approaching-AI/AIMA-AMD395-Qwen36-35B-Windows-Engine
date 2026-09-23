from pathlib import Path
import hashlib,importlib.util,json,shutil
B=Path(__file__).resolve().parent
G=Path('/Users/jiawei-macmini/projects/AIMA-explicit-gdn-layout-candidate')
C=B/'qrt-gdn-explicit-layout-20260923-r2/collected'
N=B/'native-gdn-layout-controls-windows-r2'
D=B/'linux-core-gdn-explicit-asset-import-r1';D.mkdir()
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
compiled=read(C/'aot/result.json');plan=read(N/'manifest.json')
current=read(G/'native/linux_core_port/gb10_gdn_ordered_compile.json')
sources={}
for file in ('ordered_pipeline.py','group16.py','exp2.py'):
    relative='native/providers/gdn_explicit_layout/'+file
    assert sha(G/relative)==sha(C/file)==compiled['source_hashes'][file]
    sources[relative]=sha(G/relative)
inverse='native/providers/gdn/ordered_inverse.py'
assert sha(G/inverse)==sha(C/'ordered_inverse.py')==current['source_files'][inverse]
sources[inverse]=sha(G/inverse)
images={e['name']:e for e in compiled['images']}
images['inverse']=current['compiled']['inverse']
for name,entry in images.items():
    native=plan['images']['explicit_layout/'+name]
    source=N/native['file']
    assert entry['sha256']==native['sha256']==sha(source)
    assert entry['bytes']==native['bytes']==source.stat().st_size
    assert entry['metadata']['num_warps']==native['num_warps']==4
    assert entry['metadata']['shared']==native['shared_bytes']==(1024 if name=='inverse'else 0)
    shutil.copy2(source,D/entry['file'])
spec=importlib.util.spec_from_file_location('explicit_compiler',G/'tools/compile_linux_core_gdn_explicit.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
include=module.embed(D,images)
report=dict(source_files=sources,compiler_script_sha256=sha(C/'compile_worker.py'),
    compiler_script_scope='Recorded /work/compile_worker.py from qrt-gdn-explicit-layout-20260923-r2 for seven images; inverse retained from the original64 control.',
    standalone_rebuild_script_sha256=sha(G/'tools/compile_linux_core_gdn_explicit.py'),
    triton_version='3.6.0',compiled=images,embedded_image_sha256=sha(include),embedded_image_bytes=include.stat().st_size,
    compiled_binary_bytes=sum(e['bytes']for e in images.values()),
    artifact_import=dict(importer_sha256=sha(Path(__file__)),
        group16_compilation_report_sha256=sha(C/'aot/result.json'),
        group16_dispatch_sha256=sha(C/'dispatch.json'),
        original_inverse_image_sha256=images['inverse']['sha256'],
        source_snapshot_commit='759b496e',source_snapshot_repository='amd395-windows-qwen',
        original_continuous_q7169_comparison_sha256=sha(C/'pipeline-output/result.json'),
        arithmetic_equivalence_sha256=sha(C/'equivalence-result.json'),
        pending_native_plan_sha256=sha(N/'manifest.json'),
        all_eight_images_identical_to_pending_native_plan=True),
    cpu_only=True,gpu_executed=False,amd_native_qualified=False,
    inference_acceptance=False,performance_acceptance=False)
(D/'result.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copy2(include,G/'native/linux_core_port/gb10_gdn_ordered_images.inc')
shutil.copy2(D/'result.json',G/'native/linux_core_port/gb10_gdn_ordered_compile.json')
print(json.dumps(dict(embedded_include_bytes=report['embedded_image_bytes'],compiled_binary_bytes=report['compiled_binary_bytes'],
    embedded_include_sha256=report['embedded_image_sha256'],image_count=len(images),native_executed=False)))
