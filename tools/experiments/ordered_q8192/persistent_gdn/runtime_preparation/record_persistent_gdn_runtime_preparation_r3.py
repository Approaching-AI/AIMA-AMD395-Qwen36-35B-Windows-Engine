"""Freeze checked optional runtime inputs; native and model acceptance stay pending."""
from pathlib import Path
import hashlib,json,shutil,socket,subprocess

B=Path(__file__).resolve().parent
F=B.parent.parent.parent/'AIMA-persistent-gdn-candidate'
E=F.parent/'AIMA-explicit-q8192-candidate'
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
read=lambda p:json.loads(p.read_text())
out=F/'build/persistent-gdn-preparation-controls-r3';out.mkdir()
current=F/'build/persistent-gdn-prepare-r2';control=E/'build/explicit-q8192-prepare-r1'
a,b=read(current/'prepare.json'),read(control/'prepare.json')
matching=['upstream_revision','upstream_inventory_sha256','imported_files','imported_bytes','images','image_bytes']
assert all(a[k]==b[k]for k in matching)
norm=lambda text,repo,directory:text.replace(str(directory),'<preparation>').replace(str(repo),'<checkout>')
assert [norm(p,F,current)for p in a['sources']]==[norm(p,E,control)for p in b['sources']]
generated={}
for name in ('aot_registry.cpp','decode_registry.cpp','prefill_registry.cpp','frozen_text_registry.cpp','aot_images.S'):
    assert norm((current/name).read_text(),F,current)==norm((control/name).read_text(),E,control)
    generated[name]=dict(candidate_sha256=sha(current/name),control_sha256=sha(control/name),only_checkout_paths_differ=True)
assert [x['path']for x in a['overlays']]==[x['path']for x in b['overlays']]
changed=[x['path']for x,y in zip(a['overlays'],b['overlays'])if x!=y]
assert changed==['native/src/native_linear_prefill.hip.cpp']
new=(current/'overlay'/changed[0]).read_text();old=(control/'overlay'/changed[0]).read_text()
new_marker=next(line for line in new.splitlines(True)if 'std::fprintf(stderr' in line and 'native_gdn_prefill' in line)
old_marker=next(line for line in old.splitlines(True)if 'std::fprintf(stderr' in line and 'native_gdn_prefill' in line)
assert new.replace('  const bool persistent_gdn_prefill = aima_port::gb10_persistent_gdn_prefill_enabled();\n','').replace(new_marker,old_marker)==old
# Compile the exact generated statement, including variadic argument widths.
source=out/'marker.cpp'
source.write_text('#include <cstdio>\n#include <cstddef>\nint main(){struct { std::size_t layer_index=2; } options;std::size_t tokens=8192;\n'
    'for(bool persistent_gdn_prefill:{false,true}){std::size_t native_gdn_launches=persistent_gdn_prefill?7:390;\n'
    +new_marker+'}\n}\n')
source.write_text('#include <initializer_list>\n'+source.read_text())
command=['clang++','-std=c++17','-Wall','-Wextra','-Werror=format',str(source),'-o',str(out/'marker')]
compile_result=subprocess.run(command,capture_output=True,text=True,timeout=20)
(out/'build.stderr').write_text(compile_result.stderr);compile_result.check_returncode()
result=subprocess.run([str(out/'marker')],capture_output=True,text=True,timeout=5);result.check_returncode()
markers=[json.loads(line)for line in result.stderr.splitlines()]
assert markers==[dict(event='native_gdn_prefill',layer=2,tokens=8192,stages=6 if fused else 8,
    aot_launches=7 if fused else 390,chunk_tokens=64,original_preparation=True,ordered_integer_accumulator=True,
    cold=True,persistent_recurrence=fused)for fused in (False,True)]
# Attention has a separate unchanged HIP stub; retain its actual prior check.
attention=read(E/'build/explicit-attention-host-r1/result.json')
assert attention['returncode']==0
for name,digest in attention['inputs'].items():assert sha(F/name)==digest
host=read(F/'build/persistent-gdn-host-r1/result.json')
prefill=read(F/'build/persistent-prefill-host-r1/result.json')
assert host['result']['persistent_images_and_abi_pass']
compiled=read(F/'native/linux_core_port/gb10_gdn_persistent_compile.json')
rebuild=B/'qrt-persistent-gdn-runtime-rebuild-20260923-r1'
comparison=read(rebuild/'image-comparison.json')
assert comparison['executable_layout_and_content_match']
assets=read(B/'persistent-gdn-runtime-reembed-r1/result.json')
assert assets['include_sha256']==compiled['embedded_include_sha256']
helpers=('prepare_persistent_gdn_runtime_assets_r1.py','rebuild_persistent_gdn_runtime_r1.py',
    'compare_persistent_gdn_rebuild_r1.py','compare_gdn_explicit_rebuild_r2.py','record_persistent_gdn_runtime_preparation_r1.py','record_persistent_gdn_runtime_preparation_r2.py','record_persistent_gdn_runtime_preparation_r3.py')
relative=Path('tools/experiments/ordered_q8192/persistent_gdn/runtime_preparation')
archived={}
for name in helpers:
    target=F/relative/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(B/name,target)
    archived[(relative/name).as_posix()]=dict(bytes=target.stat().st_size,sha256=sha(target))
report=dict(schema=1,classification='optional_persistent_gdn_runtime_prepared_native_pending',controller_host=socket.gethostname(),
    source_base_commit='c85259254b805788dc80680caad37685708c98a8',default_enabled=False,
    option='AIMA_PORT_NATIVE_GDN_PERSISTENT=1',requires='AIMA_PORT_NATIVE_GDN_PREFILL=1',scope='cold q8192',
    compiled_manifest_sha256=sha(F/'native/linux_core_port/gb10_gdn_persistent_compile.json'),
    native_trial_plan_sha256=compiled['native_plan_sha256'],native_trial_cases=15,
    selected_images={name:dict(bytes=row['bytes'],sha256=row['sha256'],symbol=row['metadata']['name'],
        warps=row['metadata']['num_warps'],shared=row['metadata']['shared'],resources=row['resources'])for name,row in compiled['compiled'].items()},
    image_bytes=sum(row['bytes']for row in compiled['compiled'].values()),new_device_allocation_bytes=0,
    launches_including_original_cumsum=7,previous_launches=390,extra_runtime_dependencies=0,
    host_checks=dict(gdn=host,prefill=prefill,attention_prior_report_sha256=sha(E/'build/explicit-attention-host-r1/result.json'),
        attention_inputs_unchanged=True,marker_command=command,marker_source_sha256=sha(source),markers=markers),
    build_preparation=dict(command=['python3.12','tools/prepare_linux_core_windows.py','--out',str(current),
        '--current-text-decode','--gb10-convolution','--gb10-gdn','--gb10-projections',
        '--gb10-prefill-projections','--gb10-normalization','--gb10-moe'],
        report_sha256=sha(current/'prepare.json'),control_report_sha256=sha(control/'prepare.json'),
        imported_files=a['imported_files'],base_images=len(a['images']),compilation_units=len(a['sources']),
        overlays=len(a['overlays']),unchanged_overlays=16,changed_overlay=changed[0],
        only_overlay_change='query selected owner and report exact stages, launch count and persistent_recurrence',
        identical_fields=matching,generated=generated),
    reembedding=assets,cpu_rebuild=dict(dispatch=read(rebuild/'collected/dispatch.json'),
        download_manifest_sha256=sha(rebuild/'collected/download-manifest.json'),comparison_report_sha256=sha(rebuild/'image-comparison.json'),
        executable_content_and_layout_match=True,signatures_constants_launches_match=True,code_mutation_detected=True,
        exact_native_trial_images_preserved=True),reproduction_sources=archived,
    earlier_local_preparation=dict(preserved=True,report_sha256=sha(F/'build/persistent-gdn-prepare-r1/prepare.json'),
        difference='Extra --windows-rectangular-ck was absent from the actual product builder; regenerated with its exact flags',
        runtime_sources_changed=False),
    native_gpu_execution=False,native_windows_build=False,inference_acceptance=False,performance_acceptance=False,release_qualified=False)
path=F/'native/linux_core_port/persistent_gdn_preparation.json'
with path.open('x')as file:json.dump(report,file,indent=2);file.write('\n')
print(json.dumps(dict(report=str(path),sha256=sha(path),images=6,bytes=report['image_bytes'],host_checks_pass=True,native_executed=False)))
