"""Publish preparation evidence while preserving pending native/product status."""
from pathlib import Path
import hashlib,json,shutil,socket,subprocess
from linux_core_build_inventory_r1 import inventory
from qualify_persistent_gdn_components_r1 import COMMIT,PLAN_SHA

B=Path(__file__).resolve().parent;R=B.parents[1]
P=R.parent/'AIMA-public-r1191-candidate';F=R.parent/'AIMA-persistent-gdn-candidate'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert subprocess.check_output(['git','-C',str(F),'rev-parse','HEAD'],text=True,timeout=15).strip()==COMMIT
assert not subprocess.check_output(['git','-C',str(F),'status','--porcelain'],text=True,timeout=15).strip()
preparation=read(F/'native/linux_core_port/persistent_gdn_preparation.json')
control=B/'persistent-gdn-product-entry-controls-r1';control.mkdir()
controls=[]
for index,options in enumerate(([],['--dense-tile','0','--routed-tile','0','--without-attention'],
        ['--dense-tile','32','--routed-tile','128','--without-attention'])):
    command=['python3.12',str(B/'stage_persistent_gdn_product_r1.py'),'--check',*options]
    result=subprocess.run(command,capture_output=True,text=True,timeout=30);result.check_returncode()
    observed=json.loads(result.stdout)
    assert observed['remote_calls']==0 and observed['staged']is False and observed['selection']['ready']is False
    assert observed['selection']['candidate_commit']==COMMIT
    assert observed['selection']['environment']['AIMA_PORT_NATIVE_GDN_PERSISTENT']=='1'
    (control/('selection-'+str(index)+'.json')).write_text(result.stdout)
    controls.append(dict(command=command,returncode=0,result=observed))
for file in ('linux-core-windows-persistent-gdn-source-r1.json','linux-core-windows-persistent-gdn-stage-r1.json',
        'linux-core-windows-persistent-gdn-source-r1.bundle'):
    assert not (B/file).exists()
prefix=Path('tools/experiments/ordered_q8192/persistent_gdn/runtime_preparation')
files={prefix/p.name:p for p in (F/prefix).glob('*.py')}
for name in ('qualify_persistent_gdn_components_r1.py','stage_persistent_gdn_product_r1.py',
        'dispatch_persistent_gdn_product_r1.py','qualify_explicit_q8192_components_r1.py',
        'linux_core_build_inventory_r1.py','record_persistent_gdn_product_entry_r1.py'):
    files[prefix/name]=B/name
archived={}
for relative,source in files.items():
    compile(source.read_text(),str(source),'exec')
    archived[str(relative)]=dict(bytes=source.stat().st_size,sha256=sha(source))
    for repo in (R,P):
        target=repo/relative;target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists();shutil.copyfile(source,target)
inputs=inventory(F);assert len(inputs)==441
report=dict(schema=1,classification='optional_persistent_gdn_runtime_and_conditional_product_entry_prepared',
    controller_host=socket.gethostname(),candidate_commit=COMMIT,candidate_branch='codex/persistent-gdn-prefill',
    candidate_checkout=str(F),preparation_sha256=sha(F/'native/linux_core_port/persistent_gdn_preparation.json'),
    preparation=preparation,actual_build_source_inputs=inputs,actual_build_source_input_count=len(inputs),
    immutable_original_reference_reports={name:sha(R/'benchmarks/correctness'/name)for name in (
        'persistent-gdn-explicit-controls-20260923.json','persistent-gdn-explicit-all-layers-20260923.json')},
    exact_reproduction_sources=archived,local_entry_controls=controls,
    native_trial=dict(manifest_sha256=PLAN_SHA,cases=15,staged=False,executed=False,
        prerequisite='Current full256k owner completes cleanly, then preceding native component owners complete'),
    product_entry=dict(remote_checkout='P:/projects/AIMA-public-persistent-gdn-20260923-r1',
        candidate_source_commit=COMMIT,stage_files_created=False,remote_calls=0,
        requires_original_native_outputs=True,requires_15_native_control_runtime_capture_cases=True,
        requires_original_q8192_out512_tokens_and_first_logit=True,requires_same_run_30_layer_activation_markers=True,
        build_process_timeout_seconds=1740,build_transport_timeout_seconds=1860,
        product_process_timeout_seconds=600,product_transport_timeout_seconds=720,
        dependent_component_settings='Choose 0/32/64/128 dense/routed and optional attention only after their actual native results'),
    imported_source_whitespace_preserved=['native/providers/gdn_persistent/exp2.py','native/providers/gdn_persistent/group16.py'],
    native_windows_build=False,inference_acceptance=False,performance_acceptance=False,release_qualified=False)
path=Path('benchmarks/correctness/persistent-gdn-runtime-preparation-20260923.json')
payload=json.dumps(report,indent=2,allow_nan=False)+'\n'
for repo in (R,P):
    with(repo/path).open('x')as file:file.write(payload)
print(json.dumps(dict(report=str(path),sha256=sha(R/path),bytes=len(payload),candidate_commit=COMMIT,
    actual_build_inputs=441,local_controls=len(controls),remote_calls=0,executed=False)))
