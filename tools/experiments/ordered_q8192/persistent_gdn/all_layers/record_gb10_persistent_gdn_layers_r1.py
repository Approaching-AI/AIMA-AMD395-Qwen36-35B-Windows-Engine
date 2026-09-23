from pathlib import Path
import hashlib
import json
import shutil

B=Path(__file__).resolve().parent
R=B.parents[1]
P=Path('/Users/jiawei-macmini/projects/AIMA-public-r1191-candidate')
name='qrt-gb10-persistent-gdn-layers-20260923-r1'
D=B/name
N=B/'native-gdn-persistent-windows-r1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
analysis=read(D/'operator-analysis.json')
assert analysis['all_original_values_bit_exact'] and analysis['original_576_tokens_match']
assert analysis['original_operands_and_outputs_match_previous_capture']
dispatch=read(D/'dispatch.json')
summary=read(D/'capture/q8192-out512/ordered-summary.json')
plan=read(N/'manifest.json')
abi=read(B/'native-gdn-persistent-abi-controls-r1.json')
assert abi['manifest_sha256']==sha(N/'manifest.json') and abi['worker_sha256']==sha(N/'worker.py')
layers=[]
for row in summary['layers']:
    candidate,=row['variants']
    assert candidate['variant']=='persistent'
    layers.append(dict(layer=row['layer'],tokens=8192,chunks=128,
        original_inputs=row['original_inputs'],original_outputs=row['original_outputs'],
        core_values=candidate['core']['values'],core_mismatches=candidate['core']['bit_mismatches'],
        core_sha256=candidate['core']['actual']['sha256'],
        state_values=candidate['final_state']['values'],state_mismatches=candidate['final_state']['bit_mismatches'],
        state_sha256=candidate['final_state']['actual']['sha256'],guards_pass=candidate['guards_pass'],
        copied_inputs_original_inputs_and_outputs_unchanged=True,
        complete_row_sha256=sha(D/('capture/q8192-out512/ordered-layer-%02d.json'%row['layer']))))
archive=Path('tools/experiments/ordered_q8192/persistent_gdn')
files={archive/'all_layers'/file:B/file for file in (
    'prepare_gb10_persistent_gdn_layers_r1.py','dispatch_gb10_persistent_gdn_layers_r1.py',
    'run-'+name+'.py','analyze_gb10_persistent_gdn_layers_r1.py','record_gb10_persistent_gdn_layers_r1.py')}
files.update({archive/'all_layers'/'observer.py':D/'scripts/capture_ordered_gdn_layers.py',
    archive/'all_layers'/'source-inputs.json':D/'source-inputs.json',
    archive/'all_layers'/'stage.json':B/(name+'-stage.json'),
    archive/'native'/'manifest.json':N/'manifest.json',archive/'native'/'worker.py':N/'worker.py',
    archive/'native'/'prepare_native_gdn_persistent_r1.py':B/'prepare_native_gdn_persistent_r1.py',
    archive/'native'/'dispatch_native_gdn_persistent_r1.py':B/'dispatch_native_gdn_persistent_r1.py',
    archive/'native'/'check_native_gdn_persistent_abi_r1.py':B/'check_native_gdn_persistent_abi_r1.py'})
repro={}
for relative,source in files.items():
    repro[str(relative)]=dict(bytes=source.stat().st_size,sha256=sha(source))
    for repo in (R,P):
        target=repo/relative;assert not target.exists()
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
report=dict(schema=1,classification='original_model_all_layer_persistent_gdn_cuda_comparison',
    candidate_source_commit=analysis['candidate_base_commit'],analysis=analysis,
    analysis_sha256=sha(D/'operator-analysis.json'),layers=layers,
    summary_sha256=sha(D/'capture/q8192-out512/ordered-summary.json'),
    original_576_output_tokens_preserved=True,source_manifest_sha256=sha(D/'source-inputs.json'),
    source_archive_sha256=dispatch['archive_sha256'],
    completion={k:dispatch[k]for k in ('returncode','reason','wall_seconds','container_status',
        'minimum_host_available_bytes','host_guard_pass','gpu_processes_after','original_token_matrix_qualified',
        'frozen_autotune_cache_no_misses','frozen_inductor_cache_preserved','original_inductor_kernel_source_preserved')},
    exact_reproduction_sources=repro,
    pending_native_comparison=dict(manifest_sha256=sha(N/'manifest.json'),worker_sha256=sha(N/'worker.py'),
        remote_directory=plan['remote_directory'],process_deadline_seconds=600,transport_deadline_seconds=660,
        cases=[{k:c[k]for k in ('name','tokens','recurrence','repeat')}for c in plan['cases']],
        image_bindings=len(plan['images']),original_u64_upstream_images_unchanged=True,
        actual_fused_image_sha256={k:v['sha256']for k,v in plan['images'].items()if k.startswith('persistent-')},
        expected_original_full_outputs_and_final_states=True,
        capture_expected_original_incoming_state_checkpoints=True,
        images_staged=False,amd_execution=False,host_abi_controls=abi,
        active_preceding_owner_record=plan['active_owner_local_record']),
    download=dict(verified_files=analysis['verified_download_files'],
        manifest_sha256=sha(D/'compact-download-manifest.json'),
        reclamation=read(B/(name+'-download-archive-reclamation.json')),
        complete_remote_directory='/home/qujing/'+name),
    limitations=['CUDA comparisons preserve original model outputs and never inject candidate outputs.',
        'Actual gfx1151 stability and timing remain pending; no Windows runtime binds this image.',
        'Fewer launches do not establish retained full-model performance.'],
    inference_acceptance=False,performance_acceptance=False,release_qualified=False)
relative=Path('benchmarks/correctness/persistent-gdn-explicit-all-layers-20260923.json')
payload=json.dumps(report,indent=2,allow_nan=False)+'\n'
for repo in (R,P):
    with (repo/relative).open('x')as f:f.write(payload)
print(json.dumps(dict(report=str(relative),bytes=len(payload),sha256=sha(R/relative),layers=30,
    original_576_outputs=True,native_executed=False)))
