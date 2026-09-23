"""Stage a read-only original-model/operator comparison with frozen caches."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import tarfile

B = Path(__file__).resolve().parent
P = Path('/Users/jiawei-macmini/projects/AIMA-explicit-q8192-candidate')
name = 'qrt-gb10-explicit-q8192-layers-20260923-r1'
prior_name = 'qrt-gb10-prefix256-step189-20260923-r3'
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
read = lambda p: json.loads(p.read_text())
prior = read(B / (prior_name + '-stage.json'))
base = Path(prior['source_archive_base'])
prior_stage = B / (prior_name + '-stage')
stage = B / (name + '-stage')
stage.mkdir()
source_inputs = read(prior_stage / 'source-inputs.json')
files = {}
for relative, digest in source_inputs['files'].items():
    if not (relative.startswith(('scripts/', 'contracts/', 'frozen-'))):
        continue
    source = prior_stage / relative if (prior_stage / relative).is_file() else base / relative
    assert sha(source) == digest
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.hardlink_to(source)
    files[relative] = digest

original = stage / 'scripts/capture_gb10_token_matrix.py'
runner = original.read_text()
assert runner.count('"capture_gb10_token_matrix.TokenMatrixCapture"))') == 1
runner = runner.replace('"capture_gb10_token_matrix.TokenMatrixCapture"))',
                        '"capture_explicit_q8192_layers.ExplicitQ8192Capture"))')
needle = '    cases, oracles = fixtures({7169: args.oracle_q7169, 8192: args.oracle_q8192})\n'
assert runner.count(needle) == 1
runner = runner.replace(needle, needle + '    cases = [c for c in cases if c["name"] in ("q7169-out32", "q8192-out32", "q8192-out512")]\n')
needle = 'timeout_limit = 1800 if max(len(case["prompt_token_ids"]) for case in cases) > 66560 else 600'
assert runner.count(needle) == 1
runner = runner.replace(needle, 'timeout_limit = 1800  # Bound includes 160 dense, 10 attention and 80 routed explicit-layout original-operand comparisons.')
runner_path = stage / 'scripts/capture_gb10_explicit_q8192_matrix.py'
compile(runner, str(runner_path), 'exec')
runner_path.write_text(runner)
files[str(runner_path.relative_to(stage))] = sha(runner_path)
archive_sources=B.parents[1]/'tools/experiments/ordered_q8192'
new_sources={
 'scripts/capture_explicit_q8192_layers.py':B/'capture_explicit_q8192_layers_r1.py',
 **{'scripts/capture_explicit_'+family+'_layers.py':B/('capture_explicit_'+family+'_layers_r1.py') for family in ('dense','attention','routed')},
 **{'candidate/'+file:P/'native/providers/explicit_q8192'/file for file in ('attention_explicit.py','explicit_dense.py','explicit_group16.py','explicit_exp2.py','routed_explicit.py')},
} 
for candidate_name in ['attention.py','integer_u.py','dense_selected.py','ordered_pipeline.py','pack_value.py']:
    new_sources['candidate/'+candidate_name]=archive_sources/candidate_name
for relative,source in new_sources.items():
    compile(source.read_bytes(),str(source),'exec')
    target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);target.hardlink_to(source);files[relative]=sha(target)
operator_sources={
 '/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention.py':'5a8af26832bd0b23604fefa4a0876e12b39101b85431c368649fbb7f779e8921',
 '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py':'0f7c2df8fa972a193922bad89260d6cf1b6acb97a63b9c6675bc0be362d0a1e5',
 '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py':'677fdbad739a5687fefa35cc5c72efba476b6d8e73ae2df4f03114f1d7f2b44c',
 '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2_moe.py':'58899ae017336a4ea00e37788788969e17a6178c113961486acca9e6569f4a8f',
 '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py':'607c0a459306a71ff7d01445772494367f3924098739bbd3b4f43020738297d4'}
candidate_commit = subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
manifest = dict(source_commit=source_inputs['source_commit'],
    dependency_source_commit=source_inputs['dependency_source_commit'],
    candidate_base_commit=candidate_commit,
    files=files, original_operator_sources=operator_sources,
    original_token_matrix_script_sha256=sha(original),
    observer_changes=['default worker extension selects read-only candidate hooks',
        'select two immutable controls and original q8192-out512',
        'bounded operator-comparison budget 1800 seconds'],
    model_compute_sources_modified=False, candidate_outputs_fed_to_model=False,
    controller_candidate_commit=subprocess.check_output(['git','-C',str(B.parents[1]),'rev-parse','HEAD'],text=True,timeout=10).strip(),
    variants=dict(explicit_layout='BM64 dense/routed and selected PV; unchanged explicit arithmetic source bytes'),
    composition_commit='c85259254b805788dc80680caad37685708c98a8',
    observer_equivalence_sha256=sha(B/'explicit-q8192-all-layer-observer-equivalence-r1.json'),
    dense_component_report_sha256=sha(B/'dense-selected-explicit-full-r1/result.json'),
    routed_component_report_sha256=sha(B/'routed-selected-explicit-r1/result.json'),
    complete_layer3_report_sha256=sha(B/'attention-explicit-layout-r2/result.json'),
    value_pack_report_sha256=sha(B/'attention-pack-v-r1/result.json'))
(stage / 'source-inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
archive = B / (name + '.tar')
with tarfile.open(archive, 'x', dereference=True) as tar:
    for relative in sorted([*files, 'source-inputs.json']):
        tar.add(stage / relative, arcname=relative, recursive=False)

wrapper = (B / prior['script']).read_text()
assert sha(B / prior['script']) == prior['script_sha256']
wrapper = wrapper[:wrapper.index("record['long_reference_reproduced']=False")]
wrapper = wrapper.replace(prior_name, name).replace(prior['archive_sha256'], sha(archive))
line = next(x for x in wrapper.splitlines() if x.startswith('command='))
old_command = ast.literal_eval(line[len('command='):])
prefix = old_command[:old_command.index('-w')]
clean = []
i = 0
while i < len(prefix):
    if prefix[i] == '-e' and prefix[i + 1].startswith('QRT_'):
        i += 2
        continue
    clean.append(prefix[i])
    i += 1
clean += ['-v', '/home/qujing/qrt-native-gdn-ordered-pipeline-20260923-r1:/table:ro',
          '-v', '/home/qujing/qrt-attention-ordered-u32-20260923-r1:/rcp:ro',
          '-w', '/work', '--entrypoint', 'timeout',
          'sha256:822f5c399ce6a08c0583302e0d82ff3938158b73977b6e1fbdf00ed72735a30d',
          '--signal=TERM', '--kill-after=10', '1830', 'python3',
          'scripts/capture_gb10_explicit_q8192_matrix.py',
          '--source-commit', source_inputs['source_commit'],
          '--oracle-q7169', 'contracts/arbitrary_q7169_gb10_oracle.json',
          '--oracle-q8192', 'contracts/hprefill_q8192_gb10_oracle.json',
          '--output-dir', 'capture', '--expected-host', 'aitopatom-66c4',
          '--timeout-seconds', '1800', '--execute']
wrapper = wrapper.replace(line, 'command=' + repr(clean))
wrapper += r'''
record['candidate_base_commit']=inputs['candidate_base_commit']
record['original_token_matrix_qualified']=False
record['all_original_attention_results_match']=False
record['all_original_dense_results_match']=False
record['all_original_routed_results_match']=False
if capture_path.exists():
 captured=json.loads(capture_path.read_text())
 frozen=json.loads((root/'contracts/gb10_cold_token_matrix_20260911_oracle.json').read_text())
 frozen={c['name']:c for c in frozen['cases']}
 checks=[]
 for actual in captured['cases']:
  expected=frozen[actual['name']]
  checks.append(dict(name=actual['name'],prompt_matches=actual['prompt']['u32le_sha256']==expected['prompt']['u32le_sha256'],
   all_outputs_match=actual['output_token_ids']==expected['expected']['output_token_ids'],
   first_logit_matches=abs(actual['first_token_raw_logit']-expected['expected']['first_token_raw_logit'])<=0.125))
 record['token_checks']=checks
 record['original_token_matrix_qualified']=bool(captured['completed'] and captured['controls_qualified'] and
  [c['name'] for c in checks]==['q7169-out32','q8192-out32','q8192-out512'] and
  all(all(v for k,v in c.items() if k!='name') for c in checks))
 product=next((c for c in captured['cases'] if c['name']=='q8192-out512'),None)
 if product is not None:
  report=product['worker']['ordered_attention']
  record['all_original_attention_results_match']=bool(report['layers_complete'] and report['all_original_attention_results_match']
   and report['tables_unchanged'] and report['pending_calls']==0)
  for family in ('dense','routed'):
   report=product['worker']['ordered_'+family]
   record['all_original_'+family+'_results_match']=bool(report['projections_complete'] and report['all_original_'+family+'_results_match'] and report['pending_calls']==0)

record['operator_comparison_qualified']=bool(record['original_token_matrix_qualified'] and
 record['all_original_attention_results_match'] and record['all_original_dense_results_match'] and record['all_original_routed_results_match'] and record['frozen_autotune_cache_no_misses'] and
 record['host_guard_pass'] and record['frozen_inductor_cache_preserved'] and record['original_inductor_kernel_source_preserved'])
record['inference_acceptance']=False
record['performance_acceptance']=False
(root/'dispatch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
for line in (root/'container.log').read_text(errors='replace').splitlines()[-18:]:print(line[:1500])
selected=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.log','.txt','.py','.bin','.best_config') and p.stat().st_size<4*(1<<20)]
assert len(selected)<500 and sum(p.stat().st_size for p in selected)<40*(1<<20)
download={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in selected}
(root/'compact-download-manifest.json').write_text(json.dumps(download,indent=2)+'\n')
selected.append(root/'compact-download-manifest.json')
compact=root/'compact-download.tar.gz'
with tarfile.open(compact,'x:gz',compresslevel=1,dereference=True) as tar:
 for p in selected:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
print(json.dumps(dict(compact_archive_bytes=compact.stat().st_size,
 compact_archive_sha256=hashlib.file_digest(compact.open('rb'),'sha256').hexdigest(),
 selected_files=len(selected),selected_bytes=sum(p.stat().st_size for p in selected),
 collection_pass=True,operator_comparison_qualified=record['operator_comparison_qualified'])),flush=True)
raise SystemExit(process.returncode or (0 if record['operator_comparison_qualified'] else 1))
'''
script = B / ('run-' + name + '.py')
compile(wrapper, str(script), 'exec')
script.write_text(wrapper)
dispatcher = B / 'dispatch_gb10_explicit_q8192_layers_r1.py'
dispatch = (B / 'dispatch_gb10_step189_r3.py').read_text().replace(prior_name, name)
compile(dispatch, str(dispatcher), 'exec')
dispatcher.write_text(dispatch)
record = dict(name=name, archive=archive.name, archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
    script=script.name, script_sha256=sha(script), dispatcher=dispatcher.name,
    dispatcher_sha256=sha(dispatcher), source_inputs_sha256=sha(stage / 'source-inputs.json'),
    ssh_timeout_seconds=1950, candidate_base_commit=candidate_commit,
    original_model_sources_unchanged=True, original_576_outputs_required=True,
    observer_source_base_commit=source_inputs['source_commit'], generator_sha256=sha(Path(__file__)))
(B / (name + '-stage.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
