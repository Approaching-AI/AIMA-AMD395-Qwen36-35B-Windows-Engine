"""Prepare a bounded actual-AMD comparison, without staging or remote calls."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import subprocess

B = Path(__file__).resolve().parent
R = B.parents[1]
OLD = B/'native-gdn-layout-controls-windows-r2'
C = B/'qrt-gdn-persistent-explicit-20260923-r3/collected'
D = B/'native-gdn-persistent-windows-r1'; D.mkdir()
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
read = lambda p: json.loads(p.read_text())
old = read(OLD/'manifest.json')
compile_report = read(C/'aot/result.json')
numeric = read(C/'numerical/result.json')
assert numeric['all_original_values_match'] and numeric['inputs_state_scores_unchanged'] and numeric['guards_pass']
commit = subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True,timeout=10).strip()
source = R/'tools/experiments/ordered_q8192/persistent_gdn/persistent.py'
assert subprocess.check_output(['git','-C',str(R),'show',commit+':tools/experiments/ordered_q8192/persistent_gdn/persistent.py'],timeout=15) == source.read_bytes()
assert sha(source) == numeric['source_hashes']['persistent.py']
images = {name:entry for name,entry in old['images'].items() if name=='cumsum' or name.startswith('u64/')}
for entry in images.values():
    assert sha(OLD/entry['file']) == entry['sha256']
    shutil.copyfile(OLD/entry['file'],D/entry['file'])
for entry in compile_report['images']:
    path = C/'aot'/entry['file']
    assert sha(path) == entry['sha256'] and entry['metadata']['shared'] == 6144
    shutil.copyfile(path,D/path.name)
    images[entry['name']] = dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),
        symbol=entry['metadata']['name'],num_warps=4,shared_bytes=6144,
        source_base_commit=commit,source_changes_relative_to_base_commit=False,
        source_files=numeric['source_hashes'],kernel_hash=entry['metadata']['hash'],
        signature=entry['signature'],constants=entry['constants'],options=entry['options'],
        compilation_report_sha256=sha(C/'aot/result.json'))
cases = []
for shape, repeats in [('first64',3),('continuous7169',2)]:
    template = next(c for c in old['cases'] if c['name'].startswith(shape+'-u64'))
    for repeat in range(repeats):
        variants = ['unfused_u64','persistent-runtime','persistent-capture']
        if repeat % 2: variants.reverse()
        for variant in variants:
            case = copy.deepcopy(template)
            case.update(name=shape+'-'+variant+'-repeat'+str(repeat),recurrence=variant,repeat=repeat,
                kernel_set='u64',candidate_source=commit,upstream_source=old['candidate_source_variants']['u64'])
            cases.append(case)
assert len(cases)==15
s=(OLD/'worker.py').read_text()
s=s.replace("image_key=name if name=='cumsum' else case['kernel_set']+'/'+name",
            "image_key=name if name=='cumsum' or name.startswith('persistent-') else case['kernel_set']+'/'+name")
s=s.replace("assert sha(Path(__file__))==m['worker_sha256']", "assert sha(Path(__file__))==m['worker_sha256']\nfor entry in m['source_inputs']:\n p=root/entry['file'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']")
first=s.index("  incoming=b['state0']\n")
last=s.index("  immutable=",first)
replacement="""  incoming=b['state0']
  recurrence_start=time.monotonic();recurrence_launches=0
  if case['recurrence']=='unfused_u64':
   for chunk,first in enumerate(range(0,T,64)):
    valid=min(64,T-first);grid=[(valid+7)//8,16,32]
    g=sub('g',first,32,4);k=sub('k',first,2048,2);vn=sub('v-new',first,4096,2)
    next_state=b['state1'if chunk%2==0 else 'state0']
    execute('residual',[sub('w',first,4096,2),sub('u',first,4096,2),incoming,g,b['table'],vn,b['residual']],valid,grid)
    execute('state',[k,b['residual'],incoming,g,b['table'],next_state],valid,[16,16,32])
    execute('output',[sub('q',first,2048,2),vn,incoming,g,sub('scores',first,32*64,2),b['table'],sub('core',first,4096,2)],valid,grid)
    incoming=next_state;recurrence_launches+=3
  else:
   capture=case['recurrence']=='persistent-capture'
   if capture:allocate('checkpoints',((T+63)//64)*32*128*128*2)
   execute(case['recurrence'],[b['q'],b['k'],b['w'],b['u'],b['g'],b['scores'],b['table'],b['state0'],b['state1'],b['core'],
       b['v-new'] if capture else 0,b['checkpoints'] if capture else 0],T,[32,16,1])
   incoming=b['state1'];recurrence_launches=1
  check(sync(),'recurrence completion')
  recurrence_wall_ms=(time.monotonic()-recurrence_start)*1000
  if case['recurrence']=='persistent-capture':
   raw=download(b['checkpoints'],sizes['checkpoints']);chunk_bytes=32*128*128*2
   for chunk,first in enumerate(range(0,T,64)):
    actual=digest(raw[chunk*chunk_bytes:(chunk+1)*chunk_bytes]);expected=case['checkpoints'][chunk]
    checkpoint=dict(chunk=chunk,first_position=first,tokens=min(64,T-first),incoming_bf16_sha256=actual,
        expected_bf16_sha256=expected['sha256'],bit_exact=actual==expected['sha256'])
    checkpoints.append(checkpoint)
  if case['recurrence']!='persistent-runtime':observe('v-new')
  observe('core');observe('final-state',raw=download(incoming,32*128*128*4))
  cold_initial_unchanged=case['recurrence']=='unfused_u64' or download(b['state0'],sizes['state0'])==bytes(sizes['state0'])
"""
s=s[:first]+replacement+s[last:]
s=s.replace("and immutable and guarded\n  row=dict", "and immutable and guarded and cold_initial_unchanged\n  row=dict")
s=s.replace("row=dict(name=case['name'],tokens=T,kernel_set=case['kernel_set'],", "row=dict(name=case['name'],tokens=T,kernel_set=case['kernel_set'],recurrence=case['recurrence'],repeat=case['repeat'],\n   recurrence_launches=recurrence_launches,recurrence_wall_ms=recurrence_wall_ms,recurrence_time_diagnostic_only=True,cold_initial_unchanged=cold_initial_unchanged,")
s=s.replace("report=dict(host=", "source_files_unchanged=all(sha(root/e['file'])==e['sha256'] for e in m['source_inputs'])\npassed=passed and source_files_unchanged\nreport=dict(source_files_unchanged=source_files_unchanged,host=")
s=s.replace("if save and name == 'w':", "if save and name == 'w' and T == 64:")
s=s.replace("cold_initial_unchanged=cold_initial_unchanged,", "persistent_initial_unchanged=None if case['recurrence']=='unfused_u64' else cold_initial_unchanged,")
compile(s,str(D/'worker.py'),'exec');(D/'worker.py').write_text(s)
inputs=[]
for name,digest in numeric['source_hashes'].items():
    path=C/name;assert sha(path)==digest
    filename='source-'+name;shutil.copyfile(path,D/filename)
    inputs.append(dict(file=filename,bytes=path.stat().st_size,sha256=digest))
m={key:copy.deepcopy(value) for key,value in old.items() if key in (
    'execution_checkout','execution_checkout_commit','rocm_root','python_executable','exp2_table',
    'source_model_reference','guard_file','guard_sha256','input_directory')}
m.update(candidate_source_commit=commit,worker_sha256=sha(D/'worker.py'),images=images,cases=cases,
    source_inputs=inputs,candidate_source_variants=dict(upstream_u64=old['candidate_source_variants']['u64'],
        persistent=dict(source_commit=commit,kernel_sha256=sha(source),numerical_report_sha256=sha(C/'numerical/result.json'))),
    active_owner_local_record='routed-selected-explicit-windows-r1/run/run-record.json',
    active_owner_remote_record='D:/projects/routed-selected-explicit-20260923-r1/run/run-record.json',
    remote_directory='D:/projects/native-gdn-persistent-20260923-r1',native_timeout_seconds=600,
    upstream_manifest_sha256=sha(OLD/'manifest.json'),
    original_raw_g_computed=True,staged_on_baiying=False,executed=False,
    inference_acceptance=False,performance_acceptance=False)
(D/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
for e in {e['file']:e for c in cases for e in c['inputs'].values()}.values():
    assert sha(OLD/'inputs'/e['file'])==e['sha256']
    target=D/'inputs'/e['file'];target.parent.mkdir(exist_ok=True);target.hardlink_to(OLD/'inputs'/e['file'])
dispatcher=(B/'dispatch_native_gdn_layout_controls_r2.py').read_text()
dispatcher=dispatcher.replace("D=B/'native-gdn-layout-controls-windows-r2'","D=B/'native-gdn-persistent-windows-r1'")
dispatcher=dispatcher.replace('1ff1ce5417880a0f3a1c64f08d0adfe63ef2febfa9596167afe414b92e326dce',sha(D/'manifest.json'))
dispatcher=dispatcher.replace("*[D/e['file']for e in m['images'].values()]]", "*[D/e['file']for e in m['images'].values()],*[D/e['file']for e in m['source_inputs']]]")
dispatcher=dispatcher.replace('The active full256k owner has no completed, clean host record; zero remote calls.',
    'The preceding native component has no completed, clean host record; zero remote calls.')
dispatcher=dispatcher.replace('component only: original u64/u32 controls plus single signed product reduction, internal reduction barriers, and explicit shared-free reduction layout; repeated first64 and full continuous q7169',
    'component only: original u64 upstream stages and repeated unfused/persistent/capture recurrence on first64 and full continuous q7169; timings diagnostic only')
compile(dispatcher,'dispatcher','exec');(B/'dispatch_native_gdn_persistent_r1.py').write_text(dispatcher)
print(json.dumps(dict(manifest_sha256=sha(D/'manifest.json'),worker_sha256=sha(D/'worker.py'),
    cases=len(cases),images=len(images),native_staged=False,remote_calls=0)))
