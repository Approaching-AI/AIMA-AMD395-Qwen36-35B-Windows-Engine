"""Select the committed fused runtime only after exact native numerical evidence."""
from pathlib import Path
import hashlib,json,subprocess
from linux_core_build_inventory_r1 import inventory
from qualify_explicit_q8192_components_r1 import qualify as qualify_other, COMMIT as OTHER_COMMIT

COMMIT='01d1418b28f55339f28c29b6b47821c29626c984'
PLAN_SHA='11db30bb44542a931e243b5aa1eaa7b7100861bce943cd0745ad3ae248c2cc3b'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()

def validate_case(actual,planned):
    for key in ('name','tokens','kernel_set','recurrence','repeat','candidate_source','reference'):
        assert actual[key]==planned[key],key
    assert actual['all_values_match'] and actual['inputs_unchanged'] and actual['guards_pass']
    assert actual['recurrence_time_diagnostic_only']
    baseline=planned['recurrence']=='unfused_u64'
    capture=planned['recurrence']=='persistent-capture'
    assert actual['recurrence_launches']==(3*((planned['tokens']+63)//64)if baseline else 1)
    assert actual['persistent_initial_unchanged']==(None if baseline else True)
    expected={k:v for k,v in planned['expected'].items()if k!='v-new' or planned['recurrence']!='persistent-runtime'}
    rows={r['surface']:r for r in actual['records']}
    assert len(rows)==len(actual['records'])==len(expected) and set(rows)==set(expected)
    for name,row in rows.items():
        assert row['bit_exact'] and row['sha256']==row['expected_sha256']==expected[name]['sha256']
        assert row['bytes']==expected[name]['bytes'] and row['reference']==expected[name]
    assert len(actual['checkpoints'])==((planned['tokens']+63)//64 if capture else 0)
    for index,row in enumerate(actual['checkpoints']):
        expected_checkpoint=planned['checkpoints'][index]
        assert row['chunk']==index and row['first_position']==64*index
        assert row['tokens']==min(64,planned['tokens']-64*index)
        assert row['bit_exact'] and row['incoming_bf16_sha256']==row['expected_bf16_sha256']==expected_checkpoint['sha256']

def qualify_native(base,checkout):
    root=base/'native-gdn-persistent-windows-r1'
    assert sha(root/'manifest.json')==PLAN_SHA
    plan=read(root/'manifest.json');compiled=read(checkout/'native/linux_core_port/gb10_gdn_persistent_compile.json')
    assert compiled['native_plan_sha256']==PLAN_SHA
    assert sha(root/'worker.py')==plan['worker_sha256']
    for name,expected in compiled['source_files'].items():assert sha(checkout/name)==expected
    assert sha(checkout/'tools/compile_linux_core_gdn_persistent.py')==compiled['compiler_sha256']
    assert sha(checkout/'native/linux_core_port/gb10_gdn_persistent_images.inc')==compiled['embedded_include_sha256']
    for item in [*plan['images'].values(),*plan['source_inputs']]:
        path=root/item['file'];assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes']
    for name,item in compiled['compiled'].items():
        planned=plan['images']['persistent-runtime'if name=='recurrence'else'u64/'+name]
        for key in ('sha256','bytes','signature','constants','options'):assert item[key]==planned[key],(name,key)
        assert item['metadata']['name']==planned['symbol']
        assert item['metadata']['shared']==planned['shared_bytes'] and item['metadata']['num_warps']==planned['num_warps']
    required=[root/'outputs/result.json',root/'run/run-record.json']
    if not all(p.is_file()for p in required):
        return dict(ready=False,reason='Actual 15-case native trial and clean completion are pending',
            manifest_sha256=PLAN_SHA,prepared_images_verified=True,remote_calls=0)
    result,run=map(read,required)
    assert run['host'].lower()==result['host'].lower()=='baiying'
    assert run['reason']=='completed' and run['exit_code']==0
    assert run['host_checks_pass'] and all(run['host_checks'].values()) and not run['after_processes']
    assert result['cleanup_pass'] and result['source_files_unchanged'] and result['guards_pass'] and result['exp2_table_unchanged']
    assert result['all_components_match']
    assert run['spec']['source_manifest_sha256']==result['manifest_sha256']==PLAN_SHA
    assert result['worker_sha256']==plan['worker_sha256']
    assert result['source_commit']==plan['candidate_source_commit']
    for key in ('images','candidate_source_variants','execution_checkout','execution_checkout_commit','source_model_reference'):
        assert result[key]==plan[key]
    assert len(result['cases'])==len(plan['cases'])==15
    for actual,planned in zip(result['cases'],plan['cases']):
        validate_case(actual,planned)
        assert read(root/'outputs'/actual['name']/'result.json')==actual
        if actual['tokens']==64:
            path=root/'outputs'/actual['name']/'w.bin';expected=planned['expected']['w']
            assert sha(path)==expected['sha256'] and path.stat().st_size==expected['bytes']
    groups={name:[c['name']for c in result['cases']if c['recurrence']==name]
        for name in ('unfused_u64','persistent-runtime','persistent-capture')}
    assert all(len(cases)==5 for cases in groups.values())
    return dict(ready=True,manifest_sha256=PLAN_SHA,native_host='baiying',result_sha256=sha(required[0]),
        run_sha256=sha(required[1]),cases=groups,all_selected_values_and_checkpoints_bit_exact=True,
        exact_runtime_images_match_native_trial=True,native_model_loaded=False,
        full_product_correctness_still_required=True,inference_acceptance=False,performance_acceptance=False)

def qualify(base,checkout,dense_batch=64,routed_batch=64,attention=True):
    assert dense_batch in (0,32,64,128) and routed_batch in (0,32,64,128)
    assert subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True,timeout=15).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(checkout),'status','--porcelain'],text=True,timeout=15).strip()
    other=checkout.parent/'AIMA-explicit-q8192-candidate'
    assert subprocess.check_output(['git','-C',str(other),'rev-parse','HEAD'],text=True,timeout=15).strip()==OTHER_COMMIT
    original={x['path']:x for x in inventory(other)};current={x['path']:x for x in inventory(checkout)}
    assert len(original)==438 and len(current)==441
    assert not original.keys()-current.keys()
    assert current.keys()-original.keys()=={'native/linux_core_port/'+name for name in (
        'gb10_gdn_persistent_compile.json','gb10_gdn_persistent_images.inc','persistent_gdn_preparation.json')}
    assert {p for p in original if original[p]!=current[p]}=={
        'native/linux_core_port/gb10_gdn.h','native/linux_core_port/gb10_gdn.hip.cpp',
        'native/linux_core_port/gb10_gdn_host_contract_test.cpp','tools/prepare_linux_core_windows.py'}
    preparation=read(checkout/'native/linux_core_port/persistent_gdn_preparation.json')
    assert preparation['compiled_manifest_sha256']==sha(checkout/'native/linux_core_port/gb10_gdn_persistent_compile.json')
    for entry in preparation['host_checks']['gdn']['inputs']:
        assert sha(checkout/entry['path'])==entry['sha256']
    assert sha(checkout/'tools/test_linux_core_gdn.py')==preparation['host_checks']['gdn']['test_source_sha256']
    if dense_batch or routed_batch or attention:
        extra=qualify_other(base,other,dense_batch=dense_batch,routed_batch=routed_batch,attention=attention,gdn=False)
        selections=extra['selections'];environment=extra['environment']
    else:
        selections={};environment=dict(AIMA_PORT_PREFILL_ORDERED_REPLAY='0',AIMA_PORT_ROUTED_ORDERED_REPLAY='0',
            AIMA_PORT_ORDERED_ATTENTION_PREFILL='0',AIMA_PORT_NATIVE_MOE_PREFILL='0')
    selections['persistent_gdn']=qualify_native(base,checkout)
    environment.update(AIMA_PORT_NATIVE_GDN_PREFILL='1',AIMA_PORT_NATIVE_GDN_PERSISTENT='1')
    return dict(ready=all(x['ready']for x in selections.values()),candidate_commit=COMMIT,
        selections=selections,environment=environment,unchanged_other_runtime_inputs=True,
        full_product_correctness_still_required=True,remote_calls=0,inference_acceptance=False,performance_acceptance=False)

if __name__=='__main__':
    base=Path(__file__).resolve().parent
    print(json.dumps(qualify(base,base.parent.parent.parent/'AIMA-persistent-gdn-candidate'),indent=2))
