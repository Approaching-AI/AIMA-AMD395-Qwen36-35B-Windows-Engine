from pathlib import Path
import array,ctypes as C,hashlib,json,os,platform,socket,subprocess,sys,time

assert platform.system()=='Windows' and socket.gethostname().split('.')[0].lower()=='baiying'
assert sys.byteorder=='little'
root=Path(sys.argv[1]);m=json.loads((root/'manifest.json').read_text());out=root/'outputs';out.mkdir()
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest();digest=lambda b:hashlib.sha256(b).hexdigest()
assert sha(Path(__file__))==m['worker_sha256']
assert subprocess.check_output(['git','-C',m['execution_checkout'],'rev-parse','HEAD'],text=True,timeout=15).strip()==m['execution_checkout_commit']
for e in m['source_inputs']:assert sha(root/e['file'])==e['sha256']
files=list((Path(m['rocm_root'])/'bin').glob('amdhip64*.dll'));assert len(files)==1
handle=os.add_dll_directory(str(files[0].parent));hip=C.WinDLL(str(files[0]));void=C.c_void_p;size=C.c_size_t;uint=C.c_uint

def bind(name,args):
 fn=getattr(hip,name);fn.argtypes=args;fn.restype=C.c_int;return fn

def check(code,label):
 if code:raise RuntimeError(label+': HIP '+str(code))

init=bind('hipInit',[uint]);setdev=bind('hipSetDevice',[C.c_int]);getname=bind('hipDeviceGetName',[void,C.c_int,C.c_int])
malloc=bind('hipMalloc',[C.POINTER(void),size]);free=bind('hipFree',[void]);copy=bind('hipMemcpy',[void,void,size,C.c_int]);zero=bind('hipMemset',[void,C.c_int,size]);sync=bind('hipDeviceSynchronize',[])
load=bind('hipModuleLoadData',[C.POINTER(void),void]);getfn=bind('hipModuleGetFunction',[C.POINTER(void),void,C.c_char_p]);unload=bind('hipModuleUnload',[void])
launch=bind('hipModuleLaunchKernel',[void,uint,uint,uint,uint,uint,uint,uint,void,C.POINTER(void),void])
check(init(0),'init');check(setdev(0),'device');device=C.create_string_buffer(256);check(getname(device,256,0),'name')
buffers={};bases={};sizes={};modules=[];images=[];functions={};guard=512;peak=0;records=[];passes=[]

def allocate(name,n,raw=None):
 global peak
 base=void();check(malloc(C.byref(base),n+2*guard),'allocate '+name);bases[name]=base;sizes[name]=n;buffers[name]=void(base.value+guard)
 check(zero(base,0xa5,n+2*guard),'guard '+name)
 if raw is not None:
  assert len(raw)==n;host=C.create_string_buffer(raw);check(copy(buffers[name],C.cast(host,void),n,1),'upload '+name)
 peak=max(peak,sum(sizes.values())+2*guard*len(sizes))

def download(pointer,n):
 host=C.create_string_buffer(n);check(copy(C.cast(host,void),pointer,n,2),'download');return host.raw

def content_hash(name):return digest(download(buffers[name],sizes[name]))
def guards():return all(download(base,guard)==b'\xa5'*guard and download(void(base.value+guard+sizes[name]),guard)==b'\xa5'*guard for name,base in bases.items())

def execute(name,pointers,integers,grid):
 e=m['images'][name];args=[void(p.value if isinstance(p,void)else p)for p in pointers]+[C.c_int32(v)for v in integers]+[void(0),void(0)]
 argv=(void*len(args))(*(C.addressof(v)for v in args))
 check(launch(functions[name],*grid,e['num_warps']*32,1,1,e['shared_bytes'],None,argv,None),'launch '+name)

def timed(fn):
 check(sync(),'before timing');t=time.perf_counter();fn();check(sync(),'after timing');return(time.perf_counter()-t)*1000

try:
 for name,e in m['images'].items():
  p=root/e['file'];assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256']
  image=C.create_string_buffer(p.read_bytes());images.append(image);module=void();check(load(C.byref(module),C.cast(image,void)),'load '+name);modules.append(module)
  fn=void();check(getfn(C.byref(fn),module,e['symbol'].encode()),'symbol '+name);functions[name]=fn
 for name,e in m['tables'].items():
  p=Path(e['path']);assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'];allocate(name,e['bytes'],p.read_bytes())
 for name,e in m['inputs'].items():
  p=Path(e['path']);assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'];allocate(name,e['bytes'],p.read_bytes())
 T=8192;QT=128;cells=QT*4096
 for name,n in [('value-packed',T*512*2),('scores',QT*16*T*4),('probability',QT*16*T*2),('scales',QT*16*(T//32+1)*4),('output',T*4096*2),('debug',4)]:allocate(name,n)
 allocate('counts',4,array.array('I',[cells]).tobytes());allocate('indices',cells*4,array.array('I',range(cells)).tobytes())
 unchanged={name:content_hash(name)for name in ['q','k','v','exp2','rcp','counts','indices','debug']}
 b=buffers
 golden=None
 variants=m['variants']
 for repeat in range(2):
  order=list(variants) if repeat==0 else list(reversed(variants))
  for variant in order:
   config=variants[variant]
   for name in ['output','scores','probability','scales']:check(zero(b[name],0xa5,sizes[name]),'reset '+name)
   packing=timed(lambda:execute('pack',[b['v'],b['value-packed']],[T],[16,256,1]))
   packed=content_hash('value-packed');assert packed==m['packed_value_sha256']
   timings=dict(pack=packing,qk=0.0,probability=0.0,pv=0.0)
   for case in m['cases']:
    QS=case['query_start'];assert case['queries']==QT
    destination=void(b['output'].value+QS*4096*2)
    qk=timed(lambda:execute(config['qk'],[b['q'],b['k'],b['scores']],[T,QS,QT],[QT//8,T//8,16]));timings['qk']+=qk
    qk_hash=content_hash('scores')
    prob=timed(lambda:execute(config['probability'],[b['scores'],b['probability'],b['scales'],b['exp2']],[T,QS,QT],[QT//4,16,1]));timings['probability']+=prob
    probability_hash=content_hash('probability');scale_hash=content_hash('scales')
    if config['pv_kind']=='selected':
     pv=timed(lambda:execute(config['pv'],[b['probability'],b['value-packed'],b['scales'],b['rcp'],b['counts'],b['indices'],destination,b['debug']],[T,QS,QT],[1024,1,1]))
    else:
     pv=timed(lambda:execute(config['pv'],[b['probability'],b['value-packed'],b['scales'],b['rcp'],destination],[T,QS,QT],[QT//8,256//config['pv_bn'],16]))
    timings['pv']+=pv
    raw=download(destination,cells*2);actual=digest(raw)
    row=dict(variant=variant,repeat=repeat,query_start=QS,queries=QT,qk_fp32_values=QT*16*T,
     qk_sha256=qk_hash,expected_qk_sha256=case['qk_original_mma']['sha256'],qk_bit_exact=qk_hash==case['qk_original_mma']['sha256'],
     probability_sha256=probability_hash,expected_probability_sha256=case['probability_sha256'],probability_bit_exact=probability_hash==case['probability_sha256'],
     scales_sha256=scale_hash,expected_scales_sha256=case['scales_sha256'],scales_bit_exact=scale_hash==case['scales_sha256'],
     context_bf16_values=cells,context_sha256=actual,expected_context_sha256=case['context_original']['sha256'],
     context_bit_exact=actual==case['context_original']['sha256'],host_synchronized_ms=dict(qk=qk,probability=prob,pv=pv))
    records.append(row)
    if not row['context_bit_exact'] and not(out/('first-'+variant+'-context-mismatch.bin')).exists():(out/('first-'+variant+'-context-mismatch.bin')).write_bytes(raw)
    if QS==0 or QS%2048==0 or QS==8064 or not all(row[k]for k in ['qk_bit_exact','probability_bit_exact','scales_bit_exact','context_bit_exact']):print(json.dumps(row),flush=True)
   full=content_hash('output');guarded=guards()
   passes.append(dict(variant=variant,repeat=repeat,host_synchronized_component_ms=timings,total_component_ms=sum(timings.values()),
    whole_context_sha256=full,expected_context_sha256=m['expected_context_sha256'],whole_context_bit_exact=full==m['expected_context_sha256'],
    packed_value_sha256=packed,guards_pass=guarded,performance_acceptance=False))
   if variant=='baseline' and repeat==0 and full==m['expected_context_sha256']:golden=download(b['output'],sizes['output'])
   print(json.dumps(passes[-1]),flush=True)
 queue_records=[]
 if golden is not None:
  allocate('queue-output',cells*2)
  for QS in [0,8064]:
   case=m['cases'][QS//QT]
   for name in ['probability','scales']:check(zero(b[name],0xa5,sizes[name]),'reset '+name)
   timed(lambda:execute('explicit.qk',[b['q'],b['k'],b['scores']],[T,QS,QT],[QT//8,T//8,16]))
   timed(lambda:execute('explicit.probability',[b['scores'],b['probability'],b['scales'],b['exp2']],[T,QS,QT],[QT//4,16,1]))
   input_pass=(content_hash('scores')==case['qk_original_mma']['sha256'] and content_hash('probability')==case['probability_sha256'] and content_hash('scales')==case['scales_sha256'])
   for count in [0,137]:
    selected=array.array('I',((i*7919+11)%cells for i in range(count)))
    count_host=C.create_string_buffer(array.array('I',[count]).tobytes());check(copy(b['counts'],C.cast(count_host,void),4,1),'upload count')
    if count:
     selected_host=C.create_string_buffer(selected.tobytes());check(copy(b['indices'],C.cast(selected_host,void),count*4,1),'upload selected indices')
    queue_before=(content_hash('counts'),content_hash('indices'))
    check(zero(b['queue-output'],0xa5,sizes['queue-output']),'poison partial output')
    elapsed=timed(lambda:execute('explicit.pv',[b['probability'],b['value-packed'],b['scales'],b['rcp'],b['counts'],b['indices'],b['queue-output'],b['debug']],[T,QS,QT],[17,1,1]))
    actual=download(b['queue-output'],sizes['queue-output']);expected=bytearray(b'\xa5'*len(actual))
    for index in selected:expected[index*2:index*2+2]=golden[(QS*4096+index)*2:(QS*4096+index)*2+2]
    queue_records.append(dict(query_start=QS,selected_count=count,whole_output_including_unselected_poison_matches=actual==expected,
     expected_values_from_original_hash_qualified_same_run_baseline=True,original_context_sha256=m['expected_context_sha256'],
     input_surfaces_match_original=input_pass,queues_unchanged=queue_before==(content_hash('counts'),content_hash('indices')),
     guards_pass=guards(),host_synchronized_ms=elapsed,performance_acceptance=False))
    print(json.dumps(dict(stage='selected-queue',**queue_records[-1])),flush=True)
  for name,raw in [('counts',array.array('I',[cells]).tobytes()),('indices',array.array('I',range(cells)).tobytes())]:
   host=C.create_string_buffer(raw);check(copy(b[name],C.cast(host,void),len(raw),1),'restore '+name)
 immutable=unchanged=={name:content_hash(name)for name in unchanged}
 packed_unchanged=content_hash('value-packed')==m['packed_value_sha256'];final_guards=guards()
finally:
 check(sync(),'final sync')
 for module in reversed(modules):check(unload(module),'unload')
 for pointer in bases.values():check(free(pointer),'free')
passed=len(records)==512 and len(passes)==8 and len(queue_records)==4 and all(all(r[k]for k in ['qk_bit_exact','probability_bit_exact','scales_bit_exact','context_bit_exact'])for r in records)and all(r['whole_context_bit_exact'] and r['guards_pass']for r in passes)and immutable and packed_unchanged and final_guards
passed=passed and all(all(r[k]for k in ['whole_output_including_unselected_poison_matches','input_surfaces_match_original','queues_unchanged','guards_pass'])for r in queue_records)
source_unchanged=all(sha(root/e['file'])==e['sha256']for e in m['source_inputs']);passed=passed and source_unchanged
report=dict(host=socket.gethostname(),device=device.value.decode(),command=[sys.executable,*sys.argv],
 candidate_base_commit=m['candidate_base_commit'],candidate_outside_runtime=True,execution_checkout_commit=m['execution_checkout_commit'],
 model_reference=m['model_reference'],original_reference=m['original_reference'],manifest_sha256=sha(root/'manifest.json'),worker_sha256=sha(Path(__file__)),
 hip_dll=dict(path=str(files[0]),bytes=files[0].stat().st_size,sha256=sha(files[0])),images=m['images'],records=records,passes=passes,selected_queue_controls=queue_records,
 all_components_match=passed,inputs_tables_queues_debug_unchanged=immutable,packed_value_unchanged=packed_unchanged,
 source_files_unchanged=source_unchanged,variants=m['variants'],guards_pass=final_guards,cleanup_pass=True,peak_allocated_device_bytes=peak,
 original_model_loaded=False,inference_acceptance=False,performance_acceptance=False)
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(all_components_match=passed,slabs=len(records))));sys.exit(0 if passed else 6)
