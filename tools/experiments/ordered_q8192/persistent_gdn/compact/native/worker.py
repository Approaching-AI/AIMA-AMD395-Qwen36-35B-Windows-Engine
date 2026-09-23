from pathlib import Path
import array,ctypes as C,hashlib,json,os,platform,socket,subprocess,sys,time

assert platform.system()=='Windows' and socket.gethostname().split('.')[0].lower()=='baiying'
assert sys.byteorder=='little'
root=Path(sys.argv[1]);m=json.loads((root/'manifest.json').read_text());out=root/'outputs';out.mkdir()
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
digest=lambda b:hashlib.sha256(b).hexdigest()
assert sha(Path(__file__))==m['worker_sha256']
for entry in m['source_inputs']:
 p=root/entry['file'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
assert subprocess.check_output(['git','-C',m['execution_checkout'],'rev-parse','HEAD'],text=True,timeout=15).strip()==m['execution_checkout_commit']
dlls=list((Path(m['rocm_root'])/'bin').glob('amdhip64*.dll'));assert len(dlls)==1
dll_dir=os.add_dll_directory(str(dlls[0].parent));hip=C.WinDLL(str(dlls[0]));void=C.c_void_p;size=C.c_size_t;uint=C.c_uint
def bind(name,args):
 fn=getattr(hip,name);fn.argtypes=args;fn.restype=C.c_int;return fn
def check(code,label):
 if code:raise RuntimeError(label+': HIP '+str(code))
init=bind('hipInit',[uint]);setdev=bind('hipSetDevice',[C.c_int]);getname=bind('hipDeviceGetName',[void,C.c_int,C.c_int])
malloc=bind('hipMalloc',[C.POINTER(void),size]);free=bind('hipFree',[void]);copy=bind('hipMemcpy',[void,void,size,C.c_int]);zero=bind('hipMemset',[void,C.c_int,size]);sync=bind('hipDeviceSynchronize',[])
load=bind('hipModuleLoadData',[C.POINTER(void),void]);getfn=bind('hipModuleGetFunction',[C.POINTER(void),void,C.c_char_p]);unload=bind('hipModuleUnload',[void])
launch=bind('hipModuleLaunchKernel',[void,uint,uint,uint,uint,uint,uint,uint,void,C.POINTER(void),void])
check(init(0),'init');check(setdev(0),'device');device=C.create_string_buffer(256);check(getname(device,256,0),'name')
buffers={};bases={};sizes={};modules=[];images=[];functions={};cases=[];guard_bytes=512;allocation_peak=0
def allocate(name,n,raw=None,clear=False):
 global allocation_peak
 base=void();check(malloc(C.byref(base),n+2*guard_bytes),'allocate '+name);bases[name]=base;sizes[name]=n
 buffers[name]=void(base.value+guard_bytes);check(zero(base,0xa5,n+2*guard_bytes),'guard '+name)
 if clear:check(zero(buffers[name],0,n),'clear '+name)
 if raw is not None:
  assert len(raw)==n;host=C.create_string_buffer(raw);check(copy(buffers[name],C.cast(host,void),n,1),'upload '+name)
 allocation_peak=max(allocation_peak,sum(sizes.values())+len(sizes)*2*guard_bytes)
def download(pointer,n):
 host=C.create_string_buffer(n);check(copy(C.cast(host,void),pointer,n,2),'download');return host.raw
def guards():
 return all(download(base,guard_bytes)==b'\xa5'*guard_bytes and download(void(base.value+guard_bytes+sizes[name]),guard_bytes)==b'\xa5'*guard_bytes for name,base in bases.items())
def execute(name,pointers,tokens,grid):
 image_key=name if name=='cumsum' or name.startswith('persistent-') else case['kernel_set']+'/'+name
 entry=m['images'][image_key]
 args=[void(p.value if isinstance(p,void)else p)for p in pointers]+[C.c_int32(tokens),void(0),void(0)]
 argv=(void*len(args))(*(C.addressof(v)for v in args))
 check(launch(functions[image_key],*grid,entry['num_warps']*32,1,1,entry['shared_bytes'],None,argv,None),'launch '+name)
def sub(name,first,columns,item):return void(buffers[name].value+first*columns*item)
def read_input(entry):
 p=Path(m['input_directory'])/entry['file'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'];return p.read_bytes()
try:
 table=m['exp2_table'];p=Path(table['path']);assert p.stat().st_size==table['bytes'] and sha(p)==table['sha256']
 allocate('table',table['bytes'],p.read_bytes())
 for name,entry in m['images'].items():
  p=root/entry['file'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
  image=C.create_string_buffer(p.read_bytes());images.append(image);module=void();check(load(C.byref(module),C.cast(image,void)),'load '+name);modules.append(module)
  fn=void();check(getfn(C.byref(fn),module,entry['symbol'].encode()),'symbol '+name);functions[name]=fn
 for case in m['cases']:
  start=time.monotonic();T=case['tokens'];directory=out/case['name'];directory.mkdir();records=[];checkpoints=[]
  for name,e in case['inputs'].items():allocate(name,e['bytes'],read_input(e))
  allocate('g',T*32*4)
  allocate('kkt',T*32*64*4);allocate('inverse',T*32*64*2,clear=True)
  for name in ('w','u','v-new','core'):allocate(name,T*4096*2)
  allocate('scores',T*32*64*2);allocate('residual',64*4096*2)
  allocate('state0',32*128*128*4,clear=True);allocate('state1',32*128*128*4)
  def observe(name,reference=None,raw=None,save=True):
   if raw is None:raw=download(buffers[name],sizes[name])
   entry=case['expected'][reference or name]
   row=dict(surface=name,file=name+'.bin',bytes=len(raw),sha256=digest(raw),expected_sha256=entry['sha256'],bit_exact=digest(raw)==entry['sha256'],reference=entry)
   if save and name == 'w' and T == 64:(directory/row['file']).write_bytes(raw)
   records.append(row);print(json.dumps(dict(case=case['name'],surface=name,bit_exact=row['bit_exact'])),flush=True)
  b=buffers;rows=(T+7)//8
  execute('cumsum',[b['g-raw'],b['g']],T,[(T+63)//64,32,1]);observe('g')
  execute('kkt',[b['k'],b['k'],b['beta'],b['g'],b['table'],b['kkt']],T,[rows,8,32]);observe('kkt')
  execute('inverse',[b['kkt'],b['inverse']],T,[(T+63)//64,32,1]);observe('inverse')
  execute('w',[b['inverse'],b['k'],b['beta'],b['g'],b['table'],b['w']],T,[rows,16,32]);observe('w')
  execute('u',[b['inverse'],b['v'],b['beta'],b['u'],0],T,[rows,16,32]);observe('u')
  execute('scores',[b['q'],b['k'],b['beta'],b['g'],b['table'],b['scores']],T,[rows,8,32])
  incoming=b['state0']
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
  immutable=all(digest(download(b[name],sizes[name]))==e['sha256']for name,e in case['inputs'].items())
  guarded=guards();passed=all(r['bit_exact']for r in records)and all(r['bit_exact']for r in checkpoints)and immutable and guarded and cold_initial_unchanged
  row=dict(name=case['name'],tokens=T,kernel_set=case['kernel_set'],recurrence=case['recurrence'],repeat=case['repeat'],
   recurrence_launches=recurrence_launches,recurrence_wall_ms=recurrence_wall_ms,recurrence_time_diagnostic_only=True,persistent_initial_unchanged=None if case['recurrence']=='unfused_u64' else cold_initial_unchanged,candidate_source=case['candidate_source'],reference=case['reference'],records=records,checkpoints=checkpoints,
   inputs_unchanged=immutable,guards_pass=guarded,all_values_match=passed,host_elapsed_ms=(time.monotonic()-start)*1000)
  cases.append(row);(directory/'result.json').write_text(json.dumps(row,indent=2)+'\n')
  for name in list(bases):
   if name!='table':check(free(bases.pop(name)),'case free '+name);del buffers[name];del sizes[name]
 table_unchanged=digest(download(buffers['table'],sizes['table']))==m['exp2_table']['sha256']
 final_guards=guards()
finally:
 check(sync(),'final sync')
 for module in reversed(modules):check(unload(module),'unload')
 for pointer in bases.values():check(free(pointer),'free')
passed=all(c['all_values_match']for c in cases)and table_unchanged and final_guards and len(cases)==len(m['cases'])
source_files_unchanged=all(sha(root/e['file'])==e['sha256'] for e in m['source_inputs'])
passed=passed and source_files_unchanged
report=dict(source_files_unchanged=source_files_unchanged,host=socket.gethostname(),device=device.value.decode(),command=[sys.executable,*sys.argv],
 source_commit=m['candidate_source_commit'],execution_checkout=m['execution_checkout'],execution_checkout_commit=m['execution_checkout_commit'],
 source_model_reference=m['source_model_reference'],manifest_sha256=sha(root/'manifest.json'),worker_sha256=sha(Path(__file__)),
 candidate_source_variants=m['candidate_source_variants'],
 hip_dll=dict(path=str(dlls[0]),sha256=sha(dlls[0]),bytes=dlls[0].stat().st_size),images=m['images'],cases=cases,
 exp2_table_unchanged=table_unchanged,guards_pass=final_guards,cleanup_pass=True,all_components_match=passed,
 peak_device_allocated_bytes=allocation_peak,actual_model_loaded=False,inference_acceptance=False,performance_acceptance=False)
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(all_components_match=passed,cases=len(cases))));sys.exit(0 if passed else 6)
