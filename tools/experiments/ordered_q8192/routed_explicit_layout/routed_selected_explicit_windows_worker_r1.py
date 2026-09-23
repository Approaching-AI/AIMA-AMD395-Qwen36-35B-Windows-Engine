from pathlib import Path
import array,ctypes as C,hashlib,json,os,platform,socket,struct,subprocess,sys,time

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


def content_hash(name):
 hasher=hashlib.sha256()
 for offset in range(0,sizes[name],8<<20):
  hasher.update(download(void(buffers[name].value+offset),min(8<<20,sizes[name]-offset)))
 return hasher.hexdigest()
def guards():return all(download(base,guard)==b'\xa5'*guard and download(void(base.value+guard+sizes[name]),guard)==b'\xa5'*guard for name,base in bases.items())

def execute(name,pointers,integers,grid):
 e=m['images'][name];args=[void(p.value if isinstance(p,void)else p)for p in pointers]+[C.c_int32(v)for v in integers]+[void(0),void(0)]
 argv=(void*len(args))(*(C.addressof(v)for v in args))
 check(launch(functions[name],*grid,e['num_warps']*32,1,1,e['shared_bytes'],None,argv,None),'launch '+name)

def timed(fn):
 check(sync(),'before timing');t=time.perf_counter();fn();check(sync(),'after timing');return(time.perf_counter()-t)*1000

def upload_file(name,path,offset,n,expected):
 allocate(name,n);hasher=hashlib.sha256();done=0
 with path.open('rb')as stream:
  stream.seek(offset)
  while done<n:
   raw=stream.read(min(8<<20,n-done));assert raw
   hasher.update(raw);host=C.create_string_buffer(raw)
   check(copy(void(buffers[name].value+done),C.cast(host,void),len(raw),1),'upload '+name)
   done+=len(raw)
 assert done==n and hasher.hexdigest()==expected

def load_weight(source):
 index_path=Path(m['model'])/'model.safetensors.index.json';assert sha(index_path)==m['model_index_sha256']
 index=json.loads(index_path.read_text());key=source['weight_key'];shard=index['weight_map'][key]
 assert Path(shard).name==shard and shard.endswith('.safetensors')
 path=Path(m['model'])/shard
 with path.open('rb')as stream:
  n=struct.unpack('<Q',stream.read(8))[0];assert 0<n<16<<20
  header=json.loads(stream.read(n));entry=header[key]
 assert entry['dtype']=='BF16'and entry['shape']==source['weight']['shape']
 begin,end=entry['data_offsets'];assert 0<=begin<end<=path.stat().st_size-8-n
 assert end-begin==source['weight']['bytes']
 upload_file('weight',path,8+n+begin,end-begin,source['weight']['sha256'])
 return dict(file=str(path),key=key,offset=8+n+begin,bytes=end-begin,sha256=source['weight']['sha256'])

source_records=[];control_records=[]
try:
 for name,e in m['images'].items():
  p=root/e['file'];assert p.stat().st_size==e['bytes']and sha(p)==e['sha256']
  image=C.create_string_buffer(p.read_bytes());images.append(image);module=void()
  check(load(C.byref(module),C.cast(image,void)),'load '+name);modules.append(module)
  fn=void();check(getfn(C.byref(fn),module,e['symbol'].encode()),'symbol '+name);functions[name]=fn
 for source in m['projections']:
  down=source['down'];N,K=(2048,512)if down else(1024,2048);cells=65536*N;window=4<<20;windows=cells//window+2
  weight_record=load_weight(source)
  for name,entry in [('input',source['input']),('ids',m['inputs']['ids']),('weights',m['inputs']['weights'])]:
   path=Path(entry['path']);assert path.stat().st_size==entry['bytes']and sha(path)==entry['sha256']
   upload_file(name,path,0,entry['bytes'],entry['sha256'])
  allocate('counts',windows*4,array.array('i',[0]+[window]*(windows-2)+[0]).tobytes())
  allocate('indices',windows*window*4)
  indices=array.array('I',range(cells)).tobytes();host=C.create_string_buffer(indices)
  check(copy(void(buffers['indices'].value+window*4),C.cast(host,void),len(indices),1),'upload complete middle queues')
  del indices,host
  for name,n in [('output',cells*2),('debug',4),('invalid',4)]:allocate(name,n)
  immutable={name:content_hash(name)for name in('input','weight','ids','weights','counts','indices','debug')}
  b=buffers
  for repeat in range(2):
   batches=(32,64,128)if repeat==0 else(128,64,32)
   for batch in batches:
    for variant in (('baseline','explicit')if repeat==0 else('explicit','baseline')):
     name=variant+'.'+('down'if down else'gate-up')+'-bm'+str(batch)
     check(zero(b['output'],0xa5,sizes['output']),'reset output');check(zero(b['invalid'],0,4),'reset flags')
     milliseconds=timed(lambda:execute(name,[b[k]for k in('input','weight','ids','weights','counts','indices','output','debug','invalid')],
       [65536],[256,windows,1]))
     observed=content_hash('output');flags=struct.unpack('<I',download(b['invalid'],4))[0]
     row=dict(variant=variant,projection=source['name'],down=down,batch=batch,repeat=repeat,logical_tokens=8192,logical_routes=65536,
      n=N,k=K,queue_windows=windows,empty_first_and_last_window=True,bf16_values=cells,output_sha256=observed,
      original_output_sha256=source['output']['sha256'],original_bit_exact=observed==source['output']['sha256'],invalid_flags=flags,
      guards_pass=guards(),host_synchronized_replay_ms=milliseconds,performance_acceptance=False)
     records.append(row);print(json.dumps(row),flush=True)
     (out/('record-'+variant+'-'+source['name']+'-'+str(batch)+'-'+str(repeat)+'.json')).write_text(json.dumps(row,indent=2)+'\n')
     if not row['original_bit_exact']and not(out/('mismatch-'+variant+'-'+source['name']+'-prefix.bin')).exists():
      (out/('mismatch-'+variant+'-'+source['name']+'-prefix.bin')).write_bytes(download(b['output'],4<<20))

  # The complete preceding result is tied to the original model output hash.
  # Keep only137 selected original words before poisoning the output for controls.
  selected=[(i*65537+123)%cells for i in range(137)]
  selected_words={index:download(void(b['output'].value+index*2),2)for index in selected}if records[-1]['original_bit_exact']else{}
  saved_id=download(b['ids'],4);saved_weight=download(b['weights'],4)
  def put(pointer,raw):
   host=C.create_string_buffer(raw);check(copy(pointer,C.cast(host,void),len(raw),1),'upload control')
  def poison_hash(patches):
   digest_value=hashlib.sha256()
   for offset in range(0,sizes['output'],8<<20):
    chunk=bytearray(b'\xa5'*min(8<<20,sizes['output']-offset))
    for index,word in patches.items():
     at=index*2-offset
     if 0<=at<len(chunk):chunk[at:at+2]=word
    digest_value.update(chunk)
   return digest_value.hexdigest()
  empty_hash=poison_hash({})
  controls=[('negative_count',-1,0,64),('oversize_count',window+1,0,64),
   ('negative_index',1,-1,64),('oversize_index',1,cells,64),
   ('negative_expert',1,0,8),('oversize_expert',1,0,8)]
  if down:controls.extend([('nonfinite_scale',1,0,32),('negative_scale',1,0,32),('oversize_scale',1,0,32)])
  if selected_words:controls.insert(0,('partial_shuffled',137,selected[0],0))
  for label,count,index,expected_flags in controls:
   check(zero(b['counts'],0,sizes['counts']),'clear control counts');put(b['counts'],struct.pack('<i',count))
   if label=='partial_shuffled':put(b['indices'],array.array('i',selected).tobytes())
   else:put(b['indices'],struct.pack('<i',index))
   put(b['ids'],saved_id);put(b['weights'],saved_weight)
   if label=='negative_expert':put(b['ids'],struct.pack('<i',-1))
   if label=='oversize_expert':put(b['ids'],struct.pack('<i',256))
   if label=='nonfinite_scale':put(b['weights'],struct.pack('<I',0x7fc00000))
   if label=='negative_scale':put(b['weights'],struct.pack('<f',-0.25))
   if label=='oversize_scale':put(b['weights'],struct.pack('<f',1.25))
   before=[content_hash(name)for name in('ids','weights','counts','indices','debug')]
   check(zero(b['output'],0xa5,sizes['output']),'poison control output');check(zero(b['invalid'],0,4),'clear control flags')
   milliseconds=timed(lambda:execute('explicit.'+('down'if down else'gate-up')+'-bm64',
    [b[k]for k in('input','weight','ids','weights','counts','indices','output','debug','invalid')],[65536],[3,1,1]))
   flags=struct.unpack('<I',download(b['invalid'],4))[0]
   invalid_expert_zero=True
   if expected_flags==8:invalid_expert_zero=download(b['output'],2)==b'\0\0'
   if expected_flags in(8,32):put(b['output'],b'\xa5\xa5')
   expected_hash=poison_hash(selected_words)if label=='partial_shuffled'else empty_hash
   observed=content_hash('output')
   row=dict(projection=source['name'],case=label,expected_flags=expected_flags,actual_flags=flags,
    output_sha256=observed,expected_output_sha256=expected_hash,output_and_unselected_poison_match=observed==expected_hash,
    expected_partial_words_from_original_hash_qualified_same_run_baseline=label=='partial_shuffled',
    invalid_expert_zero=invalid_expert_zero,inputs_unchanged=before==[content_hash(name)for name in('ids','weights','counts','indices','debug')],
    guards_pass=guards(),host_synchronized_ms=milliseconds,performance_acceptance=False)
   row['pass']=flags==expected_flags and all(row[k]for k in('output_and_unselected_poison_match','invalid_expert_zero','inputs_unchanged','guards_pass'))
   control_records.append(row);print(json.dumps(dict(stage='queue-control',**row)),flush=True)
  put(b['counts'],array.array('i',[0]+[window]*(windows-2)+[0]).tobytes())
  check(zero(b['indices'],0xa5,137*4),'restore inactive first queue')
  put(b['ids'],saved_id);put(b['weights'],saved_weight)
  source_records.append(dict(name=source['name'],weight=weight_record,input=source['input'],output=source['output'],
   inputs_queues_debug_unchanged=immutable=={name:content_hash(name)for name in immutable},guards_pass=guards()))
  for name,pointer in list(bases.items()):
   check(free(pointer),'case free '+name);del bases[name];del buffers[name];del sizes[name]
finally:
 check(sync(),'final sync')
 for module in reversed(modules):check(unload(module),'unload')
 for pointer in bases.values():check(free(pointer),'free')
source_unchanged=all(sha(root/e['file'])==e['sha256']for e in m['source_inputs'])
passed=len(records)==24 and len(source_records)==2 and len(control_records)==17 and all(r['pass']for r in control_records) and source_unchanged and all(
 r['original_bit_exact']and r['invalid_flags']==0 and r['guards_pass']for r in records)and all(
 r['inputs_queues_debug_unchanged']and r['guards_pass']for r in source_records)
report=dict(host=socket.gethostname(),device=device.value.decode(),command=[sys.executable,*sys.argv],
 candidate_base_commit=m['candidate_base_commit'],candidate_outside_runtime=True,execution_checkout_commit=m['execution_checkout_commit'],
 model=m['model'],model_reference=m['model_reference'],original_reference=m['original_reference'],
 manifest_sha256=sha(root/'manifest.json'),worker_sha256=sha(Path(__file__)),
 hip_dll=dict(path=str(files[0]),bytes=files[0].stat().st_size,sha256=sha(files[0])),images=m['images'],records=records,queue_controls=control_records,
 sources=source_records,all_components_match=passed,source_files_unchanged=source_unchanged,baseline_and_explicit_interleaved=True,cleanup_pass=True,
 peak_allocated_device_bytes=peak,actual_model_weight_tensors_loaded=True,original_model_engine_loaded=False,
 inference_acceptance=False,performance_acceptance=False)
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(all_components_match=passed,projections=len(source_records),records=len(records))),flush=True)
sys.exit(0 if passed else 6)
