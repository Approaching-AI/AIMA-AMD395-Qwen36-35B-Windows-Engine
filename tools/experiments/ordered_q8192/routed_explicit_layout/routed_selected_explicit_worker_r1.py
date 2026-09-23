"""Replay all real q8192 routes and preserve original malformed-queue controls."""
from pathlib import Path
import hashlib, json, re, socket, time
import torch, triton
from safetensors import safe_open
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource
import routed_selected as original
import routed_explicit as candidate
ROOT=Path('/work');WINDOW=4<<20
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def info(value):
 import torch
 raw=value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
 return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

class Pipeline:
 def __init__(self,module):
  self.module=module
  self.options=dict(num_warps=4,num_stages=2,enable_fp_fusion=False)

 def compare(self,x,w,ids,weights,original,down,sparse=False,batch=64):
  import torch
  routes=65536;N,K=(2048,512)if down else(1024,2048);cells=routes*N
  assert x.shape==(routes if down else 8192,K)and w.shape==(256,N,K)
  assert ids.shape==(8192,8)and weights.shape==ids.shape and original.shape==(8192,8,N)
  assert x.dtype==w.dtype==original.dtype==torch.bfloat16
  assert ids.dtype==torch.int32 and weights.dtype==torch.float32
  guarded=[]
  def allocate(shape,dtype):
   count=1
   for v in shape:count*=v
   backing=torch.full((count*torch.empty((),dtype=dtype).element_size()+1024,),0xa5,dtype=torch.uint8,device='cuda')
   value=backing[512:-512].view(dtype).reshape(shape);guarded.append(backing);return value
  windows=5 if sparse else cells//WINDOW
  counts=allocate((windows,),torch.int32);indices=allocate((windows,WINDOW),torch.int32)
  if sparse:
   count_values=[0,17,4056,20,0];counts.copy_(torch.tensor(count_values,dtype=torch.int32,device='cuda'))
   selected=((torch.arange(4093,dtype=torch.int64,device='cuda')*65537+123)%cells).to(torch.int32)
   indices.fill_(-1);offset=0
   for window,count in enumerate(count_values):
    indices[window,:count].copy_(selected[offset:offset+count]);offset+=count
   assert offset==4093
  else:
   counts.fill_(WINDOW);indices.copy_(torch.arange(cells,dtype=torch.int32,device='cuda').reshape(windows,WINDOW))
  output=allocate(original.shape,torch.bfloat16);debug=allocate((1,),torch.float32);invalid=allocate((1,),torch.int32);invalid.zero_()
  queues_before=(info(counts),info(indices));debug_before=info(debug)
  started=time.monotonic()
  self.module.routed_replay_kernel[(256,windows)](x,w,ids,weights,counts,indices,output,debug,invalid,routes,
   DOWN=down,BM=batch,WINDOW=WINDOW,CAPTURE_F32=False,**self.options)
  torch.cuda.synchronize();seconds=time.monotonic()-started
  actual=output.flatten()[selected.long()]if sparse else output
  expected=original.flatten()[selected.long()]if sparse else original
  different=actual.view(torch.int16)!=expected.view(torch.int16)
  row=dict(down=down,batch=batch,sparse=sparse,logical_routes=routes,n=N,k=K,windows=windows,
   compared_values=actual.numel(),bf16_mismatches=int(different.sum()),invalid_flags=int(invalid.item()),
   actual=info(actual),original=info(expected),queues_unchanged=queues_before==(info(counts),info(indices)),
   debug_unchanged=debug_before==info(debug),diagnostic_kernel_seconds=seconds,performance_acceptance=False)
  if row['bf16_mismatches']:
   first=different.flatten().nonzero()[:16,0]
   row['first_differences']=[dict(index=int(i),actual=int(actual.view(torch.int16).flatten()[i]),
    original=int(expected.view(torch.int16).flatten()[i]))for i in first]
  if sparse:
   output.view(torch.int16).flatten()[selected.long()]=-23131
   row['unselected_outputs_unchanged']=bool((output.view(torch.int16)==-23131).all())
  row['guards_pass']=all(bool((b[:512]==0xa5).all())and bool((b[-512:]==0xa5).all())for b in guarded)
  return row

 def malformed(self,x,w,ids,weights,down):
  import torch
  N=2048 if down else 1024;routes=65536;guards=[]
  def alloc(n,dtype):
   b=torch.full((n*torch.empty((),dtype=dtype).element_size()+1024,),0xa5,dtype=torch.uint8,device='cuda')
   guards.append(b);return b[512:-512].view(dtype)
  counts=alloc(1,torch.int32);queue=alloc(WINDOW,torch.int32);output=alloc(routes*N,torch.bfloat16)
  debug=alloc(1,torch.float32);invalid=alloc(1,torch.int32);local_ids=ids.clone();local_weights=weights.clone()
  cases=[('negative_count',-1,0,64),('oversize_count',WINDOW+1,0,64),
   ('negative_index',1,-1,64),('oversize_index',1,routes*N,64),
   ('negative_expert',1,0,8),('oversize_expert',1,0,8)]
  if down:cases.extend([('nonfinite_scale',1,0,32),('negative_scale',1,0,32),('oversize_scale',1,0,32)])
  rows=[]
  for name,count,index,expected in cases:
   local_ids.copy_(ids);local_weights.copy_(weights);output.view(torch.int16).fill_(-23131)
   counts.fill_(count);queue.fill_(-1);queue[0]=index;invalid.zero_()
   if name=='negative_expert':local_ids.flatten()[0]=-1
   if name=='oversize_expert':local_ids.flatten()[0]=256
   if name=='nonfinite_scale':local_weights.flatten()[0]=float('nan')
   if name=='negative_scale':local_weights.flatten()[0]=-0.25
   if name=='oversize_scale':local_weights.flatten()[0]=1.25
   before=[info(v)for v in (local_ids,local_weights,counts,queue,debug)]
   self.module.routed_replay_kernel[(3,1)](x,w,local_ids,local_weights,counts,queue,output,debug,invalid,routes,
    DOWN=down,BM=64,WINDOW=WINDOW,CAPTURE_F32=False,**self.options)
   torch.cuda.synchronize();flags=int(invalid.item())
   if expected in(8,32):
    if expected==8:assert int(output.view(torch.int16)[0])==0
    output.view(torch.int16)[0]=-23131
   row=dict(case=name,expected_flags=expected,actual_flags=flags,
    only_declared_output_touched=bool((output.view(torch.int16)==-23131).all()),
    inputs_unchanged=before==[info(v)for v in(local_ids,local_weights,counts,queue,debug)],
    guards_pass=all(bool((b[:512]==0xa5).all())and bool((b[-512:]==0xa5).all())for b in guards))
   row['pass']=flags==expected and all(row[k]for k in('only_declared_output_touched','inputs_unchanged','guards_pass'))
   rows.append(row)
  return rows


out=ROOT/'output';out.mkdir()
m=json.loads((ROOT/'manifest.json').read_text());torch.set_num_threads(4)
assert torch.cuda.get_device_capability()==(12,1)and triton.__version__=='3.6.0'
for name,digest in m['source_hashes'].items():assert sha(ROOT/name)==digest,name
ref=Path('/reference');reference=json.loads((ROOT/'reference-analysis.json').read_text())
assert reference['reference_qualified']and reference['original_576_tokens_and_full_first_logits_match']
assert sha(ref/'dispatch.json')==reference['dispatch_sha256']
assert sha(ref/'capture/capture.json')==reference['capture_sha256']
assert json.loads((ref/'dispatch.json').read_text())['operator_comparison_qualified']
model=Path('/models');index_path=model/'model.safetensors.index.json'
assert sha(index_path)==m['model_index_sha256'];index=json.loads(index_path.read_text())
options=dict(num_warps=4,num_stages=2,enable_fp_fusion=False);compiled={}
for down in(False,True):
 for batch in(32,64,128):
  c=triton.compile(GluonASTSource(candidate.routed_replay_kernel,
   signature=dict(X='*bf16',W='*bf16',RouteIds='*i32',RouteWeights='*fp32',Counts='*i32',Indices='*i32',Output='*bf16',Debug='*fp32',Invalid='*i32',Routes='i32'),
   constexprs=dict(DOWN=down,BM=batch,WINDOW=WINDOW,CAPTURE_F32=False)),
   target=GPUTarget('hip','gfx1151',32),options=options)
  name=('down'if down else'gate-up')+'-bm'+str(batch);p=out/(name+'.hsaco');p.write_bytes(c.asm['hsaco'])
  asm,ir=c.asm['amdgcn'],c.asm['ttgir'];(out/(name+'.amdgcn')).write_text(asm);(out/(name+'.ttgir')).write_text(ir)
  compiled[name]=dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p),metadata=c.metadata._asdict(),options=options,
   amdgcn_sha256=sha(out/(name+'.amdgcn')),ttgir_sha256=sha(out/(name+'.ttgir')),
   layout_conversion_count=ir.count('ttg.convert_layout'),shared_memory_instructions=len(re.findall(r'^\s*ds_(?:read|write|load|store)',asm,re.M)),
   resources={k:re.findall(re.escape(k)+r'\s*:?\s*(\d+)',asm)for k in('.vgpr_count','.vgpr_spill_count','.private_segment_fixed_size')})
  print(json.dumps(dict(compiled=name,shared=c.metadata.shared,resources=compiled[name]['resources'],layout_conversions=compiled[name]['layout_conversion_count'])),flush=True)
(out/'compilation.json').write_text(json.dumps(compiled,indent=2,default=str)+'\n')

def guarded(shape,dtype,cpu=None):
 count=1
 for d in shape:count*=d
 n=count*torch.empty((),dtype=dtype).element_size();b=torch.full((n+1024,),0xa5,dtype=torch.uint8,device='cuda')
 value=b[512:-512].view(dtype).reshape(shape)
 if cpu is not None:value.copy_(cpu)
 return b,value

def guards(backings):return all(bool((b[:512]==0xa5).all())and bool((b[-512:]==0xa5).all())for b in backings)

def load_input(e):
 p=ref/'capture/q8192-out512'/e['file'];assert p.stat().st_size==e['bytes']and sha(p)==e['sha256']
 dtype={'torch.bfloat16':torch.bfloat16,'torch.int32':torch.int32,'torch.float32':torch.float32}[e['dtype']]
 return torch.frombuffer(bytearray(p.read_bytes()),dtype=dtype).reshape(e['shape'])

ids_cpu=load_input(reference['entries']['routed-ids.bin']);weights_cpu=load_input(reference['entries']['routed-weights.bin'])
ib,ids=guarded(ids_cpu.shape,ids_cpu.dtype,ids_cpu);rb,weights=guarded(weights_cpu.shape,weights_cpu.dtype,weights_cpu)
del ids_cpu,weights_cpu
records=[];source_records=[];control_records=[];pipeline=Pipeline(candidate)
for projection in m['projections']:
 down=projection['down'];N,K=(2048,512)if down else(1024,2048);routes=65536;cells=routes*N
 x_cpu=load_input(projection['input']);xb,x=guarded(x_cpu.shape,x_cpu.dtype,x_cpu);del x_cpu
 key=projection['weight_key'];shard=index['weight_map'][key];assert Path(shard).name==shard and shard.endswith('.safetensors')
 with safe_open(str(model/shard),framework='pt',device='cpu')as tensors:w_cpu=tensors.get_tensor(key).contiguous()
 assert info(w_cpu)==projection['weight']
 wb,w=guarded(w_cpu.shape,w_cpu.dtype,w_cpu);del w_cpu
 immutable={name:info(value)for name,value in [('x',x),('w',w),('ids',ids),('weights',weights)]}
 windows=cells//WINDOW
 cb,counts=guarded((windows,),torch.int32,torch.full((windows,),WINDOW,dtype=torch.int32))
 qb,queue=guarded((windows,WINDOW),torch.int32,torch.arange(cells,dtype=torch.int32).reshape(windows,WINDOW))
 yb,expected=guarded((8192,8,N),torch.bfloat16);db,debug=guarded((1,),torch.float32);fb,invalid=guarded((1,),torch.int32);invalid.zero_()
 queue_before=(info(counts),info(queue),info(debug))
 original.routed_replay_kernel[(256,windows)](x,w,ids,weights,counts,queue,expected,debug,invalid,routes,
  DOWN=down,BM=64,WINDOW=WINDOW,CAPTURE_F32=False,**options)
 torch.cuda.synchronize();baseline=info(expected)
 assert baseline==projection['output']and int(invalid.item())==0
 assert queue_before==(info(counts),info(queue),info(debug))and guards([xb,wb,ib,rb,cb,qb,yb,db,fb])
 # The independently replayed baseline must equal the original model tensor hash
 # before it supplies expected per-element values to the new candidate controls.
 baseline_guards=[yb]
 del counts,queue,debug,invalid,cb,qb,db,fb
 for batch in(32,64,128):
  full=pipeline.compare(x,w,ids,weights,expected,down,False,batch)
  sparse=pipeline.compare(x,w,ids,weights,expected,down,True,batch)
  records.extend([full,sparse]);print(json.dumps(dict(projection=projection['name'],batch=batch,
   full_values=full['compared_values'],full_mismatches=full['bf16_mismatches'],sparse_mismatches=sparse['bf16_mismatches'],
   original_output_sha256=projection['output']['sha256'])),flush=True)
 malformed=pipeline.malformed(x,w,ids,weights,down);control_records.extend(malformed)
 after={name:info(value)for name,value in [('x',x),('w',w),('ids',ids),('weights',weights)]}
 source_records.append(dict(name=projection['name'],input=projection['input'],weight=projection['weight'],weight_key=key,
  original_output=projection['output'],baseline_sha256=baseline['sha256'],baseline_matches_original=True,
  expected_output_unchanged=info(expected)==baseline,inputs_unchanged=immutable==after,guards_pass=guards([xb,wb,ib,rb,yb])))
 (out/('projection-'+projection['name']+'.json')).write_text(json.dumps(dict(source=source_records[-1],records=records[-6:],malformed=malformed),indent=2)+'\n')
 del x,w,expected,xb,wb,yb,baseline_guards
 torch.cuda.empty_cache()
passed=len(records)==12 and len(control_records)==15 and len(source_records)==2
passed=passed and all(r['bf16_mismatches']==0 and r['invalid_flags']==0 and all(r[k]for k in('guards_pass','queues_unchanged','debug_unchanged'))and r.get('unselected_outputs_unchanged',True)for r in records)
passed=passed and all(r['pass']for r in control_records)and all(all(r[k]for k in('baseline_matches_original','expected_output_unchanged','inputs_unchanged','guards_pass'))for r in source_records)
unchanged=all(sha(ROOT/name)==value for name,value in m['source_hashes'].items());passed=passed and unchanged
report=dict(host=socket.gethostname(),device=torch.cuda.get_device_name(),manifest_sha256=sha(ROOT/'manifest.json'),
 source_hashes=m['source_hashes'],candidate_base_commit=m['candidate_base_commit'],original_reference=m['original_reference'],
 records=records,malformed_controls=control_records,sources=source_records,compiled=compiled,
 full_bf16_values=sum(r['compared_values']for r in records if not r['sparse']),
 sparse_bf16_values=sum(r['compared_values']for r in records if r['sparse']),all_components_match=bool(passed),source_files_unchanged=unchanged,
 actual_weight_tensors_loaded=True,model_engine_loaded=False,amd_gpu_executed=False,
 inference_acceptance=False,performance_acceptance=False,release_qualified=False)
(out/'result.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps(dict(all_components_match=bool(passed),full_bf16_values=report['full_bf16_values'],sparse_bf16_values=report['sparse_bf16_values'],malformed_controls=len(control_records))),flush=True)
assert passed
