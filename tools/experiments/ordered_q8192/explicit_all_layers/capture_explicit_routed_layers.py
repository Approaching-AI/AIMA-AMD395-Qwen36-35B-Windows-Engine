"""Observe original routed projections and compare an isolated raw BF16 replay."""
from pathlib import Path
import hashlib, importlib, inspect, json, sys, time
from capture_gb10_token_matrix import TokenMatrixCapture
ROOT=Path('/work');WINDOW=4<<20
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()

def info(value):
 import torch
 raw=value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
 return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

class Pipeline:
 def __init__(self):
  import torch,triton
  assert torch.cuda.get_device_capability()==(12,1)and triton.__version__=='3.6.0'
  manifest=json.loads((ROOT/'source-inputs.json').read_text())
  for name,digest in manifest['files'].items():assert sha(ROOT/name)==digest,name
  sys.path.insert(0,str(ROOT/'candidate'))
  self.module=importlib.import_module('routed_explicit')
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

class OrderedRoutedCapture(TokenMatrixCapture):
 def qrt_arm_token_matrix(self,directory,prompt_tokens):
  import torch
  result=super().qrt_arm_token_matrix(directory,prompt_tokens)
  self._or_enabled=Path(directory).name=='q8192-out512';self._or_restores=[];self._or_rows=[]
  self._or_pipeline=None;self._or_root=Path(directory);self._or_expected=[];self._or_failed=False
  if not self._or_enabled:return result
  assert prompt_tokens==8192
  self._or_pipeline=Pipeline();self._or_started=time.monotonic()
  module=importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
  assert sha(Path(inspect.getsourcefile(module)))=='607c0a459306a71ff7d01445772494367f3924098739bbd3b4f43020738297d4'
  original_dispatch=module.invoke_fused_moe_triton_kernel;signature=inspect.signature(original_dispatch)
  containers=[(name,m)for name,m in self.model_runner.model.named_modules()
   if isinstance(m,torch.nn.ModuleList)and len(m)==40 and hasattr(m[3],'self_attn')]
  assert len(containers)==1;container,layers=containers[0]
  def attach(layer,experts):
   context={};forward=experts.forward;router=experts.router;select=router.select_experts
   self._or_expected.extend('%02d-%s'%(layer,label)for label in('gate-up','down'))
   def select_original(*args,**kwargs):
    selected=select(*args,**kwargs)
    if context.get('active'):
     assert selected[0].shape==selected[1].shape==(8192,8)
     assert selected[0].dtype==torch.float32
     context['weights']=selected[0].detach().contiguous().clone()
     context['ids']=selected[1].detach().contiguous().to(torch.int32).clone()
    return selected
   def dispatch(*args,**kwargs):
    index=context['calls'];assert index<2;down=bool(index);context['calls']+=1
    bound=signature.bind(*args,**kwargs);bound.apply_defaults();values=bound.arguments
    N,K=(2048,512)if down else(1024,2048)
    a,b,c=values['A'],values['B'],values['C']
    assert a.shape==(65536 if down else 8192,K)and b.shape==(256,N,K)and c.shape==(8192,8,N)
    assert values['mul_routed_weight']==down and values['top_k']==(1 if down else 8)
    assert all(values[k]is None for k in('A_scale','B_scale','B_bias','block_shape'))
    assert not any(values[k]for k in('use_fp8_w8a8','use_int8_w8a8','use_int8_w8a16','use_int4_w4a16','per_channel_quant'))
    assert a.is_contiguous()and b.is_contiguous()and c.is_contiguous()
    assert 'ids'in context and 'weights'in context
    if values['topk_weights']is not None:assert info(values['topk_weights'])==info(context['weights'])
    ids,weights=context['ids'],context['weights']
    padded=int(values['num_tokens_post_padded'].item());block=values['config']['BLOCK_SIZE_M']
    sorted_ids=values['sorted_token_ids'][:padded].long();expert_ids=values['expert_ids'][:padded//block].repeat_interleave(block)
    mask=sorted_ids<65536;routes=sorted_ids[mask]
    assert routes.numel()==65536 and bool((routes.sort().values==torch.arange(65536,device='cuda')).all())
    assert bool((expert_ids[mask].int()==ids.flatten()[routes]).all())
    x=a.detach().contiguous().clone();w=b.detach().contiguous().clone()
    inputs=[info(v)for v in(x,w,ids,weights)]
    returned=original_dispatch(*args,**kwargs)
    original=c.detach().contiguous().clone();output_before=info(original)
    started=time.monotonic();comparisons=[self._or_pipeline.compare(x,w,ids,weights,original,down)]
    malformed=[]
    if layer==0:
     comparisons.extend(self._or_pipeline.compare(x,w,ids,weights,original,down,True,batch)for batch in(32,64,128))
     malformed=self._or_pipeline.malformed(x,w,ids,weights,down)
    unchanged=inputs==[info(v)for v in(x,w,ids,weights)]and inputs[:2]==[info(a),info(b)]
    output_unchanged=output_before==info(original)==info(c)
    passed=unchanged and output_unchanged and all(r['pass']for r in malformed)and all(
     r['bf16_mismatches']==0 and r['invalid_flags']==0 and all(r[k]for k in('guards_pass','queues_unchanged','debug_unchanged'))
     and r.get('unselected_outputs_unchanged',True)for r in comparisons)
    key='%02d-%s'%(layer,'down'if down else'gate-up')
    row=dict(key=key,layer=layer,down=down,original_config=values['config'],padded_routes=padded,
     original_router_and_sorted_dispatch_agree=True,inputs=inputs,original_output=output_before,
     comparisons=comparisons,malformed_controls=malformed,inputs_unchanged=unchanged,
     original_output_unchanged=output_unchanged,all_values_match=passed,candidate_outputs_fed_to_model=False,
     diagnostic_wall_seconds=time.monotonic()-started,performance_acceptance=False)
    self._or_rows.append(row);self._or_failed|=not passed
    (self._or_root/('ordered-routed-'+key+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(dict(event='ordered_routed_comparison',key=key,all_values_match=passed,
     bf16_mismatches=sum(r['bf16_mismatches']for r in comparisons),diagnostic_seconds=row['diagnostic_wall_seconds'])),flush=True)
    return returned
   def forward_original(*args,**kwargs):
    hidden=kwargs.get('hidden_states',args[0]if args else None)
    if hidden is None or hidden.shape[0]!=8192 or self._or_failed:return forward(*args,**kwargs)
    assert time.monotonic()-self._or_started<1400
    assert not context and module.invoke_fused_moe_triton_kernel is original_dispatch
    context.update(active=True,calls=0);module.invoke_fused_moe_triton_kernel=dispatch
    try:
     returned=forward(*args,**kwargs);assert context['calls']==2
     return returned
    finally:
     module.invoke_fused_moe_triton_kernel=original_dispatch;context.clear()
   self._or_restores.extend([(experts,'forward',forward),(router,'select_experts',select)])
   experts.forward=forward_original;router.select_experts=select_original
  for index,layer in enumerate(layers):attach(index,layer.mlp.experts)
  result['ordered_routed']=dict(enabled=True,expected_projections=self._or_expected,container=container,candidate_outputs_fed_to_model=False)
  return result

 def qrt_finish_token_matrix(self):
  for obj,name,original in reversed(self._or_restores):setattr(obj,name,original)
  result=super().qrt_finish_token_matrix()
  if self._or_enabled:
   complete=sorted(r['key']for r in self._or_rows)==sorted(self._or_expected)
   report=dict(projections_complete=complete,projections=self._or_rows,expected_projections=self._or_expected,
    pending_calls=0,all_original_routed_results_match=complete and all(r['all_values_match']for r in self._or_rows),
    original_outputs_returned_unchanged=True,candidate_outputs_fed_to_model=False,
    full_bf16_values=sum(c['compared_values']for r in self._or_rows for c in r['comparisons']if not c['sparse']),
    sparse_bf16_values=sum(c['compared_values']for r in self._or_rows for c in r['comparisons']if c['sparse']),
    malformed_controls=sum(len(r['malformed_controls'])for r in self._or_rows))
   result['ordered_routed']=report
   (self._or_root/'ordered-routed-all-layers.json').write_text(json.dumps(report,indent=2)+'\n')
  self._or_pipeline=None;self._or_restores=[]
  return result
