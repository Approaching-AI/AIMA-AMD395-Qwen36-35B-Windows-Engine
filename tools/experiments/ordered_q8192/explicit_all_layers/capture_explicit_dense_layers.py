"""Compare queued unsigned32 dots at original q8192 dense projections."""
from pathlib import Path
import hashlib, importlib.util, inspect, json, sys, time
from capture_gb10_token_matrix import TokenMatrixCapture
ROOT=Path('/work');WINDOW=1<<20
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()

def info(value):
 import torch
 raw=value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
 return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

class Pipeline:
 def __init__(self):
  import torch,triton
  assert torch.cuda.get_device_capability()==(12,1) and triton.__version__=='3.6.0'
  manifest=json.loads((ROOT/'source-inputs.json').read_text())
  for name,digest in manifest['files'].items():assert sha(ROOT/name)==digest,name
  spec=importlib.util.spec_from_file_location('all_layers_ordered_dense',ROOT/'candidate/explicit_dense.py')
  self.module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=self.module;spec.loader.exec_module(self.module)

 def compare(self,x,weight,original,layout):
  import torch
  T,K=x.shape;N,WK=weight.shape
  assert T==8192 and WK==K and K in (512,2048,4096) and N%16==0 and 0<N<=12352
  assert tuple(original.shape)==(T,N)
  guarded=[]
  def allocate(shape,dtype):
   count=1
   for v in shape:count*=v
   n=count*torch.empty((),dtype=dtype).element_size()
   backing=torch.full((n+1024,),0xa5,dtype=torch.uint8,device='cuda')
   value=backing[512:-512].view(dtype).reshape(shape);guarded.append(backing);return value
  if layout=='transposed':
   w=weight.t().contiguous();row_stride,k_stride=1,N
  else:
   assert layout=='contiguous';w=weight;row_stride,k_stride=K,1
  cells=T*N;windows=(cells+WINDOW-1)//WINDOW
  counts=allocate((windows,),torch.int32);counts.fill_(WINDOW);counts[-1]=cells-(windows-1)*WINDOW
  indices=allocate((windows,WINDOW),torch.int32)
  indices.copy_(torch.arange(windows*WINDOW,dtype=torch.int32,device='cuda').reshape(windows,WINDOW))
  queue_before=(info(counts),info(indices));weight_before=info(w)
  result=allocate((T,N),torch.bfloat16);debug=allocate((1,),torch.float32);debug_before=info(debug)
  started=time.monotonic()
  self.module.selected_replay_kernel[(256,windows)](x,w,counts,indices,result,debug,N,K,row_stride,k_stride,
   BM=64,WINDOW=WINDOW,CAPTURE_F32=False,num_warps=4,num_stages=2,enable_fp_fusion=False)
  torch.cuda.synchronize();seconds=time.monotonic()-started
  different=result.view(torch.int16)!=original.view(torch.int16)
  row=dict(tokens=T,n=N,k=K,layout=layout,queued_cells=cells,windows=windows,batch=64,
   bf16_mismatches=int(different.sum()),actual=info(result),original=info(original),
   guards_pass=all(bool((b[:512]==0xa5).all())and bool((b[-512:]==0xa5).all())for b in guarded),
   queues_unchanged=queue_before==(info(counts),info(indices)),weight_view_unchanged=weight_before==info(w),
   debug_unchanged=debug_before==info(debug),diagnostic_kernel_seconds=seconds,performance_acceptance=False)
  if row['bf16_mismatches']:
   cells=different.reshape(-1).nonzero()[:32,0]
   row['first_differences']=[dict(index=int(i),actual=int(result.view(torch.int16).reshape(-1)[i]),
    original=int(original.view(torch.int16).reshape(-1)[i]))for i in cells]
  return row

class OrderedDenseCapture(TokenMatrixCapture):
 def qrt_arm_token_matrix(self,directory,prompt_tokens):
  import torch
  result=super().qrt_arm_token_matrix(directory,prompt_tokens)
  self._od_enabled=Path(directory).name=='q8192-out512';self._od_handles=[];self._od_pending={};self._od_rows=[]
  self._od_expected=[];self._od_pipeline=None;self._od_root=Path(directory)
  if not self._od_enabled:return result
  assert prompt_tokens==8192;self._od_pipeline=Pipeline();self._od_started=time.monotonic()
  containers=[(name,module)for name,module in self.model_runner.model.named_modules()
   if isinstance(module,torch.nn.ModuleList)and len(module)==40 and hasattr(module[3],'self_attn')]
  assert len(containers)==1;container,layers=containers[0]
  def attach(layer,label,module,layout):
   key='%02d-%s'%(layer,label);assert key not in self._od_expected
   self._od_expected.append(key)
   assert hasattr(module,'weight')and module.weight.dtype==torch.bfloat16
   assert getattr(module,'bias',None) is None
   def before(m,args):
    x=args[0]
    if x.shape[0]!=8192:return
    assert key not in self._od_pending and key not in [r['key']for r in self._od_rows]
    assert time.monotonic()-self._od_started<1400
    assert x.ndim==2 and x.dtype==torch.bfloat16 and m.weight.ndim==2
    values=[x.detach().contiguous().clone(),m.weight.detach().contiguous().clone()]
    self._od_pending[key]=dict(values=values,inputs=[info(v)for v in values])
   def after(m,args,output):
    if args[0].shape[0]!=8192:return
    pending=self._od_pending.pop(key)
    value=output[0]if isinstance(output,tuple)else output
    assert value.dtype==torch.bfloat16
    original=value.detach().contiguous().clone();before_output=info(original)
    started=time.monotonic();comparison=self._od_pipeline.compare(*pending['values'],original,layout)
    unchanged=pending['inputs']==[info(v)for v in pending['values']]and pending['inputs']==[info(args[0]),info(m.weight)]
    output_unchanged=before_output==info(original)==info(value)
    passed=(comparison['bf16_mismatches']==0 and all(comparison[k]for k in
     ('guards_pass','queues_unchanged','weight_view_unchanged','debug_unchanged'))and unchanged and output_unchanged)
    row=dict(key=key,layer=layer,projection=label,inputs=pending['inputs'],comparison=comparison,
     inputs_unchanged=unchanged,original_output_unchanged=output_unchanged,all_values_match=passed,
     candidate_outputs_fed_to_model=False,diagnostic_wall_seconds=time.monotonic()-started,performance_acceptance=False)
    self._od_rows.append(row)
    (self._od_root/('ordered-dense-'+key+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(dict(event='ordered_dense_comparison',key=key,all_values_match=passed,
     bf16_mismatches=comparison['bf16_mismatches'],values=comparison['queued_cells'],
     diagnostic_seconds=row['diagnostic_wall_seconds'])),flush=True)
   self._od_handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
  for index,layer in enumerate(layers):
   if index%4==3:
    a=layer.self_attn
    assert sha(Path(inspect.getsourcefile(type(a))))=='0f7c2df8fa972a193922bad89260d6cf1b6acb97a63b9c6675bc0be362d0a1e5'
    attach(index,'attention_qkv',a.qkv_proj,'transposed');attach(index,'attention_out',a.o_proj,'contiguous')
   else:
    a=layer.linear_attn
    assert sha(Path(inspect.getsourcefile(type(a))))=='677fdbad739a5687fefa35cc5c72efba476b6d8e73ae2df4f03114f1d7f2b44c'
    assert not a.gqa_interleaved_layout and a.tp_size==1
    attach(index,'gdn_qkvz',a.in_proj_qkvz,'transposed');attach(index,'gdn_out',a.out_proj,'contiguous')
   shared=layer.mlp.shared_expert
   assert sha(Path(inspect.getsourcefile(type(shared))))=='58899ae017336a4ea00e37788788969e17a6178c113961486acca9e6569f4a8f'
   attach(index,'shared_gate_up',shared.gate_up_proj,'contiguous');attach(index,'shared_down',shared.down_proj,'contiguous')
  assert len(self._od_expected)==160
  result['ordered_dense']=dict(enabled=True,expected_projections=self._od_expected,container=container,candidate_outputs_fed_to_model=False)
  return result

 def qrt_finish_token_matrix(self):
  for handle in self._od_handles:handle.remove()
  result=super().qrt_finish_token_matrix()
  if self._od_enabled:
   complete=sorted(r['key']for r in self._od_rows)==sorted(self._od_expected)and not self._od_pending
   report=dict(projections_complete=complete,projections=self._od_rows,expected_projections=self._od_expected,
    pending_calls=len(self._od_pending),all_original_dense_results_match=complete and all(r['all_values_match']for r in self._od_rows),
    original_outputs_returned_unchanged=True,candidate_outputs_fed_to_model=False,
    bf16_values=sum(r['comparison']['queued_cells']for r in self._od_rows))
   result['ordered_dense']=report
   (self._od_root/'ordered-dense-all-layers.json').write_text(json.dumps(report,indent=2)+'\n')
  self._od_pipeline=None;self._od_handles=[]
  return result
