"""Replay all original full q8192 attention calls without replacing outputs."""
from pathlib import Path
import hashlib,importlib.util,inspect,json,sys,time
from capture_gb10_token_matrix import TokenMatrixCapture
ROOT=Path('/work');LAYERS=list(range(3,40,4))
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()


def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
 sys.modules[name]=result;spec.loader.exec_module(result);return result


def info(value):
 import torch
 raw=value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
 return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


class Pipeline:
 def __init__(self):
  import torch,triton
  assert torch.cuda.get_device_capability()==(12,1) and triton.__version__=='3.6.0'
  m=json.loads((ROOT/'source-inputs.json').read_text())
  for name,digest in m['files'].items():assert sha(ROOT/name)==digest,name
  previous={name:sys.modules.get(name)for name in ['integer_u','dense_selected','ordered_pipeline']}
  try:
   for name in previous:load_module(name,ROOT/'candidate'/(name+'.py'))
   self.a=load_module('all_layers_ordered_attention',ROOT/'candidate/attention.py')
   self.pack=load_module('all_layers_attention_pack',ROOT/'candidate/pack_value.py')
  finally:
   for name,old in previous.items():
    if old is None:sys.modules.pop(name,None)
    else:sys.modules[name]=old
  from types import SimpleNamespace
  sys.path.insert(0,str(ROOT/'candidate'))
  explicit=load_module('all_layers_explicit_attention',ROOT/'candidate/attention_explicit.py')
  self.a=SimpleNamespace(original_qk_kernel=self.a.original_qk_kernel,
   ordered_qk_kernel=explicit.ordered_qk_kernel,probability_kernel=explicit.probability_kernel,
   selected_pv_kernel=explicit.selected_pv_kernel)
  self.tables={}
  for name,path,expected in [('exp2','/table/exp2.bin','f490940df2bd80421159a96424c3e922330b7ca120d5ae7b629a973b9183730b'),
   ('rcp','/rcp/rcp.bin','d2e557543f6bc51f5141ba6414000cd8ed892e2e915eda19245c3cae22c16b39')]:
   p=Path(path);assert sha(p)==expected
   self.tables[name]=torch.frombuffer(bytearray(p.read_bytes()),dtype=torch.uint8).cuda()
  self.table_hashes={name:info(value)for name,value in self.tables.items()}
  import vllm.v1.attention.ops.triton_unified_attention as original
  assert sha(Path(original.__file__))=='5a8af26832bd0b23604fefa4a0876e12b39101b85431c368649fbb7f779e8921'

 def compare(self,q,k,v,original):
  import torch,triton
  T=8192;QT=128;guarded=[]
  def allocate(shape,dtype):
   n=1
   for x in shape:n*=x
   n*=torch.empty((),dtype=dtype).element_size()
   backing=torch.full((n+1024,),0xa5,dtype=torch.uint8,device='cuda')
   result=backing[512:-512].view(dtype).reshape(shape);guarded.append(backing);return result
  a=self.a;options=dict(num_warps=4,num_stages=2,enable_fp_fusion=False)
  packed=allocate((512,T),torch.bfloat16)
  self.pack.pack_value_kernel[(16,256)](v,packed,T,**options)
  expected_packed=v.reshape(T,512).t().contiguous()
  pack_match=torch.equal(packed.view(torch.int16),expected_packed.view(torch.int16));del expected_packed
  reference=allocate((QT,16,T),torch.float32);scores=allocate(reference.shape,torch.float32)
  probability=allocate((QT,16,T),torch.bfloat16);scales=allocate((QT,16,257),torch.float32)
  result=allocate((T,4096),torch.bfloat16);debug=allocate((1,),torch.float32);debug_before=info(debug)
  counts=torch.tensor([QT*4096],dtype=torch.int32,device='cuda');indices=torch.arange(QT*4096,dtype=torch.int32,device='cuda')
  queue_before=(info(counts),info(indices));qk_differences=0;slabs=[]
  for QS in range(0,T,QT):
   a.original_qk_kernel[(QT//2,T//32,2)](q,k,reference,T,QS,QT,num_warps=4,num_stages=3,enable_fp_fusion=True)
   a.ordered_qk_kernel[(QT//8,T//8,16)](q,k,scores,T,QS,QT,BM=8,BN=8,**options)
   bad=int((scores.view(torch.int32)!=reference.view(torch.int32)).sum());qk_differences+=bad
   a.probability_kernel[(QT//4,16)](scores,probability,scales,self.tables['exp2'],T,QS,QT,BM=4,FUSED_L=True,**options)
   a.selected_pv_kernel[(1024,)](probability,packed,scales,self.tables['rcp'],counts,indices,result[QS:QS+QT],debug,T,QS,QT,BM=32,CAPTURE_F32=False,**options)
   mismatch=int((result[QS:QS+QT].view(torch.int16)!=original[QS:QS+QT].view(torch.int16)).sum())
   slabs.append(dict(query_start=QS,queries=QT,qk_fp32_values=QT*16*T,qk_bit_mismatches=bad,context_bf16_values=QT*4096,context_bit_mismatches=mismatch))
  torch.cuda.synchronize();diff=result.view(torch.int16)!=original.view(torch.int16);mismatches=int(diff.sum())
  row=dict(slabs=slabs,qk_fp32_values=sum(x['qk_fp32_values']for x in slabs),qk_bit_mismatches=qk_differences,
   context_bf16_values=result.numel(),context_bit_mismatches=mismatches,actual=info(result),original=info(original),
   guards_pass=all(bool((b[:512]==0xa5).all()) and bool((b[-512:]==0xa5).all())for b in guarded),
   packed_value_matches=bool(pack_match),queues_unchanged=queue_before==(info(counts),info(indices)),debug_unchanged=debug_before==info(debug))
  if mismatches:
   cells=diff.reshape(-1).nonzero()[:32,0]
   row['first_differences']=[dict(index=int(i),actual=float(result.reshape(-1)[i]),original=float(original.reshape(-1)[i]))for i in cells]
  return row


class OrderedAttentionCapture(TokenMatrixCapture):
 def qrt_arm_token_matrix(self,directory,prompt_tokens):
  import torch
  import vllm.v1.attention.ops.triton_unified_attention as original
  result=super().qrt_arm_token_matrix(directory,prompt_tokens)
  self._oa_enabled=Path(directory).name=='q8192-out512';self._oa_handles=[];self._oa_pending={};self._oa_rows=[]
  self._oa_pipeline=None;self._oa_original_run=None;self._oa_active=None;self._oa_kernels={}
  self._oa_root=Path(directory)
  if not self._oa_enabled:return result
  assert prompt_tokens==8192;self._oa_pipeline=Pipeline();self._oa_started=time.monotonic()
  containers=[(n,m)for n,m in self.model_runner.model.named_modules()if isinstance(m,torch.nn.ModuleList)and len(m)==40 and hasattr(m[3],'self_attn')]
  assert len(containers)==1
  name,layers=containers[0]
  self._oa_original_run=original.kernel_unified_attention_2d.run
  original_run=self._oa_original_run
  def observed_run(*args,**kwargs):
   compiled=original_run(*args,**kwargs)
   if self._oa_active is not None:
    index=self._oa_active;assert index not in self._oa_kernels
    self._oa_kernels[index]=dict(kernel_hash=compiled.hash,metadata=json.loads(json.dumps(compiled.metadata._asdict(),default=str)),
     ptx_sha256=hashlib.sha256(compiled.asm['ptx'].encode()).hexdigest(),original_run_returned_unchanged=True)
   return compiled
  original.kernel_unified_attention_2d.run=observed_run
  def attach(index):
   attn=layers[index].self_attn
   assert (attn.num_heads,attn.num_kv_heads,attn.head_dim)==(16,2,256)
   assert sha(Path(inspect.getsourcefile(type(attn))))=='0f7c2df8fa972a193922bad89260d6cf1b6acb97a63b9c6675bc0be362d0a1e5'
   def before(module,args):
    q,k,v=args
    if q.shape[0]!=8192:return
    assert index not in self._oa_pending and index not in [r['layer']for r in self._oa_rows]
    assert self._oa_active is None and tuple(q.shape)==(8192,4096)and tuple(k.shape)==tuple(v.shape)==(8192,512)
    values=[x.detach().contiguous().clone()for x in args]
    self._oa_pending[index]=dict(values=values,inputs=[info(x)for x in values]);self._oa_active=index
   def after(module,args,output):
    if args[0].shape[0]!=8192:return
    assert self._oa_active==index and tuple(output.shape)==(8192,4096)
    assert time.monotonic()-self._oa_started<1400
    pending=self._oa_pending.pop(index);original_output=output.detach().contiguous().clone();before_output=info(original_output)
    started=time.monotonic();comparison=self._oa_pipeline.compare(*pending['values'],original_output)
    unchanged=pending['inputs']==[info(x)for x in pending['values']]and pending['inputs']==[info(x)for x in args]
    original_unchanged=before_output==info(output)==info(original_output)
    passed=(comparison['qk_bit_mismatches']==comparison['context_bit_mismatches']==0 and
     all(comparison[k]for k in ['guards_pass','packed_value_matches','queues_unchanged','debug_unchanged'])and unchanged and original_unchanged)
    row=dict(layer=index,inputs=pending['inputs'],comparison=comparison,original_kernel=self._oa_kernels[index],
     inputs_unchanged=unchanged,original_output_unchanged=original_unchanged,all_values_match=passed,
     diagnostic_wall_seconds=time.monotonic()-started,performance_acceptance=False)
    self._oa_rows.append(row);self._oa_active=None
    (self._oa_root/('ordered-attention-layer-%02d.json'%index)).write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(dict(event='ordered_attention_comparison',layer=index,all_values_match=passed,
     qk_bit_mismatches=comparison['qk_bit_mismatches'],context_bit_mismatches=comparison['context_bit_mismatches'],
     diagnostic_seconds=row['diagnostic_wall_seconds'])),flush=True)
   self._oa_handles.extend([attn.attn.register_forward_pre_hook(before),attn.attn.register_forward_hook(after)])
  for index in LAYERS:attach(index)
  result['ordered_attention']=dict(enabled=True,layers=LAYERS,container=name,candidate_outputs_fed_to_model=False)
  return result

 def qrt_finish_token_matrix(self):
  import vllm.v1.attention.ops.triton_unified_attention as original
  for handle in self._oa_handles:handle.remove()
  if self._oa_original_run is not None:original.kernel_unified_attention_2d.run=self._oa_original_run
  result=super().qrt_finish_token_matrix()
  if self._oa_enabled:
   tables_ok=self._oa_pipeline.table_hashes=={name:info(value)for name,value in self._oa_pipeline.tables.items()}
   complete=[r['layer']for r in self._oa_rows]==LAYERS and not self._oa_pending and self._oa_active is None
   report=dict(layers_complete=complete,layers=self._oa_rows,pending_calls=len(self._oa_pending),tables_unchanged=tables_ok,
    all_original_attention_results_match=complete and tables_ok and all(r['all_values_match']for r in self._oa_rows),
    original_outputs_returned_unchanged=True,candidate_outputs_fed_to_model=False,
    qk_fp32_values=sum(r['comparison']['qk_fp32_values']for r in self._oa_rows),
    context_bf16_values=sum(r['comparison']['context_bf16_values']for r in self._oa_rows))
   result['ordered_attention']=report
   (self._oa_root/'ordered-attention-all-layers.json').write_text(json.dumps(report,indent=2)+'\n')
  self._oa_pipeline=None;self._oa_handles=[]
  return result
