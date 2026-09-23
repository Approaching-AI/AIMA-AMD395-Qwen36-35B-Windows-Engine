"""Match the actual worker launch arguments to both compiled kernel ABIs."""
from pathlib import Path
import ast
import ctypes as C
import hashlib
import json
import re

B=Path(__file__).resolve().parent
D=B/'native-gdn-persistent-windows-r1'
A=B/'qrt-gdn-persistent-explicit-20260923-r3/collected/aot'
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
m=json.loads((D/'manifest.json').read_text())
assert sha(D/'worker.py')==m['worker_sha256']
tree=ast.parse((D/'worker.py').read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
records=[]
for name in ('persistent-runtime','persistent-capture'):
    text=(A/(name+'.amdgcn')).read_text()
    meta=text[text.index('.amdgpu_metadata'):]
    offsets=[int(v)for v in re.findall(r'\.offset:\s+(\d+)',meta)]
    sizes=[int(v)for v in re.findall(r'\.size:\s+(\d+)',meta)]
    kinds=re.findall(r'\.value_kind:\s+(\w+)',meta)
    assert offsets==list(range(0,96,8))+[96,104,112]
    assert sizes==[8]*12+[4,8,8]
    assert kinds==['global_buffer']*12+['by_value','global_buffer','global_buffer']
    assert re.search(r'\.kernarg_segment_size:\s+120\b',meta)
    assert sha(D/m['images'][name]['file'])==m['images'][name]['sha256']
    for tokens in (64,7169,8192):
        pointers=[0x100000+i*0x10000 for i in range(12)]
        if name=='persistent-runtime':pointers[-2:]=[0,0]
        observed=[]
        def launch(*args):
            assert args[:8]==(0x1234,32,16,1,128,1,1,6144)
            assert args[8] is None and args[10] is None
            argv=args[9]
            assert len(argv)==15
            actual=[C.c_void_p.from_address(argv[i]).value or 0 for i in range(12)]
            assert actual==pointers
            assert C.c_int32.from_address(argv[12]).value==tokens
            assert C.c_void_p.from_address(argv[13]).value is None
            assert C.c_void_p.from_address(argv[14]).value is None
            observed.append(True)
            return 0
        def check(code,label): assert code==0,label
        namespace=dict(C=C,void=C.c_void_p,case={'kernel_set':'u64'},m=m,
            functions={name:0x1234},check=check,launch=launch)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'exact-worker-execute','exec'),namespace)
        namespace['execute'](name,[C.c_void_p(v)for v in pointers],tokens,[32,16,1])
        assert observed==[True]
        records.append(dict(image=name,tokens=tokens,explicit_pointer_arguments=12,
            hidden_null_pointer_arguments=2,kernarg_bytes=120,grid=[32,16,1],block=[128,1,1],
            shared_bytes=6144,actual_worker_abi_pass=True))
report=dict(manifest_sha256=sha(D/'manifest.json'),worker_sha256=sha(D/'worker.py'),
    records=records,compiled_metadata_offsets_verified=True,host_only=True,remote_calls=0,
    candidate_amd_gpu_executed=False,inference_acceptance=False,performance_acceptance=False)
out=B/'native-gdn-persistent-abi-controls-r1.json'
with out.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(cases=len(records),actual_worker_abi_pass=True,sha256=sha(out))))
