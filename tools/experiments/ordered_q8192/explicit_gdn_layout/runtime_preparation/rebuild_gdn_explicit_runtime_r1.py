from pathlib import Path
import hashlib,json,socket,struct,subprocess,tarfile,time
B=Path(__file__).resolve().parent
G=Path('/Users/jiawei-macmini/projects/AIMA-explicit-gdn-layout-candidate')
assert socket.gethostname().split('.')[0].lower()!='baiying'
D=B/'qrt-gdn-explicit-runtime-rebuild-20260923-r1';D.mkdir()
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
files=['tools/compile_linux_core_gdn_explicit.py','native/providers/gdn/ordered_inverse.py',
       *['native/providers/gdn_explicit_layout/'+f for f in ('ordered_pipeline.py','group16.py','exp2.py')]]
manifest=dict(source_files={f:dict(bytes=(G/f).stat().st_size,sha256=sha(G/f))for f in files},
    intended_target='gfx1151',cpu_only=True,gpu_executed=False,inference_acceptance=False,performance_acceptance=False)
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=B/(D.name+'.tar')
with tarfile.open(archive,'x')as tar:
    for f in files:tar.add(G/f,arcname='project/'+f,recursive=False)
    tar.add(D/'manifest.json',arcname='manifest.json',recursive=False)
remote='/home/qujing/'+D.name
driver=r'''from pathlib import Path
import hashlib,json,socket,subprocess,tarfile,time
assert socket.gethostname()=='aitopatom-66c4'
p=Path(REMOTE);sha=lambda x:hashlib.file_digest(x.open('rb'),'sha256').hexdigest()
assert sha(p.with_suffix('.tar'))==ARCHIVE_SHA
p.mkdir()
with tarfile.open(p.with_suffix('.tar'))as tar:
 members=tar.getmembers();assert len(members)==6 and all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
 tar.extractall(p,filter='data')
manifest=json.loads((p/'manifest.json').read_text())
for f,item in manifest['source_files'].items():
 q=p/'project'/f;assert q.stat().st_size==item['bytes']and sha(q)==item['sha256']
name=p.name
command=['docker','run','--name',name,'--hostname','aitopatom-66c4','--network','none','--memory','3g','--cpus','2','--pids-limit','256',
 '-v',str(p)+':/work','-w','/work','-e','CUDA_VISIBLE_DEVICES=-1','-e','HIP_VISIBLE_DEVICES=-1','-e','ROCR_VISIBLE_DEVICES=-1',
 '--entrypoint','timeout','sha256:822f5c399ce6a08c0583302e0d82ff3938158b73977b6e1fbdf00ed72735a30d','--signal=TERM','--kill-after=5','180',
 'python3','/work/project/tools/compile_linux_core_gdn_explicit.py','--out','/work/aot']
start=time.monotonic();reason='completed'
with(p/'stdout.log').open('xb')as out,(p/'stderr.log').open('xb')as err:
 process=subprocess.Popen(command,stdout=out,stderr=err)
 try:process.wait(timeout=200)
 except subprocess.TimeoutExpired:
  reason='deadline';subprocess.run(['docker','kill',name],capture_output=True,timeout=10);process.wait(timeout=10)
state=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',name],text=True,timeout=15))
record=dict(host=socket.gethostname(),command=command,returncode=process.returncode,reason=reason,seconds=time.monotonic()-start,
 container_state=state,cleanup_pass=not state['Running'],gpu_used=False,manifest_sha256=sha(p/'manifest.json'))
(p/'dispatch.json').write_text(json.dumps(record,indent=2)+'\n')
files=[f for f in p.rglob('*')if f.is_file()and f.suffix!='.pyc']
assert len(files)<50 and sum(f.stat().st_size for f in files)<32*(1<<20)
inventory={str(f.relative_to(p)):dict(bytes=f.stat().st_size,sha256=sha(f))for f in files}
(p/'download-manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
with tarfile.open(p/'download.tar.gz','x:gz',compresslevel=1)as tar:
 for f in files+[p/'download-manifest.json']:tar.add(f,arcname=str(f.relative_to(p)),recursive=False)
print(json.dumps(dict(**record,download_sha256=sha(p/'download.tar.gz'))),flush=True)
assert record['returncode']==0 and record['cleanup_pass']
'''.replace('REMOTE',repr(remote)).replace('ARCHIVE_SHA',repr(sha(archive)))
compile(driver,'remote-driver','exec');(D/'driver.py').write_text(driver)
opts=['-o','BatchMode=yes','-o','ConnectTimeout=10']
subprocess.run(['scp',*opts,str(archive),'gb10-4t:/home/qujing/'],capture_output=True,check=True,timeout=45)
cmd=['ssh',*opts,'gb10-4t','python3 -'];start=time.monotonic()
r=subprocess.run(cmd,input=driver,capture_output=True,text=True,timeout=240)
(D/'transport.json').write_text(json.dumps(dict(command=cmd,script_sha256=sha(D/'driver.py'),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.monotonic()-start),indent=2)+'\n')
print(r.stdout,r.stderr[-1500:],flush=True)
subprocess.run(['scp',*opts,'gb10-4t:'+remote+'/download.tar.gz',str(D/'download.tar.gz')],capture_output=True,check=True,timeout=60)
assert sha(D/'download.tar.gz')==json.loads(r.stdout)['download_sha256']
C=D/'collected';C.mkdir()
with tarfile.open(D/'download.tar.gz')as tar:
 members=tar.getmembers();assert len(members)<51 and sum(m.size for m in members)<32*(1<<20)
 assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
 tar.extractall(C,filter='data')
for file,item in json.loads((C/'download-manifest.json').read_text()).items():
 p=C/file;assert p.stat().st_size==item['bytes']and sha(p)==item['sha256']
if r.returncode:print((C/'stderr.log').read_text()[-5000:])
r.check_returncode()

def allocated_sections(path):
    raw=path.read_bytes();header=struct.unpack_from('<16sHHIQQQIHHHHHH',raw)
    assert raw[:6]==b'\x7fELF\x02\x01'
    offset,step,count,string_index=header[6],header[11],header[12],header[13]
    assert step==64 and count<100 and offset+count*step<=len(raw)
    sections=[struct.unpack_from('<IIQQQQIIQQ',raw,offset+i*step)for i in range(count)]
    names=sections[string_index];strings=raw[names[4]:names[4]+names[5]]
    result={}
    for s in sections:
        if not(s[2]&2):continue
        name=strings[s[0]:].split(b'\0',1)[0].decode()
        assert s[4]+s[5]<=len(raw)
        data=raw[s[4]:s[4]+s[5]]
        result[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),type=s[1],flags=s[2])
    assert '.text'in result
    return result

selected=json.loads((G/'native/linux_core_port/gb10_gdn_ordered_compile.json').read_text())
rebuild=json.loads((C/'aot/result.json').read_text());rows=[]
for name,entry in rebuild['compiled'].items():
    prior=selected['compiled'][name]
    assert entry['signature']==prior['signature']and entry['constants']==prior['constants']and entry['options']==prior['options']
    assert entry['metadata']['shared']==prior['metadata']['shared']and entry['metadata']['num_warps']==prior['metadata']['num_warps']
    a=C/'aot'/entry['file'];b=B/'linux-core-gdn-explicit-asset-import-r1'/prior['file']
    sections=allocated_sections(a);old_sections=allocated_sections(b)
    rows.append(dict(name=name,rebuilt_sha256=sha(a),selected_sha256=sha(b),all_file_bytes_match=sha(a)==sha(b),
        allocated_sections=sections,selected_allocated_sections=old_sections,all_allocated_sections_match=sections==old_sections))
record=dict(manifest_sha256=sha(D/'manifest.json'),dispatch_sha256=sha(C/'dispatch.json'),
    report_sha256=sha(C/'aot/result.json'),source_files_match=rebuild['source_files']==selected['source_files'],
    rebuild_compiler_sha256=sha(G/'tools/compile_linux_core_gdn_explicit.py'),rows=rows,
    signatures_constants_launches_match=True,all_allocated_sections_match=all(x['all_allocated_sections_match']for x in rows),
    native_execution=False,product_images_changed=False,inference_acceptance=False,performance_acceptance=False)
(D/'image-comparison.json').write_text(json.dumps(record,indent=2)+'\n')
assert record['source_files_match']and record['all_allocated_sections_match']
print(json.dumps(dict(rebuilt=8,allocated_sections_match=True,all_bytes_match=all(x['all_file_bytes_match']for x in rows))),flush=True)
