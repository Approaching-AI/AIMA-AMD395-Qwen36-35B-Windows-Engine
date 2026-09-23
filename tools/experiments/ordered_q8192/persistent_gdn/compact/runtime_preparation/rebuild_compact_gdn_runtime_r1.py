from pathlib import Path
import hashlib,json,socket,struct,subprocess,tarfile,time
B=Path(__file__).resolve().parent
G=Path('/Users/jiawei-macmini/projects/AIMA-compact-gdn-candidate')
assert socket.gethostname().split('.')[0].lower()!='baiying'
D=B/'qrt-compact-gdn-runtime-rebuild-20260923-r1';D.mkdir()
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
files=['tools/compile_linux_core_gdn_persistent.py',*[p.relative_to(G).as_posix() for p in sorted((G/'native/providers/gdn_persistent').rglob('*.py'))]]
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
 members=tar.getmembers();assert len(members)==9 and all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
 tar.extractall(p,filter='data')
manifest=json.loads((p/'manifest.json').read_text())
for f,item in manifest['source_files'].items():
 q=p/'project'/f;assert q.stat().st_size==item['bytes']and sha(q)==item['sha256']
name=p.name
command=['docker','run','--name',name,'--hostname','aitopatom-66c4','--network','none','--memory','3g','--cpus','2','--pids-limit','256',
 '-v',str(p)+':/work','-w','/work','-e','CUDA_VISIBLE_DEVICES=-1','-e','HIP_VISIBLE_DEVICES=-1','-e','ROCR_VISIBLE_DEVICES=-1',
 '--entrypoint','timeout','sha256:822f5c399ce6a08c0583302e0d82ff3938158b73977b6e1fbdf00ed72735a30d','--signal=TERM','--kill-after=5','300',
 'python3','/work/project/tools/compile_linux_core_gdn_persistent.py','--out','/work/aot']
start=time.monotonic();reason='completed'
with(p/'stdout.log').open('xb')as out,(p/'stderr.log').open('xb')as err:
 process=subprocess.Popen(command,stdout=out,stderr=err)
 try:process.wait(timeout=320)
 except subprocess.TimeoutExpired:
  reason='deadline';subprocess.run(['docker','kill',name],capture_output=True,timeout=10);process.wait(timeout=10)
state=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',name],text=True,timeout=15))
record=dict(host=socket.gethostname(),command=command,returncode=process.returncode,reason=reason,seconds=time.monotonic()-start,
 container_state=state,cleanup_pass=not state['Running'],gpu_used=False,manifest_sha256=sha(p/'manifest.json'))
(p/'dispatch.json').write_text(json.dumps(record,indent=2)+'\n')
files=[f for f in p.rglob('*')if f.is_file()and f.suffix!='.pyc']
assert len(files)<100 and sum(f.stat().st_size for f in files)<64*(1<<20)
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
r=subprocess.run(cmd,input=driver,capture_output=True,text=True,timeout=360)
(D/'transport.json').write_text(json.dumps(dict(command=cmd,script_sha256=sha(D/'driver.py'),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.monotonic()-start),indent=2)+'\n')
print(r.stdout,r.stderr[-1500:],flush=True)
subprocess.run(['scp',*opts,'gb10-4t:'+remote+'/download.tar.gz',str(D/'download.tar.gz')],capture_output=True,check=True,timeout=60)
assert sha(D/'download.tar.gz')==json.loads(r.stdout)['download_sha256']
C=D/'collected';C.mkdir()
with tarfile.open(D/'download.tar.gz')as tar:
 members=tar.getmembers();assert len(members)<101 and sum(m.size for m in members)<64*(1<<20)
 assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
 tar.extractall(C,filter='data')
for file,item in json.loads((C/'download-manifest.json').read_text()).items():
 p=C/file;assert p.stat().st_size==item['bytes']and sha(p)==item['sha256']
if r.returncode:print((C/'stderr.log').read_text()[-5000:])
r.check_returncode()
