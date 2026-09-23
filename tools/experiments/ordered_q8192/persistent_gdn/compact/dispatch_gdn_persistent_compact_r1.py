from pathlib import Path
import hashlib,json,socket,subprocess,tarfile,time
B=Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0].lower()!='baiying'
D=B/'qrt-gdn-persistent-compact-20260923-r1'
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
inputs=[p for p in D.iterdir()if p.is_file()]
assert len(inputs)==7
archive=B/(D.name+'.tar')
with tarfile.open(archive,'x')as tar:
    for p in inputs:tar.add(p,arcname=p.name,recursive=False)
remote='/home/qujing/'+D.name
driver=r'''from pathlib import Path
import hashlib,json,socket,subprocess,tarfile,time
assert socket.gethostname()=='aitopatom-66c4'
root=Path(REMOTE);archive=root.with_suffix('.tar')
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert sha(archive)==ARCHIVE_SHA
root.mkdir()
with tarfile.open(archive)as tar:
 entries=tar.getmembers();assert len(entries)==7 and all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in entries)
 tar.extractall(root,filter='data')
image='sha256:822f5c399ce6a08c0583302e0d82ff3938158b73977b6e1fbdf00ed72735a30d'
probe=['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']
records=[]
for action,worker,seconds in [('compile','compile_worker.py',300),('pipeline','numerical_worker.py',600)]:
 name=root.name+'-'+action;gpu=action!='compile'
 if gpu:assert not subprocess.check_output(probe,text=True,timeout=15).strip()
 command=['docker','run','--name',name,'--hostname','aitopatom-66c4','--network','none','--memory','8g'if gpu else'3g','--cpus','4'if gpu else'2','--pids-limit','256','--shm-size','1g','-v',str(root)+':/work','-w','/work']
 if gpu:command+=['--gpus','all']
 if action=='pipeline':
  command+=['-v','/home/qujing/qrt-native-gdn-ordered-pipeline-matrix-20260923-r1/inputs:/work/inputs:ro',
   '-v','/home/qujing/qrt-native-gdn-ordered-pipeline-20260923-r1:/table:ro',
   '-v','/home/qujing/qrt-native-gdn-ordered-inverse-matrix-20260923-r1/inputs:/inverse:ro',
   '-v','/home/qujing/qrt-native-gdn-integer-u-matrix-20260923-r1/inputs:/baseu:ro']
 command+=['--entrypoint','timeout',image,'--signal=TERM','--kill-after=5',str(seconds),'python3',worker]
 start=time.monotonic();reason='completed'
 with(root/(action+'.stdout.log')).open('xb')as out,(root/(action+'.stderr.log')).open('xb')as err:
  process=subprocess.Popen(command,stdout=out,stderr=err)
  try:process.wait(timeout=seconds+20)
  except subprocess.TimeoutExpired:
   reason='deadline';subprocess.run(['docker','kill',name],capture_output=True,timeout=10);process.wait(timeout=10)
 state=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',name],text=True,timeout=15))
 after=subprocess.check_output(probe,text=True,timeout=15).strip()if gpu else None
 record=dict(action=action,host=socket.gethostname(),command=command,returncode=process.returncode,reason=reason,container_state=state,seconds=time.monotonic()-start,gpu_used=gpu,gpu_processes_after=after,cleanup_pass=not state['Running']and(not gpu or not after))
 records.append(record)
 (root/'dispatch.json').write_text(json.dumps(dict(host=socket.gethostname(),source_archive_sha256=ARCHIVE_SHA,actions=records,model_loaded=False,inference_acceptance=False,performance_acceptance=False),indent=2)+'\n')
 print(json.dumps(record),flush=True)
 if process.returncode or not record['cleanup_pass']:break
selected=[p for p in root.rglob('*')if p.is_file()and p.suffix!='.pyc'and 'inputs'not in p.relative_to(root).parts]
assert len(selected)<100 and sum(p.stat().st_size for p in selected)<64*(1<<20)
inventory={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in selected}
(root/'download-manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
with tarfile.open(root/'download.tar.gz','x:gz',compresslevel=1)as tar:
 for p in selected+[root/'download-manifest.json']:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
assert len(records)==2 and all(r['returncode']==0 and r['cleanup_pass']for r in records)
'''.replace('REMOTE',repr(remote)).replace('ARCHIVE_SHA',repr(sha(archive)))
compile(driver,'remote-driver','exec')
(D/'driver.py').write_text(driver)
opts=['-o','BatchMode=yes','-o','ConnectTimeout=10']
subprocess.run(['scp',*opts,str(archive),'gb10-4t:/home/qujing/'],capture_output=True,check=True,timeout=40)
command=['ssh',*opts,'gb10-4t','python3 -'];started=time.monotonic()
r=subprocess.run(command,input=driver,capture_output=True,text=True,timeout=1020)
(D/'transport.json').write_text(json.dumps(dict(command=command,remote_script_sha256=sha(D/'driver.py'),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,wall_seconds=time.monotonic()-started),indent=2)+'\n')
print(r.stdout[-6000:],r.stderr[-1500:],flush=True)
download=D/'download.tar.gz'
subprocess.run(['scp',*opts,'gb10-4t:'+remote+'/download.tar.gz',str(download)],capture_output=True,check=True,timeout=60)
out=D/'collected';out.mkdir()
with tarfile.open(download)as tar:
 entries=tar.getmembers();assert len(entries)<101 and sum(e.size for e in entries)<64*(1<<20)
 assert all(e.isfile()and not Path(e.name).is_absolute()and '..'not in Path(e.name).parts for e in entries)
 tar.extractall(out,filter='data')
inventory=json.loads((out/'download-manifest.json').read_text())
for file,item in inventory.items():
 p=out/file;assert p.stat().st_size==item['bytes']and sha(p)==item['sha256']
print(json.dumps(dict(files_verified=len(inventory),archive_sha256=sha(download))),flush=True)
if r.returncode:
 for p in out.glob('*.stderr.log'):
  print(p.name,p.read_text()[-6000:],flush=True)
r.check_returncode()
