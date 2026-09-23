"""Run paired full-q8192 routed trials after the current AMD owner is clean."""
from pathlib import Path
import base64
import hashlib
import json
import socket
import subprocess
import sys

B = Path(__file__).resolve().parent
D = B / 'routed-selected-explicit-windows-r1'
assert socket.gethostname().split('.')[0].lower() != 'baiying'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(D / 'manifest.json') == '82d3aaa6e4667f28b7a7b78968353fe9988835a15abc3bf5acfd674f7e15564b'
m = read(D / 'manifest.json')
assert sha(D / 'worker.py') == m['worker_sha256']
local_entries = list(m['images'].values())
for e in local_entries:
    p = D / e['file']
    assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
for e in m['source_inputs']:
    p=D/e['file'];assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256']
for e in m['inputs'].values():
    p=B/m['local_inputs_directory']/e['file'];assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256']
prior_path = B / m['active_owner_local_record']
ready = prior_path.exists()
if ready:
    prior = read(prior_path)
    ready = prior['host_checks_pass'] and all(prior['host_checks'].values()) and not prior['after_processes']
if sys.argv[1:] == ['--check']:
    print(json.dumps(dict(prepared_manifest_verified=True, full256k_owner_complete_and_clean=bool(ready),
        remote_calls=0, executed=False)))
    sys.exit(0)
assert not sys.argv[1:] and ready, 'Active full256k owner has no completed clean host record; zero remote calls.'
remote = m['remote_directory']
opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
quote = lambda s: "'" + str(s).replace("'", "''") + "'"
wrapper = "$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue';[Console]::InputEncoding=[Text.UTF8Encoding]::new($false);[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);$s=[Console]::In.ReadToEnd();& ([ScriptBlock]::Create($s))"
command = 'powershell.exe -NoProfile -EncodedCommand ' + base64.b64encode(wrapper.encode('utf-16le')).decode()


def ps(script, timeout=30):
    return subprocess.run(['ssh', *opts, 'baiying', command], input=script, capture_output=True,
                          text=True, errors='replace', timeout=timeout)


def save(name, value):
    with (D / name).open('x') as file:
        json.dump(value, file, indent=2)
        file.write('\n')


def record(result):
    return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


admission = "if(-not [Environment]::MachineName.Equals('baiying',[StringComparison]::OrdinalIgnoreCase)){throw 'Unexpected host'}\n"
admission += "function Verify([string]$p,[string]$s){if((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() -ne $s){throw ('Changed file '+$p)}}\n"
admission += 'Verify ' + quote(m['active_owner_remote_record']) + ' ' + quote(sha(prior_path)) + '\n'
admission += '$prior=Get-Content -Raw -LiteralPath ' + quote(m['active_owner_remote_record']) + '|ConvertFrom-Json;if(-not $prior.host_checks_pass -or @($prior.after_processes).Count -ne 0){throw "Previous owner was not clean"}\n'
setup = admission + 'if(Test-Path ' + quote(remote) + "){throw 'Replay directory exists'};$null=New-Item -ItemType Directory -Path " + quote(remote) + ' -Force\n'
result = ps(setup)
save('setup.json', dict(command_script=setup, **record(result)))
result.check_returncode()
for path in [D / 'manifest.json', D / 'worker.py', *[D / e['file'] for e in local_entries]]:
    subprocess.run(['scp', *opts, str(path), 'baiying:' + remote + '/' + path.name], capture_output=True,
                   text=True, check=True, timeout=60)
subprocess.run(['scp', *opts, '-r', str(D/'source'), 'baiying:' + remote + '/'], capture_output=True, text=True, check=True, timeout=45)
spec = dict(executable=m['python_executable'], arguments=[remote + '/worker.py', remote],
    working_directory=m['execution_checkout'], repo_commit=m['execution_checkout_commit'],
    source_manifest_sha256=sha(D / 'manifest.json'), command_file=remote + '/dispatch.ps1',
    model=m['model_reference'], model_loaded=False, actual_original_model_operands_loaded=True, actual_model_weight_tensors_loaded=True,
    purpose='Full q8192 routed gate/up and weighted down: interleaved original and explicit images, three tiles and two passes;17 malformed and partial-queue controls',
    inference_acceptance=False, performance_acceptance=False)
script = admission + 'Verify ' + quote(m['guard_file']) + ' ' + quote(m['guard_sha256']) + '\n'
script += 'Verify ' + quote(remote + '/manifest.json') + ' ' + quote(sha(D / 'manifest.json')) + '\n'
script += 'Verify ' + quote(remote + '/worker.py') + ' ' + quote(m['worker_sha256']) + '\n'
script += '$env:PATH=' + quote(m['rocm_root'] + '/bin;') + '+$env:PATH\n'
script += "$spec=@'\n" + json.dumps(spec) + "\n'@\n[IO.File]::WriteAllText(" + quote(remote + '/spec.json') + ",$spec,[Text.UTF8Encoding]::new($false))\n"
script += '& ' + quote(m['guard_file']) + ' -SpecPath ' + quote(remote + '/spec.json') + ' -OutDir ' + quote(remote + '/run') + ' -TimeoutSeconds ' + str(m['native_timeout_seconds']) + '|Out-Null\n'
script += '$r=Get-Content -Raw -LiteralPath ' + quote(remote + '/run/run-record.json') + '|ConvertFrom-Json;if($r.reason -ne "completed" -or $r.exit_code -notin @(0,6) -or -not $r.host_checks_pass -or @($r.after_processes).Count -ne 0){throw "Component host execution failed"};$r.host_checks|ConvertTo-Json -Compress\n'
with (D / 'dispatch.ps1').open('x') as file:
    file.write(script)
subprocess.run(['scp', *opts, str(D / 'dispatch.ps1'), 'baiying:' + remote + '/dispatch.ps1'],
               capture_output=True, text=True, check=True, timeout=30)
result = ps('& ' + quote(remote + '/dispatch.ps1'), m['native_timeout_seconds'] + 60)
save('dispatch.json', dict(command_file=remote + '/dispatch.ps1', command_file_sha256=sha(D / 'dispatch.ps1'),
                           **record(result)))
collection = []
for folder, names in [('run', ['run-record.json', 'preflight.json', 'launch.json', 'product.stdout.jsonl',
                              'product.stderr.log', 'telemetry.jsonl']), ('outputs', ['result.json'])]:
    (D / folder).mkdir()
    for name in names:
        r = subprocess.run(['scp', *opts, 'baiying:' + remote + '/' + folder + '/' + name, str(D / folder / name)],
                           capture_output=True, text=True, timeout=40)
        collection.append(dict(file=folder + '/' + name, **record(r)))
save('collection.json', collection)
result.check_returncode()
assert all(r['returncode'] == 0 for r in collection)
run, report = read(D / 'run/run-record.json'), read(D / 'outputs/result.json')
assert run['host_checks_pass'] and all(run['host_checks'].values()) and not run['after_processes']
assert report['manifest_sha256'] == sha(D / 'manifest.json') and report['cleanup_pass'] and report['source_files_unchanged']
assert len(report['records']) == 24 and len(report['sources']) == 2 and len(report['queue_controls']) == 17
print(json.dumps(dict(host=report['host'], all_components_match=report['all_components_match'],
    native_exit_code=run['exit_code'], result_sha256=sha(D / 'outputs/result.json'),
    run_sha256=sha(D / 'run/run-record.json'), inference_acceptance=False)), flush=True)
