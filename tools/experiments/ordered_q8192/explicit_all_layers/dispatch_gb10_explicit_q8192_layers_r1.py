from pathlib import Path
import hashlib, json, socket, subprocess, tarfile, time, shutil

B = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0].lower() != 'baiying'
m = json.loads((B / 'qrt-gb10-explicit-q8192-layers-20260923-r1-stage.json').read_text())
opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
name = m['name']
D = B / name
assert not D.exists() and not (B / (name + '-transport.json')).exists()
for key in ('archive', 'script'):
    assert hashlib.file_digest((B / m[key]).open('rb'), 'sha256').hexdigest() == m[key + '_sha256']
subprocess.run(['scp', *opts, str(B / m['archive']), str(B / m['script']),
    'gb10-4t:/home/qujing/'], check=True, timeout=60)
command = ['ssh', *opts, 'gb10-4t', 'python3 /home/qujing/' + m['script']]
started = time.monotonic()
try:
    result = subprocess.run(command, capture_output=True, text=True, errors='replace', timeout=m['ssh_timeout_seconds'])
    record = dict(command=command, returncode=result.returncode, stdout=result.stdout,
        stderr=result.stderr, wall_seconds=time.monotonic() - started)
except subprocess.TimeoutExpired as error:
    record = dict(command=command, returncode=None, timeout=True,
        stdout=(error.stdout or b'').decode(errors='replace') if isinstance(error.stdout, bytes) else error.stdout,
        stderr=(error.stderr or b'').decode(errors='replace') if isinstance(error.stderr, bytes) else error.stderr,
        wall_seconds=time.monotonic() - started)
    with (B / (name + '-transport.json')).open('x') as f: json.dump(record, f, indent=2); f.write('\n')
    raise
with (B / (name + '-transport.json')).open('x') as f: json.dump(record, f, indent=2); f.write('\n')
print('reference_dispatch', result.returncode, result.stdout[-4500:], result.stderr[-2000:], flush=True)
metadata = json.loads(result.stdout.splitlines()[-1])
assert shutil.disk_usage(B).free > metadata['compact_archive_bytes'] + metadata['selected_bytes'] + (64 << 20), 'Completed reference retained remotely; insufficient controller space'
archive = B / (name + '-compact.tar.gz')
assert not archive.exists()
subprocess.run(['scp', *opts, 'gb10-4t:/home/qujing/' + name + '/compact-download.tar.gz',
    str(archive)], check=True, timeout=180)
assert archive.stat().st_size == metadata['compact_archive_bytes']
assert hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest() == metadata['compact_archive_sha256']
D.mkdir()
with tarfile.open(archive) as tar:
 members=tar.getmembers();assert len(members)<1800 and sum(m.size for m in members)<400*(1<<20)
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 tar.extractall(D, filter='data')
manifest = json.loads((D / 'compact-download-manifest.json').read_text())
for relative, expected in manifest.items():
    p = D / relative
    assert p.stat().st_size == expected['bytes']
    assert hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() == expected['sha256'], relative
print('compact_files_verified', len(manifest), flush=True)
receipt = dict(archive=archive.name, bytes=archive.stat().st_size, sha256=metadata['compact_archive_sha256'], remote_archive_retained=True, all_downloaded_members_hash_verified=True, removed_local_duplicate=False)
receipt_path = B / (name + '-download-archive-reclamation.json')
with receipt_path.open('x') as f: json.dump(receipt, f, indent=2); f.write('\n')
archive.unlink()
receipt['removed_local_duplicate']=True
receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
result.check_returncode()
