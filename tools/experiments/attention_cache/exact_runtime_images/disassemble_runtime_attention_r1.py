"""Read existing AMD code objects with the installed disassembler; no GPU work."""
from pathlib import Path
import hashlib
import json
import socket
import subprocess

assert socket.gethostname().split('.')[0].lower() != 'baiying'
B = Path(__file__).resolve().parent
D = B / 'runtime-attention-codeobjects-r1'
remote = 'D:/projects/runtime-attention-codeobjects-20260923-r1'
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
comparison = json.loads((D / 'comparison.json').read_text())
options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
records = []


def call(args, timeout):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    records.append(dict(command=args, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr))
    (D / 'disassembly-transport.json').write_text(json.dumps(records, indent=2) + '\n')
    r.check_returncode()
    return r


setup = "$ErrorActionPreference='Stop';if(Test-Path '" + remote + "'){throw 'Directory exists'};New-Item -ItemType Directory -Path '" + remote + "'|Out-Null"
call(['ssh', *options, 'baiying', 'powershell.exe -NoProfile -Command "' + setup + '"'], 20)
jobs = []
for label in ('observed', 'qualified_probe'):
    source = D / (label + '-verified.hsaco')
    expected = comparison['sources'][label]['bundle']['image_sha256']
    assert sha(source) == expected
    call(['scp', *options, str(source), 'baiying:' + remote + '/' + source.name], 40)
    for symbol, item in comparison['sources'][label]['functions'].items():
        if item['type'] != 2 or not any(word in symbol for word in ('6scores', '14segment_kernel')):
            continue
        kind = 'scores' if '6scores' in symbol else 'segment'
        jobs.append(dict(file=source.name, sha256=expected, tool='llvm-objdump.exe',
                         arguments=['-d', '--mcpu=gfx1151', '--disassemble-symbols=' + symbol],
                         output=label + '-' + kind + '.isa'))
    jobs.append(dict(file=source.name, sha256=expected, tool='llvm-readobj.exe',
                     arguments=['--notes'], output=label + '-notes.txt'))
script = r'''$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
if(-not [Environment]::MachineName.Equals('baiying',[StringComparison]::OrdinalIgnoreCase)){throw 'Wrong host'}
$root='REMOTE'
$jobs=@'
JOBS
'@|ConvertFrom-Json
$results=@()
foreach($job in $jobs){
 $inputFile=Join-Path $root $job.file
 if((Get-FileHash -LiteralPath $inputFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $job.sha256){throw 'Input changed'}
 $tool=Join-Path 'C:/Program Files/AMD/ROCm/7.1/bin' $job.tool
 $info=New-Object Diagnostics.ProcessStartInfo
 $info.FileName=$tool;$info.UseShellExecute=$false
 $info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
 $info.Arguments=(@($job.arguments)+@($inputFile)|ForEach-Object{ '"'+$_+'"' }) -join ' '
 $process=New-Object Diagnostics.Process;$process.StartInfo=$info
 $null=$process.Start();$out=$process.StandardOutput.ReadToEndAsync();$err=$process.StandardError.ReadToEndAsync()
 if(-not $process.WaitForExit(20000)){try{$process.Kill()}catch{};throw 'Disassembler deadline'}
 $stdout=$out.GetAwaiter().GetResult();$stderr=$err.GetAwaiter().GetResult()
 if($process.ExitCode -ne 0){throw ('Disassembler failed: '+$stderr)}
 $output=Join-Path $root $job.output
 [IO.File]::WriteAllText($output,$stdout,[Text.UTF8Encoding]::new($false))
 if((Get-Item $output).Length -gt 8MB){throw 'Unexpected disassembly size'}
 $results+=@([ordered]@{tool=$tool;tool_sha256=(Get-FileHash $tool -Algorithm SHA256).Hash.ToLowerInvariant();
   arguments=$info.Arguments;input_sha256=$job.sha256;exit_code=$process.ExitCode;stderr=$stderr;
   output=$job.output;output_bytes=(Get-Item $output).Length;
   output_sha256=(Get-FileHash $output -Algorithm SHA256).Hash.ToLowerInvariant()})
}
$record=[ordered]@{host=[Environment]::MachineName;utc=[DateTime]::UtcNow.ToString('o');
 command_file=$PSCommandPath;command_file_sha256=(Get-FileHash $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant();
 results=$results;gpu_api_called=$false;model_loaded=$false;native_inference_executed=$false}
[IO.File]::WriteAllText((Join-Path $root 'disassembly.json'),($record|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
$record|ConvertTo-Json -Depth 8 -Compress
'''.replace('REMOTE', remote).replace('JOBS', json.dumps(jobs))
path = D / 'disassemble.ps1'
path.write_text(script)
call(['scp', *options, str(path), 'baiying:' + remote + '/disassemble.ps1'], 30)
result = call(['ssh', *options, 'baiying', 'powershell.exe -NoProfile -File ' + remote + '/disassemble.ps1'], 160)
record = json.loads(result.stdout)
assert record['command_file_sha256'] == sha(path) and not record['gpu_api_called']
for row in record['results']:
    output = D / row['output']
    call(['scp', *options, 'baiying:' + remote + '/' + row['output'], str(output)], 30)
    assert sha(output) == row['output_sha256'] and output.stat().st_size == row['output_bytes']
(D / 'disassembly.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(files=len(record['results']), bytes=sum(r['output_bytes'] for r in record['results']),
                     native_execution=False, gpu_api_called=False)))
