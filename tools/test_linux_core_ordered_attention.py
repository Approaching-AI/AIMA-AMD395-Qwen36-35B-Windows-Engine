#!/usr/bin/env python3
"""Check exact gfx1151 image bindings, q8192 scratch ownership and failures.

The recording HIP API executes no GPU arithmetic and loads no model.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--out',type=Path,required=True)
 parser.add_argument('--cxx',default='clang++')
 args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
 source=ROOT/'native/linux_core_port/gb10_ordered_attention_host_test.cpp'
 stub=ROOT/'tests/native/ordered_attention_fake_hip'
 command=[args.cxx,'-std=c++17','-O1','-ffp-contract=off','-fsanitize=address,undefined','-fno-sanitize-recover=all',
  '-I',str(stub),'-I',str(ROOT/'third_party/aima_linux/native/include'),str(source),
  str(ROOT/'third_party/aima_linux/native/src/sha256.cpp'),str(ROOT/'third_party/aima_linux/native/src/aot_kernel.hip.cpp'),
  '-o',str(out/'host-contract')]
 build=subprocess.run(command,capture_output=True,text=True,timeout=90)
 (out/'build.stderr').write_text(build.stderr);build.check_returncode()
 run=subprocess.run([str(out/'host-contract')],capture_output=True,text=True,timeout=30)
 (out/'run.stdout').write_text(run.stdout);(out/'run.stderr').write_text(run.stderr)
 files=[source,ROOT/'native/linux_core_port/gb10_ordered_attention.h',
  ROOT/'native/linux_core_port/gb10_ordered_attention.cpp',ROOT/'native/linux_core_port/gb10_ordered_attention_images.inc',
  stub/'hip/hip_runtime_api.h',Path(__file__),ROOT/'third_party/aima_linux/native/src/sha256.cpp',
  ROOT/'third_party/aima_linux/native/src/aot_kernel.hip.cpp',ROOT/'third_party/aima_linux/native/include/aima/aot_kernel.h',
  ROOT/'third_party/aima_linux/native/include/aima/sha256.h']
 report=dict(build_command=command,returncode=run.returncode,
  inputs={str(p.relative_to(ROOT)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest()for p in files},
  host_only=True,gpu_arithmetic_executed=False,inference_acceptance=False,performance_acceptance=False)
 if run.returncode==0:report['result']=json.loads(run.stdout)
 (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report.get('result',dict(returncode=run.returncode))))
 run.check_returncode()
if __name__=='__main__':main()
