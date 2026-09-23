from pathlib import Path
import hashlib
import json
import os
import re
import time
for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
    os.environ[key] = '-1'
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource
from persistent import persistent_kernel

root = Path('/work'); out = root/'aot'; out.mkdir()
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
manifest = json.loads((root/'manifest.json').read_text())
assert triton.__version__ == '3.6.0'
for file, digest in manifest['source_hashes'].items(): assert sha(root/file) == digest
signature = dict(Q='*bf16', K='*bf16', W='*bf16', U='*bf16', G='*fp32', Scores='*bf16',
                 Table='*i8', Initial='*fp32', Final='*fp32', Out='*bf16', VNew='*bf16', Checkpoints='*bf16', T='i32')
records = []; started = time.monotonic()
for capture in (False, True):
    options = dict(num_warps=4, num_stages=2, enable_fp_fusion=False)
    kernel = triton.compile(GluonASTSource(persistent_kernel, signature=signature, constexprs=dict(CAPTURE=capture)),
        target=GPUTarget('hip', 'gfx1151', 32), options=options)
    name = 'persistent-capture' if capture else 'persistent-runtime'
    image = out/(name+'.hsaco'); image.write_bytes(kernel.asm['hsaco'])
    for kind in ('amdgcn', 'ttgir', 'llir'): (out/(name+'.'+kind)).write_text(kernel.asm[kind])
    resources = {key: re.findall(re.escape(key)+r'\s*:?\s*(\d+)', kernel.asm['amdgcn'])
        for key in ('.vgpr_count', '.vgpr_spill_count', '.private_segment_fixed_size')}
    record = dict(name=name, capture=capture, file=image.name, bytes=image.stat().st_size, sha256=sha(image),
        signature=signature, constants=dict(CAPTURE=capture), options=options,
        metadata=kernel.metadata._asdict(), resources=resources,
        layout_conversion_count=kernel.asm['ttgir'].count('ttg.convert_layout'),
        shared_memory_bytes=kernel.metadata.shared)
    records.append(record); print(json.dumps(dict(name=name, bytes=image.stat().st_size,
        resources=resources, shared=kernel.metadata.shared)), flush=True)
report = dict(source_manifest_sha256=sha(root/'manifest.json'), source_hashes=manifest['source_hashes'],
    images=records, wall_seconds=time.monotonic()-started, gpu_used=False, amd_gpu_executed=False,
    inference_acceptance=False, performance_acceptance=False)
(out/'result.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
