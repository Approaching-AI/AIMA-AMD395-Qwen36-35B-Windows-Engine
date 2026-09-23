from pathlib import Path
import hashlib, importlib.util, json, socket, sys, time
import torch, triton
from safetensors import safe_open
from triton.experimental.gluon._runtime import GluonASTSource as ASTSource
from triton.backends.compiler import GPUTarget

root = Path('/work')
out = root / 'output'
out.mkdir()
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
manifest = json.loads((root / 'manifest.json').read_text())
assert torch.cuda.get_device_capability() == (12, 1) and triton.__version__ == '3.6.0'
for name, digest in manifest['source_hashes'].items():
    assert sha(root / name) == digest
reference = Path('/reference/capture/q8192-out512.json')
assert sha(reference) == manifest['reference_case_sha256']
case = json.loads(reference.read_text())
assert case['control_pass'] and case['full_matrix_case_pass'] and len(case['output_token_ids']) == 512
index_path = Path('/models/model.safetensors.index.json')
assert sha(index_path) == '41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83'
index = json.loads(index_path.read_text())
spec = importlib.util.spec_from_file_location('dense_selected_candidate', root / 'kernel.py')
kernel = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kernel
spec.loader.exec_module(kernel)
options = dict(num_warps=4, num_stages=2, enable_fp_fusion=False)
WINDOW = 1 << 20
records = []
sources = []
compiled = {}


def raw(value):
    return value.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def tensor_info(value):
    data = raw(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def guarded(shape, dtype, cpu=None):
    cells = 1
    for extent in shape:
        cells *= extent
    size = cells * torch.empty((), dtype=dtype).element_size()
    backing = torch.full((size + 1024,), 0xa5, dtype=torch.uint8, device='cuda')
    value = backing[512:-512].view(dtype).reshape(shape)
    if cpu is not None:
        value.copy_(cpu)
    return backing, value


def safe(buffers):
    return all(bool((b[:512] == 0xa5).all()) and bool((b[-512:] == 0xa5).all()) for b in buffers)


def read_operand(entry):
    path = Path('/reference/capture/q8192-out512') / entry['file']
    assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).reshape(entry['shape'])


torch.set_num_threads(4)
for ordinal, projection in enumerate(manifest['projections']):
    x_cpu = read_operand(projection['input'])
    expected = read_operand(projection['output'])
    key = projection['weight']
    shard = index['weight_map'][key]
    assert Path(shard).name == shard and shard.endswith('.safetensors')
    with safe_open(str(Path('/models') / shard), framework='pt', device='cpu') as tensors:
        weight = tensors.get_tensor(key).contiguous()
    T, K = x_cpu.shape
    N, WK = weight.shape
    assert WK == K and expected.shape == (T, N) and weight.dtype == torch.bfloat16
    assert T == 71 and (N, K) == ((8192, 2048) if ordinal == 0 else (2048, 4096))
    source = dict(name=projection['name'], input=projection['input'], output=projection['output'],
        weight_key=key, weight_shard=shard, weight_tensor=tensor_info(weight), model_index_sha256=sha(index_path))
    sources.append(source)
    xb, x = guarded(x_cpu.shape, torch.bfloat16, x_cpu)
    cells = T * N
    generator = torch.Generator().manual_seed(20260923 + ordinal)
    permutation = torch.randperm(cells, generator=generator, dtype=torch.int32)
    counts_cpu = torch.tensor([0, 17, cells - 37, 20, 0], dtype=torch.int32)
    queue_cpu = torch.full((5, WINDOW), -1, dtype=torch.int32)
    consumed = 0
    for window, count in enumerate(counts_cpu.tolist()):
        queue_cpu[window, :count] = permutation[consumed:consumed + count]
        consumed += count
    assert consumed == cells
    cb, counts = guarded(counts_cpu.shape, torch.int32, counts_cpu)
    ib, queue = guarded(queue_cpu.shape, torch.int32, queue_cpu)
    fixed_inputs = dict(x=tensor_info(x), counts=tensor_info(counts), queue=tensor_info(queue))
    first_f32 = None
    for layout in ('contiguous', 'transposed'):
        w_cpu = weight if layout == 'contiguous' else weight.t().contiguous()
        wb, w = guarded(w_cpu.shape, torch.bfloat16, w_cpu)
        weight_info = tensor_info(w)
        row_stride, k_stride = (K, 1) if layout == 'contiguous' else (1, N)
        for BM in (32, 64, 128):
            yb, output = guarded(expected.shape, torch.bfloat16)
            db, debug = guarded(expected.shape, torch.float32)
            begun = time.monotonic()
            kernel.selected_replay_kernel[(256, 5)](x, w, counts, queue, output, debug,
                N, K, row_stride, k_stride, BM=BM, WINDOW=WINDOW, CAPTURE_F32=True, **options)
            torch.cuda.synchronize()
            observed = output.cpu()
            different = observed.view(torch.int16) != expected.view(torch.int16)
            fp32_info = tensor_info(debug)
            if first_f32 is None:
                first_f32 = fp32_info
            row = dict(projection=projection['name'], layout=layout, batch=BM, grid=[256, 5],
                counts=counts_cpu.tolist(), values=cells, bf16_mismatches=int(different.sum()),
                maximum_absolute_error=float((observed.float() - expected.float()).abs().max()),
                output=tensor_info(output), original=projection['output'], fp32=fp32_info,
                fp32_matches_other_configurations=fp32_info == first_f32,
                guards_pass=safe((xb, wb, cb, ib, yb, db)),
                input_unchanged=tensor_info(x) == fixed_inputs['x'],
                weight_unchanged=tensor_info(w) == weight_info,
                counts_unchanged=tensor_info(counts) == fixed_inputs['counts'],
                queue_unchanged=tensor_info(queue) == fixed_inputs['queue'],
                wall_seconds=time.monotonic() - begun, performance_acceptance=False)
            if row['bf16_mismatches']:
                indices = different.reshape(-1).nonzero()[:32, 0]
                row['first_differences'] = [dict(index=int(i), actual=int(observed.view(torch.int16).reshape(-1)[i]),
                    original=int(expected.view(torch.int16).reshape(-1)[i])) for i in indices]
            records.append(row)
            print(json.dumps({k: row[k] for k in ('projection', 'layout', 'batch', 'values', 'bf16_mismatches', 'guards_pass', 'wall_seconds')}), flush=True)
            del output, debug, yb, db
        del w, wb
    del x, counts, queue, xb, cb, ib

for BM in (32, 64, 128):
    compiled_kernel = triton.compile(ASTSource(kernel.selected_replay_kernel,
        signature=dict(X='*bf16', W='*bf16', Counts='*i32', Indices='*i32', Output='*bf16', Debug='*fp32',
                       N='i32', K='i32', W_ROW='i32', W_K='i32'),
        constexprs=dict(BM=BM, WINDOW=WINDOW, CAPTURE_F32=False)),
        target=GPUTarget('hip', 'gfx1151', 32), options=options)
    path = out / ('dense-selected-explicit-bm%d.hsaco' % BM)
    path.write_bytes(compiled_kernel.asm['hsaco'])
    assembly = compiled_kernel.asm['amdgcn']
    (out / ('dense-selected-explicit-bm%d.amdgcn' % BM)).write_text(assembly)
    (out / ('dense-selected-explicit-bm%d.ttgir' % BM)).write_text(compiled_kernel.asm['ttgir'])
    import re
    compiled[str(BM)] = dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path),
        metadata=compiled_kernel.metadata._asdict(), kernel_hash=compiled_kernel.hash,
        signature=dict(X='*bf16', W='*bf16', Counts='*i32', Indices='*i32', Output='*bf16', Debug='*fp32', N='i32', K='i32', W_ROW='i32', W_K='i32'),
        constants=dict(BM=BM, WINDOW=WINDOW, CAPTURE_F32=False),
        layout_conversion_count=compiled_kernel.asm['ttgir'].count('ttg.convert_layout'),
        shared_memory_instructions=len(re.findall(r'^\s*ds_(?:load|store)', assembly, re.M)),
        options=options, resources={key: re.findall(re.escape(key) + r'\s*:?\s*(\d+)', assembly)
            for key in ('.vgpr_count', '.vgpr_spill_count', '.private_segment_fixed_size')})
passed = all(r['bf16_mismatches'] == 0 and r['fp32_matches_other_configurations'] and
    all(r[k] for k in ('guards_pass', 'input_unchanged', 'weight_unchanged', 'counts_unchanged', 'queue_unchanged')) for r in records)
report = dict(host=socket.gethostname(), device=torch.cuda.get_device_name(), manifest_sha256=sha(root / 'manifest.json'),
    source_hashes=manifest['source_hashes'], candidate_base_commit=manifest['candidate_base_commit'],
    reference_case_sha256=manifest['reference_case_sha256'], source_model_reference='/mnt/data/models/Qwen3.6-35B-A3B mounted at /models',
    sources=sources, records=records, compiled=compiled, all_original_bf16_values_match=passed,
    total_bf16_values=sum(r['values'] for r in records), model_loaded=False, actual_weight_tensors_loaded=True,
    amd_gpu_executed=False, inference_acceptance=False, performance_acceptance=False)
(out / 'result.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
print(json.dumps(dict(all_original_bf16_values_match=passed, total_bf16_values=report['total_bf16_values'],
    compiled=len(compiled), resources={k: v['resources'] for k, v in compiled.items()})), flush=True)
assert passed
