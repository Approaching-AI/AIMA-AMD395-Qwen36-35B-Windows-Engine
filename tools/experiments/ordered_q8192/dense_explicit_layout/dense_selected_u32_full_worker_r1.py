from pathlib import Path
import hashlib, importlib.util, json, socket, sys, time
import torch, triton
from safetensors import safe_open

root = Path('/work')
out = root / 'output'
out.mkdir()
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
manifest = json.loads((root / 'manifest.json').read_text())
assert torch.cuda.get_device_capability() == (12, 1) and triton.__version__ == '3.6.0'
torch.set_num_threads(4)
for name, digest in manifest['source_hashes'].items():
    assert sha(root / name) == digest
ref = Path('/reference')
assert sha(ref / 'dispatch.json') == manifest['reference_dispatch_sha256']
assert json.loads((ref / 'dispatch.json').read_text())['reference_boundary_qualified']
assert sha(ref / 'capture/q8192-out512.json') == manifest['reference_case_sha256']
case = json.loads((ref / 'capture/q8192-out512.json').read_text())
assert case['control_pass'] and case['worker']['dense_capture']['complete']
assert len(case['output_token_ids']) == 512
index_path = Path('/models/model.safetensors.index.json')
assert sha(index_path) == '41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83'
index = json.loads(index_path.read_text())
spec = importlib.util.spec_from_file_location('dense_selected_full_candidate', root / 'kernel.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
prior = json.loads((root / 'small-component.json').read_text())
assert prior['all_original_bf16_values_match'] and prior['source_hashes']['kernel.py'] == sha(root / 'kernel.py')
WINDOW = 1 << 20
options = dict(num_warps=4, num_stages=2, enable_fp_fusion=False)
records = []
sources = []


def info(tensor):
    raw = tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def guarded(shape, dtype, cpu=None):
    count = 1
    for size in shape:
        count *= size
    n = count * torch.empty((), dtype=dtype).element_size()
    backing = torch.full((n + 1024,), 0xa5, dtype=torch.uint8, device='cuda')
    tensor = backing[512:-512].view(dtype).reshape(shape)
    if cpu is not None:
        tensor.copy_(cpu)
    return backing, tensor


def guard_pass(buffers):
    return all(bool((b[:512] == 0xa5).all()) and bool((b[-512:] == 0xa5).all()) for b in buffers)


def read_tensor(entry):
    path = ref / 'capture/q8192-out512' / entry['file']
    assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).reshape(entry['shape'])


for ordinal, projection in enumerate(manifest['projections']):
    x_cpu = read_tensor(projection['input'])
    expected = read_tensor(projection['output'])
    key = projection['weight']
    shard = index['weight_map'][key]
    assert Path(shard).name == shard and shard.endswith('.safetensors')
    with safe_open(str(Path('/models') / shard), framework='pt', device='cpu') as tensors:
        weight = tensors.get_tensor(key).contiguous()
    T, K = x_cpu.shape
    N, WK = weight.shape
    assert T == 8192 and WK == K and expected.shape == (T, N)
    assert (N, K) == [(8192, 2048), (4096, 2048), (2048, 4096)][ordinal]
    source = dict(name=projection['name'], input=projection['input'], output=projection['output'],
        weight_key=key, weight_shard=shard, weight_tensor=info(weight), model_index_sha256=sha(index_path))
    sources.append(source)
    layout = projection['layout']
    w_cpu = weight if layout == 'contiguous' else weight.t().contiguous()
    row_stride, k_stride = (K, 1) if layout == 'contiguous' else (1, N)
    xb, x = guarded(x_cpu.shape, torch.bfloat16, x_cpu)
    wb, w = guarded(w_cpu.shape, torch.bfloat16, w_cpu)
    cells = T * N
    generator = torch.Generator().manual_seed(20260923 + ordinal)
    permutation = torch.randperm(cells, generator=generator, dtype=torch.int32)
    counts_cpu = torch.tensor([0] + [min(WINDOW, cells - first) for first in range(0, cells, WINDOW)] + [0], dtype=torch.int32)
    windows = counts_cpu.numel()
    queue_cpu = torch.full((windows, WINDOW), -1, dtype=torch.int32)
    offset = 0
    for i, count in enumerate(counts_cpu.tolist()):
        queue_cpu[i, :count] = permutation[offset:offset + count]
        offset += count
    assert offset == cells
    cb, counts = guarded(counts_cpu.shape, torch.int32, counts_cpu)
    ib, queue = guarded(queue_cpu.shape, torch.int32, queue_cpu)
    immutable = dict(x=info(x), w=info(w), counts=info(counts), indices=info(queue))
    del permutation, queue_cpu, x_cpu, w_cpu, weight
    for BM in (32, 64, 128):
        yb, result = guarded(expected.shape, torch.bfloat16)
        db, debug = guarded((1,), torch.float32)
        untouched_debug = info(debug)
        started = time.monotonic()
        module.selected_replay_kernel[(256, windows)](x, w, counts, queue, result, debug,
            N, K, row_stride, k_stride, BM=BM, WINDOW=WINDOW, CAPTURE_F32=False, **options)
        torch.cuda.synchronize()
        execution_seconds = time.monotonic() - started
        actual = result.cpu()
        different = actual.view(torch.int16) != expected.view(torch.int16)
        row = dict(projection=projection['name'], tokens=T, n=N, k=K, layout=layout,
            batch=BM, grid=[256, windows], windows=windows, queued_cells=cells,
            bf16_mismatches=int(different.sum()), output=info(actual), original=projection['output'],
            guards_pass=guard_pass((xb, wb, cb, ib, yb, db)),
            unused_debug_unchanged=info(debug) == untouched_debug, fp32_capture=False,
            diagnostic_execution_seconds=execution_seconds, performance_acceptance=False)
        if row['bf16_mismatches']:
            at = different.reshape(-1).nonzero()[:32, 0]
            row['first_differences'] = [dict(index=int(i), actual=int(actual.view(torch.int16).reshape(-1)[i]),
                original=int(expected.view(torch.int16).reshape(-1)[i])) for i in at]
        records.append(row)
        (out / ('record-%s-%d.json' % (projection['name'], BM))).write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps({k: row[k] for k in ('projection', 'tokens', 'batch', 'queued_cells', 'bf16_mismatches',
            'guards_pass', 'diagnostic_execution_seconds')}), flush=True)
        del result, actual, different, debug, yb, db
    after = dict(x=info(x), w=info(w), counts=info(counts), indices=info(queue))
    source['device_inputs_unchanged'] = after == immutable
    source['device_inputs'] = immutable
    assert source['device_inputs_unchanged']
    del x, w, counts, queue, xb, wb, cb, ib, expected
    torch.cuda.empty_cache()

passed = len(records) == 9 and all(r['bf16_mismatches'] == 0 and r['guards_pass'] and r['unused_debug_unchanged'] for r in records)
report = dict(host=socket.gethostname(), device=torch.cuda.get_device_name(),
    manifest_sha256=sha(root / 'manifest.json'), source_hashes=manifest['source_hashes'],
    candidate_base_commit=manifest['candidate_base_commit'], reference_case_sha256=manifest['reference_case_sha256'],
    reference_dispatch_sha256=manifest['reference_dispatch_sha256'],
    model='/mnt/data/models/Qwen3.6-35B-A3B mounted at /models',
    sources=sources, records=records, compiled=prior['compiled'],
    all_original_bf16_values_match=passed, total_bf16_values=sum(r['queued_cells'] for r in records),
    model_engine_loaded=False, actual_weight_tensors_loaded=True, amd_gpu_executed=False,
    inference_acceptance=False, performance_acceptance=False)
(out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(all_original_bf16_values_match=passed, total_bf16_values=report['total_bf16_values'])), flush=True)
assert passed
