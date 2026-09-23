"""Check fused recurrence against original q7169 tensors and an unfused seed."""
from pathlib import Path
import hashlib
import json
import socket
import time
import torch
import triton
from persistent import persistent_kernel
import ordered_pipeline as baseline

root = Path('/work')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
manifest = json.loads((root / 'manifest.json').read_text())
assert triton.__version__ == '3.6.0' and torch.cuda.get_device_capability() == (12, 1)
for name, digest in manifest['source_hashes'].items():
    assert sha((root / name).read_bytes()) == digest
out = root / 'numerical'; out.mkdir()
refs = {}
for name in ('q-normalized', 'k-normalized', 'w', 'u', 'g-cumsum', 'initial_state',
             'v-new', 'output', 'state', 'chunk-state'):
    item = manifest['inputs'][name]
    raw = (Path(item['mount']) / item['file']).read_bytes()
    assert len(raw) == item['bytes'] and sha(raw) == item['sha256']
    dtype = torch.bfloat16 if item['dtype'] == 'bfloat16' else torch.float32
    refs[name] = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(item['shape'])
raw = Path('/table/exp2.bin').read_bytes()
assert len(raw) == 183174448 and sha(raw) == 'f490940df2bd80421159a96424c3e922330b7ca120d5ae7b629a973b9183730b'
table = torch.frombuffer(bytearray(raw), dtype=torch.uint8).cuda(); del raw
tensors = {name: refs[name].cuda() for name in ('q-normalized', 'k-normalized', 'w', 'u', 'g-cumsum', 'initial_state')}
options = dict(num_warps=4, num_stages=2, enable_fp_fusion=False)
guarded = []
def output(shape, dtype):
    count = 1
    for dim in shape: count *= dim
    item = 2 if dtype == torch.bfloat16 else 4
    backing = torch.full((count * item + 1024,), 0xa5, dtype=torch.uint8, device='cuda')
    guarded.append(backing)
    return backing[512:-512].view(dtype).reshape(shape)
def guards():
    return all(bool((x[:512] == 0xa5).all()) and bool((x[-512:] == 0xa5).all()) for x in guarded)
def compare(actual, expected):
    actual = actual.cpu().contiguous(); expected = expected.cpu().contiguous()
    bits = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    return dict(values=actual.numel(), bit_mismatches=int((actual.view(bits) != expected.view(bits)).sum()),
        actual_sha256=sha(actual.view(torch.uint8).numpy().tobytes()),
        original_sha256=sha(expected.view(torch.uint8).numpy().tobytes()))
T = 7169
scores = output((1, T, 32, 64), torch.bfloat16)
baseline.gram_kernel[(triton.cdiv(T, 8), 8, 32)](tensors['q-normalized'], tensors['k-normalized'],
    tensors['g-cumsum'], tensors['g-cumsum'], table, scores, T, BM=8, BN=8, SCORE=True, **options)
torch.cuda.synchronize()
scores_hash = sha(scores.cpu().view(torch.uint8).numpy().tobytes())
records = []
def candidate(label, q, k, w, u, g, s, initial, expected_core, expected_state, expected_v=None, expected_h=None, capture=False):
    count = q.shape[1]
    core = output((1, count, 32, 128), torch.bfloat16)
    final = output((1, 32, 128, 128), torch.float32)
    vn = output(core.shape, torch.bfloat16) if capture else None
    history = output((1, triton.cdiv(count, 64), 32, 128, 128), torch.bfloat16) if capture else None
    start = time.monotonic()
    kernel = persistent_kernel[(32, 16)](q, k, w, u, g, s, table, initial, final, core, vn, history, count,
                                         CAPTURE=capture, **options)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    comparisons = dict(core=compare(core, expected_core), state=compare(final, expected_state))
    if capture:
        comparisons['v_new'] = compare(vn, expected_v)
        comparisons['incoming_state_checkpoints'] = compare(history, expected_h)
    row = dict(name=label, tokens=count, chunks=triton.cdiv(count, 64), capture=capture,
        comparisons=comparisons, guards_pass=guards(), diagnostic_seconds=elapsed,
        kernel_hash=kernel.hash, kernel_metadata=kernel.metadata._asdict(),
        original_values_match=all(v['bit_mismatches'] == 0 for v in comparisons.values()),
        performance_acceptance=False)
    records.append(row)
    (out / (label + '.json')).write_text(json.dumps(row, indent=2, default=str) + '\n')
    print(json.dumps(dict(name=label, seconds=elapsed, mismatches={k:v['bit_mismatches'] for k,v in comparisons.items()},
                         guards_pass=row['guards_pass'])), flush=True)
    assert row['original_values_match'] and row['guards_pass']
    return core, final

candidate('original-q7169-capture', tensors['q-normalized'], tensors['k-normalized'], tensors['w'], tensors['u'],
    tensors['g-cumsum'], scores, tensors['initial_state'], refs['output'], refs['state'],
    refs['v-new'], refs['chunk-state'], capture=True)
candidate('original-q7169-runtime', tensors['q-normalized'], tensors['k-normalized'], tensors['w'], tensors['u'],
    tensors['g-cumsum'], scores, tensors['initial_state'], refs['output'], refs['state'])

# A source-qualified, unfused prefix supplies an independently materialized
# nonzero incoming FP32 state. The fused invocation must carry it through a
# complete chunk and a one-token tail without resetting the state.
core = output((1, 129, 32, 128), torch.bfloat16)
vn = output(core.shape, torch.bfloat16); residual = output(core.shape, torch.bfloat16)
current = tensors['initial_state']
seed = None
for first in (0, 64, 128):
    count = min(64, 129 - first); end = first + count
    final = output((1, 32, 128, 128), torch.float32)
    g = tensors['g-cumsum'][:, first:end]
    grid = (triton.cdiv(count, 8), 16, 32)
    baseline.residual_kernel[grid](tensors['w'][:, first:end], tensors['u'][:, first:end], current,
        g, table, vn[:, first:end], residual[:, first:end], count, BM=8, BN=8, **options)
    baseline.state_kernel[(16, 16, 32)](tensors['k-normalized'][:, first:end], residual[:, first:end],
        current, g, table, final, count, BM=8, BN=8, **options)
    baseline.output_kernel[grid](tensors['q-normalized'][:, first:end], vn[:, first:end], current,
        g, scores[:, first:end], table, core[:, first:end], count, BM=8, BN=8, **options)
    current = final
    if first == 0: seed = final
torch.cuda.synchronize()
assert compare(core, refs['output'][:, :129])['bit_mismatches'] == 0
assert compare(seed.to(torch.bfloat16), refs['chunk-state'][:, 1])['bit_mismatches'] == 0
seed_hash = sha(seed.cpu().view(torch.uint8).numpy().tobytes())
candidate('original-first64-runtime', tensors['q-normalized'][:, :64], tensors['k-normalized'][:, :64],
    tensors['w'][:, :64], tensors['u'][:, :64], tensors['g-cumsum'][:, :64], scores[:, :64],
    tensors['initial_state'], refs['output'][:, :64], seed)
candidate('nonzero-seeded65-runtime', tensors['q-normalized'][:, 64:129], tensors['k-normalized'][:, 64:129],
    tensors['w'][:, 64:129], tensors['u'][:, 64:129], tensors['g-cumsum'][:, 64:129], scores[:, 64:129],
    seed, refs['output'][:, 64:129], current)
assert sha(seed.cpu().view(torch.uint8).numpy().tobytes()) == seed_hash
immutable = all(torch.equal(x.cpu().view(torch.uint8), refs[name].view(torch.uint8)) for name, x in tensors.items())
assert immutable and guards() and sha(scores.cpu().view(torch.uint8).numpy().tobytes()) == scores_hash
report = dict(host=socket.gethostname(), device=torch.cuda.get_device_name(), source_hashes=manifest['source_hashes'],
    source_manifest_sha256=sha((root/'manifest.json').read_bytes()), records=records,
    all_original_values_match=True, inputs_state_scores_unchanged=True, guards_pass=True,
    original_reference=manifest['reference_model'], original_matrix_result_sha256=manifest['original_matrix_result_sha256'],
    seeded_fp32_control_sha256=seed_hash, seeded_control_matches_original_bf16_state=True,
    launch_geometry=[32,16], q8192_recurrence_launches_before=384, q8192_recurrence_launches_candidate=1,
    model_loaded=False, amd_gpu_executed=False, inference_acceptance=False, performance_acceptance=False)
(out/'result.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
print(json.dumps(dict(all_original_values_match=True, cases=len(records))), flush=True)
