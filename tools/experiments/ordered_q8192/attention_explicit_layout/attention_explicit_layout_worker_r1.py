"""Full q8192 original-operand comparison and gfx1151 compile inspection."""
from pathlib import Path
import hashlib
import json
import re
import socket
import time

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource
import attention_explicit as candidate
import attention as original

root = Path('/work')
out = root / 'output'
out.mkdir()
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
m = json.loads((root / 'manifest.json').read_text())
torch.set_num_threads(4)
assert torch.cuda.get_device_capability() == (12, 1) and triton.__version__ == '3.6.0'
for name, value in m['source_hashes'].items():
    assert sha(root / name) == value
ref = Path('/reference')
assert sha(ref / 'dispatch.json') == m['reference_dispatch_sha256']
assert json.loads((ref / 'dispatch.json').read_text())['reference_boundary_qualified']
assert sha(ref / 'capture/q8192-out512.json') == m['reference_case_sha256']
case = json.loads((ref / 'capture/q8192-out512.json').read_text())
capture = case['worker']['attention_capture']
assert case['control_pass'] and capture['complete'] and len(case['output_token_ids']) == 512
prior = json.loads((root / 'prior-component.json').read_text())
assert prior['all_components_match'] and len(prior['records']) == 64
for name in ['attention.py', 'integer_u.py', 'dense_selected.py', 'ordered_pipeline.py']:
    assert prior['source_hashes'][name] == sha(root / name)
prior_rows = {r['query_start']: r for r in prior['records']}
options = dict(num_warps=4, num_stages=2, enable_fp_fusion=False)


def raw(value):
    return value.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def info(value):
    content = raw(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest())


def guarded(shape, dtype, cpu=None):
    count = 1
    for d in shape:
        count *= d
    count *= torch.empty((), dtype=dtype).element_size()
    backing = torch.full((count + 1024,), 0xa5, dtype=torch.uint8, device='cuda')
    value = backing[512:-512].view(dtype).reshape(shape)
    if cpu is not None:
        value.copy_(cpu)
    return backing, value


def guards(items):
    return all(bool((x[:512] == 0xa5).all()) and bool((x[-512:] == 0xa5).all()) for x in items)


def load_ref(name):
    e = capture['files'][name]
    path = ref / 'capture/q8192-out512' / e['file']
    assert path.stat().st_size == e['bytes'] and sha(path) == e['sha256']
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).reshape(e['shape'])


def compare(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    a, b = actual.cpu().view(dtype), expected.cpu().view(dtype)
    different = a != b
    count = int(different.sum())
    result = dict(values=a.numel(), bit_mismatches=count, actual=info(actual), expected=info(expected))
    if count:
        indices = different.reshape(-1).nonzero()[:12, 0]
        result['first_differences'] = [dict(index=int(i), actual_bits=int(a.reshape(-1)[i]),
            expected_bits=int(b.reshape(-1)[i])) for i in indices]
    return result


compiled = {}
configs = [
    ('qk', candidate.ordered_qk_kernel, dict(Q='*bf16', K='*bf16', Scores='*fp32', T='i32', QS='i32', QT='i32'), dict(BM=8, BN=8)),
    ('probability', candidate.probability_kernel, dict(Scores='*fp32', Probability='*bf16', Scales='*fp32', Table='*u8', T='i32', QS='i32', QT='i32'), dict(BM=4, FUSED_L=True)),
    ('pv', candidate.selected_pv_kernel, dict(Probability='*bf16', Value='*bf16', Scales='*fp32', RcpTable='*u8', Counts='*i32', Indices='*i32', Output='*bf16', RawAccumulator='*fp32', T='i32', QS='i32', QT='i32'), dict(BM=32, CAPTURE_F32=False)),
]
for bm, bn in [(8, 8), (8, 16)]:
    configs.append((f'tiled{bm}x{bn}', candidate.tiled_pv_kernel,
        dict(Probability='*bf16', Value='*bf16', Scales='*fp32', RcpTable='*u8', Output='*bf16', T='i32', QS='i32', QT='i32'), dict(BM=bm, BN=bn)))
for name, fn, signature, constants in configs:
    result = triton.compile(GluonASTSource(fn, signature=signature, constexprs=constants),
                           target=GPUTarget('hip', 'gfx1151', 32), options=options)
    path = out / (name + '.hsaco')
    path.write_bytes(result.asm['hsaco'])
    asm, ir = result.asm['amdgcn'], result.asm['ttgir']
    (out / (name + '.amdgcn')).write_text(asm)
    (out / (name + '.ttgir')).write_text(ir)
    compiled[name] = dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path),
        signature=signature, constants=constants, metadata=result.metadata._asdict(), options=options,
        layout_conversion_count=ir.count('ttg.convert_layout'),
        shared_memory_instructions=len(re.findall(r'^\s*ds_(?:read|write|load|store)', asm, re.M)),
        resources={key: re.findall(re.escape(key) + r'\s*:?\s*(\d+)', asm)
                   for key in ('.vgpr_count', '.vgpr_spill_count', '.private_segment_fixed_size')},
        amdgcn_sha256=sha(out / (name + '.amdgcn')), ttgir_sha256=sha(out / (name + '.ttgir')))
    print(json.dumps(dict(compiled=name, metadata=compiled[name]['metadata'],
        layout_conversions=compiled[name]['layout_conversion_count'],
        shared_instructions=compiled[name]['shared_memory_instructions']), default=str), flush=True)
(out / 'compilation.json').write_text(json.dumps(compiled, indent=2, default=str) + '\n')

qc, kc, vc, expected = (load_ref(name) for name in ('q', 'k', 'v', 'context'))
qb, q = guarded(qc.shape, qc.dtype, qc)
kb, k = guarded(kc.shape, kc.dtype, kc)
vc = vc.reshape(8192, 2, 256).permute(1, 2, 0).contiguous()
vb, v = guarded(vc.shape, vc.dtype, vc)
tables = {}
table_backings = []
for name, path in [('exp2', Path('/table/exp2.bin')), ('rcp', root / 'rcp.bin')]:
    entry = m['tables'][name]
    assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
    content = path.read_bytes()
    backing, value = guarded((len(content),), torch.uint8,
                             torch.frombuffer(bytearray(content), dtype=torch.uint8))
    tables[name] = value
    table_backings.append(backing)
del qc, kc, vc, content
et, rt = tables['exp2'], tables['rcp']
fixed_inputs = [('q', q), ('k', k), ('v', v), ('exp2', et), ('rcp', rt)]
unchanged = {name: info(value) for name, value in fixed_inputs}
T, QT = 8192, 128
sb, scores = guarded((QT, 16, T), torch.float32)
rb, original_scores = guarded((QT, 16, T), torch.float32)
pb, probability = guarded((QT, 16, T), torch.bfloat16)
opb, original_probability = guarded((QT, 16, T), torch.bfloat16)
ab, scales = guarded((QT, 16, T // 32 + 1), torch.float32)
oab, original_scales = guarded((QT, 16, T // 32 + 1), torch.float32)
db, debug = guarded((1,), torch.float32)
cb, counts = guarded((1,), torch.int32, torch.tensor([QT * 4096], dtype=torch.int32))
ib, indices = guarded((QT * 4096,), torch.int32, torch.arange(QT * 4096, dtype=torch.int32))
outputs = {name: guarded((QT, 4096), torch.bfloat16) for name in ('selected32', 'tiled8x8', 'tiled8x16')}
all_backings = [qb, kb, vb, sb, rb, pb, opb, ab, oab, db, cb, ib, *table_backings,
                *[x[0] for x in outputs.values()]]
debug_hash = info(debug)
digests = {name: hashlib.sha256() for name in outputs}
records, queue_records = [], []
assert m['query_starts'] == list(range(0, T, QT))
for slab, QS in enumerate(m['query_starts']):
    started = time.monotonic()
    original.original_qk_kernel[(QT // 2, T // 32, 2)](q, k, original_scores, T, QS, QT,
        num_warps=4, num_stages=3, enable_fp_fusion=True)
    candidate.ordered_qk_kernel[(QT // 8, T // 8, 16)](q, k, scores, T, QS, QT, BM=8, BN=8, **options)
    torch.cuda.synchronize()
    qk = compare(scores, original_scores)
    assert qk['expected']['sha256'] == prior_rows[QS]['qk_original_mma']['sha256']
    original.probability_kernel[(QT // 4, 16)](original_scores, original_probability, original_scales,
        et, T, QS, QT, BM=4, FUSED_L=True, **options)
    candidate.probability_kernel[(QT // 4, 16)](scores, probability, scales, et, T, QS, QT,
        BM=4, FUSED_L=True, **options)
    torch.cuda.synchronize()
    # Unused future tiles retain the same initialized poison in both arrays.
    prob = compare(probability, original_probability)
    scale = compare(scales, original_scales)
    before_pv = (info(probability), info(scales))
    pv = {}
    names = list(outputs)
    names = names[slab % 3:] + names[:slab % 3]
    for name in names:
        backing, output = outputs[name]
        output.view(torch.uint8).fill_(0xa5)
        if name == 'selected32':
            candidate.selected_pv_kernel[(1024,)](probability, v, scales, rt, counts, indices,
                output, debug, T, QS, QT, BM=32, CAPTURE_F32=False, **options)
        else:
            bn = 8 if name == 'tiled8x8' else 16
            candidate.tiled_pv_kernel[(QT // 8, 256 // bn, 16)](probability, v, scales, rt,
                output, T, QS, QT, BM=8, BN=bn, **options)
        torch.cuda.synchronize()
        pv[name] = compare(output, expected[QS:QS + QT])
        assert pv[name]['expected']['sha256'] == prior_rows[QS]['context_original']['sha256']
        digests[name].update(raw(output))
    if QS in (0, 8064):
        for count in (0, 137):
            # A fixed permutation touches multiple heads, rows and partial tiles.
            selected = (torch.arange(count, dtype=torch.int32, device='cuda') * 7919 + 11) % (QT * 4096)
            counts.fill_(count)
            indices[:count].copy_(selected)
            queue_before = (info(counts), info(indices))
            output = outputs['selected32'][1]
            output.view(torch.uint8).fill_(0xa5)
            candidate.selected_pv_kernel[(17,)](probability, v, scales, rt, counts, indices,
                output, debug, T, QS, QT, BM=32, CAPTURE_F32=False, **options)
            torch.cuda.synchronize()
            selected_cpu = selected.cpu().long()
            expected_selected = expected[QS:QS + QT].reshape(-1)[selected_cpu]
            selected_result = compare(output.reshape(-1)[selected.long()], expected_selected)
            untouched = output.cpu().reshape(-1).view(torch.int16).clone()
            untouched[selected_cpu] = -23131  # 0xa5a5 interpreted as signed16.
            queue_records.append(dict(query_start=QS, selected_count=count,
                comparison=selected_result, unselected_output_unchanged=bool((untouched == -23131).all()),
                queues_unchanged=queue_before == (info(counts), info(indices)), guards_pass=guards(all_backings)))
        counts.fill_(QT * 4096)
        indices.copy_(torch.arange(QT * 4096, dtype=torch.int32, device='cuda'))
    row = dict(query_start=QS, qk=qk, probability=prob, scales=scale, pv=pv,
        probability_and_scales_unchanged=before_pv == (info(probability), info(scales)),
        guards_pass=guards(all_backings), unused_debug_unchanged=info(debug) == debug_hash,
        diagnostic_wall_seconds=time.monotonic() - started, performance_acceptance=False)
    records.append(row)
    (out / ('case-' + str(QS) + '.json')).write_text(json.dumps(row, indent=2) + '\n')
    if QS % 1024 == 0 or QS == 8064 or qk['bit_mismatches'] or any(x['bit_mismatches'] for x in pv.values()):
        print(json.dumps(dict(query_start=QS, qk_mismatches=qk['bit_mismatches'],
            probability_mismatches=prob['bit_mismatches'], scales_mismatches=scale['bit_mismatches'],
            pv_mismatches={name: x['bit_mismatches'] for name, x in pv.items()}, guards_pass=row['guards_pass'])), flush=True)
after = {name: info(value) for name, value in fixed_inputs}
whole = {name: value.hexdigest() for name, value in digests.items()}
passed = len(records) == 64 and len(queue_records) == 4 and unchanged == after and guards(all_backings)
passed = passed and all(v == capture['files']['context']['sha256'] for v in whole.values())
passed = passed and all(r['guards_pass'] and r['unused_debug_unchanged'] and r['probability_and_scales_unchanged'] and
    all(r[k]['bit_mismatches'] == 0 for k in ('qk', 'probability', 'scales')) and
    all(x['bit_mismatches'] == 0 for x in r['pv'].values()) for r in records)
passed = passed and all(r['comparison']['bit_mismatches'] == 0 and r['unselected_output_unchanged'] and
                       r['queues_unchanged'] and r['guards_pass'] for r in queue_records)
source_unchanged = all(sha(root / name) == value for name, value in m['source_hashes'].items())
passed = passed and source_unchanged
report = dict(host=socket.gethostname(), device=torch.cuda.get_device_name(),
    manifest_sha256=sha(root / 'manifest.json'), source_hashes=m['source_hashes'],
    candidate_base_commit=m['candidate_base_commit'], original_reference=m['original_reference'],
    records=records, selected_queue_controls=queue_records, compiled=compiled,
    complete_context_sha256=whole, original_context_sha256=capture['files']['context']['sha256'],
    qk_fp32_values=sum(r['qk']['values'] for r in records),
    probability_bf16_values=sum(r['probability']['values'] for r in records),
    scale_fp32_values=sum(r['scales']['values'] for r in records),
    context_bf16_values=sum(x['values'] for r in records for x in r['pv'].values()),
    all_components_match=bool(passed), inputs_and_tables_unchanged=unchanged == after,
    sources_unchanged=source_unchanged, guards_pass=guards(all_backings),
    original_model_loaded=False, amd_gpu_executed=False, inference_acceptance=False,
    performance_acceptance=False, release_qualified=False)
(out / 'result.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
print(json.dumps(dict(all_components_match=bool(passed), qk_fp32_values=report['qk_fp32_values'],
                     context_bf16_values=report['context_bf16_values'])), flush=True)
assert passed
