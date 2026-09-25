#!/usr/bin/env python3
"""Rebuild explicit-layout q8192 images for gfx1151 without a GPU.

Triton 3.6.0 is a build dependency. These images still require native component
and complete original-token product qualification before runtime selection.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = Path('native/providers/explicit_q8192')
SOURCE_NAMES = ('attention_explicit.py', 'explicit_dense.py', 'routed_explicit.py',
                'explicit_group16.py', 'explicit_exp2.py', 'pack_value.py')


def jobs():
    sys.path.insert(0, str(ROOT / PROVIDER))
    import attention_explicit as attention
    import explicit_dense as dense
    import routed_explicit as routed
    from pack_value import pack_value_kernel
    result = []
    for batch in (32, 64, 128):
        result.append((f'dense{batch}', dense.selected_replay_kernel,
            dict(X='*bf16', W='*bf16', Counts='*i32', Indices='*i32', Output='*bf16', Debug='*fp32',
                 N='i32', K='i32', W_ROW='i32', W_K='i32'),
            dict(BM=batch, WINDOW=1 << 20, CAPTURE_F32=False)))
    for down in (False, True):
        for batch in (32, 64, 128):
            name = ('down' if down else 'gate-up') + f'-bm{batch}'
            result.append((name, routed.routed_replay_kernel,
                dict(X='*bf16', W='*bf16', RouteIds='*i32', RouteWeights='*fp32', Counts='*i32',
                     Indices='*i32', Output='*bf16', Debug='*fp32', Invalid='*i32', Routes='i32'),
                dict(DOWN=down, BM=batch, WINDOW=4 << 20, CAPTURE_F32=False)))
    result.extend([
        ('pack', pack_value_kernel, dict(Source='*bf16', Packed='*bf16', T='i32'), {}),
        ('qk', attention.ordered_qk_kernel,
         dict(Q='*bf16', K='*bf16', Scores='*fp32', T='i32', QS='i32', QT='i32'), dict(BM=8, BN=8)),
        ('probability', attention.probability_kernel,
         dict(Scores='*fp32', Probability='*bf16', Scales='*fp32', Table='*u8', T='i32', QS='i32', QT='i32'),
         dict(BM=4, FUSED_L=True)),
        ('pv', attention.selected_pv_kernel,
         dict(Probability='*bf16', Value='*bf16', Scales='*fp32', RcpTable='*u8', Counts='*i32',
              Indices='*i32', Output='*bf16', RawAccumulator='*fp32', T='i32', QS='i32', QT='i32'),
         dict(BM=32, CAPTURE_F32=False)),
    ])
    for bn in (8, 16):
        result.append((f'tiled8x{bn}', attention.tiled_pv_kernel,
            dict(Probability='*bf16', Value='*bf16', Scales='*fp32', RcpTable='*u8', Output='*bf16',
                 T='i32', QS='i32', QT='i32'), dict(BM=8, BN=bn)))
    return result


def embed(out, compiled):
    """Embed exact selected bytes; diagnostic tiled PV remains separate."""
    groups = {
        'prefill': [f'dense{batch}' for batch in (32, 64, 128)],
        'routed': [f'{projection}-bm{batch}' for projection in ('gate-up', 'down') for batch in (32, 64, 128)],
        'attention': ['pack', 'qk', 'probability', 'pv'],
    }
    paths = {}
    for group, names in groups.items():
        lines = ['// SPDX-License-Identifier: Apache-2.0',
                 '// Explicit-layout experiment; AMD and full-model qualification remain required.']
        if group == 'prefill':
            lines.append('struct OrderedReplayImage { const unsigned char* data; std::size_t bytes; const char* sha256; unsigned batch, warps, shared; };')
        elif group == 'routed':
            lines.append('struct RoutedOrderedImage { unsigned batch; bool down; const unsigned char* data; std::size_t bytes; const char* sha256; unsigned warps, shared; };')
        for ordinal, name in enumerate(names):
            entry = compiled[name]
            data = (out / entry['file']).read_bytes()
            assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
            lines.append(f'alignas(64) static constexpr unsigned char explicit_{group}_{ordinal}[] = {{')
            lines.extend('  ' + ','.join(str(v) for v in data[i:i + 24]) + ',' for i in range(0, len(data), 24))
            lines.append('};')
        kind, variable = {'prefill': ('OrderedReplayImage', 'ordered_replay_images'),
            'routed': ('RoutedOrderedImage', 'routed_ordered_images'),
            'attention': ('OrderedAttentionImage', 'ordered_attention_images')}[group]
        lines.append(f'static constexpr {kind} {variable}[] = {{')
        for ordinal, name in enumerate(names):
            entry = compiled[name]
            meta = entry['metadata']
            array = f'explicit_{group}_{ordinal}'
            if group == 'prefill':
                fields = f'{array}, sizeof({array}), "{entry["sha256"]}", {entry["constants"]["BM"]}'
            elif group == 'routed':
                fields = f'{entry["constants"]["BM"]}, {str(entry["constants"]["DOWN"]).lower()}, {array}, sizeof({array}), "{entry["sha256"]}"'
            else:
                fields = f'{array}, sizeof({array}), "{meta["name"]}", "{entry["sha256"]}"'
            lines.append(f'  {{{fields}, {meta["num_warps"]}, {meta["shared"]}}},')
        lines.append('};')
        filename = {'prefill': 'gb10_prefill_ordered_images.inc', 'routed': 'gb10_routed_ordered_images.inc',
                    'attention': 'gb10_ordered_attention_images.inc'}[group]
        path = out / filename
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        paths[group] = path
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for name in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        if os.environ.get(name) != '-1':
            raise ValueError('Offline compilation requires ' + name + '=-1')
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.experimental.gluon._runtime import GluonASTSource
    if triton.__version__ != '3.6.0':
        raise ValueError('The pinned Triton 3.6.0 compiler is required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    options = dict(num_warps=4, num_stages=2, enable_fp_fusion=False)
    compiled = {}
    for name, fn, signature, constants in jobs():
        source_type = ASTSource if name == 'pack' else GluonASTSource
        kernel = triton.compile(source_type(fn, signature=signature, constexprs=constants),
                                target=GPUTarget('hip', 'gfx1151', 32), options=options)
        data, assembly, ttgir = kernel.asm['hsaco'], kernel.asm['amdgcn'], kernel.asm['ttgir']
        (out / (name + '.hsaco')).write_bytes(data)
        (out / (name + '.amdgcn')).write_text(assembly)
        (out / (name + '.ttgir')).write_text(ttgir)
        resources = {key: re.findall(re.escape(key) + r'\s*:?\s*(\d+)', assembly)
                     for key in ('.vgpr_count', '.vgpr_spill_count', '.private_segment_fixed_size')}
        compiled[name] = dict(file=name + '.hsaco', bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
            metadata=kernel.metadata._asdict(), signature=signature, constants=constants, options=options,
            resources=resources, layout_conversion_count=ttgir.count('convert_layout'),
            amdgcn_sha256=hashlib.sha256(assembly.encode()).hexdigest(),
            ttgir_sha256=hashlib.sha256(ttgir.encode()).hexdigest())
    includes = embed(out, compiled)
    sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
    report = dict(source_files={(PROVIDER / name).as_posix(): sha(ROOT / PROVIDER / name) for name in SOURCE_NAMES},
        compiler_script_sha256=sha(Path(__file__)), triton_version=triton.__version__, compiled=compiled,
        embedded_includes={name: dict(file=p.name, bytes=p.stat().st_size, sha256=sha(p)) for name, p in includes.items()},
        diagnostic_pv_tiles_embedded=False, wall_seconds=time.monotonic() - started,
        cpu_only=True, gpu_executed=False, inference_acceptance=False, performance_acceptance=False)
    (out / 'result.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps(dict(compiled=len(compiled), embedded_includes=report['embedded_includes'], gpu_executed=False)))


if __name__ == '__main__':
    main()
