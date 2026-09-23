"""Bind an exact-image layer19 replay to existing original-model evidence."""
from pathlib import Path
import ctypes as C
import hashlib
import importlib.util
import json
import os
import re
import shutil

B = Path(__file__).resolve().parent
SOURCE = B / 'runtime-attention-codeobjects-r1'
TARGET = B / 'runtime-attention-images-windows-r1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def metadata(text, symbol):
    marker = re.search(r'^    \.name:\s+' + re.escape(symbol) + '$', text, re.M)
    assert marker
    start = text.rfind('  - .args:', 0, marker.start())
    end = text.find('  - .args:', marker.end())
    block = text[start:end if end >= 0 else len(text)]
    arguments = block[:block.index('    .group_segment_fixed_size:')]
    offsets = [int(x) for x in re.findall(r'\.offset:\s+(\d+)', arguments)]
    sizes = [int(x) for x in re.findall(r'\.size:\s+(\d+)', arguments)]
    kinds = re.findall(r'\.value_kind:\s+(\w+)', arguments)
    fields = {}
    for name in ['kernarg_segment_size', 'group_segment_fixed_size', 'private_segment_fixed_size',
                 'wavefront_size', 'vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count']:
        fields[name] = int(re.search(r'^    \.' + name + r':\s+(\d+)', block, re.M)[1])
    return dict(offsets=offsets, sizes=sizes, value_kinds=kinds, **fields)


def main():
    comparison_path = SOURCE / 'comparison.json'
    assert sha(comparison_path) == '5b16362ac532ce9bd24b0a733e268bb96c9989796ebed2d5dfed9456abcfc876'
    comparison = read(comparison_path)
    assert comparison['all_attention_instructions_and_descriptor_resources_match_prior']
    assert not comparison['all_attention_instructions_and_descriptor_resources_match_probe']
    disassembly = read(SOURCE / 'disassembly.json')
    assert len(disassembly['results']) == 6 and not disassembly['gpu_api_called']
    for entry in disassembly['results']:
        path = SOURCE / entry['output']
        assert entry['exit_code'] == 0 and not entry['stderr']
        assert path.stat().st_size == entry['output_bytes'] and sha(path) == entry['output_sha256']
    original_manifest_path = B / 'layer19-segmented-attention-windows-r1/manifest.json'
    assert sha(original_manifest_path) == 'b090e097a734692c4322d6d83fe54c18c25ef392e28dbc8188683079317d3552'
    original_manifest = read(original_manifest_path)
    native = read(B / 'layer19-segmented-attention-windows-r1/comparison.json')
    assert native['host_checks_pass'] and native['native_exit_code'] == 0
    assert all(r['bit_mismatches'] == 0 for r in native['report']['comparisons'])
    qk_path = B / 'qrt-gb10-layer19-qk-capture-20260923-r1/comparison.json'
    qk = read(qk_path)
    for key in ['original_source_recovers_exactly_after_removing_observer',
                'original_fp32_context_and_segments_bit_exact', 'score_guards_pass', 'all_inputs_unchanged']:
        assert qk[key]
    assert original_manifest['original608_outputs_qualified']
    symbols = {
        'scores': '_ZN22qrt_sm121_q1_attention6scoresEPKfPKtS3_Pfjjj',
        'segment': '_ZN32qrt_sm121_q1_segmented_attention14segment_kernelEPKfPKtS3_PfS4_S4_jjjPKh',
        'merge': '_ZN32qrt_sm121_q1_segmented_attention12merge_kernelEPKfS1_S1_PfPKhS4_',
    }
    specs = {
        'scores': dict(argument_types=['pointer'] * 4 + ['u32'] * 3,
                       offsets=[0, 8, 16, 24, 32, 36, 40], sizes=[8]*4+[4]*3,
                       kernarg_bytes=44, static_shared_bytes=0),
        'segment': dict(argument_types=['pointer'] * 6 + ['u32'] * 3 + ['pointer'],
                        offsets=[0, 8, 16, 24, 32, 40, 48, 52, 56, 64], sizes=[8]*6+[4]*3+[8],
                        kernarg_bytes=72, static_shared_bytes=176),
        'merge': dict(argument_types=['pointer'] * 6, offsets=[0, 8, 16, 24, 32, 40],
                      sizes=[8]*6, kernarg_bytes=48, static_shared_bytes=68),
    }
    kernels = {}
    for name, spec in specs.items():
        records = {}
        for image in ('observed', 'qualified_probe'):
            record = metadata((SOURCE / (image + '-notes.txt')).read_text(), symbols[name])
            assert record['offsets'] == spec['offsets'] and record['sizes'] == spec['sizes']
            assert record['value_kinds'] == ['global_buffer' if t == 'pointer' else 'by_value' for t in spec['argument_types']]
            assert record['kernarg_segment_size'] == spec['kernarg_bytes']
            assert record['group_segment_fixed_size'] == spec['static_shared_bytes']
            assert record['wavefront_size'] == 32 and record['sgpr_spill_count'] == record['vgpr_spill_count'] == 0
            records[image] = record
        fields = [('a' + str(i), C.c_void_p if t == 'pointer' else C.c_uint32)
                  for i, t in enumerate(spec['argument_types'])]
        packed = type(name.title() + 'Args', (C.Structure,), {'_fields_': fields})
        assert [getattr(packed, f).offset for f, _ in fields] == spec['offsets']
        assert spec['offsets'][-1] + spec['sizes'][-1] == spec['kernarg_bytes']
        kernels[name] = dict(symbol=symbols[name], **spec, metadata=records,
                             block=[256, 1, 1], dynamic_shared_bytes=0,
                             launch_argument_offsets_host_verified=True)
    module_spec = importlib.util.spec_from_file_location('image_worker', B / 'runtime_attention_image_worker_r1.py')
    worker = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(worker)
    raw = bytes(16 * 2 * 4)
    assert worker.compare(raw, raw, [16, 2])['bit_exact']
    changed = bytearray(raw)
    changed[0], changed[-1] = 1, 0x80
    negative = worker.compare(bytes(changed), raw, [16, 2])
    assert negative['bit_mismatches'] == 2 and not negative['bit_exact']
    assert [x['coordinates'] for x in negative['first_differences']] == [[0, 0], [15, 1]]
    TARGET.mkdir()
    images = {}
    for name in ('observed', 'qualified_probe'):
        entry = comparison['sources'][name]
        source = Path(entry['image_file'])
        assert source.stat().st_size == entry['bundle']['image_bytes']
        assert sha(source) == entry['bundle']['image_sha256']
        os.link(source, TARGET / source.name)
        images[name] = dict(file=source.name, bytes=source.stat().st_size, sha256=sha(source),
            binary_source_commit=entry['source_commit'], binary_source_file=entry['source_file'],
            binary_source_sha256=entry['source_sha256'], extraction=entry['bundle'],
            functions={symbol: entry['functions'][symbol] for symbol in symbols.values()})
    inputs, tables = {}, {}
    for original in original_manifest['files']:
        entry = {k: original[k] for k in ['path', 'bytes', 'sha256']}
        (tables if original['role'] in ('exp2', 'rcp') else inputs)[original['role']] = entry
    score = next(x for x in qk['artifacts'] if x['file'] == 'scores-f32.bin')
    path = qk_path.parent / score['file']
    assert path.stat().st_size == score['bytes'] and sha(path) == score['sha256']
    os.link(path, TARGET / 'scores-f32.bin')
    inputs['expected_scores'] = dict(file='scores-f32.bin', bytes=score['bytes'], sha256=score['sha256'])
    shutil.copyfile(B / 'runtime_attention_image_worker_r1.py', TARGET / 'worker.py')
    previous = read(B / 'dense-selected-explicit-windows-r1/manifest.json')
    manifest = {k: previous[k] for k in ['python_executable', 'rocm_root', 'guard_file', 'guard_sha256',
                                        'active_owner_local_record', 'active_owner_remote_record']}
    manifest.update(schema=1, host='baiying', tokens=263292, prefixes=[263168, 263291],
        strides=[263309, 263681], repeats=2, images=images, kernels=kernels,
        inputs=inputs, tables=tables, worker_sha256=sha(TARGET / 'worker.py'),
        execution_checkout='P:/projects/AIMA-public-q1-cache-capture-20260923-r1',
        execution_checkout_commit='b2b7c0bea3f3fe41ada0fc2d99abb48079e64478',
        remote_directory='D:/projects/runtime-attention-images-20260923-r1',
        model_reference='D:/models/Qwen3.6-35B-A3B', native_timeout_seconds=300,
        original_reference=dict(host='gb10-4t', model='/mnt/data/models/Qwen3.6-35B-A3B',
            source_commit=qk['source_commit'], input_manifest_sha256=sha(original_manifest_path),
            capture_sha256=original_manifest['original_capture_sha256'],
            original_608_outputs_qualified=True,
            original_qualified_dispatch_sha256=original_manifest['original_qualified_dispatch_sha256'],
            original_replay_sha256=original_manifest['original_replay_sha256'],
            original_qk_observer_sha256=sha(qk_path)),
        code_comparison_sha256=sha(comparison_path),
        disassembly_sha256=sha(SOURCE / 'disassembly.json'),
        host_controls=dict(argument_offsets_match_both_actual_images=True,
            exact_comparison_positive_pass=True, changed_first_last_words_detected=True),
        expectation=dict(qk_cases=16, pv_cases=48, full_scores_fp32_values_per_qk_case=4212672,
            independent_original_scores_replay_cases=16, other_pv_cases_use_actual_qk_image_output=True,
            expected_context_and_segments_host_only=True),
        native_execution=False, original_model_loaded=False, inference_acceptance=False,
        performance_acceptance=False, release_qualified=False)
    with (TARGET / 'manifest.json').open('x') as file:
        json.dump(manifest, file, indent=2)
        file.write('\n')
    print(json.dumps(dict(manifest_sha256=sha(TARGET / 'manifest.json'),
        worker_sha256=manifest['worker_sha256'], qk_cases=16, pv_cases=48, native_execution=False)))


if __name__ == '__main__':
    main()
