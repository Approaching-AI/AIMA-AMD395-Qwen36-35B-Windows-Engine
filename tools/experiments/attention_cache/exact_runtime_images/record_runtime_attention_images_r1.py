"""Record binary comparability and the pending original-operand replay."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import shutil

B = Path(__file__).resolve().parent
ROOT = B.parent.parent
PUBLIC = ROOT.parent / 'AIMA-public-r1191-candidate'
D = B / 'runtime-attention-codeobjects-r1'
TRIAL = B / 'runtime-attention-images-windows-r1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def instructions(path):
    rows = []
    for line in path.read_text().splitlines():
        if re.match(r'^\s+[a-z][a-z0-9_]+\s', line) and re.search(r'// [0-9A-F]+:', line):
            rows.append(line.split('//')[0].strip())
    assert len(rows) > 100
    return rows


def main():
    comparison = read(D / 'comparison.json')
    disassembly = read(D / 'disassembly.json')
    for e in disassembly['results']:
        p = D / e['output']
        assert p.stat().st_size == e['output_bytes'] and sha(p) == e['output_sha256']
    instruction_records = {}
    for name in ('scores', 'segment'):
        counts = {}
        for source in ('observed', 'qualified_probe'):
            rows = instructions(D / (source + '-' + name + '.isa'))
            count = Counter(row.split()[0] for row in rows)
            counts[source] = dict(instructions=len(rows), opcodes=dict(sorted(count.items())),
                fp_opcode_counts={k: v for k, v in sorted(count.items()) if re.search(r'_(?:f32|f64|f16)(?:_|$)', k)},
                pc_relative_or_indirect_call_instructions=[r for r in rows if r.startswith(('s_getpc', 's_swappc', 's_call', 's_setpc'))])
        instruction_records[name] = dict(images=counts,
            fp_opcode_counts_equal=counts['observed']['fp_opcode_counts'] == counts['qualified_probe']['fp_opcode_counts'],
            numerical_equivalence_established=False)
        assert instruction_records[name]['fp_opcode_counts_equal']
    build = read(B / 'q1-cache-capture-whole-build-r1/build-provenance.json')
    probe = read(B / 'segmented-attention-native-r2/qrt-segmented-attention-build-spec.json')
    flags = ['-fno-fast-math', '-fno-reciprocal-math', '-ffp-contract=off']
    assert all(f in probe['arguments'] and f not in build['provider_compile_arguments'] for f in flags)
    plan = read(TRIAL / 'manifest.json')
    assert sha(TRIAL / 'worker.py') == plan['worker_sha256']
    archived = {}
    relative = Path('tools/experiments/attention_cache/exact_runtime_images')
    for name in ['compare_runtime_attention_code_r1.py', 'disassemble_runtime_attention_r1.py',
                 'runtime_attention_image_worker_r1.py', 'prepare_runtime_attention_images_r1.py',
                 'dispatch_runtime_attention_images_r1.py', Path(__file__).name]:
        source = B / name
        compile(source.read_bytes(), str(source), 'exec')
        target = relative / name
        for repo in (ROOT, PUBLIC):
            (repo / target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, repo / target)
        archived[target.as_posix()] = dict(bytes=source.stat().st_size, sha256=sha(source))
    report = dict(schema=1, classification='actual_binary_attention_comparability_and_pending_replay',
        controller_host='JiaweiMnideMini.lan', target_host='baiying', model_reference=plan['model_reference'],
        comparison=comparison, comparison_sha256=sha(D / 'comparison.json'),
        disassembly=disassembly, disassembly_sha256=sha(D / 'disassembly.json'),
        instruction_inspection=instruction_records,
        compiler_evidence=dict(whole_build_provenance_sha256=sha(B / 'q1-cache-capture-whole-build-r1/build-provenance.json'),
            whole_provider_arguments=build['provider_compile_arguments'],
            standalone_build_spec_sha256=sha(B / 'segmented-attention-native-r2/qrt-segmented-attention-build-spec.json'),
            standalone_arguments=probe['arguments'], additional_standalone_math_flags=flags,
            translation_unit_and_other_arguments_also_differ=True, root_cause_established=False),
        pending_native_replay=dict(manifest_sha256=sha(TRIAL / 'manifest.json'),
            worker_sha256=plan['worker_sha256'], images=plan['images'], kernels=plan['kernels'],
            tokens=plan['tokens'], prefixes=plan['prefixes'], strides=plan['strides'], repeats=plan['repeats'],
            original_reference=plan['original_reference'], expected_cases=plan['expectation'],
            host_controls=plan['host_controls'], process_timeout_seconds=plan['native_timeout_seconds'],
            exact_runtime_attention_code_and_descriptors_match_failed_prior_build=True,
            expected_context_and_segments_are_never_device_inputs=True,
            independent_gb10_score_input_is_labelled=True, dispatched=False),
        exact_reproduction_sources=archived,
        findings=[
            'Observed and prior failed DLLs have identical selected attention instruction bytes and descriptor resources.',
            'Standalone QK scores and segmented PV instruction bytes differ from the actual runtime images despite matching source headers.',
            'Append and merge instruction bytes match all three artifacts.',
            'Floating-point opcode counts match in the disassembled QK and segment functions; counts alone do not establish numerical equivalence or a cause.',
            'The prepared module replay compares actual images on qualified original operands after the full256k owner completes cleanly.'
        ],
        limitations=[
            'No extracted image has executed in this pending trial.',
            'Original current-token operands cannot establish equality of the native historical KV; that independent comparison awaits the active capture.',
            'Standalone success does not by itself qualify the actual runtime image or real-model continuation.',
            'No compiler flag, image replacement, runtime arithmetic, token oracle, numerical tolerance or release threshold changed.'
        ], native_execution=False, inference_acceptance=False, performance_acceptance=False, release_qualified=False)
    path = Path('benchmarks/correctness/layer19-runtime-attention-images-20260923.json')
    raw = json.dumps(report, indent=2) + '\n'
    for repo in (ROOT, PUBLIC):
        (repo / path).write_text(raw)
    print(json.dumps(dict(report_sha256=sha(ROOT / path), bytes=len(raw.encode()),
                         reproduction_sources=len(archived), native_execution=False)))


if __name__ == '__main__':
    main()
