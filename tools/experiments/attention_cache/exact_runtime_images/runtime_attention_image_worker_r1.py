"""Replay the exact embedded HIP kernels, independently of the model runtime.

This diagnostic deliberately includes PV-only runs on captured gb10 scores.
Those rows are labelled separately from end-to-end QK/PV component runs.
"""
from pathlib import Path
import array
import ctypes as C
import hashlib
import json
import os
import platform
import socket
import struct
import subprocess
import sys
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def compare(actual, expected, dimensions):
    assert len(actual) == len(expected) and len(actual) % 4 == 0
    result = dict(fp32_values=len(actual) // 4, sha256=digest(actual),
                  expected_sha256=digest(expected), bit_exact=actual == expected,
                  bit_mismatches=0, first_differences=[])
    if actual == expected:
        return result
    a, b = memoryview(actual).cast('I'), memoryview(expected).cast('I')
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        result['bit_mismatches'] += 1
        if len(result['first_differences']) < 12:
            position, coordinates = i, []
            for dimension in reversed(dimensions):
                coordinates.append(position % dimension)
                position //= dimension
            assert position == 0
            result['first_differences'].append(dict(index=i, coordinates=list(reversed(coordinates)),
                actual_bits=x, expected_bits=y,
                actual=struct.unpack('<f', struct.pack('<I', x))[0],
                expected=struct.unpack('<f', struct.pack('<I', y))[0]))
    return result


def main():
    assert platform.system() == 'Windows' and socket.gethostname().split('.')[0].lower() == 'baiying'
    assert sys.byteorder == 'little' and C.sizeof(C.c_void_p) == 8
    root = Path(sys.argv[1])
    manifest = root / 'manifest.json'
    m = json.loads(manifest.read_text())
    assert sha(Path(__file__)) == m['worker_sha256']
    assert subprocess.check_output(['git', '-C', m['execution_checkout'], 'rev-parse', 'HEAD'],
                                   text=True, timeout=15).strip() == m['execution_checkout_commit']
    out = root / 'outputs'
    out.mkdir()
    started = time.perf_counter()
    evidence = []
    for entry in [*m['inputs'].values(), *m['tables'].values(), *m['images'].values()]:
        path = root / entry['file'] if 'file' in entry else Path(entry['path'])
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], str(path)
        evidence.append((path, entry))
    tokens = m['tokens']
    assert tokens == 263292 and m['prefixes'] == [263168, 263291]
    assert m['strides'] == [263309, 263681] and m['repeats'] == 2
    inputs = {name: (root / e['file'] if 'file' in e else Path(e['path'])).read_bytes()
              for name, e in m['inputs'].items()}
    assert len(inputs['query']) == 4096 * 2
    assert len(inputs['keys']) == len(inputs['values']) == tokens * 512 * 2
    assert len(inputs['expected_scores']) == 16 * tokens * 4
    query = array.array('H')
    query.frombytes(inputs['query'])
    rope = array.array('I', (x << 16 for x in query))
    rope.extend([0] * (9216 - 4096))
    assert rope.itemsize == 4
    expected = {
        'context': inputs['expected_output'],
        'segment_output': inputs['expected_segment_output'],
        'segment_max': inputs['expected_segment_max'],
        'segment_sum': inputs['expected_segment_sum'],
    }
    dimensions = {'context': [16, 256], 'segment_output': [16, 16, 256],
                  'segment_max': [16, 16], 'segment_sum': [16, 16]}
    dlls = list((Path(m['rocm_root']) / 'bin').glob('amdhip64*.dll'))
    assert len(dlls) == 1
    dll_handle = os.add_dll_directory(str(dlls[0].parent))
    hip = C.WinDLL(str(dlls[0]))
    void, size, uint = C.c_void_p, C.c_size_t, C.c_uint

    def bind(name, args):
        fn = getattr(hip, name)
        fn.argtypes, fn.restype = args, C.c_int
        return fn

    def check(status, label):
        if status:
            raise RuntimeError(f'{label}: HIP {status}')

    init = bind('hipInit', [uint])
    setdev = bind('hipSetDevice', [C.c_int])
    getname = bind('hipDeviceGetName', [void, C.c_int, C.c_int])
    malloc = bind('hipMalloc', [C.POINTER(void), size])
    free = bind('hipFree', [void])
    copy = bind('hipMemcpy', [void, void, size, C.c_int])
    memset = bind('hipMemset', [void, C.c_int, size])
    sync = bind('hipDeviceSynchronize', [])
    load = bind('hipModuleLoadData', [C.POINTER(void), void])
    getfn = bind('hipModuleGetFunction', [C.POINTER(void), void, C.c_char_p])
    unload = bind('hipModuleUnload', [void])
    launch = bind('hipModuleLaunchKernel', [void, uint, uint, uint, uint, uint, uint, uint,
                                          void, C.POINTER(void), void])
    create_stream = bind('hipStreamCreateWithFlags', [C.POINTER(void), uint])
    destroy_stream = bind('hipStreamDestroy', [void])
    stream_sync = bind('hipStreamSynchronize', [void])
    check(init(0), 'init')
    check(setdev(0), 'device')
    device = C.create_string_buffer(256)
    check(getname(device, 256, 0), 'device name')
    stream = void()
    check(create_stream(C.byref(stream), 1), 'nonblocking stream')
    buffers, bases, sizes = {}, {}, {}
    modules, image_storage, functions = [], [], {}
    immutable = {}
    guard, peak = 512, 0
    qk_records, pv_records = [], []

    def download(pointer, count):
        host = C.create_string_buffer(count)
        check(copy(C.cast(host, void), pointer, count, 2), 'download')
        return host.raw

    def upload(pointer, raw):
        host = C.create_string_buffer(raw)
        check(copy(pointer, C.cast(host, void), len(raw), 1), 'upload')

    def allocate(name, count, raw=None, readonly=False):
        nonlocal peak
        assert name not in bases and count > 0
        base = void()
        check(malloc(C.byref(base), count + 2 * guard), 'allocate ' + name)
        bases[name], sizes[name] = base, count
        buffers[name] = void(base.value + guard)
        check(memset(base, 0xa5, count + 2 * guard), 'initialize ' + name)
        if raw is not None:
            assert len(raw) == count
            upload(buffers[name], raw)
            if readonly:
                immutable[name] = digest(raw)
        peak = max(peak, sum(sizes.values()) + len(sizes) * 2 * guard)

    def guards():
        return all(download(base, guard) == b'\xa5' * guard and
                   download(void(base.value + guard + sizes[name]), guard) == b'\xa5' * guard
                   for name, base in bases.items())

    def content(name):
        return download(buffers[name], sizes[name])

    def release(name):
        check(free(bases.pop(name)), 'free ' + name)
        buffers.pop(name)
        sizes.pop(name)
        immutable.pop(name, None)

    def execute(image, name, values, grid):
        # Each kernelParams element points to one typed argument. HSA metadata
        # determines the 8-byte alignment of segment's final pointer at offset64.
        signature = m['kernels'][name]['argument_types']
        assert len(values) == len(signature)
        args = [void(value.value if isinstance(value, void) else value) if kind == 'pointer'
                else uint(value) for kind, value in zip(signature, values)]
        argv = (void * len(args))(*(C.addressof(value) for value in args))
        check(launch(functions[image][name], *grid, 256, 1, 1, 0, stream, argv, None),
              'launch ' + image + '/' + name)

    def timed(fn):
        check(stream_sync(stream), 'before timing')
        t = time.perf_counter()
        fn()
        check(stream_sync(stream), 'after timing')
        return (time.perf_counter() - t) * 1000

    def save_mismatch(label, raw, comparison):
        if comparison['bit_exact']:
            return
        path = out / (label + '.bin')
        with path.open('xb') as file:
            file.write(raw)

    all_guards = True
    all_immutable = True
    cleanup_errors = []
    try:
        for name, entry in m['images'].items():
            raw = C.create_string_buffer((root / entry['file']).read_bytes())
            image_storage.append(raw)
            module = void()
            check(load(C.byref(module), C.cast(raw, void)), 'load ' + name)
            modules.append(module)
            functions[name] = {}
            for kernel, spec in m['kernels'].items():
                fn = void()
                check(getfn(C.byref(fn), module, spec['symbol'].encode()), 'symbol ' + kernel)
                functions[name][kernel] = fn
        for name, entry in m['tables'].items():
            allocate(name, entry['bytes'], Path(entry['path']).read_bytes(), readonly=True)
        allocate('rope', len(rope) * 4, rope.tobytes(), readonly=True)
        for name, raw in expected.items():
            allocate(name, len(raw))
        for prefix in m['prefixes']:
            split = prefix * 512 * 2
            for name in ('keys', 'values'):
                allocate('prefix_' + name, split, inputs[name][:split], readonly=True)
                allocate('tail_' + name, len(inputs[name]) - split, inputs[name][split:], readonly=True)
            for stride in m['strides']:
                allocate('scores', 16 * stride * 4)
                padding = struct.pack('<I', 0xff800000) * (stride - tokens)
                padded_expected = b''.join(inputs['expected_scores'][h*tokens*4:(h+1)*tokens*4] + padding
                                           for h in range(16))
                for repeat in range(m['repeats']):
                    # Swap order on the second pass to detect order-sensitive failures.
                    producers = ['qualified_probe', 'observed'] if repeat == 0 else ['observed', 'qualified_probe']
                    for producer in ['gb10_captured_scores', *producers]:
                        if producer == 'gb10_captured_scores':
                            upload(buffers['scores'], padded_expected)
                        else:
                            check(memset(buffers['scores'], 0xa5, sizes['scores']), 'poison scores')
                            elapsed = timed(lambda: execute(producer, 'scores', [buffers['rope'],
                                buffers['prefix_keys'], buffers['tail_keys'], buffers['scores'],
                                prefix, tokens, stride], [stride, 1, 1]))
                            padded = content('scores')
                            raw = b''.join(padded[h*stride*4:(h*stride+tokens)*4] for h in range(16))
                            comparison = compare(raw, inputs['expected_scores'], [16, tokens])
                            row = dict(prefix=prefix, stride=stride, repeat=repeat, producer=producer,
                                host_synchronized_ms=elapsed, scores=comparison,
                                padding_negative_infinity=all(padded[(h*stride+tokens)*4:(h+1)*stride*4] == padding
                                                              for h in range(16)))
                            qk_records.append(row)
                            save_mismatch(f'p{prefix}-s{stride}-r{repeat}-{producer}-scores', raw, comparison)
                            print(json.dumps(dict(stage='qk', **row)), flush=True)
                        score_hash = digest(content('scores'))
                        for consumer in producers:
                            for name in expected:
                                check(memset(buffers[name], 0xa5, sizes[name]), 'poison ' + name)

                            def pv():
                                execute(consumer, 'segment', [buffers['scores'], buffers['prefix_values'],
                                    buffers['tail_values'], buffers['segment_output'], buffers['segment_max'],
                                    buffers['segment_sum'], prefix, tokens, stride, buffers['exp2']], [16, 16, 1])
                                execute(consumer, 'merge', [buffers['segment_output'], buffers['segment_max'],
                                    buffers['segment_sum'], buffers['context'], buffers['exp2'], buffers['rcp']], [16, 1, 1])

                            elapsed = timed(pv)
                            comparisons = {}
                            for name, oracle in expected.items():
                                raw = content(name)
                                comparisons[name] = compare(raw, oracle, dimensions[name])
                                save_mismatch(f'p{prefix}-s{stride}-r{repeat}-{producer}-{consumer}-{name}',
                                              raw, comparisons[name])
                            intact = digest(content('scores')) == score_hash
                            guarded = guards()
                            all_guards = all_guards and guarded
                            row = dict(prefix=prefix, stride=stride, repeat=repeat, score_producer=producer,
                                pv_image=consumer, oracle_scores_used_as_input=producer == 'gb10_captured_scores',
                                comparisons=comparisons, host_synchronized_ms=elapsed,
                                score_input_unchanged=intact, guards_pass=guarded)
                            pv_records.append(row)
                            print(json.dumps(dict(stage='pv', **row)), flush=True)
                release('scores')
            for name in ('prefix_keys', 'tail_keys', 'prefix_values', 'tail_values'):
                all_immutable = all_immutable and digest(content(name)) == immutable[name]
                release(name)
        all_immutable = all_immutable and all(digest(content(name)) == value for name, value in immutable.items())
        all_guards = all_guards and guards()
    finally:
        status = sync()
        if status:
            cleanup_errors.append(dict(operation='device_sync', status=status))
        for module in reversed(modules):
            status = unload(module)
            if status:
                cleanup_errors.append(dict(operation='module_unload', status=status))
        for base in bases.values():
            status = free(base)
            if status:
                cleanup_errors.append(dict(operation='free', status=status))
        status = destroy_stream(stream)
        if status:
            cleanup_errors.append(dict(operation='stream_destroy', status=status))
        dll_handle.close()
    source_unchanged = all(path.stat().st_size == e['bytes'] and sha(path) == e['sha256'] for path, e in evidence)
    source_unchanged = source_unchanged and sha(Path(__file__)) == m['worker_sha256']
    passed = len(qk_records) == 16 and len(pv_records) == 48 and all_guards and all_immutable and source_unchanged
    passed = passed and not cleanup_errors and all(r['scores']['bit_exact'] and r['padding_negative_infinity'] for r in qk_records)
    passed = passed and all(r['score_input_unchanged'] and all(v['bit_exact'] for v in r['comparisons'].values()) for r in pv_records)
    report = dict(host=socket.gethostname(), device=device.value.decode(), command=[sys.executable, *sys.argv],
        execution_checkout_commit=m['execution_checkout_commit'], model_reference=m['model_reference'],
        manifest_sha256=sha(manifest), worker_sha256=sha(Path(__file__)),
        hip_dll=dict(path=str(dlls[0]), bytes=dlls[0].stat().st_size, sha256=sha(dlls[0])),
        images=m['images'], kernels=m['kernels'], original_reference=m['original_reference'],
        qk_records=qk_records, pv_records=pv_records, all_components_match=bool(passed),
        inputs_and_tables_unchanged=all_immutable, sources_unchanged=source_unchanged,
        guards_pass=all_guards, cleanup_pass=not cleanup_errors, cleanup_errors=cleanup_errors,
        peak_device_allocation_bytes=peak, wall_seconds=time.perf_counter()-started,
        original_model_loaded=False, inference_acceptance=False, performance_acceptance=False,
        release_qualified=False)
    with (out / 'result.json').open('x') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps(dict(all_components_match=bool(passed), qk_cases=len(qk_records), pv_cases=len(pv_records))), flush=True)
    return 0 if passed else 6


if __name__ == '__main__':
    sys.exit(main())
