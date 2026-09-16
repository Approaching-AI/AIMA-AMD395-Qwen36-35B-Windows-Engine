#!/usr/bin/env python3
"""Build isolated wave32/wave64 scalar objects and a wave32 QK fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path):
    with path.open('rb') as source:
        value = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
        return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', required=True, type=Path)
    parser.add_argument('--build', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = args.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = root / 'native/providers/ck_fmha/wave_qk_probe.cpp'
    fixture = root / 'tests/native/wave_qk_selftest.cpp'
    executable = build / 'wave-qk.exe'
    objects = [build / ('wave-qk-' + str(width) + '.obj') for width in (32, 64)]
    record_file = build / 'build-steps.json'
    for path in [executable, record_file, *objects]:
        if path.exists():
            raise RuntimeError('Build evidence already exists: ' + str(path))
    if not args.compiler.is_file() or not (build / 'hipblaslt.lib').is_file():
        raise RuntimeError('Compiler or verified hipBLASLt import library missing')
    common = [str(args.compiler), '-std=c++17', '-O3',
              '-DQRT_SM121_DPP_REDUCTION=1', '-DQRT_SM121_COMPACT_NORMALIZE=1',
              '-DQRT_CK_SM121_INTERPOLATED_EXP2=1', '--offload-arch=gfx1151']
    commands = []
    for width, obj in zip((32, 64), objects):
        flag = '-mwavefrontsize64' if width == 64 else '-mno-wavefrontsize64'
        commands.append(common + ['-Xarch_device', flag,
                                  '-DQRT_QK_WAVE_BITS=' + str(width),
                                  '-c', str(source), '-o', str(obj)])
    commands.append(common + ['-Xarch_device', '-mno-wavefrontsize64',
                              '-L', str(build), str(fixture),
                              *map(str, objects), '-o', str(executable), '-lhipblaslt'])
    record = dict(source=str(source), source_sha256=digest(source),
                  fixture=str(fixture), fixture_sha256=digest(fixture),
                  per_command_timeout_seconds=90, steps=[], completed=False)
    code = 1
    try:
        for index, command in enumerate(commands):
            begin = time.monotonic()
            result = subprocess.run(command, cwd=root, capture_output=True,
                                    text=True, errors='replace', timeout=90)
            step = dict(index=index, command=command, returncode=result.returncode,
                        wall_seconds=time.monotonic() - begin,
                        stdout=result.stdout, stderr=result.stderr)
            record['steps'].append(step)
            print(json.dumps({key: step[key] for key in
                              ('index', 'command', 'returncode', 'wall_seconds')}), flush=True)
            if result.returncode:
                print(result.stdout, result.stderr, flush=True)
                code = result.returncode
                return code
        record['artifacts'] = [dict(file=str(path), bytes=path.stat().st_size,
                                    sha256=digest(path)) for path in [*objects, executable]]
        record['completed'] = True
        code = 0
        return code
    except subprocess.TimeoutExpired as error:
        record['timeout_command'] = error.cmd
        record['failure'] = 'compiler process timeout'
        code = 124
        return code
    finally:
        record['returncode'] = code
        with record_file.open('x', encoding='utf-8') as output:
            json.dump(record, output, indent=2)
            output.write('\n')


if __name__ == '__main__':
    raise SystemExit(main())
