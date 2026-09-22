#!/usr/bin/env python3
"""Exercise the actual probe's output-only collector with CPU recording HIP calls."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    probe = ROOT/'native/linux_core_port/probe.cpp'
    source = probe.read_text()
    collector = source[source.index('class Observation {'):source.index('\nint run(')]
    prefix = r'''
#include "probe_input.h"
#include "aima/sha256.h"
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
namespace aima {
enum class DecodeTensorDtype { kNone, kBfloat16, kFloat32, kInt32 };
FULL_ATTENTION_OBSERVATION_DECLARATION
struct NativeResidentRequestOptions {
  std::optional<std::size_t> decode_layer_observer_output_index;
  std::size_t decode_linear_observer_layer_index = 0;
  std::function<void(std::size_t,const void*,std::uint64_t,const void*,std::uint64_t)> prefill_linear_state_observer;
  std::function<void(std::size_t,const void*)> decode_layer_observer;
  std::function<void(const char*,const void*,std::uint64_t,DecodeTensorDtype)> decode_linear_layer0_observer;
  decltype(decode_linear_layer0_observer) decode_layer0_tail_observer;
  std::function<void(const NativeDecodeFullAttentionObservation&)> decode_full_attention_observer;
};
}
constexpr int hipSuccess = 0;
constexpr int hipMemcpyDeviceToHost = 2;
bool fail_sync = false, fail_copy = false;
unsigned sync_calls = 0, copy_calls = 0;
int hipDeviceSynchronize() { ++sync_calls; return fail_sync ? 1 : 0; }
int hipMemcpy(void* destination, const void* source, std::size_t bytes, int kind) {
  if (kind != hipMemcpyDeviceToHost || sync_calls <= copy_calls) std::abort();
  ++copy_calls;
  if (fail_copy) return 1;
  std::memcpy(destination, source, bytes); return 0;
}
void require(bool ok) { if (!ok) throw std::runtime_error("Collector contract failed"); }
unsigned rejected = 0;
template<class F> void rejects(F callback) {
  try { callback(); } catch (const std::exception&) { ++rejected; return; }
  throw std::runtime_error("Invalid collector operation accepted");
}
'''
    full_header = (ROOT/'third_party/aima_linux/native/include/aima/native_full_layer.h').read_text()
    begin = full_header.index('struct NativeDecodeFullAttentionObservation {')
    declaration = full_header[begin:full_header.index('\n};', begin) + 3]
    prefix = prefix.replace('FULL_ATTENTION_OBSERVATION_DECLARATION', declaration)
    suffix = r'''
int main(int argc, char** argv) {
  require(argc == 2);
  const std::filesystem::path root(argv[1]);
  require(std::filesystem::create_directory(root));
  std::vector<unsigned char> device((8192 + 511) * 512 * 2);
  for (std::size_t i = 0; i < device.size(); ++i) device[i] = (i * 71 + 13 + (i >> 10) + (i >> 19)) & 255;
  const auto original = aima::sha256_bytes(device.data(), device.size());
  {
    Observation observed(root/"good", 115, 0);
    aima::NativeResidentRequestOptions request;
    observed.bind(request);
    require(!request.decode_full_attention_observer);
    require(request.decode_layer_observer_output_index == 115 && request.decode_linear_observer_layer_index == 0);
    request.prefill_linear_state_observer(1, device.data(), 32, device.data(), 64);
    require(copy_calls == 0);
    request.prefill_linear_state_observer(0, device.data(), 49152, device.data(), 2097152);
    for (std::size_t layer = 0; layer <= 40; ++layer) request.decode_layer_observer(layer, device.data());
    request.decode_linear_layer0_observer("recurrent_state_before", device.data(), 2097152, aima::DecodeTensorDtype::kFloat32);
    request.decode_layer0_tail_observer("router_indices", device.data(), 32, aima::DecodeTensorDtype::kInt32);
    require(copy_calls == 45 && sync_calls == 45);
    rejects([&]{ request.decode_layer_observer(0, device.data()); });
    rejects([&]{ request.decode_layer_observer(41, device.data()); });
    rejects([&]{ request.decode_linear_layer0_observer("bad", device.data(), 4, aima::DecodeTensorDtype::kNone); });
    rejects([&]{ request.decode_linear_layer0_observer(nullptr, device.data(), 4, aima::DecodeTensorDtype::kFloat32); });
    for (const char* bad : {"", "../outside", "/outside", "back\\slash", "UPPER"})
      rejects([&]{ observed.capture(bad, device.data(), 4, "f32"); });
    rejects([&]{ observed.capture("null", nullptr, 4, "f32"); });
    rejects([&]{ observed.capture("empty", device.data(), 0, "f32"); });
    rejects([&]{ observed.capture("huge", device.data(), (8 << 20) + 1, "f32"); });
  }
  rejects([&]{ Observation duplicate(root/"good", 1, 0); });
  {
    Observation failed(root/"copy-errors", 1, 0);
    fail_sync = true;
    const auto copies_before = copy_calls;
    rejects([&]{ failed.capture("sync", device.data(), 16, "f32"); });
    require(copy_calls == copies_before); fail_sync = false;
    fail_copy = true;
    rejects([&]{ failed.capture("copy", device.data(), 16, "f32"); });
    fail_copy = false;
    require(!std::filesystem::exists(root/"copy-errors/sync.bin"));
    require(!std::filesystem::exists(root/"copy-errors/copy.bin"));
  }
  {
    Observation bytes(root/"byte-limit", 1, 0);
    for (unsigned i = 0; i < 4; ++i) bytes.capture("payload-"+std::to_string(i), device.data(), 8 << 20, "f32");
    rejects([&]{ bytes.capture("over-budget", device.data(), 4, "f32"); });
  }
  {
    Observation files(root/"file-limit", 1, 0);
    for (unsigned i = 0; i < 128; ++i) files.capture("tiny-"+std::to_string(i), device.data(), 4, "f32");
    rejects([&]{ files.capture("over-count", device.data(), 4, "f32"); });
  }
  for (const std::size_t index : {std::size_t(1), std::size_t(511)}) {
    Observation observed(root/("full-" + std::to_string(index)), index, 0, 3);
    aima::NativeResidentRequestOptions request;
    observed.bind(request);
    require(!request.prefill_linear_state_observer && !request.decode_linear_layer0_observer &&
            !request.decode_layer0_tail_observer && bool(request.decode_full_attention_observer));
    aima::NativeDecodeFullAttentionObservation value;
    ASSIGN_FULL_ATTENTION_POINTERS
    value.layer_index = 7; value.cache_end = 8192 + index;
    const auto before = copy_calls;
    request.decode_full_attention_observer(value);
    require(copy_calls == before);
    value.layer_index = 3; value.cache_end -= 1;
    rejects([&]{ request.decode_full_attention_observer(value); });
    require(copy_calls == before);
    value.cache_end += 1; value.key_cache = nullptr;
    rejects([&]{ request.decode_full_attention_observer(value); });
    require(copy_calls == before); value.key_cache = device.data();
    request.decode_full_attention_observer(value);
    for (std::size_t layer = 0; layer <= 40; ++layer) request.decode_layer_observer(layer, device.data());
    require(copy_calls == before + 67);
  }
  for (const std::size_t layer : {std::size_t(0), std::size_t(40)})
    rejects([&]{ Observation invalid(root/"invalid-selector", 1, 0, layer); });
  for (const std::size_t index : {std::size_t(0), std::size_t(512)})
    rejects([&]{ Observation invalid(root/"invalid-selector", index, 0, 3); });
  require(aima::sha256_bytes(device.data(), device.size()) == original);
  std::cout << "{\"passed\":true,\"rejected_controls\":" << rejected
            << ",\"input_unchanged\":true,\"native_inference\":false}" << std::endl;
}
'''
    pointers = [line.strip().split()[2] for line in declaration.splitlines()
                if line.strip().startswith('const void* ')]
    assert len(pointers) == 24
    suffix = suffix.replace('ASSIGN_FULL_ATTENTION_POINTERS',
                            '\n    '.join(f'value.{name} = device.data();' for name in pointers))
    harness = out/'collector.cpp'
    harness.write_text(prefix + collector + suffix)
    compiler = shutil.which('clang++')
    if not compiler:
        raise RuntimeError('clang++ is required')
    include = ROOT/'native/linux_core_port'
    upstream = ROOT/'third_party/aima_linux/native'
    flags = ['-std=c++17','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
    commands = []
    def run(label, command):
        result = subprocess.run([str(x) for x in command], capture_output=True, text=True, timeout=90)
        (out/(label+'.stdout.txt')).write_text(result.stdout)
        (out/(label+'.stderr.txt')).write_text(result.stderr)
        commands.append(dict(label=label, command=[str(x) for x in command], exit_code=result.returncode,
                             stdout_sha256=sha(out/(label+'.stdout.txt')), stderr_sha256=sha(out/(label+'.stderr.txt'))))
        result.check_returncode()
        return result
    run('collector-build', [compiler,*flags,'-I',include,'-I',upstream/'include',harness,
                           upstream/'src/sha256.cpp','-o',out/'collector'])
    result = json.loads(run('collector-run',[out/'collector',out/'fixtures']).stdout)
    manifest = [json.loads(x) for x in (out/'fixtures/good/manifest.jsonl').read_text().splitlines()]
    assert len(manifest) == 45 and len({x['file'] for x in manifest}) == 45
    for entry in manifest:
        path = out/'fixtures/good'/entry['file']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        assert path.read_bytes() == bytes((i*71+13+(i>>10)+(i>>19))&255 for i in range(entry['bytes']))
    full_verified = 0
    for index in (1, 511):
        directory = out/f'fixtures/full-{index}'
        full = [json.loads(x) for x in (directory/'manifest.jsonl').read_text().splitlines()]
        assert len(full) == 67 and sum(x['bytes'] for x in full) <= 32 << 20
        assert len({x['file'] for x in full}) == 67
        for entry in full:
            path = directory/entry['file']
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
            assert entry['full_attention_layer'] == 3 and entry['selected_decode_output_index'] == index
            assert 'linear_layer' not in entry and 0 < entry['bytes'] <= 8 << 20
            offset = 8192*512*2 if entry['file'].endswith('-decode.bin') else 0
            expected = bytes((i*71+13+(i>>10)+(i>>19))&255 for i in range(offset, offset+entry['bytes']))
            assert path.read_bytes() == expected
        for key in ('k', 'v'):
            assert (directory/f'decode-full-cache-{key}-prefill.bin').stat().st_size == 8192*512*2
            assert (directory/f'decode-full-cache-{key}-decode.bin').stat().st_size == index*512*2
        full_verified += len(full)
    run('host-contract-build',[compiler,*flags,include/'host_contract_test.cpp','-o',out/'host-contract'])
    host = json.loads(run('host-contract-run',[out/'host-contract',out/'host-fixtures']).stdout)
    assert host['passed'] and result['passed'] and result['rejected_controls'] == 25
    report = dict(probe_sha256=sha(probe), harness_sha256=sha(harness), commands=commands,
                  host_contract=host, collector=result, verified_files=len(manifest)+full_verified,
                  generated_collector_from_actual_probe=True, gpu_executed=False,
                  inference_acceptance=False, performance_acceptance=False)
    with (out/'result.json').open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,collector_files=179,rejected_controls=25,host_contract_pass=True)))


if __name__ == '__main__':
    main()
