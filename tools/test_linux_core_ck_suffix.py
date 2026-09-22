#!/usr/bin/env python3
"""Exercise the generated CK loader against CPU-only ABI recording libraries."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

from prepare_linux_core_windows import make_overlays, verify_import

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "0a57516e8c7c1dba55a077eba9d38b6155a0620e"

RECORD_HEADER = r"""
#pragma once
#include <cstdint>
struct Record {
  unsigned prepares, releases, calls, kind, context, prefix, queries, kv;
  std::uintptr_t q, k, v, suffix_k, suffix_v, output, stream;
};
"""

MOCK_LIBRARY = r"""
#include "record.h"
#include "ck_suffix_adapter.h"
namespace { Record value{}; int failure = 0; }
extern "C" {
const Record* test_record() { return &value; }
void test_failure(int status) { failure = status; }
int observe(unsigned kind, const void* q, const void* k, const void* v,
            void* output, void* stream, unsigned queries, unsigned kv) {
  ++value.calls; value.kind = kind;
  value.q = reinterpret_cast<std::uintptr_t>(q);
  value.k = reinterpret_cast<std::uintptr_t>(k);
  value.v = reinterpret_cast<std::uintptr_t>(v);
  value.output = reinterpret_cast<std::uintptr_t>(output);
  value.stream = reinterpret_cast<std::uintptr_t>(stream);
  value.queries = queries; value.kv = kv;
  value.prefix = value.suffix_k = value.suffix_v = 0;
  return failure;
}
int qrt_ck_fmha_q8192_prepare() { ++value.prepares; return 0; }
int qrt_ck_fmha_q8192_release() { ++value.releases; return 0; }
int qrt_ck_fmha_q8192_bf16_launch(const void* q, const void* k, const void* v,
                                void* output, void* stream) {
  return observe(1, q, k, v, output, stream, 8192, 8192);
}
#ifndef OMIT_DYNAMIC
int qrt_ck_fmha_dynamic_bf16_launch(const void* q, const void* k, const void* v,
                                   void* output, void* stream, unsigned queries) {
  return observe(2, q, k, v, output, stream, queries, queries);
}
#endif
#ifndef OMIT_SUFFIX
int qrt_ck_fmha_sm121_suffix_bf16_v1(
    const std::uint16_t* q, const std::uint16_t* k, const std::uint16_t* v,
    const std::uint16_t* sk, const std::uint16_t* sv, float* output,
    void* stream, unsigned prefix, unsigned queries) {
  const int status = observe(3, q, k, v, output, stream, queries, prefix + queries);
  value.prefix = prefix;
  value.suffix_k = reinterpret_cast<std::uintptr_t>(sk);
  value.suffix_v = reinterpret_cast<std::uintptr_t>(sv);
  return status;
}
#endif
#ifdef GENERIC
int qrt_ck_fmha_prepare(unsigned context) {
  ++value.prepares; value.context = context; return 0;
}
int qrt_ck_fmha_release() { ++value.releases; return 0; }
int qrt_ck_fmha_bf16_launch_ex(const void* q, const void* k, const void* v,
                              void* output, unsigned queries, unsigned kv,
                              void* stream) {
  return observe(4, q, k, v, output, stream, queries, kv);
}
int qrt_ck_fmha_bf16_launch(const void* q, const void* k, const void* v,
                           void* output, unsigned queries, void* stream) {
  return observe(5, q, k, v, output, stream, queries, queries);
}
#endif
}
"""

MAIN = r"""
#include "record.h"
#include <iostream>
#include <utility>

namespace {
unsigned rejected = 0, checked = 0;
void require(bool ok) {
  if (!ok) throw std::runtime_error("CK adapter contract failed");
  ++checked;
}
template <class Fn> void rejects(Fn fn) {
  try { fn(); }
  catch (const std::exception&) { ++rejected; return; }
  throw std::runtime_error("Invalid CK call was accepted");
}
void* address(std::uintptr_t value) { return reinterpret_cast<void*>(value); }
struct Library {
  void* handle;
  const Record* (*record)();
  void (*failure)(int);
  explicit Library(const char* path) : handle(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
    require(handle != nullptr);
    record = reinterpret_cast<decltype(record)>(dlsym(handle, "test_record"));
    failure = reinterpret_cast<decltype(failure)>(dlsym(handle, "test_failure"));
    require(record && failure);
  }
  ~Library() { dlclose(handle); }
};
}

int main(int argc, char** argv) {
  try {
    require(argc == 5 && sizeof(std::uintptr_t) == 8);
    // Opaque addresses are never read or written by these recording libraries.
    void* q = address(0x100000000ULL);
    void* k = address(0x200000000ULL);
    void* v = address(0x300000000ULL);
    void* output = address(0x400000000ULL);
    void* stream = address(0x500000000ULL);
    const auto base_k = reinterpret_cast<std::uintptr_t>(k);
    const auto base_v = reinterpret_cast<std::uintptr_t>(v);
    {
      Library library(argv[1]);
      aima::NativeQ8192CkProvider provider;
      const auto loaded = provider.load(argv[1], 32768);
      require(!loaded.generic_context_abi && loaded.rectangular_context_abi);
      provider.launch(q, k, v, output, 8192, 8192, stream);
      require(library.record()->kind == 1);
      provider.launch(q, k, v, output, 7168, 7168, stream);
      require(library.record()->kind == 2);
      provider.launch(q, k, v, output, 32768, 32768, stream);
      require(library.record()->kind == 2 && library.record()->queries == 32768);
      const std::pair<unsigned, unsigned> shapes[] = {
          {1, 2}, {1, 7169}, {1024, 17408}, {8192, 16384},
          {8192, 262144}, {1, 262144}, {1024, 17408}};
      for (auto shape : shapes) {
        provider.launch(q, k, v, output, shape.first, shape.second, stream);
        const auto& r = *library.record();
        const unsigned prefix = shape.second - shape.first;
        require(r.kind == 3 && r.prefix == prefix && r.queries == shape.first &&
                r.kv == shape.second && r.q == reinterpret_cast<std::uintptr_t>(q) &&
                r.k == base_k && r.v == base_v &&
                r.suffix_k == base_k + std::uint64_t(prefix) * 1024 &&
                r.suffix_v == base_v + std::uint64_t(prefix) * 1024 &&
                r.output == reinterpret_cast<std::uintptr_t>(output) &&
                r.stream == reinterpret_cast<std::uintptr_t>(stream));
      }
      auto calls = library.record()->calls;
      auto launches = provider.metrics().launches;
      rejects([&] { provider.launch(q, k, v, output, 8193, 16385, stream); });
      rejects([&] { provider.launch(q, k, v, output, 1024, 263168, stream); });
      rejects([&] { provider.launch(q, k, v, output, 8192, 8191, stream); });
      rejects([&] { provider.launch(q, k, v, q, 1024, 17408, stream); });
      require(library.record()->calls == calls && provider.metrics().launches == launches);
      library.failure(37);
      rejects([&] { provider.launch(q, k, v, output, 1024, 17408, stream); });
      require(library.record()->calls == calls + 1 && provider.metrics().launches == launches);
      library.failure(0);
      provider.reset(); provider.reset();
      require(library.record()->releases == 1 && !provider.loaded() &&
              !provider.metrics().rectangular_context_abi);
      rejects([&] { provider.launch(q, k, v, output, 1024, 17408, stream); });
    }
    {
      Library library(argv[2]);
      aima::NativeQ8192CkProvider provider;
      require(!provider.load(argv[2], 32768).rectangular_context_abi);
      rejects([&] { provider.launch(q, k, v, output, 1024, 17408, stream); });
      require(library.record()->calls == 0);
      provider.launch(q, k, v, output, 32768, 32768, stream);
      require(library.record()->kind == 2);
    }
    {
      Library library(argv[3]);
      aima::NativeQ8192CkProvider provider;
      rejects([&] { provider.load(argv[3], 4096); });
      require(!provider.loaded() && library.record()->calls == 0);
      require(provider.load(argv[3], 8192).rectangular_context_abi);
      provider.launch(q, k, v, output, 8192, 8192, stream);
      require(library.record()->kind == 1);
      rejects([&] { provider.launch(q, k, v, output, 4096, 4096, stream); });
      provider.launch(q, k, v, output, 1024, 17408, stream);
      require(library.record()->kind == 3);
    }
    {
      Library library(argv[4]);
      aima::NativeQ8192CkProvider provider;
      const auto loaded = provider.load(argv[4], 32768);
      require(loaded.generic_context_abi && loaded.rectangular_context_abi &&
              library.record()->context == 32768);
      provider.launch(q, k, v, output, 16384, 32768, stream);
      require(library.record()->kind == 4 && library.record()->queries == 16384 &&
              library.record()->kv == 32768);
      provider.launch(q, k, v, output, 8192, 8192, stream);
      require(library.record()->kind == 5);
    }
    for (const auto shape : {std::pair<std::size_t, std::size_t>{0, 8192},
                             {8193, 16385}, {8192, 8192}, {8192, 8191},
                             {1024, 263168}, {1, std::size_t(-1)}}) {
      rejects([&] { aima_port::ck_suffix_views(q, k, v, output, shape.first, shape.second); });
    }
    const auto maximum = std::numeric_limits<std::uintptr_t>::max();
    for (unsigned index = 0; index < 4; ++index) {
      for (void* invalid : {static_cast<void*>(nullptr), address(3), address(maximum - 3)}) {
        void* pointers[] = {q, k, v, output}; pointers[index] = invalid;
        rejects([&] { aima_port::ck_suffix_views(pointers[0], pointers[1],
                                                pointers[2], pointers[3], 1024, 17408); });
      }
    }
    for (void* alias : {q, k, v, address(base_k + 17408ULL * 1024 - 4)}) {
      rejects([&] { aima_port::ck_suffix_views(q, k, v, alias, 1024, 17408); });
    }
    const auto adjacent = aima_port::ck_suffix_views(
        q, k, k, address(base_k + 17408ULL * 1024), 1024, 17408);
    require(adjacent.prefix_k == adjacent.prefix_v &&
            adjacent.suffix_k == adjacent.suffix_v && adjacent.prefix_tokens == 16384);
    std::cout << "{\"checked\":" << checked << ",\"rejections\":" << rejected
              << ",\"gpu_executed\":false,\"inference_acceptance\":false}" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl; return 1;
  }
}
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()
    if sys.platform == "win32":
        raise SystemExit("This POSIX dynamic-loader test does not qualify the Win32 loader")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    verify_import()
    frozen = subprocess.check_output(
        ["git", "show", BASELINE + ":tools/prepare_linux_core_windows.py"],
        cwd=ROOT, timeout=20)
    namespace = {"__file__": str(ROOT / "tools/prepare_linux_core_windows.py"),
                 "__name__": "frozen_linux_core_preparer"}
    exec(compile(frozen, "frozen_preparer.py", "exec"), namespace)
    baseline = namespace["make_overlays"]()
    if baseline != make_overlays():
        raise ValueError("Default overlays changed from the queued experiment")
    observer = ROOT / "tools/check_linux_core_q8192.py"
    if observer.read_bytes() != subprocess.check_output(
            ["git", "show", BASELINE + ":tools/check_linux_core_q8192.py"],
            cwd=ROOT, timeout=20):
        raise ValueError("Queued observer changed")
    stages = []

    def run(label, command, timeout=90):
        result = subprocess.run([str(x) for x in command], cwd=ROOT,
                                capture_output=True, text=True, timeout=timeout)
        (out / (label + ".json")).write_text(json.dumps(dict(command=list(map(str, command)),
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), indent=2) + "\n")
        result.check_returncode()
        stages.append(label)
        return result

    prepared = out / "prepared"
    run("prepare", [sys.executable, ROOT / "tools/prepare_linux_core_windows.py",
                    "--out", prepared, "--windows-rectangular-ck"])
    header = (prepared / "overlay/native/include/aima/native_full_prefill.h").read_text()
    source = (prepared / "overlay/native/src/native_full_prefill.hip.cpp").read_text()
    declaration = header[header.index("struct NativeQ8192CkProviderMetrics {"):
                         header.index("struct NativeFullPrefillMetrics {")]
    methods = source[source.index("NativeQ8192CkProvider::~NativeQ8192CkProvider()"):
                     source.index("NativeFullPrefillOracleResult probe_native_q8192_full_prefill_oracle(")]
    translation = ('#include "ck_suffix_adapter.h"\n#include <filesystem>\n'
                   '#include <string>\n#include <dlfcn.h>\n'
                   'constexpr int hipSuccess = 0, hipErrorInvalidValue = 1;\n'
                   'namespace aima {\n' + declaration + methods + '}\n' + MAIN)
    (out / "provider_contract.cpp").write_text(translation)
    (out / "record.h").write_text(RECORD_HEADER)
    (out / "recording_library.cpp").write_text(MOCK_LIBRARY)
    common = [args.cxx, "-std=c++17", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
              "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
              "-I", ROOT / "native/linux_core_port", "-I", out]
    shared = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
    libraries = []
    for name, defines in (("suffix", []), ("no-suffix", ["-DOMIT_SUFFIX"]),
                          ("no-dynamic", ["-DOMIT_DYNAMIC"]), ("generic", ["-DGENERIC"])):
        target = out / (name + (".dylib" if sys.platform == "darwin" else ".so"))
        run("build-" + name, common + shared + defines + [out / "recording_library.cpp", "-o", target])
        libraries.append(target)
    binary = out / "provider-contract"
    run("build-contract", common + [out / "provider_contract.cpp", "-o", binary] +
        ([] if sys.platform == "darwin" else ["-ldl"]))
    contract = json.loads(run("contract", [binary, *libraries], 30).stdout)
    report = dict(schema=1, baseline_source_commit=BASELINE,
        default_overlay_count=len(baseline), default_overlays_byte_identical=True,
        queued_observer_byte_identical=True, queued_observer_sha256=sha(observer.read_bytes()),
        adapter_sha256=sha((ROOT / "native/linux_core_port/ck_suffix_adapter.h").read_bytes()),
        generated_provider_declaration_sha256=sha(declaration.encode()),
        generated_provider_methods_sha256=sha(methods.encode()),
        stages=stages, contract=contract,
        recording_library_variants=len(libraries), numerical_comparison=False,
        windows_build_qualified=False, model_correctness_qualified=False,
        performance_acceptance=False)
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
