#!/usr/bin/env python3
"""Generate auditable Windows overlays and COFF inputs for the pinned core.

This prepares a standalone q8192 text experiment, not a replacement runtime.
The imported tree is immutable. Only generated files receive OS/ABI changes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/aima_linux"
INVENTORY_SHA256 = "00737aa8d71b9ade06dd663171cf2e02f45e7de514ce79bfd870cafea97345c2"
CONTEXTS = (1024, 2048, 4096, 7168, 7680, 8191, 8192, 16384, 32768)
EXTRA_AOT = (
    "q1024-text-v151", "vl-unified-attention-v0.1.0",
    "unified-attention-decode-v0.1.0", "vl-recompute-w-u-q131-v0.1.0",
    "packed-linear-decode-v0.1.0", "causal-conv-decode-v0.1.0",
    "linear-gated-norm-decode-v0.1.0", "routed-moe-decode-v0.1.0",
    "routed-moe-exact-hybrid-v0.1.0", "vision-attention-v0.2.0",
)
# Same computation units as build-native-runtime.sh. Media transports,
# tokenizer and server frontends are outside this pretokenized experiment.
EXCLUDED = {
    "main.cpp", "aot_kernel_probe.hip.cpp", "aot_registry_probe.hip.cpp",
    "decode_schedule_probe.cpp", "native_aotriton_fmha_provider.hip.cpp",
    "native_chat_protocol.cpp", "native_doctor.cpp", "native_http_server.cpp",
    "native_http_support.cpp", "native_image_decoder.cpp", "native_media.cpp",
    "native_q16384_hybrid_fmha_provider.hip.cpp", "native_remote_media.cpp",
    "native_tokenizer.cpp", "native_video_decoder.cpp", "native_vl_request.cpp",
    "native_vision_block_stack.hip.cpp",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f"Upstream adaptation anchor changed: {old[:90]!r}")
    return text.replace(old, new)


def replace_re(text, pattern, replacement, count):
    result, changed = re.subn(pattern, lambda _: replacement, text)
    if changed != count:
        raise ValueError(f"Upstream adaptation count changed: {pattern!r}: {changed}")
    return result


def loader_overlay(text):
    for header in ("fcntl.h", "sys/stat.h", "unistd.h"):
        text = replace(text, f"#include <{header}>\n", "")
    text = replace(text, "#include <hip/hip_runtime.h>",
                   '#include <hip/hip_runtime.h>\n#include "file_io.h"')
    start = text.index("ssize_t pread_retry(")
    end = text.index("\n}\n", start) + 3
    text = replace(text, text[start:end],
        "std::int64_t pread_retry(aima_port::file_handle descriptor, void* buffer,\n"
        "                          std::size_t bytes, std::uint64_t offset) {\n"
        "  return aima_port::read_file_at(descriptor, buffer, bytes, offset);\n}")
    text = replace(text,
        "      struct stat status {};\n"
        "      if (stat(paths.back().c_str(), &status) != 0 ||\n"
        "          static_cast<std::uint64_t>(status.st_size) != shard_bytes[shard]) {",
        "      if (!aima_port::file_size_matches(paths.back().c_str(), shard_bytes[shard])) {")
    text = replace(text, "int descriptor = -1;",
                   "aima_port::file_handle descriptor = aima_port::invalid_file;")
    text = replace_re(text, r"open\(paths\[shard\]\.c_str\(\),\s*O_RDONLY \| O_DIRECT \| O_CLOEXEC\)",
                      "aima_port::open_file(paths[shard].c_str(), true)", 1)
    text = replace_re(text, r"open\(paths\[shard\]\.c_str\(\),\s*O_RDONLY \| O_CLOEXEC\)",
                      "aima_port::open_file(paths[shard].c_str(), false)", 3)
    text = replace(text, "descriptor < 0", "descriptor == aima_port::invalid_file", 4)
    text = replace(text, "descriptor >= 0", "descriptor != aima_port::invalid_file")
    text = replace(text, "descriptor = -1;", "descriptor = aima_port::invalid_file;")
    text = replace(text, "close(descriptor)", "aima_port::close_file(descriptor)", 4)
    text = replace(text, "ssize_t amount", "std::int64_t amount")
    text = replace(text,
        "(void)posix_fadvise(descriptor, static_cast<off_t>(offset),\n"
        "                                    static_cast<off_t>(valid_bytes),\n"
        "                                    POSIX_FADV_DONTNEED);",
        "aima_port::drop_file_cache(descriptor, offset, valid_bytes);")
    return replace(text, "std::ofstream output(output_path);",
                   "std::ofstream output(std::filesystem::u8path(output_path));", 2)


def full_prefill_overlay(text):
    text = replace(text,
        "      if (context_tokens != 8192) {",
        '      dynamic_launch_ = reinterpret_cast<DynamicLaunchFn>(\n'
        '          dlsym(handle_, "qrt_ck_fmha_dynamic_bf16_launch"));\n'
        "      if (context_tokens != 8192 &&\n"
        "          (context_tokens > 8192 || dynamic_launch_ == nullptr)) {")
    text = replace(text,
        "  } else {\n    if (query_tokens != metrics_.context_tokens) {",
        "  } else if (query_tokens != 8192 && dynamic_launch_ != nullptr) {\n"
        "    status = dynamic_launch_(q_bf16, k_bf16, v_bf16, output_f32, stream,\n"
        "                             static_cast<unsigned int>(query_tokens));\n"
        "  } else {\n    if (query_tokens != 8192) {")
    return replace(text, "  legacy_launch_ = nullptr;",
                   "  legacy_launch_ = nullptr;\n  dynamic_launch_ = nullptr;")


def rectangular_ck_overlay(header, source):
    """Opt-in mapping of contiguous Linux KV to the existing Windows ABI."""
    header = replace(header, '#include "aima/native_decode_executor.h"',
                     '#include "aima/native_decode_executor.h"\n#include "ck_suffix_adapter.h"')
    header = replace(header, "  DynamicLaunchFn dynamic_launch_ = nullptr;",
                     "  DynamicLaunchFn dynamic_launch_ = nullptr;\n"
                     "  aima_port::CkSuffixLaunch suffix_launch_ = nullptr;")
    source = replace(source,
        '          dlsym(handle_, "qrt_ck_fmha_dynamic_bf16_launch"));\n'
        "      if (context_tokens != 8192 &&\n"
        "          (context_tokens > 8192 || dynamic_launch_ == nullptr)) {",
        '          dlsym(handle_, "qrt_ck_fmha_dynamic_bf16_launch"));\n'
        "      suffix_launch_ = reinterpret_cast<aima_port::CkSuffixLaunch>(\n"
        '          dlsym(handle_, "qrt_ck_fmha_sm121_suffix_bf16_v1"));\n'
        "      metrics_.rectangular_context_abi = suffix_launch_ != nullptr;\n"
        "      if (context_tokens != 8192 && dynamic_launch_ == nullptr) {")
    source = replace(source,
        "    if (rectangular_launch_ == nullptr) {\n"
        "      throw std::runtime_error(\n"
        '          "native FMHA provider lacks the rectangular context ABI");\n'
        "    }\n"
        "    status = rectangular_launch_(\n"
        "        q_bf16, k_bf16, v_bf16, output_f32,\n"
        "        static_cast<unsigned int>(query_tokens),\n"
        "        static_cast<unsigned int>(kv_tokens), stream);",
        "    if (rectangular_launch_ != nullptr) {\n"
        "      status = rectangular_launch_(\n"
        "          q_bf16, k_bf16, v_bf16, output_f32,\n"
        "          static_cast<unsigned int>(query_tokens),\n"
        "          static_cast<unsigned int>(kv_tokens), stream);\n"
        "    } else if (suffix_launch_ != nullptr) {\n"
        "      const auto views = aima_port::ck_suffix_views(\n"
        "          q_bf16, k_bf16, v_bf16, output_f32, query_tokens, kv_tokens);\n"
        "      status = suffix_launch_(views.q, views.prefix_k, views.prefix_v,\n"
        "          views.suffix_k, views.suffix_v, views.output, stream,\n"
        "          views.prefix_tokens, views.query_tokens);\n"
        "    } else {\n"
        "      throw std::runtime_error(\n"
        '          "native FMHA provider lacks the rectangular context ABI");\n'
        "    }")
    source = replace(source, "  dynamic_launch_ = nullptr;",
                     "  dynamic_launch_ = nullptr;\n  suffix_launch_ = nullptr;")
    return header, source


def current_text_decode_overlay(text):
    """Reuse current upstream decode arithmetic with ordinary text positions.

    The upstream boolean named use_mrope also selects projection, recurrence,
    normalization and MoE arithmetic. Its current path uses in-place state;
    selecting only some of these call sites would leave incompatible swaps.
    No M-RoPE plan or visual request is synthesized, and prefill is unchanged.
    """
    text = replace(text,
        "    const NativeDecodePrepareMetrics prepared =\n"
        "        mrope_plan != nullptr\n"
        "            ? prepare_native_decode_step(\n"
        "                  position, rotary_position, metrics.output_token_ids.back(),\n"
        "                  impl_->weights, impl_->decode_invocations)\n"
        "            : prepare_native_decode_step(\n"
        "                  position, metrics.output_token_ids.back(), impl_->weights,\n"
        "                  impl_->decode_invocations);",
        "    // Windows current-text experiment: ordinary text keeps\n"
        "    // rotary_position == position, with the current BF16 RoPE cache.\n"
        "    const NativeDecodePrepareMetrics prepared =\n"
        "        prepare_native_decode_step(\n"
        "            position, rotary_position, metrics.output_token_ids.back(),\n"
        "            impl_->weights, impl_->decode_invocations);")
    return replace(text,
        "        mrope_plan != nullptr, impl_->decode_shared_gate_plan.get(),\n"
        "        mrope_plan != nullptr ? &impl_->decode_cross_layer_norms : nullptr);",
        "        true, impl_->decode_shared_gate_plan.get(),\n"
        "        &impl_->decode_cross_layer_norms);")


def make_overlays(*, rectangular_ck=False, current_text_decode=False):
    read = lambda p: (UPSTREAM / p).read_text(encoding="utf-8")
    sources = {}
    loader = "benchmarks/shape-lab/native/src/torch_owned_safetensors_loader.hip.cpp"
    sources[loader] = loader_overlay(read(loader))
    header = "native/include/aima/native_resident_engine.h"
    sources[header] = replace(read(header), "  bool first_token_certified = false;",
        "  // Output-only observation of the certified greedy BF16 logit.\n"
        "  float first_token_raw_logit = 0.0f;\n  bool first_token_certified = false;")
    engine = "native/src/native_resident_engine.hip.cpp"
    sources[engine] = replace(read(engine),
        "  metrics.first_token_certified = first.certified;",
        "  metrics.first_token_raw_logit = first.top1_logit;\n"
        "  metrics.first_token_certified = first.certified;")
    sources[engine] = replace(sources[engine],
        "        default_fmha_provider(4096), 4096);",
        "        options.ck_provider.empty() ? default_fmha_provider(4096)\n"
        "                                    : options.ck_provider, 4096);")
    if current_text_decode:
        sources[engine] = current_text_decode_overlay(sources[engine])
    weights = "native/src/native_weight_store.hip.cpp"
    sources[weights] = replace(read(weights), "shard_storage.push_back(path.string());",
                             "shard_storage.push_back(path.u8string());")
    sources[weights] = replace(sources[weights], "options.native_report.c_str());",
                             "options.native_report.u8string().c_str());")
    header = "native/include/aima/native_full_prefill.h"
    sources[header] = replace(read(header), "  using ReleaseFn = int (*)();",
        "  using DynamicLaunchFn = int (*)(const void*, const void*, const void*,\n"
        "                                    void*, void*, unsigned int);\n"
        "  using ReleaseFn = int (*)();")
    sources[header] = replace(sources[header], "  LegacyLaunchFn legacy_launch_ = nullptr;",
        "  LegacyLaunchFn legacy_launch_ = nullptr;\n"
        "  DynamicLaunchFn dynamic_launch_ = nullptr;")
    prefill = "native/src/native_full_prefill.hip.cpp"
    sources[prefill] = full_prefill_overlay(read(prefill))
    if rectangular_ck:
        sources[header], sources[prefill] = rectangular_ck_overlay(
            sources[header], sources[prefill])
    # These two upstream enum-to-string functions have no media I/O dependency.
    media = read("native/src/native_media.cpp")
    names = media[media.index("std::string_view native_media_kind_name("):]
    sources["native/src/media_names.cpp"] = (
        '// SPDX-License-Identifier: Apache-2.0\n'
        '// Copyright 2026 Approaching AI Authors\n'
        '#include "aima/native_media.h"\nnamespace aima {\n' + names)
    return sources


def verify_import():
    raw = (UPSTREAM / "UPSTREAM.json").read_bytes()
    if digest(raw) != INVENTORY_SHA256:
        raise ValueError("Pinned upstream inventory changed")
    inventory = json.loads(raw)
    for item in inventory["files"]:
        data = (UPSTREAM / item["path"]).read_bytes()
        if len(data) != item["bytes"] or digest(data) != item["sha256"]:
            raise ValueError("Imported upstream file changed: " + item["path"])
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows-rectangular-ck", action="store_true",
                        help="Generate the optional Windows suffix ABI mapping; not model-qualified")
    parser.add_argument("--current-text-decode", action="store_true",
                        help="Use current upstream decode arithmetic for text; not model-qualified")
    args = parser.parse_args()
    inventory = verify_import()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit("Output already exists; preserve it and choose a fresh directory")
    overlays = make_overlays(rectangular_ck=args.windows_rectangular_ck,
                             current_text_decode=args.current_text_decode)
    out.mkdir(parents=True)
    adapted = []
    for relative, text in overlays.items():
        target = out / "overlay" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        original = UPSTREAM / relative
        adapted.append(dict(path=relative, output_sha256=digest(target.read_bytes()),
                            original_sha256=digest(original.read_bytes()) if original.exists() else None))
    def run_generator(name, arguments):
        result = subprocess.run([sys.executable, str(UPSTREAM / "scripts" / name), *map(str, arguments)],
                                check=True, capture_output=True, text=True, timeout=90)
        return json.loads(result.stdout)
    aot = UPSTREAM / "native/aot/gfx1151"
    contexts = [aot / (f"q{n}-output2" if n == 8192 else f"q{n}-output1") for n in CONTEXTS]
    manifests = [d / "manifest.json" for d in contexts + [aot / name for name in EXTRA_AOT]]
    aot_args = [argument for p in manifests for argument in ("--manifest", p)]
    aot_report = run_generator("generate-native-aot-registry.py", aot_args + [
        "--output-cpp", out / "aot_registry.cpp", "--output-plan", out / "aot_objects.tsv"])
    decode = aot / "q8192-output2"
    reports = [run_generator("generate-native-decode-registry.py", [
        "--schedule", decode / "decode-schedule.json", "--aot-manifest", decode / "manifest.json",
        "--output-cpp", out / "decode_registry.cpp"])]
    prefill_args = [argument for d in contexts for argument in (
        "--schedule", d / "prefill-schedule.json", "--aot-manifest", d / "manifest.json")]
    reports.append(run_generator("generate-native-decode-registry.py", [
        "--phase", "prefill", *prefill_args, "--output-cpp", out / "prefill_registry.cpp"]))
    frozen = aot / "q1024-text-v151"
    reports.append(run_generator("generate-native-decode-registry.py", [
        "--phase", "prefill", "--prefill-registry", "frozen-text",
        "--schedule", frozen / "prefill-schedule.json", "--aot-manifest", frozen / "manifest.json",
        "--output-cpp", out / "frozen_text_registry.cpp"]))
    images, symbols = [], set()
    assembly = ['// Generated COFF read-only image data. No executable host code.', '.section .rdata,"dr"']
    for line in (out / "aot_objects.tsv").read_text().splitlines():
        image_path, object_name, filename = line.split("\t")
        path = Path(image_path)
        symbol = "_binary_" + re.sub(r"[^A-Za-z0-9_]", "_", filename) + "_start"
        if symbol in symbols:
            raise ValueError("Duplicate binary image symbol")
        symbols.add(symbol)
        # Forward slashes and quoted .incbin paths are accepted by LLVM's
        # Windows assembler. Refuse characters that could become directives.
        incbin = path.as_posix()
        if any(c in incbin for c in '\"\r\n\t'):
            raise ValueError("Unrepresentable assembler input path")
        assembly.extend([".p2align 8", f".globl {symbol}", f"{symbol}:", f'.incbin "{incbin}"'])
        images.append(dict(path=path.relative_to(UPSTREAM).as_posix(), symbol=symbol,
                           bytes=path.stat().st_size, sha256=digest(path.read_bytes())))
    (out / "aot_images.S").write_text("\n".join(assembly) + "\n", encoding="utf-8", newline="\n")
    sources = []
    for p in sorted((UPSTREAM / "native/src").glob("*.cpp")):
        if p.name in EXCLUDED:
            continue
        relative = p.relative_to(UPSTREAM).as_posix()
        sources.append(str(out / "overlay" / relative if relative in overlays else p))
    sources.extend(str(out / "overlay" / p) for p in (
        "benchmarks/shape-lab/native/src/torch_owned_safetensors_loader.hip.cpp",
        "native/src/media_names.cpp"))
    sources.extend(str(out / p) for p in (
        "aot_registry.cpp", "decode_registry.cpp", "prefill_registry.cpp", "frozen_text_registry.cpp"))
    sources.append(str(ROOT / "native/linux_core_port/probe.cpp"))
    generated = [dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size,
                      sha256=digest(p.read_bytes())) for p in sorted(out.rglob("*")) if p.is_file()]
    report = dict(schema=1, upstream_revision=inventory["revision"],
        upstream_inventory_sha256=INVENTORY_SHA256, imported_files=len(inventory["files"]),
        imported_bytes=sum(p["bytes"] for p in inventory["files"]),
        overlays=adapted, sources=sources, images=images, generated=generated,
        aot_registry=aot_report, schedule_registries=reports,
        image_bytes=sum(x["bytes"] for x in images),
        vision_image=str(aot / "vision-attention-v0.3.0/kernels/d09fefdcb1ddb6cb-_fwd_kernel.hsaco"),
        windows_build_qualified=False, model_correctness_qualified=False, performance_qualified=False)
    if args.windows_rectangular_ck:
        adapter = ROOT / "native/linux_core_port/ck_suffix_adapter.h"
        report["optional_adaptations"] = dict(windows_rectangular_ck=True,
            adapter_path=adapter.relative_to(ROOT).as_posix(),
            adapter_sha256=digest(adapter.read_bytes()),
            maximum_suffix_queries=8192, maximum_total_kv_tokens=262144)
    if args.current_text_decode:
        report.setdefault("optional_adaptations", {})["current_text_decode"] = dict(
            arithmetic="upstream current-vLLM decode",
            rotary_positions="ordinary text position, existing M-RoPE plan when supplied",
            rotary_cache="BF16 rounded", linear_state="in place; no historical ping-pong swaps",
            prefill_changed=False, model_qualified=False)
    (out / "prepare.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("upstream_revision", "imported_files", "imported_bytes", "image_bytes")}
                     | dict(images=len(images), compilation_units=len(sources), overlays=len(adapted))))


if __name__ == "__main__":
    main()
