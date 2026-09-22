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


def gb10_gdn_overlays(linear, prefill):
    linear = replace(linear, '#include "gb10_convolution.h"',
                     '#include "gb10_convolution.h"\n#include "gb10_gdn.h"')
    linear = replace(linear,
        "  launch_packed_recurrent(layer_index, workspace, invocations, executor, stream);",
        "  aima_port::gb10_decode_gdn(layer_index, projected_qkv, a_projection,\n"
        "      b_projection, recurrent_output, recurrent_state, stream);\n"
        "  metrics.native_pointwise_launches += 3;")
    linear = replace(linear, "  metrics.aot_launches += 2;\n  launch_bf16_wvsplitk(",
                     "  metrics.aot_launches += 1;\n  launch_bf16_wvsplitk(")
    prefill = replace(prefill, '#include "gb10_convolution.h"',
                      '#include "gb10_convolution.h"\n#include "gb10_gdn.h"')
    prefill = replace(prefill,
        "  const auto started = std::chrono::steady_clock::now();",
        "  // This optional experiment owns complete q8192 GDN arithmetic.\n"
        "  // Imported intermediate-AOT observations/checkpoints no longer\n"
        "  // describe this provider and must not read its retired scratch.\n"
        "  if (!q8192_schedule || tokens != 8192 || comparison_tokens != tokens ||\n"
        "      options.seed_layer_input || options.collect_oracle_comparisons ||\n"
        "      !options.checkpoints.empty() || !fixture.empty() ||\n"
        "      !boundary_fixture.empty() || !tail_fixture.empty() || !sequence_fixture.empty())\n"
        "    throw std::invalid_argument(\"GB10 GDN experiment requires unpadded q8192 without AOT fixtures\");\n"
        "  const auto started = std::chrono::steady_clock::now();")
    begin = prefill.index("  launch_attention_aot(2);")
    end = prefill.index("  launch_attention_aot(8);", begin) + len("  launch_attention_aot(8);")
    prefill = replace(prefill, prefill[begin:end],
        "  aima_port::gb10_prefill_gdn(options.layer_index,\n"
        "      invocations.tensor_pointer(base + 1, \"o_ptr\"),\n"
        "      a, b, core, final_state, tokens, options.has_initial_state);\n"
        "  // Two conversion kernels surround one existing FLA provider call.\n"
        "  result.layer.native_pointwise_launches += 2;")
    prefill = replace(prefill, "  if (q8192_schedule) {\n    aima_port::gb10_prefill_convolution(",
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-input-norm-sampled\", h1, 2048, tokens);\n"
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-qkv-sampled\", qkv, 8192, tokens);\n"
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-z-sampled\", z, 4096, tokens);\n"
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-a-sampled\", a, 32, tokens);\n"
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-b-sampled\", b, 32, tokens);\n"
        "  if (q8192_schedule) {\n    aima_port::gb10_prefill_convolution(")
    prefill = replace(prefill,
        "  output_plan.launch(gated, output_weight.device_pointer, attention_output);",
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-gated-sampled\", gated, 4096, tokens);\n"
        "  output_plan.launch(gated, output_weight.device_pointer, attention_output);\n"
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-attention-out-sampled\", attention_output, 2048, tokens);",
        count=2)
    prefill = replace(prefill, "      (q8192_schedule ? 1 : 0);",
                      "      (q8192_schedule ? 8 : 0);")
    return linear, prefill


def gb10_projection_overlays(sources, read):
    engine = "native/src/native_resident_engine.hip.cpp"
    sources[engine] = replace(sources[engine], '#include "aima/native_resident_engine.h"',
        '#include "aima/native_resident_engine.h"\n#include "gb10_projection.h"')
    sources[engine] = replace(sources[engine],
        "    const NativeDecodePrepareMetrics prepared =",
        "    aima_port::set_gb10_decode_token(metrics.output_token_ids.back());\n"
        "    const NativeDecodePrepareMetrics prepared =")
    linear = "native/src/native_linear_layer.hip.cpp"
    sources[linear] = replace(sources[linear], '#include "gb10_gdn.h"',
        '#include "gb10_gdn.h"\n#include "gb10_projection.h"')
    sources[linear] = replace(sources[linear],
        "    launch_prefill_rmsnorm_2048(\n"
        "        input.device_pointer, input_norm_weight.device_pointer, input_norm, 1,\n"
        "        stream);",
        "    if (layer_index == 0) {\n"
        "      aima_port::gb10_embedding_norm(input.device_pointer,\n"
        "          input_norm_weight.device_pointer, input_norm, 1, stream);\n"
        "    } else {\n"
        "      launch_prefill_rmsnorm_2048(input.device_pointer,\n"
        "          input_norm_weight.device_pointer, input_norm, 1, stream);\n"
        "    }")
    prefill = "native/src/native_linear_prefill.hip.cpp"
    sources[prefill] = replace(sources[prefill], '#include "gb10_gdn.h"',
        '#include "gb10_gdn.h"\n#include "gb10_projection.h"')
    sources[prefill] = replace(sources[prefill],
        "  if (use_vl_rmsnorm) {\n    launch_prefill_rmsnorm_2048(",
        "  if (options.layer_index == 0) {\n"
        "    aima_port::gb10_embedding_norm(x, input_norm_weight.device_pointer, h1, tokens);\n"
        "    ++result.layer.native_pointwise_launches;\n"
        "  } else if (use_vl_rmsnorm) {\n    launch_prefill_rmsnorm_2048(")
    sources[prefill] = replace(sources[prefill],
        "      (q8192_schedule ? 8 : 0);",
        "      (q8192_schedule ? 8 : 0) -\n"
        "      (options.layer_index == 0 && !use_vl_rmsnorm ? 1 : 0);")
    path = "native/src/bf16_wvsplitk.hip.cpp"
    sources[path] = replace(read(path), '#include "aima/bf16_wvsplitk.h"',
        '#include "aima/bf16_wvsplitk.h"\n#include "gb10_projection.h"')
    sources[path] = replace(sources[path],
        '    throw std::invalid_argument("BF16 wvSplitK dimensions exceed int32");\n'
        "  }\n  const int wave_limit = small_wave_limit(m, k);",
        '    throw std::invalid_argument("BF16 wvSplitK dimensions exceed int32");\n'
        "  }\n  aima_port::gb10_projection(weight_mk, activation_1k, bias_m, output_1m, m, k, stream);\n"
        "  return;\n  const int wave_limit = small_wave_limit(m, k);")
    sources[path] = replace(sources[path],
        "  const int active_waves = minimum_divisor(\n"
        "      static_cast<int>(total_rows), cu_count * kYTile, kGroupedWaves);",
        "  aima_port::gb10_projection_group(projections, projection_count, activation_1k, k, stream);\n"
        "  return;\n  const int active_waves = minimum_divisor(\n"
        "      static_cast<int>(total_rows), cu_count * kYTile, kGroupedWaves);")


def make_overlays(*, rectangular_ck=False, current_text_decode=False,
                  gb10_convolution=False, gb10_gdn=False, gb10_projections=False):
    if gb10_convolution and not current_text_decode:
        raise ValueError("GB10 convolution requires current text decode ownership")
    if gb10_gdn and not gb10_convolution:
        raise ValueError("GB10 GDN requires GB10 convolution and current text decode")
    if gb10_projections and not gb10_gdn:
        raise ValueError("GB10 projections require the GB10 GDN experiment")
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
    if gb10_convolution:
        linear = "native/src/native_linear_layer.hip.cpp"
        sources[linear] = replace(read(linear),
            '#include "aima/native_linear_layer.h"',
            '#include "aima/native_linear_layer.h"\n#include "gb10_convolution.h"')
        sources[linear] = replace(sources[linear],
            "  launch_current_causal_conv(projected_qkv, conv_weight.device_pointer,\n"
            "                             conv_state_before, direct_conv_state_index,\n"
            "                             executor, stream);",
            "  aima_port::gb10_decode_convolution(\n"
            "      projected_qkv, conv_weight.device_pointer, conv_state_before, stream);\n"
            "  ++metrics.native_pointwise_launches;")
        sources[linear] = replace(sources[linear],
            "  metrics.aot_launches += 3;", "  metrics.aot_launches += 2;")
        linear_prefill = "native/src/native_linear_prefill.hip.cpp"
        sources[linear_prefill] = replace(read(linear_prefill),
            '#include "aima/native_linear_prefill.h"',
            '#include "aima/native_linear_prefill.h"\n#include "gb10_convolution.h"')
        sources[linear_prefill] = replace(sources[linear_prefill],
            "  launch_attention_aot(1);",
            "  if (q8192_schedule) {\n"
            "    aima_port::gb10_prefill_convolution(\n"
            "        qkv, invocations.tensor_pointer(base + 1, \"w_ptr\"),\n"
            "        invocations.tensor_pointer(base + 1, \"initial_states_ptr\"),\n"
            "        invocations.tensor_pointer(base + 1, \"o_ptr\"),\n"
            "        tokens, options.has_initial_state);\n"
            "    result.layer.native_pointwise_launches += 2;\n"
            "  } else {\n"
            "    launch_attention_aot(1);\n"
            "  }")
        sources[linear_prefill] = replace(sources[linear_prefill],
            "      attention_launches - (use_vl_rmsnorm ? 2 : 0);",
            "      attention_launches - (use_vl_rmsnorm ? 2 : 0) -\n"
            "      (q8192_schedule ? 1 : 0);")
        if gb10_gdn:
            sources[linear], sources[linear_prefill] = gb10_gdn_overlays(
                sources[linear], sources[linear_prefill])
    if gb10_projections:
        gb10_projection_overlays(sources, read)
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
    parser.add_argument("--gb10-convolution", action="store_true",
                        help="Use RNE BF16 convolution products and the qualified SiLU table")
    parser.add_argument("--gb10-gdn", action="store_true",
                        help="Use existing Windows FLA and original GB10 Q2 decode arithmetic")
    parser.add_argument("--gb10-projections", action="store_true",
                        help="Use SM121 decode projections and full-vocabulary embedding RMS scales")
    args = parser.parse_args()
    inventory = verify_import()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit("Output already exists; preserve it and choose a fresh directory")
    overlays = make_overlays(rectangular_ck=args.windows_rectangular_ck,
                             current_text_decode=args.current_text_decode,
                             gb10_convolution=args.gb10_convolution, gb10_gdn=args.gb10_gdn,
                             gb10_projections=args.gb10_projections)
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
    if args.gb10_gdn:
        sources.append(str(ROOT / "native/linux_core_port/gb10_gdn.hip.cpp"))
    if args.gb10_projections:
        sources.append(str(ROOT / "native/linux_core_port/gb10_projection.hip.cpp"))
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
    if args.gb10_convolution:
        report["optional_adaptations"]["gb10_convolution"] = dict(
            product_rounding="BF16 round-to-nearest ties-to-even",
            accumulation="sequential FP32", prefill_tokens=8192, decode_tokens=1,
            prefill_changed=True,
            silu_table_sha256="673f8dd1280700578c1e8743afd2e3b4da134b1fbd463c890527e1c4d9f796b8",
            model_qualified=False)
    if args.gb10_gdn:
        report["optional_adaptations"]["gb10_gdn"] = dict(
            prefill="existing Windows FLA provider 1d11bf7 with exact q8192 arithmetic",
            decode="original GB10 Q2 recurrence applied to one accepted token",
            state_layout="value-head, value, key; FP32 and in place",
            decode_beta="FP32", prefill_beta="BF16",
            prefill_tokens=8192, decode_tokens=1, intermediate_aot_observations=False,
            provider_calls_per_linear_prefill=1, model_qualified=False)
    if args.gb10_projections:
        report["optional_adaptations"]["gb10_projections"] = dict(
            decode="existing Windows K16 width-26 SM121 projection arithmetic",
            scope="singleton wvSplitK call sites, including grouped projections",
            embedding_norm="live token IDs select full-vocabulary model inverse-RMS scales",
            embedding_table_bytes=993280, embedding_device_bytes=1026048,
            embedding_table_sha256="f4e37f759c586bfc8fcc4d74cefdd89235f0f0c0c90cd286147e331e87509e67",
            prefill_dense_changed=False, model_qualified=False)
    (out / "prepare.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("upstream_revision", "imported_files", "imported_bytes", "image_bytes")}
                     | dict(images=len(images), compilation_units=len(sources), overlays=len(adapted))))


if __name__ == "__main__":
    main()
